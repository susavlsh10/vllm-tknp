#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Identical-arrival NDJSON client for real replicas and P/D experiments."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time
import uuid


def make_workload(count, lengths, output_lengths, rate, seed, patterns=None, bos_token_id=128000):
    rng = random.Random(seed)
    arrival_rng = random.Random(seed)
    arrival = 0.0
    rows = []
    for index in range(count):
        if index and rate > 0:
            # Keep paired arrival traces unchanged when context lengths vary.
            arrival += arrival_rng.expovariate(rate)
        length = lengths[index % len(lengths)]
        # Valid, deliberately varied token prompts; not linguistic data.
        tokens = [bos_token_id] + [rng.randrange(300, 30000) for _ in range(length - 1)]
        if patterns:
            prefix, suffixes = patterns
            suffix = suffixes[index % len(suffixes)]
            if len(suffix) + 1 >= length:
                raise ValueError("Natural prompt length is too short for the question")
            body_length = length - 1 - len(suffix)
            tokens = [bos_token_id] + (prefix * ((body_length + len(prefix) - 1) // len(prefix)))[:body_length] + suffix
        rows.append({"request_id": f"request-{index}", "arrival_s": arrival,
                     "prompt_token_ids": tokens,
                     "max_tokens": output_lengths[index % len(output_lengths)]})
    return rows


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def validate_staged_residency(workload, health, routing):
    """Conservative resident-only preflight; a cache barrier cannot queue waves."""
    from vllm.v1.core.sched.tknp_placement import PlacementRequest, place_fifo
    endpoints = [[] for _ in health]
    loads = [0] * len(health)
    for index, req in enumerate(workload):
        owner = min(range(len(health)), key=lambda i: (loads[i], i)) if routing == "least-work" else index % len(health)
        endpoints[owner].append(req)
        loads[owner] += len(req["prompt_token_ids"]) + req["max_tokens"]
    for rows, endpoint in zip(endpoints, health):
        if endpoint["config"].get("pp", 1) > 1:
            raise ValueError("Resident pause/resume has not been validated with asynchronous PP batch queues")
        limits = endpoint["resident_limits"]
        requests = [PlacementRequest(r["request_id"], len(r["prompt_token_ids"]), r["max_tokens"]) for r in rows]
        # Token-granular check is conservative up to one allocator block per
        # request, reserved below. Owner assignment mirrors the cost planner.
        block = endpoint["reservation_block_tokens"]
        free = [max(0, n["capacity_tokens"] // block - len(rows)) for n in limits]
        assignments = place_fifo(requests, free, block,
                                 request_limits=[n["request_slots"] for n in limits],
                                 policy=endpoint["config"].get("placement", "cost")
                                 if endpoint["config"].get("placement") != "legacy" else "cost")
        if len(assignments) != len(rows):
            raise ValueError("Staged workload cannot all fit resident; use ordinary queued E2E mode")


def summarize(rows, duration, ttft_slo, tpot_slo):
    completed = [r for r in rows if not r.get("error") and r.get("output_tokens") == r["requested_tokens"]]
    good = [r for r in completed if r["ttft_s"] <= ttft_slo and
            (r["tpot_s"] is None or r["tpot_s"] <= tpot_slo)]
    result = {"submitted": len(rows), "completed": len(completed), "failed": len(rows) - len(completed),
              "duration_s": duration, "output_tokens": sum(r["output_tokens"] for r in completed),
              "system_output_tps": sum(r["output_tokens"] for r in completed) / duration,
              "request_goodput_rps": len(good) / duration,
              "token_goodput_tps": sum(r["output_tokens"] for r in good) / duration,
              "slo": {"ttft_s": ttft_slo, "tpot_s": tpot_slo,
                      "ttft_basis": "intended arrival", "tpot_statistic": "per-request mean ITL"}}
    for key in ("ttft_s", "tpot_s", "e2e_s", "prefill_s", "dispatch_lateness_s"):
        vals = [r[key] for r in completed if r.get(key) is not None]
        result[key] = {f"p{int(q * 100)}": percentile(vals, q) for q in (0.5, 0.95, 0.99)}
    gaps = [gap for row in completed for gap in row.get("intertoken_s", [])]
    result["intertoken_s"] = {f"p{int(q * 100)}": percentile(gaps, q) for q in (0.5, 0.95, 0.99)}
    return result


async def collect(args):
    import aiohttp
    data = args.workload.read_bytes()
    workload = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not workload or len({r["request_id"] for r in workload}) != len(workload):
        raise ValueError("Workload needs unique IDs and at least one request")
    for req in workload:
        if req["arrival_s"] < 0 or not req["prompt_token_ids"] or req["max_tokens"] <= 0:
            raise ValueError("Invalid request workload")
    args.output.mkdir(parents=True, exist_ok=False)
    pending = [0] * len(args.decode_urls)
    prefix = uuid.uuid4().hex
    t0 = time.perf_counter()
    rows = []
    staged = bool(getattr(args, "staged_decode", False))
    admission_gated=bool(getattr(args,"admission_gated_decode",False))
    if admission_gated and (staged or args.prefill_urls or any(r["arrival_s"] for r in workload)):
        raise ValueError("Admission-gated decode requires a cached synthetic zero-arrival burst")
    admission_release_time=None
    release_time = None
    ready_ids = [set() for _ in args.decode_urls]
    health = []
    prefill_barrier = bool(getattr(args, "prefill_barrier", False))
    prefill_ready = set()
    prefill_release = asyncio.Event()
    decode_wave_start = None
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    # Prefill can outlast Uvicorn's five-second HTTP keepalive. Reusing the
    # initial decode /health connection at that boundary can disconnect the
    # first POST before admission. One connection per stream avoids ambiguous
    # retries of stateful generation requests; token streaming is unchanged.
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
        capacities = []
        for url in args.decode_urls:
            async with session.get(url.rstrip("/") + "/health") as response:
                response.raise_for_status()
                endpoint = await response.json()
                health.append(endpoint)
                capacities.append(endpoint.get("admission_max_tokens"))
        if staged:
            validate_staged_residency(workload, health, args.routing)
        if prefill_barrier:
            producer_health = []
            for url in args.prefill_urls:
                async with session.get(url.rstrip("/") + "/health") as response:
                    response.raise_for_status()
                    producer_health.append(await response.json())
            if not producer_health:
                raise ValueError("A real prefill barrier needs producer endpoints")
            validate_staged_residency([dict(r, max_tokens=1) for r in workload], producer_health, "round-robin")
        t0 = time.perf_counter()
        async def stream(url, body):
            async with session.post(url.rstrip("/") + "/generate", json=body) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
                async for line in response.content:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    yield event

        async def one(index, request, phase="measurement"):
            await asyncio.sleep(max(0, t0 + request["arrival_s"] - time.perf_counter()))
            sent = time.perf_counter()
            body = {k: v for k, v in request.items() if k != "arrival_s"}
            body["request_id"] = prefix + "-" + phase + "-" + request["request_id"]
            if staged and phase == "measurement":
                if request["max_tokens"] < 2:
                    raise ValueError("Staged measurement needs one priming token and subsequent decode")
                body["pause_after_tokens"] = 1
            result = {"request_id": request["request_id"], "arrival_s": request["arrival_s"],
                      "server_request_id": body["request_id"],
                      "requested_tokens": request["max_tokens"], "prompt_tokens": len(request["prompt_token_ids"]),
                      "dispatch_lateness_s": sent - (t0 + request["arrival_s"]),
                      "prefill_s": None, "token_ids": [], "token_times_s": []}
            result["phase"] = phase
            # Real independent endpoints, not one replica multiplied by N.
            reservation = len(request["prompt_token_ids"]) + request["max_tokens"]
            feasible = [i for i, limit in enumerate(capacities) if limit is None or reservation <= limit]
            owner = (min(feasible, key=lambda i: (pending[i], i)) if args.routing == "least-work"
                     else feasible[index % len(feasible)]) if feasible else index % len(pending)
            pending[owner] += reservation
            result["decode_replica"] = owner
            try:
                if capacities[owner] and reservation > capacities[owner]:
                    raise RuntimeError("Request lifetime exceeds this decode endpoint's empty KV capacity")
                if args.prefill_urls:
                    pbody = dict(body, max_tokens=1,
                                 kv_transfer_params={"do_remote_decode": True})
                    pbody.pop("pause_after_tokens", None)
                    transfer = None
                    async for event in stream(args.prefill_urls[index % len(args.prefill_urls)], pbody):
                        if event.get("kv_transfer_params"):
                            transfer = event["kv_transfer_params"]
                    result["prefill_s"] = time.perf_counter() - sent
                    if not transfer or not transfer.get("remote_block_ids"):
                        raise RuntimeError("Prefill did not export real KV blocks")
                    body["kv_transfer_params"] = transfer
                    if prefill_barrier and phase == "measurement":
                        prefill_ready.add(body["request_id"])
                        await prefill_release.wait()
                terminal = False
                async for event in stream(args.decode_urls[owner], body):
                    now = time.perf_counter()
                    tokens = event.get("token_ids", [])
                    result["token_ids"].extend(tokens)
                    result["token_times_s"].extend([now - sent] * len(tokens))
                    if staged and phase == "measurement" and result["token_ids"]:
                        ready_ids[owner].add(body["request_id"])
                    terminal |= event.get("finished", False)
                if not terminal:
                    raise RuntimeError("Stream ended without a terminal event")
                if len(result["token_ids"]) != request["max_tokens"]:
                    raise RuntimeError("Output token count mismatch")
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                pending[owner] -= reservation
            result["e2e_from_dispatch_s"] = time.perf_counter() - sent
            result["e2e_s"] = result["e2e_from_dispatch_s"] + result["dispatch_lateness_s"]
            result["output_tokens"] = len(result["token_ids"])
            times = result["token_times_s"]
            result["ttft_from_dispatch_s"] = times[0] if times else None
            result["ttft_s"] = times[0] + result["dispatch_lateness_s"] if times else None
            result["tpot_s"] = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
            result["intertoken_s"] = [b - a for a, b in zip(times, times[1:])]
            rows.append(result)

        warmup_count = args.warmup_requests
        if warmup_count:
            warmup = [dict(workload[index % len(workload)], request_id=f"warmup-{index}", arrival_s=0)
                      for index in range(warmup_count)]
            if getattr(args, "warmup_output_tokens", None):
                warmup = [dict(request, max_tokens=min(request["max_tokens"], args.warmup_output_tokens))
                          for request in warmup]
            await asyncio.gather(*(one(i, request, "warmup") for i, request in enumerate(warmup)))
            (args.output / "warmup_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            if any(r.get("error") for r in rows):
                raise RuntimeError("Warmup failed; measurement was not started")
            rows.clear()
            t0 = time.perf_counter()
        async def snapshots():
            async def snapshot(url):
                async with session.get(url.rstrip("/")+"/health") as response:
                    response.raise_for_status();return await response.json()
            return await asyncio.gather(*(snapshot(url) for url in args.decode_urls))
        async def gate_control(action):
            async def send(url):
                async with session.post(url.rstrip("/")+"/decode-gate",json=dict(action=action,key=prefix)) as response:
                    response.raise_for_status()
            await asyncio.gather(*(send(url) for url in args.decode_urls))
            deadline=time.perf_counter()+args.timeout
            while True:
                state=await snapshots()
                if all(s.get("last_decode_gate")==prefix+":"+action and not s.get("decode_gate_pending") for s in state):
                    return state
                if time.perf_counter()>deadline:raise TimeoutError("Decode gate control timed out")
                await asyncio.sleep(.005)
        if admission_gated:
            deadline=time.perf_counter()+args.timeout
            while not all(s.get("sweep_drained") for s in await snapshots()):
                if time.perf_counter()>deadline:raise TimeoutError("Warmup did not drain before decode gating")
                await asyncio.sleep(.005)
            await gate_control("hold")
            t0=time.perf_counter()
        tasks = [asyncio.create_task(one(i, request)) for i, request in enumerate(workload)]
        if admission_gated:
            try:
                deadline=time.perf_counter()+args.timeout
                while True:
                    state=await snapshots()
                    if rows:raise RuntimeError("A request failed or produced output before gated release")
                    if sum(s["outstanding_requests"] for s in state)==len(workload):break
                    if time.perf_counter()>deadline:raise TimeoutError("CPU request-admission barrier timed out")
                    await asyncio.sleep(.005)
                if any(not s["decode_gate_held"] or s["kv_free_blocks"]!=s["decode_gate_initial_free_blocks"] for s in state):
                    raise RuntimeError("Request KV was allocated before the decode timer")
                admission_release_time=time.perf_counter()
                await gate_control("release")
            except BaseException:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                await gate_control("release")
                raise
        if prefill_barrier:
            deadline = time.perf_counter() + args.timeout
            while len(prefill_ready) < len(workload):
                if any(row.get("error") for row in rows) or any(t.done() and t.exception() for t in tasks):
                    (args.output / "prefill_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                    raise RuntimeError("Prefill barrier failed; no wave timing collected")
                if time.perf_counter() > deadline:
                    raise TimeoutError("Producer export-capacity barrier timed out")
                await asyncio.sleep(0.005)
            decode_wave_start = time.perf_counter()
            prefill_release.set()
        if staged:
            deadline = time.perf_counter() + args.timeout
            while sum(map(len, ready_ids)) < len(workload):
                if any(task.done() and task.exception() for task in tasks):
                    raise RuntimeError("Staging task failed")
                if any(row.get("error") for row in rows):
                    (args.output / "staging_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                    raise RuntimeError("Staging request failed; no decode timing collected")
                if time.perf_counter() > deadline:
                    raise TimeoutError("Resident staging barrier timed out")
                await asyncio.sleep(0.005)
            for url, ids in zip(args.decode_urls, ready_ids):
                if not ids:
                    continue
                async with session.get(url.rstrip("/") + "/health") as response:
                    snapshot = await response.json()
                    if not ids.issubset(set(snapshot["paused_requests"])):
                        raise RuntimeError("First tokens observed without a resident pause barrier")
            release_time = time.perf_counter()
            async def release(url, ids):
                if ids:
                    async with session.post(url.rstrip("/") + "/release", json={"request_ids": sorted(ids)}) as response:
                        response.raise_for_status()
            await asyncio.gather(*(release(url, ids) for url, ids in zip(args.decode_urls, ready_ids)))
        await asyncio.gather(*tasks)
    finished_time = time.perf_counter()
    duration = finished_time - t0
    rows.sort(key=lambda r: r["request_id"])
    (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows, duration, args.ttft_slo, args.tpot_slo)
    summary.update(workload_sha256=hashlib.sha256(data).hexdigest(),
                   decode_urls=args.decode_urls, prefill_urls=args.prefill_urls,
                   routing=args.routing, real_kv_handoff=bool(args.prefill_urls),
                   warmup_requests=args.warmup_requests,
                   measurement="client wall time; prefill, handoff, queuing, drain included")
    if admission_gated:
        preparation=admission_release_time-t0
        decode_seconds=finished_time-admission_release_time
        release_ttft=[r["arrival_s"]+r["dispatch_lateness_s"]+r["token_times_s"][0]-preparation
                      for r in rows if not r.get("error")]
        summary["admission_gated_decode"]=dict(seconds=decode_seconds,gate_key=prefix,
            output_tokens=summary["output_tokens"],output_tps=summary["output_tokens"]/decode_seconds,
            preparation_seconds=preparation,cpu_admitted_requests=len(workload),
            ttft_from_release_s={f"p{int(q*100)}":percentile(release_ttft,q) for q in (.5,.95,.99)},
            scope="All CPU requests queued before timer; no request KV allocated before release; includes KV admission/waves, decode, streaming and drain; excludes HTTP body preparation and preinitialized random-KV fill; NOT prefill or online arrivals")
        summary["measurement"]="Use admission_gated_decode for cached decode batch makespan; ordinary TTFT fields include intentional preparation hold"
    if staged:
        decode_seconds = finished_time - release_time
        summary["staged_decode"] = {
            "scope": "same live engines; one priming token/request excluded; resident-only, NOT queueing/E2E speedup",
            "seconds": decode_seconds,
            "output_tokens": sum(max(0, r["output_tokens"] - 1) for r in rows if not r.get("error")),
            "output_tps": sum(max(0, r["output_tokens"] - 1) for r in rows if not r.get("error")) / decode_seconds}
        for row in rows:
            times = row["token_times_s"]
            row["steady_intertoken_s"] = [b - a for a, b in zip(times[1:], times[2:])]
        summary["staged_decode"]["steady_intertoken_s"] = {
            f"p{int(q * 100)}": percentile([v for r in rows for v in r["steady_intertoken_s"]], q)
            for q in (0.5, 0.95, 0.99)}
        (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["measurement"] = "controlled resident-cache pause experiment; ordinary TPOT/SLO fields include the intentional pause; use staged_decode for steady timing"
    if prefill_barrier:
        summary["decode_wave"] = {
            "scope": "real KV exports prepared before synchronized decode admission; includes handoff, decoder queueing and drain, NOT normal online arrivals",
            "seconds": finished_time - decode_wave_start,
            "output_tps": summary["output_tokens"] / (finished_time - decode_wave_start),
            "preparation_seconds": decode_wave_start - t0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--count", type=int, default=16)
    gen.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    gen.add_argument("--output-lengths", type=int, nargs="+", default=[8, 16, 24, 32])
    gen.add_argument("--rate", type=float, default=0, help="Poisson rps; zero is one burst")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--natural", action="store_true", help="Cached tokenizer, repeated background plus a question")
    gen.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    bench = sub.add_parser("run")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--decode-urls", nargs="+", required=True)
    bench.add_argument("--prefill-urls", nargs="+", default=[])
    bench.add_argument("--routing", choices=["least-work", "round-robin"], default="least-work")
    bench.add_argument("--warmup-requests", type=int, default=0,
                       help="Complete a separate burst before replay; use zero for cold-start studies")
    bench.add_argument("--warmup-output-tokens", type=int,
                       help="Optional bounded warmup output length; never changes measured requests")
    bench.add_argument("--staged-decode", action="store_true",
                       help="Prime one token and hold real or synthetic KV resident, then release together")
    bench.add_argument("--admission-gated-decode",action="store_true",
                       help="Queue all CPU requests, then time natural KV admission/queueing and decode")
    bench.add_argument("--prefill-barrier", action="store_true",
                       help="Prepare real KV exports before a decode burst; producer exports must fit")
    bench.add_argument("--timeout", type=float, default=600)
    bench.add_argument("--ttft-slo", type=float, default=10)
    bench.add_argument("--tpot-slo", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "generate":
        if args.count <= 0 or min(args.lengths + args.output_lengths) <= 0 or args.rate < 0:
            parser.error("Positive request counts and lengths, nonnegative rate required")
        patterns = None
        bos_token_id = 128000
        if args.natural:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            bos_token_id = tokenizer.bos_token_id
            if bos_token_id is None:
                parser.error("A model-specific BOS token is required for this controlled generator")
            prefix = tokenizer.encode("Computers use processors and memory to perform calculations. Networks connect machines and exchange information. ", add_special_tokens=False)
            suffixes = [tokenizer.encode("\nIgnore the background notes. " + question + "\nAnswer:", add_special_tokens=False)
                        for question in ["What is the capital of France?", "What is two plus two?",
                                         "Name the largest planet in the solar system.", "What is the chemical formula of water?"]]
            patterns = (prefix, suffixes)
        rows = make_workload(args.count, args.lengths, args.output_lengths, args.rate, args.seed, patterns, bos_token_id)
        with args.output.open("x") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
    else:
        if args.prefill_barrier and (args.staged_decode or not args.prefill_urls):
            parser.error("Use a prefill barrier only with real producers, separately from resident staging")
        if args.warmup_requests < 0 or args.timeout <= 0 or args.ttft_slo <= 0 or args.tpot_slo <= 0:
            parser.error("Nonnegative warmup count and positive timeout/SLOs required")
        asyncio.run(collect(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge all-rank PyTorch traces, retaining originals and host clock evidence.

No GPU imports or allocation. Clock probes use an existing Slurm allocation.
Offsets are bounded RTT midpoint estimates, not a hardware clock sync claim.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time


def clock_server():
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        received = time.time_ns()
        request = json.loads(line)
        print(json.dumps({'sequence': request['sequence'], 'received_ns': received,
                          'sent_ns': time.time_ns()}), flush=True)


def read_line(process, timeout=60):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError('Clock probe stopped responding')
        line = process.stdout.readline()
    if not line:
        raise RuntimeError('Clock probe exited without a response')
    return json.loads(line)


def calibrate_nodes(job, nodes, script, samples=25):
    """Remote minus controller wall clock; scheduling startup is untimed."""
    result = {'controller_time_ns': time.time_ns(), 'hosts': {},
              'method': 'minimum network RTT midpoint; clocks sampled before/after profiled engine',
              'caveat': 'RTT bounds assume no clock steps; sub-bound cross-host ordering is uncertain'}
    for host in nodes:
        command = ['srun', '--jobid=' + str(job), '--overlap', '--nodes=1', '--ntasks=1',
                   '--ntasks-per-node=1', '--cpus-per-task=1', '--cpu-bind=none',
                   '--mpi=none', '--nodelist=' + host, 'python3', '-u', str(script),
                   '--clock-server']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, bufsize=1)
        try:
            if not read_line(process).get('ready'):
                raise RuntimeError('Clock server did not initialize')
            points = []
            for sequence in range(samples):
                start = time.time_ns()
                process.stdin.write(json.dumps({'sequence': sequence}) + '\n')
                process.stdin.flush()
                reply = read_line(process)
                end = time.time_ns()
                if reply['sequence'] != sequence:
                    raise RuntimeError('Clock probe sequence mismatch')
                received, sent = reply['received_ns'], reply['sent_ns']
                network = end - start - (sent - received)
                if network < 0 or end < start or sent < received:
                    raise RuntimeError('Clock stepped backwards during calibration')
                points.append(dict(controller_mid_ns=(start+end)//2,
                    offset_ns=((received-start)+(sent-end))//2,
                    uncertainty_ns=(network+1)//2, round_trip_ns=end-start))
            process.stdin.close()
            if process.wait(timeout=30):
                raise RuntimeError('Clock server failed')
            best = min(points, key=lambda p: p['uncertainty_ns'])
            result['hosts'][host] = dict(best, samples=points)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    return result


def host_correction(host, epoch_ns, clocks):
    if not clocks:
        return 0, None
    host = host.split('.')[0]
    samples = sorted((c['hosts'][host] for c in clocks), key=lambda p: p['controller_mid_ns'])
    first, last = samples[0], samples[-1]
    # Convert the worker timestamp approximately to controller domain before
    # interpolating offset drift. Never change durations or force rank starts.
    relative = epoch_ns - first['offset_ns'] - first['controller_mid_ns']
    span = last['controller_mid_ns'] - first['controller_mid_ns']
    fraction = min(1.0, max(0.0, relative / span)) if span else 0.0
    correction = round(first['offset_ns'] + fraction*(last['offset_ns']-first['offset_ns']))
    return correction, max(first['uncertainty_ns'], last['uncertainty_ns'])


def read_trace(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def merge_traces(directory, output, expected_ranks, hosts=None, clocks=None):
    paths = sorted(Path(directory).rglob('*.pt.trace.json*'))
    if not paths:
        raise ValueError('No PyTorch trace files')
    hosts, clocks = hosts or {}, clocks or []
    ranks, events, rank_info = set(), [], []
    origin = None
    pid_map, process_names = {}, {}
    for path in paths:
        trace = read_trace(path)
        distributed = trace.get('distributedInfo', {})
        rank = distributed.get('rank')
        if not isinstance(rank, int) or rank in ranks:
            raise ValueError('Missing or duplicate distributed rank: ' + str(path))
        if distributed.get('world_size') != expected_ranks:
            raise ValueError('Unexpected distributed world size: ' + str(path))
        ranks.add(rank)
        host = hosts.get(rank, hosts.get(str(rank), 'unknown'))
        base = trace.get('baseTimeNanoseconds')
        if not isinstance(base, int):
            raise ValueError('Missing profiler wall-clock base')
        kernel_count, replay_count = 0, 0
        uncertainty = None
        for raw in trace['traceEvents']:
            event = copy.deepcopy(raw)
            if 'pid' in event:
                key = rank, str(event['pid'])
                if key not in pid_map:
                    pid_map[key] = len(pid_map) + 1
                event['pid'] = pid_map[key]
            if event.get('ph') == 'M' and event.get('name') == 'process_name':
                original = event.get('args', {}).get('name', '')
                process_names[event['pid']] = f'Rank {rank:02d} @ {host} — {original}'
                # Emit one clean metadata record for each process below.
                continue
            if 'ts' in event and event.get('ph') != 'M':
                absolute_ns = base + round(event['ts']*1000)
                correction, uncertainty = host_correction(host, absolute_ns, clocks)
                event['ts'] = absolute_ns - correction  # integers until common origin subtraction
                origin = event['ts'] if origin is None else min(origin, event['ts'])
            # Flow IDs are only process-local in independent PyTorch exports.
            for field in ('id', 'bind_id'):
                if field in event:
                    event[field] = f'r{rank}:{event[field]}'
            if 'id2' in event:
                event['id2'] = {k: f'r{rank}:{v}' for k, v in event['id2'].items()}
            if event.get('cat') == 'kernel':
                kernel_count += 1
            if 'cudagraphlaunch' in event.get('name', '').lower():
                replay_count += 1
            events.append(event)
        if not kernel_count or not replay_count:
            raise ValueError('Rank has no GPU kernels or graph launches: ' + str(rank))
        rank_info.append(dict(rank=rank, host=host, raw_trace=str(path),
            base_time_ns=base, kernel_events=kernel_count, cuda_graph_launches=replay_count,
            clock_uncertainty_ns=uncertainty))
    if ranks != set(range(expected_ranks)):
        raise ValueError('Incomplete ranks: ' + str(sorted(ranks)))
    if origin is None:
        raise ValueError('No timestamped trace events')
    metadata = []
    for (rank, original_pid), pid in pid_map.items():
        host = hosts.get(rank, hosts.get(str(rank), 'unknown'))
        metadata.extend([
            dict(ph='M', name='process_name', pid=pid, tid=0,
                 args=dict(name=process_names.get(pid, f'Rank {rank:02d} @ {host} — process {original_pid}'))),
            dict(ph='M', name='process_sort_index', pid=pid, tid=0,
                 args=dict(sort_index=rank*100000+pid))])
    for event in events:
        if 'ts' in event and event.get('ph') != 'M':
            event['ts'] = (event['ts'] - origin)/1000
    index = dict(passed=True, ranks=sorted(rank_info, key=lambda p: p['rank']),
        expected_ranks=expected_ranks, common_origin_epoch_ns=origin,
        time_alignment='wall-clock bases + measured host offset interpolation' if clocks
                       else 'wall-clock bases only; host synchronization not calibrated',
        clocks=clocks, performance_valid=False,
        caveats=['Profiler timing is not unprofiled throughput or SM utilization.',
                 'Cross-host ordering closer than calibration uncertainty is not established.',
                 'No artificial alignment of first kernels; original durations are unchanged.'])
    output = Path(output)
    merged = dict(schemaVersion=1, displayTimeUnit='ms', traceEvents=metadata+events,
                  tknp_trace_metadata=index)
    with output.open('x') as stream:
        json.dump(merged, stream, separators=(',', ':'))
    with output.open('rb') as source, gzip.open(str(output)+'.gz', 'xb') as target:
        import shutil
        shutil.copyfileobj(source, target)
    with output.with_suffix('.index.json').open('x') as stream:
        json.dump(index, stream, indent=2)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clock-server', action='store_true')
    p.add_argument('--traces', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--ranks', type=int)
    p.add_argument('--hosts', type=Path, help='JSON rank-to-host mapping')
    p.add_argument('--clocks', type=Path, nargs='*')
    args = p.parse_args()
    if args.clock_server:
        clock_server()
        return
    if not (args.traces and args.output and args.ranks):
        p.error('--traces, --output and --ranks required')
    result = merge_traces(args.traces, args.output, args.ranks,
        json.loads(args.hosts.read_text()) if args.hosts else {},
        [json.loads(c.read_text()) for c in args.clocks or []])
    print('MERGED_RANKS', len(result['ranks']), flush=True)


if __name__ == '__main__':
    main()

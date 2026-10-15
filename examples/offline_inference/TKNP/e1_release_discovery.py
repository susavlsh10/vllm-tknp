#!/usr/bin/env python3
"""Release one explicitly supplied discovery allocation after its valid controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--job", type=int, required=True)
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
methods = ["tknp", "tknp-tp2", "dp-native", "dp-lifetime", "dp-tp2", "tp", "pp"]
wanted = {"e2e-c65536-b25-o1025-" + m for m in methods}
deadline = time.monotonic() + 3 * 3600
while time.monotonic() < deadline:
    path = a.root / "e2e/suite.json"
    try:
        state = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(15)
        continue
    completed = {r["name"] for r in state["attempts"] if r["status"] == "completed"}
    if wanted <= completed:
        record = dict(job=a.job, time=time.time(), completed_controls=sorted(wanted),
                      reason="Prefill-bound 32K/64K discovery controls finished; prioritize balanced-prefill long-output matrix and four-node DCP study")
        (a.root / "discovery_release.json").write_text(json.dumps(record, indent=2) + "\n")
        subprocess.run(["scancel", str(a.job)], check=True)
        print("DISCOVERY_RELEASED", record, flush=True)
        break
    active = subprocess.check_output(["squeue", "-h", "-j", str(a.job), "-o", "%T"], text=True).strip()
    if not active:
        print("Allocation already ended; no action", flush=True)
        break
    time.sleep(15)

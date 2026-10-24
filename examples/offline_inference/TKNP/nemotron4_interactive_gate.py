#!/usr/bin/env python3
"""Wait for the existing interactive node, then run its fail-closed gate once.

Does not allocate or submit jobs. On failure preserve the attempt and stop;
interactive diagnostics/repairs must decide whether to resume the gate.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    previous=None
    while True:
        status=subprocess.check_output(['squeue','-h','-j',args.job,'-o','%T|%N'],text=True).strip()
        if status!=previous:print('INTERACTIVE_ALLOCATION',args.job,status,flush=True);previous=status
        if not status:raise RuntimeError('Existing interactive allocation ended; no replacement allocated')
        state,node=status.split('|',1)
        if state=='RUNNING':break
        if state!='PENDING':raise RuntimeError('Allocation not usable: '+status)
        time.sleep(15)
    if not node or ',' in node or '[' in node:
        raise RuntimeError('Expected exactly one concrete allocated node: '+node)
    subprocess.run([sys.executable,'-u',str(args.source/'examples/offline_inference/TKNP/nemotron4_gate.py'),
        '--source',str(args.source),'--parent',str(args.parent),'--output',str(args.output),
        '--job',args.job,'--node',node],check=True)


if __name__=='__main__':main()

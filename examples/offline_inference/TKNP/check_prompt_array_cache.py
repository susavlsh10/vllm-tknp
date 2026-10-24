#!/usr/bin/env python3
"""CPU-only prompt re-insertion check; run outside timed GPU measurements."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from vllm.v1.worker.gpu_input_batch import CachedRequestState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    requests=[CachedRequestState(str(i),[(i+j)%128000 for j in range(32768)],
        [],None,None,([],),32767,[7,9]) for i in range(17)]
    array=np.zeros((17,32770),dtype=np.int32)
    times={}
    for enabled in (False,True):
        start=time.perf_counter()
        for _ in range(128):
            for i,r in enumerate(requests):
                array[i,:32768]=r.prompt_token_ids_np if enabled else r.prompt_token_ids
                array[i,32768:]=r.output_token_ids
        times[str(enabled)]=(time.perf_counter()-start)/128
        for i,r in enumerate(requests):
            np.testing.assert_array_equal(array[i],r.prompt_token_ids+r.output_token_ids)
            assert not r.prompt_token_ids_np.flags.writeable
            assert not np.shares_memory(r.prompt_token_ids_np,array)
    result=dict(passed=True,request_count=17,prompt_length=32768,
        seconds_per_rotating_batch=times,speedup=times["False"]/times["True"],
        scope="CPU copy microbenchmark, not end-to-end throughput; immutable prompts, live output tokens")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()

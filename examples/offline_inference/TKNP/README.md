# Token Parallel Inference Benchmarks

This directory contains benchmarking scripts for measuring **decode performance** in vLLM with support for Token Parallelism (TKNP), Tensor Parallelism (TP), Pipeline Parallelism (PP), and Decode Context Parallelism (DCP).

---

## Overview

The benchmark script [`tknp_inference_benchmarks.py`](tknp_inference_benchmarks.py) measures **pure decode throughput and latency** — the time to generate new tokens after the KV cache has been pre-filled. Prefill is intentionally skipped by using the `DecodeBenchConnector`, which fills the KV cache with dummy constant values. This isolates decode performance from prefill costs.

---

## Prerequisites

- vLLM installed from this repository
- Access to the target model (Hugging Face hub or local path), or use `--load-format dummy` for random weights

---

## Running the Benchmarks

All commands must be launched with `torchrun`.

### Token Parallelism (TKNP)

Distributes the **KV cache and decode steps** across GPUs, keeping weights replicated. Each GPU holds a contiguous slice of the sequence's KV cache.

```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 1 \
    --token-parallel-size 4 \
    --batch-size 32 \
    --seq-length 32768
```

### Tensor Parallelism (TP)

Distributes **model weights** (attention heads and FFN layers) across GPUs.

```bash
torchrun --nproc-per-node=8 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 8 \
    --token-parallel-size 1 \
    --batch-size 32 \
    --seq-length 16384
```

### Pipeline Parallelism (PP)

Distributes **model layers** across GPUs in a pipeline.

```bash
torchrun --nproc-per-node=8 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --batch-size 32 \
    --seq-length 32768
```

### Decode Context Parallelism (DCP)

Splits the **KV cache attention computation** across GPUs during the decode step (ring attention).

```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 4 \
    --token-parallel-size 1 \
    --decode-context-parallel-size 2 \
    --batch-size 32 \
    --seq-length 32768
```

### Combined: TP + TKNP

```bash
torchrun --nproc-per-node=8 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 2 \
    --token-parallel-size 4 \
    --batch-size 64 \
    --seq-length 65536
```

---

## Systematic Data Collection

Use `--collect-data` to sweep over a set of predefined batch sizes and sequence lengths automatically. Configurations that exceed KV cache capacity are skipped.

```bash
torchrun --nproc-per-node=4 \
    examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 1 \
    --token-parallel-size 4 \
    --collect-data \
    --output-dir examples/offline_inference/TKNP/tknp_data
```

The default sweep covers:

| Batch Size | Sequence Lengths |
|---|---|
| 32 | 32768, 65536, 98304, 114688 |
| 64 | 32768, 65536, 98304, 114688 |
| 128 | 16384, 32768, 65536, 98304 |
| 256 | 16384, 32768, 65536, 98304 |
| 512 | 16384, 32768 |

To customize the sweep, edit the `batch_seq_configs` dictionary in `run_data_collection()`.

---

## Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Number of tensor parallel GPUs |
| `--pipeline-parallel-size` | `1` | Number of pipeline stages |
| `--data-parallel-size` | `1` | Number of data parallel replicas |
| `--token-parallel-size` | `1` | Number of token parallel GPUs (TKNP) |
| `--decode-context-parallel-size` | `1` | Number of context parallel GPUs for decode |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face model ID or local path |
| `--batch-size` | `8` | Number of concurrent requests |
| `--seq-length` | `128` | Prompt length in tokens |
| `--decode-tokens` | `1000` | Number of tokens to decode per request |
| `--load-format` | `dummy` | Weight loading format: `dummy` (random, fast) or `auto` (real weights) |
| `--skip-prefill` | `True` | Pre-fill KV cache with dummy values to benchmark decode-only |
| `--collect-data` | off | Run the full batch × seq_length sweep |
| `--output-dir` | `examples/offline_inference/TKNP/tknp_data` | Directory for CSV result files |
| `--print-outputs` | off | Print generated text (rank 0 only) |
| `--seed` | `1` | Random seed for reproducibility |

> **Note:** `--load-format dummy` uses randomly initialized weights and is the recommended default for benchmarking. Use `--load-format auto` to load real model weights.

---

## Supported Models

| Family | Models |
|---|---|
| **Llama 3** | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct` |
| **Qwen** | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-32B`, `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-72B-Instruct` |
| **Mistral / Ministral** | `ministral/Ministral-3b-instruct`, `mistralai/Devstral-Small-2-24B-Instruct-2512`, `mistralai/Mistral-Large-Instruct-557161` |

---

## Output Format

Results are appended to a CSV file in `--output-dir`. The filename encodes the configuration:

```
{ModelName}_{GPU}_{TP}_{PP}_{TKNP}_{DCP}.csv
```

Example: `Llama-3.1-8B_H100_TP_1_PP_1_TKNP_4_DCP_1.csv`

Each row contains:

```
batch_size, seq_length, decode_time_ms, decode_tokens, sys_decode_tps,
decode_tps_per_gpu, avg_decode_latency_ms, decode_tps_per_user
```

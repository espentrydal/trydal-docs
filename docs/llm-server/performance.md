# Performance Benchmarks

LLM inference benchmarks across ai-smil1/2 using DeepSeek V4 (DS4) on IBM POWER9 (ppc64le) nodes with 4× V100 32GB GPUs.

## Hardware

- **ai-smil1:** ppc64le, 4× V100 32GB (NVLink pairs)
- **ai-smil2:** ppc64le, 4× V100 32GB (NVLink pairs)

Both nodes have NVLink pairs GPU0-GPU1 and GPU2-GPU3; cross-pair GPU traffic is `SYS`.

---

## Current Production Performance

The best verified one-node decode result with all scratch-locality optimizations enabled:

| Node | Tokens | Generation Rate |
|------|--------|-----------------|
| ai-smil1 | 64 | 21.58 t/s |
| ai-smil1 | 200 | 21.48 t/s |
| ai-smil2 | 64 | 21.66 t/s |
| ai-smil2 | 200 | 21.47 t/s |

!!! tip "Production serving context"
    Production serving defaults to `DS4_CTX=131072` (128K context) to reduce VRAM pressure. Historical measurements below were mostly taken at 262K context unless otherwise noted — use them as kernel/runtime comparisons, not as a claim that 262K is the preferred production context.

Service-path checks (warm 200-token non-thinking chat requests) logged **12.5–12.6 tok/s** wall-clock on earlier builds, with server-side decode reporting **12.98–13.05 tok/s**. After the scratch-locality changes, server-side service path improved substantially.

---

## Build Configuration

### Build Command

```sh
DS4_CUDA_SPLIT_OUTPUT_HEAD=1
DS4_CUDA_DEVICES=0,1,2,3
DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=512
make cuda-v100  # includes -Xptxas=-maxrregcount=64
```

Use `scripts/server-ds4-flash-v100-1node.sh` for OpenAI/Anthropic-compatible serving, or `scripts/profile-ds4-flash-v100-1node.sh` for a reproducible decode profile run.

### Scratch Locality Defaults

These env vars keep per-layer scratch tensors on each layer's assigned GPU, which is the primary mechanism behind the 21+ tok/s result:

| Variable | Purpose |
|----------|---------|
| `DS4_CUDA_LOCAL_ROUTED_SCRATCH=1` | Routed-MoE scratch/output |
| `DS4_CUDA_LOCAL_ATTN_SCRATCH=1` | Attention heads, low-rank output, buffers |
| `DS4_CUDA_LOCAL_QKV_SCRATCH=1` | Q, K, V projections and pre-HC scratch |
| `DS4_CUDA_LOCAL_COMPRESSOR_SCRATCH=1` | Compressor KV/gate, indexer query/weights |
| `DS4_CUDA_LOCAL_HC_FFN_SCRATCH=1` | HC pre/post, FFN, shared expert, router |
| `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` | Separate shared-down + HC-post paths |

!!! note "Determinism verified"
    All locality changes preserved the existing 64-token and 200-token output hashes (`cmp` exit 0).

### V100 Register Cap

The V100 target uses `NVCC_PTXAS_FLAGS=-Xptxas=-maxrregcount=64`. This is V100-specific for occupancy/register-pressure; other CUDA targets keep plain NVCC flags.

The 64-register cap was chosen after a sweep of `maxrregcount` values in the 56–96 range (see [benchmark history](#benchmark-history)). ptxas verbose output confirmed **zero spills** in the active one-token MoE decode kernels under this cap.

### Topology-Aware Device Ordering

The 43 decode layers map contiguously as:

| GPU | Layers | Interconnect |
|-----|--------|-------------|
| GPU0 | 0–10 | NVLink → GPU1 |
| GPU1 | 11–21 | SYS → GPU2 |
| GPU2 | 22–31 | NVLink → GPU3 |
| GPU3 | 32–42 | — |

Two of three inter-GPU handoffs stay inside NVLink pairs; only the middle handoff crosses the slower `SYS` path. The equal split weights (`8.2,8.2,8.2,8.2`) ensure contiguous layer chunks.

!!! tip "Three-GPU layouts rejected"
    Running on 3 GPUs causes CUDA arena allocation failures for model/cache residency, then generates only ~6–7 t/s. All four V100s are necessary even though it leaves one cross-pair handoff.

---

## Benchmark History

This section documents the chronological improvement of direct decode throughput from the initial baseline (~11 tok/s) through successive kernel and configuration changes.

### Baseline and Initial Tuning (~11 t/s)

The initial decode path on this model/CUDA split:

```text
MTP disabled baseline:                 generation: 11.03 t/s
MTP draft=1:                           generation: 10.98 t/s
MTP draft=2:                           generation:  4.71 t/s
generic MoE down path default:          generation: 11.20 t/s
reference attention-output path default: generation: 11.48 t/s
fused attention-output opt-in:          generation: 11.24 t/s
```

!!! example "MTP not useful on this setup"
    The optional MTP GGUF is usable with `--mtp`, but on this one-node V100 setup it did not produce a decode-speed win. `--mtp-draft 2` also triggered CUDA allocation failures.

### Cached F16/cuBLAS for Q8 Projections (~12.6 t/s)

One-token decode now uses cached F16/cuBLAS for eligible Q8 projections by default:

```sh
DS4_CUDA_NO_Q8_F16_GEMV=1  # opt-out
```

This cut the profiled `q_path` to **0.187 ms/layer** and moved the 96-token check from **11.48 t/s** to **12.58 t/s**.

### Shared-Down/HC Fusion Opt-Out (~13.2 t/s)

Disabling the fused shared-down/HC path with `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1`:

| Node | Before | After |
|------|--------|-------|
| ai-smil1 | 12.64 t/s | 13.20 t/s |
| ai-smil2 | 12.55 t/s | 13.09 t/s |

The profile explains why: the default fused path measured `shared_down=0.175 ms/layer` + `ffn_hc_post=0.008 ms/layer`, while the opt-out measured `shared_down=0.068 ms/layer` + `ffn_hc_post=0.064 ms/layer`. The separate path is faster overall.

!!! tip "Service path verification"
    The same setting was verified through `ds4-server.service` on ai-smil2. Systemd process environment included the flag; warm 128K-context 200-token request logged 13.28 t/s (first chunk) and 12.99 t/s average.

### V100 Register Cap Sweep (~13.4 t/s)

A 2026-05-16 build-flag sweep found a clear decode gain by capping ptxas register allocation:

```text
maxrregcount=56  13.26 t/s
maxrregcount=60  13.37-13.49 t/s
maxrregcount=62  13.43 t/s
maxrregcount=64  13.40-13.43 t/s   ← chosen default
maxrregcount=68  13.35 t/s
maxrregcount=72  13.37 t/s
maxrregcount=80  13.33 t/s
maxrregcount=96  12.86 t/s
```

Follow-up 500-token swapped checks confirmed 64 as the safer default:

| Register cap | ai-smil1 (500 tok) | ai-smil2 (500 tok) |
|-------------|--------------------|--------------------|
| 60 | 13.44 t/s | 13.33 t/s |
| 64 | 13.46 t/s | 13.33 t/s |

Other compile-flag experiments (ptxas `-dlcm=ca`/`-dlcm=cg`, `--extra-device-vectorization`) showed no consistent improvement.

### 128-Row MoE Decode Down Kernel (~14.6 t/s)

The MoE decode down kernel was reworked through successive row shapes:

| Rows per block | Threads | MoE down | Routed MoE |
|----------------|---------|----------|------------|
| 32 (old default) | 256 | 0.318 ms/layer | — |
| 64 | 256 | 0.236 ms/layer | — |
| **128** | **1024** | **0.162 ms/layer** | **0.506 ms/layer** |

Direct 200-token checks: **14.61–14.66 t/s** on ai-smil1, **14.62–14.64 t/s** on ai-smil2. The 128-row shape is now the default.

```sh
DS4_CUDA_MOE_DOWN_ROWS64=1   # comparison only
DS4_CUDA_MOE_DOWN_ROWS32=1   # comparison only
```

### One-Token cuBLAS for Attention-Output A (~14.8 t/s)

The `attn_output_a` F16/cuBLAS path now runs for one-token decode (not just prefill/batch):

```sh
DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=1  # default threshold
```

200-token direct: **14.85 t/s** (ai-smil1), **14.79 t/s** (ai-smil2); 500-token: **14.75 t/s**. Output matched the old Q8 decode path exactly.

### 512-Thread MoE Gate/Up+Midq Kernel (~15.1 t/s)

The final step to 15 tok/s was increasing the fused MoE gate/up+midq decode kernel from 256 to 512 threads, keeping the same 256-row Q8_K output tile:

| Threads | Quarter-warps | Serial rows/quarter-warp | Throughput |
|---------|--------------|------------------------|------------|
| 256 (old) | 32 | 8 | — |
| **512** | **64** | **4** | **15.02–15.09 t/s** |

The profile:

```text
total   0.407 ms/layer
gateup  0.212 ms/layer
down    0.165 ms/layer
xq      0.016 ms/layer
sum     0.009 ms/layer
midq    0.002 ms/layer
sort    0.002 ms/layer
```

```sh
DS4_CUDA_MOE_GATEUP_THREADS256=1  # comparison only
```

After selected upstream runtime/server commits were cherry-picked into `ppc64le`, the 4-GPU direct benchmark still measured **15.06 t/s** on ai-smil1 and **15.05 t/s** on ai-smil2.

### Scratch Locality Steps (16.5 → 21.6 t/s)

Each locality env var was enabled incrementally, preserving deterministic output hashes at every step:

| Step | Env Vars Added | ai-smil1 (64 tok) | ai-smil1 (200 tok) |
|------|---------------|-------------------|--------------------|
| Baseline (after 512-thread gate/up) | — | — | ~15.1 t/s |
| + Routed-MoE local | `LOCAL_ROUTED_SCRATCH=1` | 16.56 t/s | 16.47 t/s |
| + Attention local | `LOCAL_ATTN_SCRATCH=1` | 16.64 t/s | 16.54 t/s |
| + Q/KV local | `LOCAL_QKV_SCRATCH=1` | 17.23 t/s | 17.26 t/s |
| + Compressor local | `LOCAL_COMPRESSOR_SCRATCH=1` | 17.83 t/s | 17.83 t/s |
| + HC/FFN/shared local | `LOCAL_HC_FFN_SCRATCH=1` | **21.58 t/s** | **21.48 t/s** |

ai-smil2 matched at **21.66 t/s** (64 tok) and **21.47 t/s** (200 tok).

---

## Remaining Bottlenecks

### Current Synchronized Profile

The profile after all locality defaults:

```text
routed_moe          0.282 ms/layer
attn_output         0.238 ms/layer
q_path              0.164 ms/layer
attn_hc_pre         0.102 ms/layer
compressor_indexer  0.096 ms/layer
shared_gate_up      0.080 ms/layer
ffn_hc_pre          0.069 ms/layer
attention           0.062 ms/layer
shared_down         0.055 ms/layer
router              0.052 ms/layer
```

The locality changes significantly cut host-device scratch effects:

| Bucket | Before locality | After locality |
|--------|----------------|----------------|
| `ffn_hc_pre` | 0.124 ms/layer | 0.069 ms/layer |
| `shared_gate_up` | 0.103 ms/layer | 0.080 ms/layer |
| `shared_down` | 0.069 ms/layer | 0.055 ms/layer |
| HC post stages | ~0.064 ms/layer | 0.021–0.025 ms/layer |

### Future Routes

The most promising source-level routes for further improvement:

1. **MoE gate/up+midq** — the largest remaining bucket at 0.282 ms/layer. A new fused shape or down shape that beats the current rows128/512-thread configuration would have the most impact.

2. **Attention-output pipeline** — the second-largest bucket at 0.238 ms/layer. A purpose-built A/B/HC expansion pipeline that avoids extra intermediate traffic without repeating the slower older fused path.

3. **Compressor/cache-update fusion** — smaller but still material at 0.096 ms/layer. The current short-prompt benchmark does not exercise top-k selection (threshold is 512, ratio-4 needs ~2048 tokens), so this bucket is mostly compressor projection/update and cache maintenance.

4. **Pure-kernel CUDA graph islands** — cuBLAS graph capture failed on this CUDA stack, but pure-kernel islands remain plausible for non-cuBLAS groups.

5. **Two-node execution** — cross-node tensor parallelism over InfiniBand is unlikely to improve single-stream decode latency. A layer-pipeline split is possible but requires a larger scheduler/runtime change.

### Reference: Luce Megakernel Lessons

The [Luce megakernel work](https://github.com/Luce-Org/lucebox-hub/tree/main/megakernel) is directionally useful but not directly portable (hand-written batch-1 BF16 single-model path for 0.8B params). Practical lessons for this V100 branch:

- Minimize CPU launch and layer-boundary overhead for small pure-kernel steps
- Prefer scoped persistent/fused islands over a full forward-pass megakernel
- Treat register pressure as a first-class constraint
- Avoid grid-wide synchronization inside tight recurrence/token loops

### Low-Priority or Already Tested

| Route | Outcome |
|-------|---------|
| MTP / speculative decoding | No useful single-stream win |
| vLLM | May help serving/batching but not expected to beat DS4 CUDA without substantial PPC64/CUDA integration |
| F16 output split cache | Regressed |
| Shared gate/up F16 fallback | Slower than native pair kernel |
| Existing MoE env toggles | Already swept; no better mode found |
| Q8 warp-row geometry | Tested rows16/32, both below baseline |
| Fused Q norm+RoPE | Noise-level improvement, source reverted |
| Compressor emit fused | Noise-level, source reverted |
| Cost-balanced tensor split | All changed output hash and regressed |

### Profile Diagnostics

#### Stage Profiler

```sh
DS4_METAL_GRAPH_TOKEN_PROFILE=1
DS4_METAL_DECODE_STAGE_PROFILE=1
DS4_CUDA_MOE_PROFILE=1
```

!!! warning "Profiler overhead"
    The stage profiler is slower than normal generation because it synchronizes frequently. The profiler log of a 13.20 t/s build reported only 10.76 t/s generation.

#### cuBLAS Graph Capture Triage

The llama.cpp "accelerate DeepSeek V4 CUDA graph islands" commit adds custom DSV4 CUDA op islands (HC split/sinkhorn, weighted sum, expand, FP8 KV quantize, rope tail). DS4 already has equivalent or more specialized CUDA kernels for those paths.

An opt-in CUDA Graph capture experiment around the current F16 cuBLAS path was rejected:

```text
ds4: CUDA f16 cuBLAS graph capture disabled: operation not permitted when stream is capturing
```

The fallback measured **10.70 t/s** — the experiment was reverted. Future graph work should target pure-kernel islands or use a cuBLASLt path known to be capture-compatible.

#### Output Head Profiler

```sh
DS4_OUTPUT_HEAD_PROFILE=1   # opt-in, diagnostic only
DS4_CUDA_SPLIT_OUTPUT_PROFILE=1  # split-output enqueue/sync breakdown
```

Warm split-output logits are only ~1.35 ms — output logits are not a meaningful route for further decode improvement.

```text
first split-output call:  1052.8 ms total, about 263 ms enqueue per device
warm split-output call:      1.35 ms total, about 1.14 ms waiting on dev0
output F16 cache:        generation 13.29 t/s, cold-inclusive logits 69.181 ms/token
half-warp output Q8:     generation 13.39 t/s, cold-inclusive logits 67.024 ms/token
full-block DP4A output:  generation 13.36 t/s, cold-inclusive logits 67.092 ms/token
```

#### nvprof Confirmation

Warm decode MoE kernels confirmed by `nvprof`:

| Kernel | Time |
|--------|------|
| `moe_down_qwarp32_kernel` | ~293 µs/layer |
| `moe_gate_up_midq_decode_lut_qwarp32_kernel` | ~256 µs/layer |

!!! note "Hardware counters blocked"
    `ncu` cannot collect hardware counters on these nodes (`ERR_NVGPUCTRPERM`).

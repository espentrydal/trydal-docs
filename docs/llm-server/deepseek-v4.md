# DeepSeek V4 Flash on ppc64le

Running DeepSeek V4 Flash on a 2-node ppc64le cluster with 8× V100-SXM2-32GB.

## Cluster Hardware

| Property | ai-smil1 | ai-smil2 |
|---|---|---|
| CPU | 160 ppc64le cores | 160 ppc64le cores |
| GPU | 4× V100-SXM2-32GB | 4× V100-SXM2-32GB |
| VRAM | 128 GB | 128 GB |
| NVLink | GPU0-GPU1 pair, GPU2-GPU3 pair | GPU0-GPU1 pair, GPU2-GPU3 pair |
| Interconnect | InfiniBand ib0 (192.168.200.0/24) | InfiniBand ib0 (192.168.200.0/24) |
| CUDA | 12.4 at `/usr/local/cuda` | 12.4 at `/usr/local/cuda` |
| GCC | 13 at `/opt/rh/gcc-toolset-13` (system GCC 8.5 too old) | 13 at `/opt/rh/gcc-toolset-13` |

### Combined resources

- **Total VRAM**: 256 GB (8× V100-32GB)
- **Reference speed** (single-node IQ2XXS): ~6 tok/s
- **Two-node speed** (Q4KExperts): ~1.2–1.6 tok/s (see [Speed analysis](#speed-analysis))

## Build

Both nodes must be built identically from the `power64le` branch:

```bash
cd ~/llama.cpp-deepseek-v4-flash
source /opt/rh/gcc-toolset-13/enable
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_RPC=ON
make -j16
```

Or with explicit GCC path for non-interactive sessions:

```bash
CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc \
CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++ \
make -j16 -C ~/llama.cpp-deepseek-v4-flash/build
```

When adding new CUDA source files, you must re-run `cmake ..` because the CUDA backend uses a configured `file(GLOB "*.cu")` — without reconfigure, new `.cu` objects are not linked.

### Source

- **Repository**: `~/llama.cpp-deepseek-v4-flash/` on both nodes
- **Branch**: `power64le`
- **Fork**: https://github.com/espentrydal/llama.cpp-deepseek-v4-flash
- **Head commit**: `f9b41797f` (FATTN_EXT crash fix)
- **Patches**: 9 total, all committed and pushed

## Required Patches

All patches are applied on the `power64le` branch.

### 1. GGML_SCHED_MAX_SPLIT_INPUTS: 30 → 128

**File**: `ggml/src/ggml-backend.cpp`, line 757

DeepSeek V4 graph exceeds 30 inputs per scheduler split on 4 GPUs.

```diff
-#define GGML_SCHED_MAX_SPLIT_INPUTS 30
+#define GGML_SCHED_MAX_SPLIT_INPUTS 128
```

### 2. CUDA concat F16 support

**File**: `ggml/src/ggml-cuda/concat.cu`

CUDA concat only handles F32. DeepSeek V4 uses F16 intermediates. Adds cudaMemcpy-based F16 path (~50 lines added).

### 3. PowerPC IQ2XXS VSX disable

**File**: `ggml/src/ggml-cpu/arch/powerpc/quants.c`, line 1437

VSX SIMD path for IQ2XXS dequant produces wrong values on ppc64le. Fall back to portable C implementation.

```diff
-#if defined(__POWER9_VECTOR__)
+#if 0  // VSX IQ2XXS broken on ppc64le
```

Only line 1437 needs changing. K-quant VSX paths all work correctly. A subsequent commit (`4d70a4b28`) fixed the IQ VSX path with corrected signed-byte multiply behavior instead of fully disabling it.

### 4. CUDA backend: mark DSV4 ops as unsupported

**File**: `ggml/src/ggml-cuda/ggml-cuda.cu`, after line 5174

DeepSeek V4 has 5 custom ops (HC_SPLIT_SINKHORN, HC_WEIGHTED_SUM, HC_EXPAND, FP8_KV_QUANTIZE, ROPE_TAIL) handled by CPU backend. The CUDA backend's op-support check must explicitly return false for these, otherwise they fall through to the default which can crash.

```diff
+        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
+        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
+        case GGML_OP_DSV4_HC_EXPAND:
+        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
+        case GGML_OP_DSV4_ROPE_TAIL:
+            return false;
```

### 5. RPC backend: mark DSV4 ops as unsupported (root cause of 2-node crash)

**File**: `ggml/src/ggml-rpc/ggml-rpc.cpp`, function `ggml_backend_rpc_device_supports_op`

**This was the root cause of the 2-node RPC crash.** `ggml_backend_rpc_device_supports_op` has a TODO and unconditionally returns `true` for all ops. This causes the node-1 scheduler to assign DSV4 ops to the RPC backends (node-2 GPUs), which then crash in `ggml_cuda_graph_evaluate_and_capture`.

!!! warning "Misleading diagnostic"
    `strings rpc-server | grep DSV4` returns empty because DSV4 op name strings live in `libggml-base.so`, not the `rpc-server` binary. Check `strings libggml-base.so | grep DSV4` instead.

```diff
 static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
     GGML_UNUSED(dev);
-    GGML_UNUSED(op);
     //TODO: call the remote backend and cache the results
-    return true;
+    // DSV4 custom ops are CPU-only; must not be scheduled on remote CUDA via RPC
+    switch (op->op) {
+        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
+        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
+        case GGML_OP_DSV4_HC_EXPAND:
+        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
+        case GGML_OP_DSV4_ROPE_TAIL:
+            return false;
+        default:
+            return true;
+    }
 }
```

### 6. RPC graph_compute: synchronize device after compute

**File**: `ggml/src/ggml-rpc/ggml-rpc.cpp`, function `rpc_server::graph_compute`

After `ggml_backend_graph_compute`, add a device synchronize so the next operation sees completed GPU state.

```diff
     ggml_status status = ggml_backend_graph_compute(backends[device], graph);
     GGML_ASSERT(status == GGML_STATUS_SUCCESS && ...);
+    ggml_backend_synchronize(backends[device]);
     stored_graphs[device].graph = graph;
```

!!! note "Synchronize removal tested (not a performance win)"
    Direct A/B testing removed this synchronize line and found no throughput improvement. It is not a performance bottleneck. The working tree on both nodes has the line removed; this can be included in a future cleanup commit as simplification, not a speed fix.

### 7. RPC copy_tensor sync — REVERTED (commit b601e96d2)

Initially synced all backends before copy_tensor to fix a warmup race. The sync-all is too aggressive on ppc64le with 8 GPUs over IB and moves the crash from copy_tensor to get_tensor. Patch 6 (graph_compute sync) alone is sufficient.

### 8. CUDA + RPC supports_op: REPEAT must be F32/F16 only

**Files**: `ggml/src/ggml-cuda/ggml-cuda.cu` and `ggml/src/ggml-rpc/ggml-rpc.cpp`

`ggml_cuda_op_bin_bcast` asserts `src1->type` is F32 or F16. The original supports_op only rejected I32/I16, letting Q8_0/Q4_K reach the op and crash. Both CUDA and RPC supports_op now use a positive allowlist:

```cpp
case GGML_OP_REPEAT: {
    ggml_type src0_type = op->src[0]->type;
    return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
}
```

### 9. RPC supports_op: FLASH_ATTN_EXT must respect CUDA constraints

**Files**: `ggml/src/ggml-cuda/fattn.cu` and `ggml/src/ggml-rpc/ggml-rpc.cpp`

`ggml_cuda_get_best_fattn_kernel` returns `BEST_FATTN_KERNEL_NONE` for DSV4's indexer attention (head_dim=512, K.ne[1] not divisible by FATTN_KQ_STRIDE=256, so `gqa_opt_applies=false`). Without this check, RPC `supports_op` returns true → ops sent to node-2 → `GGML_ABORT` at `fattn.cu:512`.

Fix in two parts:

1. `fattn.cu` adds a C-linkage wrapper at file end:
   ```cpp
   extern "C" bool ggml_cuda_flash_attn_ext_supported_for_rpc(
       const struct ggml_tensor * dst) {
       if (ggml_cuda_info().device_count == 0) return false;
       return ggml_cuda_flash_attn_ext_supported(0, dst);
   }
   ```

2. `ggml-rpc.cpp` adds a weak extern reference and calls it:
   ```cpp
   extern "C" bool ggml_cuda_flash_attn_ext_supported_for_rpc(
       const struct ggml_tensor * dst) __attribute__((weak));

   case GGML_OP_FLASH_ATTN_EXT:
       if (ggml_cuda_flash_attn_ext_supported_for_rpc) {
           return ggml_cuda_flash_attn_ext_supported_for_rpc(op);
       }
       return false;
   ```

Weak resolution from `libggml-cuda.so` at runtime; null means no CUDA. Uses device 0 as representative for homogeneous V100 cluster.

!!! warning "Performance impact"
    This fix prevents the FLASH_ATTN_EXT crash but forces DSV4's indexer ops (K.ne[1]=1, head_dim=512) to fall back to CPU. V100 has no flash-attention kernel for head_dim=512 with K.ne[1] not divisible by 256 — both TILE and MMA_F16 require `use_gqa_opt`. This is the dominant bottleneck (1.2–1.6 tok/s). See [Speed analysis](#speed-analysis).

## CUDA Custom DSV4 Ops (Route 2)

To reduce graph fragmentation, CUDA implementations were added for the 5 DSV4 custom ops:

### Files added/modified

- `ggml/src/ggml-cuda/dsv4.cu` — CUDA implementations for:
    - `GGML_OP_DSV4_HC_SPLIT_SINKHORN`
    - `GGML_OP_DSV4_HC_WEIGHTED_SUM`
    - `GGML_OP_DSV4_HC_EXPAND`
    - `GGML_OP_DSV4_FP8_KV_QUANTIZE`
    - `GGML_OP_DSV4_ROPE_TAIL`
- `ggml/src/ggml-cuda/dsv4.cuh` — declarations and RPC support wrapper
- `ggml/src/ggml-cuda/ggml-cuda.cu` — dispatch + `supports_op` shape gates
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — RPC `supports_op` now delegates DSV4 op checks to `ggml_cuda_dsv4_supported_for_rpc`

### Results

Graph splits improved dramatically:

| Metric | Before Route 2 | After Route 2 |
|---|---|---|
| Splits (bs=512) | 11,907 | **177** |
| Splits (bs=1) | 699 | **95** |

Short decode (16 tokens) average: ~1.59 tok/s. The fragmentation reduction is real but not yet a throughput win on short decode — the remaining bottleneck is likely kernel launch/RPC overhead in many small kernels and/or the small-K FlashAttention/indexer path.

## Indexer Graph Rewrite (Route 3)

A correctness-preserving graph-builder optimization: when `top_k >= n_comp`, the indexer can't prune anything, so the existing index causal mask can be reused directly, skipping score/top-k/mask construction.

**File**: `src/models/deepseek4.cpp`

After building/storing `index_kv`, compute `top_k = min(indexer_top_k, n_comp)`. If `top_k >= n_comp`, set `comp_mask = index_mask`. Otherwise keep the original prefill path.

**Result**: Viable simplification, graph splits unchanged (177/95). Not a throughput win (~1.43 tok/s average). A more aggressive "bypass the indexer entirely" approach would turn sparse compressed attention into dense compressed attention and should be treated as a separate quality experiment.

## Speculative Decoding (Route 4)

Tested with `--spec-type ngram-simple --draft-max 16 --draft-min 0`. The no-draft ngram implementation initializes but does not draft any tokens for the chat benchmark — `#gen drafts = 0`, `#acc drafts = 0`. Real speculative decoding would need a small DSV4-family draft model or a trained/spec-compatible draft head, which does not currently exist for this cluster.

## Scripts

All in `~/scripts/` on both nodes.

### Single node (IQ2XXS)

| Script | Model | Port |
|---|---|---|
| `server-dsv4-flash-1node.sh` | IQ2XXS (81 GB) | 8080 |

### Two node (Q4KExperts)

| Script | Node | Purpose |
|---|---|---|
| `rpc-server.sh` | ai-smil2 | RPC backend, IB port 50052 |
| `server-dsv4-flash-2node.sh` | ai-smil1 | Q4K server, port 8080 |

### Other

| Script | Purpose |
|---|---|
| `server-qwen-1node.sh` | Qwen 3.6 35B-A3B server, port 8080 |

## Models

| File | Quant | Size | Use |
|---|---|---|---|
| `DeepSeek-V4-Flash-IQ2XXS-...chat-v2.gguf` | IQ2_XXS | 81 GB | Single node |
| `DeepSeek-V4-Flash-Q4KExperts-...chat-v2.gguf` | Q4K/F16/Q8 | 165 GB | Two node |

Both in `~/models/deepseek-v4-flash/`.

## Production Service Configuration

Production serving uses systemd user services. Scripts are kept for development, one-off overrides, and customized launches.

### DS4 Service

| Setting | Value |
|---|---|
| Node | `ai-smil2` |
| Service | `ds4-server.service` |
| Script | `~/ds4/scripts/server-ds4-flash-v100-1node.sh` |
| Port | `8080` |
| Model | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| Context | `131072` |
| KV disk budget | `16384 MiB` |
| GPU split | `DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2` |

The DS4 context default is intentionally 128K. DS4 decode speed is sensitive to VRAM pressure because larger context buffers compete with CUDA-side caches. Use 262K only when the task needs it:

```bash
DS4_CTX=262144 systemctl --user restart ds4-server.service
```

For one-off development runs:

```bash
DS4_CTX=262144 ~/ds4/scripts/server-ds4-flash-v100-1node.sh
```

#### Scratch environment defaults

These CUDA local scratch environment variables are enabled by default and improve VRAM management:

| Variable | Purpose |
|---|---|
| `DS4_CUDA_LOCAL_ROUTED_SCRATCH=1` | Routed MoE scratch |
| `DS4_CUDA_LOCAL_ATTN_SCRATCH=1` | Attention-output scratch |
| `DS4_CUDA_LOCAL_QKV_SCRATCH=1` | Q/KV scratch |
| `DS4_CUDA_LOCAL_COMPRESSOR_SCRATCH=1` | Compressor/indexer scratch |
| `DS4_CUDA_LOCAL_HC_FFN_SCRATCH=1` | HC/FFN/shared scratch |
| `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` | Decode-speed override |

### Qwen Service

| Setting | Value |
|---|---|
| Node | `ai-smil1` |
| Primary service | `qwen-gpu01.service` |
| Optional second service | `qwen-gpu23.service` |
| Script | `~/scripts/server-qwen-gpu01.sh` |
| Primary port | `8080` |
| Optional second port | `8081` |
| Context | `262144` |

Qwen uses 256K context by default because it has enough speed headroom for convenience.

### Deprecated runtime

The old llama.cpp DeepSeek/DSV4 services and scripts are retired. They are kept only as historical references under `~/scripts/deprecated/` and `~/.config/systemd/user/deprecated/`. The current DeepSeek production path is the `~/ds4` runtime.

## API Access

### SSH tunnel (from workstation)

```bash
ssh -L 8080:127.0.0.1:8080 -J espen@forsker-pc espen@ai-smil1 -N
```

OpenAI-compatible at `http://localhost:8080/v1`. No API key required.

```bash
curl -s --noproxy '*' http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Sampling notes

!!! warning "Temperature must be > 0"
    Do NOT use `temperature: 0` — greedy decoding hits a buggy code path on ppc64le that produces training-data fragments. Default sampling (temp > 0) works correctly.

The model defaults to reasoning/thinking mode. The content field is empty until thinking completes. Give 50+ max_tokens for full response.

## Performance Benchmarks

### Scratch defaults improvement timeline

On 2026-05-17, after all CUDA local scratch variables became script defaults, direct 32K validation preserved deterministic output hashes and measured:

| Node | Tokens | Speed |
|---|---|---|
| ai-smil1 | 200 | **21.48 tok/s** |
| ai-smil2 | 64 | **21.66 tok/s** |
| ai-smil2 | 200 | **21.47 tok/s** |

Systemd uses the same script, so the service inherits these settings on restart.

### DS4 service speed progression

Warm 200-token chat requests at 128K context on `ds4-server.service`:

| Date | Optimisation | First 50 chunk | 200-token avg |
|---|---|---|---|
| 2026-05-15 | Baseline (after killing stale RPC server) | ~12.98 tok/s server-side | 12.5–12.6 tok/s |
| 2026-05-16 | `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` default | 13.28 tok/s | 12.99 tok/s |
| 2026-05-16 | 64-row MoE decode down default | 14.30 tok/s | 13.81 tok/s |
| 2026-05-16 | 128-row MoE down + one-token cuBLAS attn-out A | 14.97 tok/s | 14.61 tok/s |
| 2026-05-16 | 512-thread MoE gate/up+midq default | **15.48 tok/s** | **15.00 tok/s** |
| 2026-05-17 | All CUDA scratch defaults | (same script as direct CLI) | 21.47–21.48 tok/s |

Cold service startup includes model/cache loading and is not representative of warm decode speed.

### Deprecated RPC server interference

On 2026-05-15, the old llama.cpp `rpc-server` was still using ~360 MiB per GPU. After stopping it and restarting `ds4-server.service`, warm 200-token non-thinking chat requests improved from ~11.5–11.7 tok/s wall-clock to 12.5–12.6 tok/s wall-clock, with server-side decode logging 12.98–13.05 tok/s.

## Speed Analysis

### Single-node IQ2XXS: ~6 tok/s

Limited by IQ2 dequant complexity on V100s — complex 256-entry LUT per weight byte.

### Two-node Q4KExperts: 1.2–1.6 tok/s

Measured 2026-05 (3 consecutive requests, FATTN fix in place). Experts use K-quant (fast dequant), attention/shared/output use Q8_0 (trivial), hyper-connections/compressors/indexers use F16 (zero overhead).

### The V100 indexer ceiling

DSV4 Flash uses DeepSeek-V3.2-style sparse attention with a "lightning indexer". The indexer dispatches FLASH_ATTN_EXT ops with shape `K.ne=[512, 1, 1, 1]` (head_dim=512, single key per decode step). All Flash variants on the cluster (Q4KExperts, Q3_K_M, Q2_K, IQ2XXS) have this architecture — there is no indexer-less variant.

V100 (compute capability 7.0) has no flash-attention kernel for head_dim=512 unless `K.ne[1]` is divisible by `FATTN_KQ_STRIDE=256`:

- `case 512` in `ggml_cuda_get_best_fattn_kernel` early-returns NONE when `gqa_opt_applies` is false
- TILE kernel `launch_fattn_tile_switch_ncols2<512, 512>` only has `use_gqa_opt` paths; non-aligned K calls `GGML_ABORT`
- MMA_F16 kernel for head_dim=512 has the same constraint

Tested experimentally: relaxing the `case 512` early-reject lets the dispatcher select the TILE kernel, but the TILE kernel itself only handles `use_gqa_opt` ncols2 ∈ {4, 8}; for `use_gqa_opt=false` it falls through to `GGML_ABORT("fatal error")`.

These indexer ops therefore run on CPU. With ~43 layers × indexer per decode step, this is the dominant cost (~800 ms/token).

!!! tip "Realistic ceiling"
    IQ2XXS (81 GB) single-node hits ~5–6 tok/s — same indexer architecture, same V100 kernel gap, but no RPC overhead. This is the realistic performance ceiling for this hardware family with the Flash architecture and the indexer running on CPU.

### Options to exceed the indexer ceiling

None are free; honest cost/benefit estimates:

| Option | Target speed | Cost | Risk |
|---|---|---|---|
| 1. Custom small-K CUDA kernel (GEMV+softmax+GEMV) | **5–15 tok/s** | 1–2 days | Low |
| 2. Custom kernel + speculative decoding | **15–30 tok/s** | (1) + draft model hunt | Medium |
| 3. Bypass indexer → dense attention (ctx ~32K) | **20–40 tok/s** | 3–5 days + A/B validation | Quality unknown |
| 4. Implement missing non-gqa-opt TILE/MMA path | ~same as (1) | 1–2 weeks | V100 shmem budget uncertain |
| 5. Add A100/H100 node | Full speed | Hardware cost | N/A |

Approaches that **don't** help materially:

- Smaller quants (Q3_K_M, Q2_K): same indexer ceiling, ~1.5–2 tok/s at best for 2-node
- Co-locating indexer layers on one node via tensor-split: shaves per-op RPC roundtrip but indexer is still on CPU

### Small-K FlashAttention kernel (Option A — implemented but not the bottleneck)

A new `BEST_FATTN_KERNEL_SMALL_K` kernel was implemented for `head_dim=512` with `K.ne[1] < FATTN_KQ_STRIDE`. It's a direct GEMV → softmax → GEMV CUDA kernel, 128 threads, IPT=4 for D=512. Templated only on D; runtime branches for `logit_softcap`, `max_bias`, mask.

**Result**: Speed went from 1.21 → 1.25 tok/s. The kernel runs and produces correct output, but GPU utilization during decode remains 0–4% on all 8 V100s. The bottleneck is not GPU compute — it's elsewhere (graph fragmentation, RPC sync, CPU ops).

Bugs found and fixed during integration:

- `slope = 0` when `max_bias == 0` zeros out the mask. Fix: always use `get_alibi_slope(...)` — it returns 1.0 when `max_bias <= 0`.

## Known Issues

### 1. `--parallel > 1` crashes

`ggml_reshape_3d` in deepseek4 graph builder. Use `--parallel 1 --cont-batching`.

!!! note "`--parallel 16` test"
    Server startup failed during graph reservation with `ggml_abort -> ggml_reshape_3d -> llm_build_deepseek4 -> graph_reserve`. The aggregate-throughput route via parallel is not viable in the current DeepSeek4 graph builder.

### 2. `temperature: 0` broken on ppc64le

Produces training-data fragments. Always use temperature > 0.

### 3. GPU memory leaks on killed SSH sessions

Zombie processes hold VRAM:

```bash
fuser /dev/nvidia0  # find PID
kill <PID>
nvidia-smi  # verify free
```

### 4. Corporate transparent proxy on ai-smil1

Squid mangles HTTP POST bodies. Always use SSH tunnel from workstation.

### 5. RPC server start order

RPC server must be started on ai-smil2 **before** launching the 2-node server. It crashes on client disconnect and must be manually restarted.

### 6. Model downloads

Download models with `hf download`, never wget/curl. Never clear huggingface cache during active downloads.

### 7. SSH session gotchas

- `pkill -f llama-server` from inside an SSH session kills the SSH session itself. Use the PID directly or `nohup ... & disown` patterns
- ai-smil2 sometimes OOMs during SSH-with-keychain. Use `ssh espen@192.168.200.2` via ai-smil1's `ProxyCommand` config; if `keychain` fails, run with `bash --norc --noprofile -c`
- After any node-2 crash, the `rpc-server` process exits. Restart it **before** re-launching `llama-server`, or all layers will try to load locally causing a phantom OOM

## Quick Smoke Test

```bash
# from jarvis
ssh ai-smil1 'curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":20}"' \
  | python3 -c 'import sys, json; d=json.loads(sys.stdin.read()); t=d["timings"]; \
    print(f"gen: {t[\"predicted_per_second\"]:.2f} tok/s ({t[\"predicted_n\"]} tokens)")'
```

Expected: ~1.2–1.6 tok/s (two-node Q4K). If you see a crash or much-lower number, check both `/tmp/llama-server.log` (ai-smil1) and `/tmp/rpc-server.log` (ai-smil2).

## Patch File Index

All files modified on the `power64le` branch of the fork:

| File | Changes |
|---|---|
| `ggml/src/ggml-backend.cpp` | `SCHED_MAX_SPLIT_INPUTS` 30→128 |
| `ggml/src/ggml-cuda/concat.cu` | F16 support |
| `ggml/src/ggml-cpu/arch/powerpc/quants.c` | IQ VSX path fix |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | DSV4 ops + REPEAT supports_op (positive allowlist) |
| `ggml/src/ggml-cuda/fattn.cu` | `ggml_cuda_flash_attn_ext_supported_for_rpc` wrapper; small-K kernel |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | DSV4 + REPEAT + FLASH_ATTN_EXT cases in `supports_op`; graph_compute sync |
| `ggml/src/ggml-cuda/dsv4.cu` | CUDA ops for 5 DSV4 custom ops |
| `ggml/src/ggml-cuda/dsv4.cuh` | Declarations and RPC support wrapper |
| `src/models/deepseek4.cpp` | Indexer graph rewrite (no-op when top_k >= n_comp) |

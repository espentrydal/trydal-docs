# Services

Systemd (Linux) and launchd (macOS) service files for Hermes Agent and ai-smil infrastructure. Production LLM serving uses systemd user services; scripts are kept for development, one-off overrides, and customized launches.

## Tunnel Services (smil-tunnel-*)

SSH tunnels from a **client** machine exposing LLM and dashboard endpoints through the gateway.

### smil-tunnel-ds4 — DeepSeek V4 API

Forwards `ai-smil2:8080` (ds4 API) to `localhost:2080`.

| File | Platform |
|------|----------|
| `smil-tunnel-ds4.service` | Linux (systemd user) |
| `smil-tunnel-ds4.plist` | macOS (launchd) |

### smil-tunnel-qwen01 — Qwen GPU 0,1

Forwards `ai-smil1:8080` (Qwen GPU 0,1) to `localhost:1080`.

| File | Platform |
|------|----------|
| `smil-tunnel-qwen01.service` | Linux (systemd user) |
| `smil-tunnel-qwen01.plist` | macOS (launchd) |

### smil-tunnel-qwen23 — Qwen GPU 2,3

Forwards `ai-smil1:8081` (Qwen GPU 2,3) to `localhost:1081`.

| File | Platform |
|------|----------|
| `smil-tunnel-qwen23.service` | Linux (systemd user) |
| `smil-tunnel-qwen23.plist` | macOS (launchd) |

### smil-tunnel-chat — DeepSeek chat web UI

Forwards `ai-smil2:8081` (ds4 chat web UI) to `localhost:2081`.

### smil-tunnel — CLI helper script

Interactive script for starting tunnels manually. Place anywhere in `PATH`.

```bash
smil-tunnel qwen       # Qwen GPU 0,1  → localhost:1080
smil-tunnel qwen23     # Qwen GPU 2,3  → localhost:1081
smil-tunnel ds4        # ds4 API       → localhost:2080
smil-tunnel chat       # ds4 web UI    → localhost:2081
smil-tunnel dashboard  # Kanban        → localhost:9119
smil-tunnel all        # All five
```

## LLM Server Services

### ds4-server — DeepSeek V4 Flash

- **Node:** `ai-smil2`
- **Service:** `ds4-server.service`
- **Script:** `~/ds4/scripts/server-ds4-flash-v100-1node.sh`
- **Port:** `8080`
- **Model:** `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- **Default context:** `131072` (128K)
- **KV disk budget:** `16384 MiB`
- **GPU split:** `DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2`

#### Environment Variables (script defaults)

```bash
DS4_CUDA_LOCAL_ROUTED_SCRATCH=1
DS4_CUDA_LOCAL_ATTN_SCRATCH=1
DS4_CUDA_LOCAL_QKV_SCRATCH=1
DS4_CUDA_LOCAL_COMPRESSOR_SCRATCH=1
DS4_CUDA_LOCAL_HC_FFN_SCRATCH=1
DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1
```

!!! note "Context tuning"

    The DS4 context default is intentionally 128K. DS4 decode speed is sensitive to VRAM pressure because larger context buffers compete with CUDA-side caches. Use 262K only when the task needs it:

    ```bash
    DS4_CTX=262144 systemctl --user restart ds4-server.service
    ```

    For one-off development runs:

    ```bash
    DS4_CTX=262144 ~/ds4/scripts/server-ds4-flash-v100-1node.sh
    ```

#### Performance history

| Date | Chunk | Tok/s | Notes |
|------|-------|-------|-------|
| 2026-05-15 | 200-token warm, 128K ctx | ~12.5–12.6 (wall-clock) / 12.98–13.05 (server-side) | After stopping deprecated `rpc-server` |
| 2026-05-16 (early) | 50-token first chunk | 13.28 | After `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` default |
| 2026-05-16 (early) | 200-token warm avg, 128K ctx | 12.99 | Same config |
| 2026-05-16 | 32K CLI benchmark, 200 tok | ~13.20 | Direct benchmark |
| 2026-05-16 | 50-token first chunk | 14.30 | After 64-row MoE decode down kernel default |
| 2026-05-16 | 200-token warm avg, 128K ctx | 13.81 | Same config |
| 2026-05-16 | 32K CLI benchmark | 14.04–14.09 | Direct benchmark |
| 2026-05-16 (later) | 50-token first chunk | 14.97 | After 128-row MoE down default + one-token cuBLAS attn-out |
| 2026-05-16 (later) | 200-token avg | 14.61 | Same config |
| 2026-05-16 (later) | 32K CLI benchmark | 14.83–14.88 | Direct benchmark |
| 2026-05-16 (final) | 32K CLI benchmark, ai-smil1 | 15.09 | After 512-thread MoE gate/up+midq default |
| 2026-05-16 (final) | 32K CLI benchmark, ai-smil2 | 15.02 | Same config |
| 2026-05-16 (final) | 50-token first chunk, ai-smil2 | 15.48 | Warm 128K service request |
| 2026-05-16 (final) | 200-token avg, ai-smil2 | 15.00 | Same config |
| 2026-05-17 | 200-token, ai-smil1 | 21.48 | After all CUDA local scratch env vars became script defaults |
| 2026-05-17 | 64-token, ai-smil2 | 21.66 | Same config |
| 2026-05-17 | 200-token, ai-smil2 | 21.47 | Same config |

!!! warning "No deprecated rpc-server alongside production DS4"

    Do not run the deprecated llama.cpp `rpc-server` beside production DS4. On 2026-05-15 it was still using about 360 MiB per GPU; after stopping it and restarting `ds4-server.service`, warm 200-token non-thinking chat requests improved from about 11.5–11.7 tok/s wall-clock to 12.5–12.6 tok/s wall-clock, with server-side decode logging 12.98–13.05 tok/s.

### qwen-gpu01 — Qwen (GPU 0,1)

- **Node:** `ai-smil1`
- **Service:** `qwen-gpu01.service`
- **Script:** `~/scripts/server-qwen-gpu01.sh`
- **Port:** `8080`
- **Default context:** `262144` (256K)

### qwen-gpu23 — Qwen (GPU 2,3)

- **Node:** `ai-smil1`
- **Service:** `qwen-gpu23.service`
- **Port:** `8081`
- **Default context:** `262144` (256K)

!!! tip "Qwen context"

    Qwen uses 256K context by default because it has enough speed headroom for convenience. The older comments that called this "1M ctx" were stale; the actual runtime argument is 256K.

## Dashboard Service

### hermes-dashboard — Kanban dashboard tunnel

SSH tunnel from a **client** machine to jarvis, forwarding the Hermes Kanban dashboard.

| File | Platform |
|------|----------|
| `hermes-dashboard.service` | Linux (systemd user) |
| `hermes-dashboard.plist` | macOS (launchd) |

After install, the Kanban dashboard is available at [http://localhost:9119](http://localhost:9119).

!!! info "Prerequisite"

    `hermes-dashboard.service` must be running on jarvis (serving on `127.0.0.1:9119`). Installed at `~/.config/systemd/user/hermes-dashboard.service` on jarvis.

## Port Map

| Local port | Remote port | Host      | Service             |
|------------|-------------|-----------|---------------------|
| 1080       | 8080        | ai-smil1  | Qwen GPU 0,1        |
| 1081       | 8081        | ai-smil1  | Qwen GPU 2,3        |
| 2080       | 8080        | ai-smil2  | DS4 API             |
| 2081       | 8081        | ai-smil2  | DS4 chat web UI     |
| 9119       | 9119        | jarvis    | Kanban dashboard    |

## Installation

### Linux (systemd user)

```bash
cp <service-file>.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now <service-file>.service
```

### macOS (launchd)

```bash
cp <service-file>.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/<service-file>.plist
```

## Removed / Deprecated

The old llama.cpp DeepSeek / DSV4 services and scripts are retired. They are kept only as historical references under `~/scripts/deprecated/` and `~/.config/systemd/user/deprecated/`.

Current DeepSeek production path is the `~/ds4` runtime, not the older `~/llama.cpp-deepseek-v4-flash` runtime.

## Related Docs

- [LLM Server — DeepSeek V4](../llm-server/deepseek-v4.md) — Server-side service files and launch scripts for ai-smil
- [LLM Server — Qwen](../llm-server/qwen.md)
- [LLM Server — Performance](../llm-server/performance.md)
- [Networking](./networking.md)
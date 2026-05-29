# Qwen on GPU

Running Qwen models on ai-smil1 GPUs.

## Instances

| Instance    | GPU   | Port  | Tunnel |
|-------------|-------|-------|--------|
| qwen-gpu01  | GPU 0 | 8080  | 1080   |
| qwen-gpu23  | GPU 2 | 8081  | 1081   |

## Service Files

- `qwen-gpu01.service`
- `qwen-gpu23.service`

Both run llama.cpp server with GGUF quantizations.

## Starting Both

Use the combined script to start both instances simultaneously.
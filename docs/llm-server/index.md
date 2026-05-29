# LLM Server

DeepSeek V4 and Qwen models running on ai-smil1/2 (ppc64le with 4× NVIDIA V100 32GB each).

## Machines

| Host      | IP            | GPUs            | Models                     |
|-----------|---------------|-----------------|----------------------------|
| ai-smil1  | 10.85.110.66  | 4× V100 32GB    | Qwen 01, Qwen 23           |
| ai-smil2  | 10.85.110.68  | 4× V100 32GB    | DeepSeek V4 Flash           |

## Architecture

Models run via llama.cpp server with GGUF quantizations. SSH reverse tunnels expose endpoints through the Hermes gateway. See the subpages for model-specific setup details and performance benchmarks.
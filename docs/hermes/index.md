# Hermes Agent

Hermes agent profiles and configuration across the infrastructure.

## Profiles

| Profile        | Host         | Purpose                   |
|----------------|-------------|---------------------------|
| default (jarvis) | caemlin    | Primary agent             |
| ai-smil        | ai-smil1     | LLM server management     |
| proxmox        | baerlon      | Proxmox cluster admin     |
| arch-vm        | arch-vm     | Arch Linux VM agent       |
| forsker        | forsker-pc   | Research workstation      |

## Provider Config

LLM providers configured via Hermes — SMIL providers for local models, OpenRouter for cloud models.
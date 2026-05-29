# Infrastructure

Overview of the Trydal homelab infrastructure.

## Hardware

| Machine     | Role           | Specs                        |
|-------------|----------------|------------------------------|
| caemlin     | Proxmox host   | Primary node, WAN gateway    |
| baerlon     | Proxmox host   | Secondary node, CTs, PBS     |
| ai-smil1    | LLM server     | ppc64le, 4× V100 32GB        |
| ai-smil2    | LLM server     | ppc64le, 4× V100 32GB        |

## Network

- Internal VLAN 20: `10.20.0.0/24`
- Tailscale for remote access
- Internal DNS via Technitium on CT 104
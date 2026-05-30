# Trydal Docs

Internal documentation for the Trydal homelab and research infrastructure.

---

## Contents

### :material-server: Infrastructure

Networking (VLANs, Tailscale, SSH tunnels), internal DNS, systemd services, and the Proxmox cluster on caemlin + baerlon.

[:octicons-arrow-right-24: Browse infrastructure](infrastructure/index.md)

- [Networking](infrastructure/networking.md) — VLAN layout, Tailscale, SSH tunnels
- [DNS](infrastructure/dns.md) — Technitium on CT 104, internal zones
- [Services](infrastructure/services.md) — All systemd services, port map
- [Proxmox](infrastructure/proxmox.md) — Cluster, containers, PBS backups

---

### :material-chip: LLM Server

DeepSeek V4 and Qwen running on ai-smil1/2 (ppc64le, 4× V100 32GB each).

[:octicons-arrow-right-24: Browse LLM Server](llm-server/index.md)

- [DeepSeek V4](llm-server/deepseek-v4.md) — Build, patches, architecture, 2-node RPC
- [Qwen](llm-server/qwen.md) — GPU instances and service config
- [Performance](llm-server/performance.md) — Benchmarks, kernel optimisations, locality tuning

---

### :material-brain: PhD Workspace

Espen Trydal's doctoral research — dementia diagnosis via AI and diffusion MRI.

[:octicons-arrow-right-24: Browse PhD Workspace](phd-workspace/README.md)

- [Onboarding](phd-workspace/docs/onboarding/00-first-30-minutes.md) — New lab member guide
- [Runbooks](phd-workspace/docs/runbooks/command-catalog.md) — Command reference, path conventions, hardware routing
- [Plans](phd-workspace/plans/README.md) — Active and completed project plans

---

### :material-robot: Hermes

Agent profiles, provider configs, and profile-specific documentation.

[:octicons-arrow-right-24: Browse Hermes](hermes/index.md)

- [ML Intern](hermes/ml-intern.md) — Profile for automated ML research tasks

---

### :material-book-open-variant: Runbook

Operational tasks, troubleshooting guides, and maintenance procedures.

[:octicons-arrow-right-24: Browse Runbook](runbook/index.md)

---

## Quick Links

| What | Where |
|------|-------|
| LLM endpoints (tunnels) | [Services → Port map](infrastructure/services.md#port-map) |
| Server not responding? | [DeepSeek V4 → Smoke test](llm-server/deepseek-v4.md#quick-smoke-test) |
| Proxmox container list | [Proxmox → Containers](infrastructure/proxmox.md#containers) |
| Training a model? | [PhD Workspace → Onboarding](phd-workspace/docs/onboarding/00-first-30-minutes.md) |
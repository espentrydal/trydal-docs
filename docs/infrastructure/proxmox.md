# Proxmox

## Cluster

Two-node cluster: `caemlin` (primary) + `baerlon` (secondary).

## Containers

| VMID | Name         | Node    | Purpose              | IP          |
|------|-------------|---------|----------------------|-------------|
| 100  | debian-headless | baerlon | General Debian     | DHCP        |
| 104  | dns          | baerlon | Technitium DNS       | 10.20.0.104 |
| 110  | jarvis       | baerlon | Hermes agent host    | 10.20.0.110 |
| 111  | actualbudget | baerlon | Actual Budget         | DHCP        |
| 112  | trydal-web   | baerlon | Web server (nginx)   | 10.20.0.112 |

## PBS (Proxmox Backup Server)

VM 108 on baerlon. Datastore: 10TB zvol on rpool.

## API Access

Token: `jarvis@pve!hermes-token` (HermesAgent role — audit + power + snapshot).
Config at `~/.secrets/hermes/proxmox.env`.
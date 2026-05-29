# Networking

## VLAN Architecture

| VLAN | Subnet        | Purpose            |
|------|---------------|--------------------|
| 10   | 10.10.0.0/24  | Management (native) |
| 20   | 10.20.0.0/24  | Services            |
| 30   | 10.30.0.0/24  | (reserved)          |
| 40   | 10.40.0.0/24  | (reserved)          |

## Tailscale

All machines are connected via Tailscale. Internal DNS is handled by Technitium, not Tailscale DNS.

## SSH Tunnels

LLM serving endpoints are exposed through SSH reverse tunnels from ai-smil1/2 to the gateway:

| Tunnel      | Local Port | Remote Port |
|-------------|-----------|-------------|
| qwen01      | 8080      | 1080        |
| qwen23      | 8081      | 1081        |
| ds4         | 8080      | 2080        |
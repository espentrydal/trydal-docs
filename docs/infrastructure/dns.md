# DNS

Internal DNS is provided by Technitium running in CT 104 on baerlon.

- **Web UI:** `http://10.20.0.104:5380`
- **Tailscale:** `http://100.85.18.127:5380`

## Internal Zones

| Zone        | Purpose                    |
|-------------|----------------------------|
| trydal.io   | Internal web services      |
| local.trydal.io | Machine hostnames      |

## Important

Do not use Tailscale DNS for internal resolution. Technitium is the authoritative source for `.trydal.io` zones.
# Atlas Edge Node — Progress Log

## 2026-02-08: Tailscale Auto-Connect

**Problem**: Edge node hardcoded `192.168.1.19` (LAN IP) to reach Atlas Brain.
Moving to a different network would break connectivity. Both machines are on Tailscale.

**Root cause**: `DNSStubListener=no` in `/etc/systemd/resolved.conf` disabled the
stub resolver. Python used LAN DNS (`192.168.1.1`) directly, bypassing systemd-resolved's
split-DNS routing for Tailscale domains.

**Changes**:
- `/etc/systemd/resolved.conf`: `DNSStubListener=yes` (was `no`)
- `/opt/atlas-node/.env.local`: Created with `ATLAS_BRAIN_HOST=atlas-brain.tailc7bd29.ts.net`
- `/etc/systemd/system/atlas-node.service`: Replaced 3 inline `Environment=` lines with `EnvironmentFile=/opt/atlas-node/.env.local`
- `/opt/atlas-node/atlas_node/config.py`: Default `ATLAS_BRAIN_HOST` → `atlas-brain.tailc7bd29.ts.net`
- `/opt/atlas-node/web/monitor.html`: Default WS URL → `atlas-brain.tailc7bd29.ts.net`

**Verified**: DNS resolves `atlas-brain.tailc7bd29.ts.net` → `100.112.150.53`,
WS connects successfully via Tailscale FQDN.

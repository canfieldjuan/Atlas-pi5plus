# Atlas Node Source of Truth

This document defines the single source of truth for the Orange Pi node runtime.

## Canonical Source

- Canonical node code repository: `https://github.com/canfieldjuan/Atlas-pi5plus`
- Canonical branch: `main`
- Canonical runtime checkout path on node: `/opt/atlas-node`
- Canonical systemd service: `atlas-node.service`

If code is not committed to `Atlas-pi5plus/main`, it is not considered durable node truth.

## Runtime Truth vs Repo Truth

- Repo truth: tracked files committed in `Atlas-pi5plus/main`
- Runtime config truth: `/opt/atlas-node/.env.local` (deployment-specific, git-ignored)
- Deployment unit truth: `/etc/systemd/system/atlas-node.service`

This means `/home/juan-canfield/Desktop/Atlas` can be a development input, but it is not the node source of truth unless changes are ported and committed to `Atlas-pi5plus`.

## Required Invariants

- `atlas-node.service` must execute from `/opt/atlas-node`
- `/opt/atlas-node` must be on a committed SHA from `origin/main`
- `/opt/atlas-node` working tree must be clean during normal operation
- All production changes must be represented by commits in GitHub

## Change Workflow (Required)

1. Make changes in `/opt/atlas-node` (or port them in).
2. Run validation checks (syntax/tests/service checks).
3. Commit focused changes with clear messages.
4. Push to `origin/main`.
5. Restart `atlas-node.service` if runtime code/config changed.
6. Verify health (`systemctl status`, recent `journalctl` logs, WS connection, expected handlers).

## Hotfix Workflow (Allowed, but must be reconciled)

If emergency edits are made directly on node:

1. Apply the minimal fix.
2. Validate service health.
3. Commit and push immediately after stabilization.
4. Confirm `/opt/atlas-node` returns to clean state.

No long-lived uncommitted runtime changes are allowed.

## Drift Check Commands

Use these commands to confirm source-of-truth compliance:

```bash
cd /opt/atlas-node
git fetch origin
git status --short
git rev-parse HEAD
git rev-parse origin/main
systemctl is-active atlas-node
```

Expected:

- `git status --short` prints nothing
- `HEAD` matches `origin/main`
- service is `active`

## Current Decision (2026-02-17)

- Keep `Atlas-pi5plus` as node source of truth.
- Do not treat Desktop `Atlas` repo as node runtime truth unless explicitly ported and committed.

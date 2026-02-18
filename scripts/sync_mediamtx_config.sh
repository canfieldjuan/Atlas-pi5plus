#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_PATH="${MEDIAMTX_TEMPLATE_PATH:-$REPO_ROOT/ops/mediamtx/mediamtx.yml.tmpl}"
CONFIG_DST="${MEDIAMTX_CONFIG_DST:-/opt/mediamtx/mediamtx.yml}"
CONTAINER_NAME="${MEDIAMTX_CONTAINER_NAME:-mediamtx}"
RECORD_DELETE_AFTER="${MEDIAMTX_RECORD_DELETE_AFTER:-72h}"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "ERROR: template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

if ! grep -q '${MEDIAMTX_RECORD_DELETE_AFTER}' "$TEMPLATE_PATH"; then
  echo "ERROR: template missing MEDIAMTX_RECORD_DELETE_AFTER placeholder" >&2
  exit 1
fi

rendered_file="$(mktemp)"
trap 'rm -f "$rendered_file"' EXIT

sed "s|\${MEDIAMTX_RECORD_DELETE_AFTER}|${RECORD_DELETE_AFTER}|g" "$TEMPLATE_PATH" > "$rendered_file"

if grep -q '${MEDIAMTX_RECORD_DELETE_AFTER}' "$rendered_file"; then
  echo "ERROR: unresolved template placeholders remain" >&2
  exit 1
fi

if ! grep -q '^[[:space:]]*recordDeleteAfter:' "$rendered_file"; then
  echo "ERROR: rendered config missing recordDeleteAfter" >&2
  exit 1
fi

if [[ -f "$CONFIG_DST" ]]; then
  backup_path="${CONFIG_DST}.bak.$(date +%Y%m%d%H%M%S)"
  cp "$CONFIG_DST" "$backup_path"
  echo "Backed up existing config: $backup_path"
fi

install -m 0644 "$rendered_file" "$CONFIG_DST"
echo "Applied config: $CONFIG_DST"

if command -v docker >/dev/null 2>&1; then
  docker restart "$CONTAINER_NAME" >/dev/null
  echo "Restarted container: $CONTAINER_NAME"
fi

ss -ltnp | grep -E "(:8554|:8888|:8889|:9997)" || true

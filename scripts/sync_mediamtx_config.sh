#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ATLAS_ENV_FILE:-$REPO_ROOT/.env.local}"
TEMPLATE_PATH="${MEDIAMTX_TEMPLATE_PATH:-$REPO_ROOT/ops/mediamtx/mediamtx.yml.tmpl}"
CONFIG_DST="${MEDIAMTX_CONFIG_DST:-/opt/mediamtx/mediamtx.yml}"
CONTAINER_NAME="${MEDIAMTX_CONTAINER_NAME:-mediamtx}"
RECORD_DELETE_AFTER="${MEDIAMTX_RECORD_DELETE_AFTER:-72h}"
READER_USER="${MEDIAMTX_READ_USER:-atlas}"
READER_PASS="${MEDIAMTX_READ_PASS:-}"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "ERROR: template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

if [[ -z "$READER_PASS" && -f "$ENV_FILE" ]]; then
  READER_PASS="$(grep -E '^DASHBOARD_API_TOKEN=' "$ENV_FILE" | tail -n 1 | cut -d= -f2-)"
fi

if [[ -z "$READER_PASS" ]]; then
  echo "ERROR: MEDIAMTX_READ_PASS is empty and DASHBOARD_API_TOKEN was not found" >&2
  exit 1
fi

if ! grep -q '\${MEDIAMTX_RECORD_DELETE_AFTER}' "$TEMPLATE_PATH"; then
  echo "ERROR: template missing MEDIAMTX_RECORD_DELETE_AFTER placeholder" >&2
  exit 1
fi
if ! grep -q '\${MEDIAMTX_READ_USER}' "$TEMPLATE_PATH"; then
  echo "ERROR: template missing MEDIAMTX_READ_USER placeholder" >&2
  exit 1
fi
if ! grep -q '\${MEDIAMTX_READ_PASS}' "$TEMPLATE_PATH"; then
  echo "ERROR: template missing MEDIAMTX_READ_PASS placeholder" >&2
  exit 1
fi

escape_sed_replacement() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//&/\\&}"
  value="${value//|/\\|}"
  printf '%s' "$value"
}

rendered_file="$(mktemp)"
trap 'rm -f "$rendered_file"' EXIT

record_delete_escaped="$(escape_sed_replacement "$RECORD_DELETE_AFTER")"
reader_user_escaped="$(escape_sed_replacement "$READER_USER")"
reader_pass_escaped="$(escape_sed_replacement "$READER_PASS")"

sed \
  -e "s|\${MEDIAMTX_RECORD_DELETE_AFTER}|${record_delete_escaped}|g" \
  -e "s|\${MEDIAMTX_READ_USER}|${reader_user_escaped}|g" \
  -e "s|\${MEDIAMTX_READ_PASS}|${reader_pass_escaped}|g" \
  "$TEMPLATE_PATH" > "$rendered_file"

if grep -q '\${MEDIAMTX_' "$rendered_file"; then
  echo "ERROR: unresolved MEDIAMTX placeholders remain" >&2
  exit 1
fi
if ! grep -q '^[[:space:]]*recordDeleteAfter:' "$rendered_file"; then
  echo "ERROR: rendered config missing recordDeleteAfter" >&2
  exit 1
fi
if ! grep -q '^[[:space:]]*authMethod:' "$rendered_file"; then
  echo "ERROR: rendered config missing authMethod" >&2
  exit 1
fi
if ! grep -q '^[[:space:]]*authInternalUsers:' "$rendered_file"; then
  echo "ERROR: rendered config missing authInternalUsers" >&2
  exit 1
fi

if [[ -f "$CONFIG_DST" ]]; then
  backup_path="${CONFIG_DST}.bak.$(date +%Y%m%d%H%M%S)"
  cp "$CONFIG_DST" "$backup_path"
  echo "Backed up existing config: $backup_path"
fi

install -m 0644 "$rendered_file" "$CONFIG_DST"
echo "Applied config: $CONFIG_DST"

echo "Using MediaMTX read credentials: user=$READER_USER pass=[hidden]"

if command -v docker >/dev/null 2>&1; then
  docker restart "$CONTAINER_NAME" >/dev/null
  echo "Restarted container: $CONTAINER_NAME"
fi

ss -ltnp | grep -E "(:8554|:8888|:8889|:9997)" || true

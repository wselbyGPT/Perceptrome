#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PATCH_FILE="${2:-genostream_protein_ce_v4.patch}"

cd "$ROOT"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[upgrade] patch not found: $PATCH_FILE" >&2
  exit 2
fi

# Clean python caches that can confuse diffs / imports
find genostream -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

echo "[upgrade] Dry-run patch..."
patch --dry-run -p0 < "$PATCH_FILE" >/dev/null

echo "[upgrade] Applying patch..."
patch -p0 < "$PATCH_FILE" >/dev/null

python3 -m py_compile genostream/*.py stream_train.py

echo "[upgrade] OK. Categorical CE loss + protein generation improvements installed."

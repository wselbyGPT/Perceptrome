#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-"$PWD"}
PATCH_FILE=${2:-"genostream_protein_grammar_v1.patch"}

cd "$ROOT"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[upgrade] ERROR: patch not found: $PATCH_FILE" >&2
  echo "[upgrade] Put genostream_protein_grammar_v1.patch in: $ROOT" >&2
  exit 1
fi

# sanity checks
for f in stream_train.py genostream/cli.py genostream/encoding.py genostream/training.py genostream/config.py genostream/io_utils.py; do
  if [[ ! -f "$f" ]]; then
    echo "[upgrade] ERROR: expected file missing: $f" >&2
    echo "[upgrade] Are you running this from the genostream project root?" >&2
    exit 1
  fi
done

# Dry-run first
if patch --dry-run -p0 < "$PATCH_FILE" >/dev/null; then
  echo "[upgrade] patch dry-run OK (-p0)"
  patch -p0 < "$PATCH_FILE"
else
  echo "[upgrade] dry-run failed with -p0; trying -p1" >&2
  patch --dry-run -p1 < "$PATCH_FILE" >/dev/null
  echo "[upgrade] patch dry-run OK (-p1)"
  patch -p1 < "$PATCH_FILE"
fi

python3 -m py_compile stream_train.py genostream/*.py

echo "[upgrade] OK: protein grammar upgrades applied and compiled."

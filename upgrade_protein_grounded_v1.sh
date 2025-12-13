#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"
PATCH_FILE="${2:-genostream_protein_grounded_v1.patch}"

if [[ ! -d "$REPO_DIR/genostream" ]]; then
  echo "[upgrade] ERROR: '$REPO_DIR' does not look like the genostream repo (missing genostream/)." >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  # also try relative to repo dir
  if [[ -f "$REPO_DIR/$PATCH_FILE" ]]; then
    PATCH_FILE="$REPO_DIR/$PATCH_FILE"
  else
    echo "[upgrade] ERROR: patch not found: $PATCH_FILE" >&2
    exit 1
  fi
fi

cd "$REPO_DIR"

echo "[upgrade] Applying patch: $PATCH_FILE"
# -p1 because the patch was generated from two directory trees (gbce_before/... -> gbce_strict/...)
patch -p1 --forward --backup --suffix=.bak_grounded_v1 < "$PATCH_FILE"

echo "[upgrade] Quick syntax check"
python3 -m py_compile genostream/*.py

echo "[upgrade] Done. New options (aa + source=genbank):"
cat <<'MSG'
  --strict-cds / --no-strict-cds
  --require-translation / --no-require-translation
  --x-free / --no-x-free
  --require-start-m / --no-require-start-m
  --reject-partial-cds / --no-reject-partial-cds
  --min-protein-aa N
  --max-protein-aa N

Notes:
- These options affect BOTH encoding (cache key) and training, so changing them will re-encode.
- For strict, grounded data: start with
    --tokenizer aa --source genbank --strict-cds --x-free --require-start-m --reject-partial-cds
MSG

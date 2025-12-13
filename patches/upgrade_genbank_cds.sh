#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-"$(pwd)"}
PATCH_FILE=${2:-"genostream_genbank_cds.patch"}

cd "$ROOT"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[upgrade] patch not found: $PATCH_FILE" >&2
  echo "[upgrade] usage: ./upgrade_genbank_cds.sh /path/to/genostream genostream_genbank_cds.patch" >&2
  exit 2
fi

# sanity
if [[ ! -f "stream_train.py" || ! -d "genostream" ]]; then
  echo "[upgrade] not a genostream project root: $ROOT" >&2
  exit 2
fi

echo "[upgrade] applying patch: $PATCH_FILE"
patch -p0 < "$PATCH_FILE"

echo "[upgrade] ensuring cache dirs"
python3 - <<'PY'
from pathlib import Path
Path('cache/genbank').mkdir(parents=True, exist_ok=True)
PY

echo "[upgrade] quick compile"
python3 -m py_compile genostream/cli.py genostream/encoding.py genostream/ncbi_fetch.py genostream/config.py genostream/io_utils.py genostream/training.py

echo "[upgrade] done"

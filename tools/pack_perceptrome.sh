#!/usr/bin/env bash
set -euo pipefail

# Genostream packer:
# - stages a clean copy into a temp dir with excludes
# - writes manifest.txt (git hash, status, env, tree snapshot)
# - produces a timestamped .zip (or .tar.gz fallback)
#
# Usage:
#   ./tools/pack_perceptrome.sh
#   ./tools/pack_perceptrome.sh --outdir /path/to/out --name perceptrome
#   ./tools/pack_perceptrome.sh --include-runs
#   ./tools/pack_perceptrome.sh --include-cache
#   ./tools/pack_perceptrome.sh --include-venv

OUTDIR="artifacts"
NAME="perceptrome"
INCLUDE_RUNS=0
INCLUDE_CACHE=0
INCLUDE_VENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --include-runs) INCLUDE_RUNS=1; shift;;
    --include-cache) INCLUDE_CACHE=1; shift;;
    --include-venv) INCLUDE_VENV=1; shift;;
    -h|--help)
      cat <<USAGE
Usage: $0 [--outdir DIR] [--name NAME] [--include-runs] [--include-cache] [--include-venv]
Defaults:
  --outdir artifacts
  --name   perceptrome
Excludes by default: venv/.venv, __pycache__, bench_runs, .git, node_modules, caches, large artifacts.
USAGE
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# Resolve repo root: prefer git root, else current dir
if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel)"
else
  ROOT="$(pwd)"
fi

mkdir -p "$OUTDIR"

TS="$(date +%Y%m%d_%H%M%S)"
GIT_SHA="nogit"
GIT_DIRTY="clean"
if command -v git >/dev/null 2>&1 && (cd "$ROOT" && git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
  GIT_SHA="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo nogit)"
  if [[ -n "$(cd "$ROOT" && git status --porcelain 2>/dev/null || true)" ]]; then
    GIT_DIRTY="dirty"
  fi
fi

BASENAME="${NAME}_${TS}_${GIT_SHA}_${GIT_DIRTY}"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# Build exclude list for rsync
EXCLUDES=(
  ".git/"
  "$OUTDIR/"
  "__pycache__/"
  "*.pyc"
  ".pytest_cache/"
  ".mypy_cache/"
  ".ruff_cache/"
  ".cache/"
  "node_modules/"
  ".DS_Store"
  "Thumbs.db"
  "*.log"
  "*.tmp"
)

if [[ "$INCLUDE_VENV" -eq 0 ]]; then
  EXCLUDES+=("venv/" ".venv/")
fi

if [[ "$INCLUDE_RUNS" -eq 0 ]]; then
  EXCLUDES+=("bench_runs/" "runs/" "wandb/" "tensorboard/" "tb/" "checkpoints/")
fi

if [[ "$INCLUDE_CACHE" -eq 0 ]]; then
  EXCLUDES+=("cache/" "caches/" "data_cache/" "ncbi_cache/" "downloads/" "tmp/")
fi

# Stage a clean snapshot
RSYNC_ARGS=(-a --delete)
for pat in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=(--exclude "$pat")
done

echo "[pack] Root: $ROOT"
echo "[pack] Out:  $OUTDIR"
echo "[pack] Name: $BASENAME"
echo "[pack] Excluding: ${#EXCLUDES[@]} patterns"

rsync "${RSYNC_ARGS[@]}" "$ROOT"/ "$TMP"/

# Manifest
MANIFEST="$TMP/manifest.txt"
{
  echo "Project: $NAME"
  echo "Packed:  $(date -Iseconds)"
  echo "Root:    $ROOT"
  echo "Git:     $GIT_SHA ($GIT_DIRTY)"
  echo
  echo "=== System ==="
  uname -a || true
  echo
  echo "=== Python ==="
  (python3 -V 2>/dev/null || true)
  echo
  echo "=== Pip Freeze (if available) ==="
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>/dev/null || true
import sys, subprocess
try:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False)
except Exception:
    pass
PY
  fi
  echo
  echo "=== Git Status (if available) ==="
  if command -v git >/dev/null 2>&1 && (cd "$ROOT" && git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
    (cd "$ROOT" && git status --porcelain || true)
  fi
  echo
  echo "=== Tree (depth 4, if available) ==="
  if command -v tree >/dev/null 2>&1; then
    (cd "$TMP" && tree -L 4) || true
  else
    echo "(tree not installed)"
  fi
} > "$MANIFEST"

# Make archive
ZIP_PATH="$OUTDIR/$BASENAME.zip"
TAR_PATH="$OUTDIR/$BASENAME.tar.gz"

if command -v zip >/dev/null 2>&1; then
  (cd "$TMP" && zip -rq "$ROOT/$ZIP_PATH" .)
  echo "[pack] Wrote: $ZIP_PATH"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ZIP_PATH" > "$ZIP_PATH.sha256"
    echo "[pack] Wrote: $ZIP_PATH.sha256"
  fi
else
  echo "[pack] 'zip' not found; falling back to tar.gz" >&2
  (cd "$TMP" && tar -czf "$ROOT/$TAR_PATH" .)
  echo "[pack] Wrote: $TAR_PATH"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$TAR_PATH" > "$TAR_PATH.sha256"
    echo "[pack] Wrote: $TAR_PATH.sha256"
  fi
  echo "[pack] Tip: sudo apt-get update && sudo apt-get install -y zip"
fi

echo "[pack] Done."

#!/usr/bin/env bash
# build.sh — bootstrap the perceptrome  project tree + docs
# Run from: /home/wselby/perceptrome  (or wherever your repo root is)

set -euo pipefail

ROOT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$ROOT_DIR")"

echo "Initializing project tree in: $ROOT_DIR"
echo "Project: $PROJECT_NAME"

# ---------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------
mkdir -p "$ROOT_DIR/config"
mkdir -p "$ROOT_DIR/state"
mkdir -p "$ROOT_DIR/cache/fasta"
mkdir -p "$ROOT_DIR/cache/encoded"
mkdir -p "$ROOT_DIR/cache/genbank"
mkdir -p "$ROOT_DIR/generated"
mkdir -p "$ROOT_DIR/model/checkpoints"
mkdir -p "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/patches"
mkdir -p "$ROOT_DIR/tools"
mkdir -p "$ROOT_DIR/docs"
mkdir -p "$ROOT_DIR/examples"

# ---------------------------------------------------------------------
# Create config/plasmids_5.txt (only if it doesn't exist)
# ---------------------------------------------------------------------
PLASMID_LIST="$ROOT_DIR/config/plasmids_5.txt"
if [[ ! -f "$PLASMID_LIST" ]]; then
  cat > "$PLASMID_LIST" << 'EOF'
# Five example plasmid accessions for streaming prototype
# accession    # comment (human-readable name)
L09137.2      # pUC19
J01749.1      # pBR322
U13853.1      # pGEX-4T-1
EF442785.1    # pET28a-LIC
CP060383.1    # pRK100 (large conjugative plasmid)
EOF
  echo "Created: $PLASMID_LIST"
else
  echo "Exists, not overwriting: $PLASMID_LIST"
fi

# ---------------------------------------------------------------------
# Create config/stream_config.yaml (only if it doesn't exist)
# ---------------------------------------------------------------------
STREAM_CONFIG="$ROOT_DIR/config/stream_config.yaml"
if [[ ! -f "$STREAM_CONFIG" ]]; then
  cat > "$STREAM_CONFIG" << 'EOF'
# stream_config.yaml — default configuration for perceptrome 

ncbi:
  email: "you@example.com"      # TODO: set to your email for NCBI
  api_key: null                 # optional NCBI API key
  max_retries: 3
  backoff_seconds: 2

training:
  steps_per_plasmid: 50         # gradient steps per catalog item visit
  batch_size: 16
  window_size: 512
  stride: 256
  max_stream_epochs: 100
  shuffle_catalog: true

io:
  cache_fasta_dir: "cache/fasta"
  cache_genbank_dir: "cache/genbank"
  cache_encoded_dir: "cache/encoded"
  generated_dir: "generated"
  model_dir: "model"
  checkpoints_dir: "model/checkpoints"
  logs_dir: "logs"
  state_file: "state/progress.json"
EOF
  echo "Created: $STREAM_CONFIG"
else
  echo "Exists, not overwriting: $STREAM_CONFIG"
fi

# ---------------------------------------------------------------------
# Initialize state/progress.json (only if it doesn't exist)
# ---------------------------------------------------------------------
STATE_FILE="$ROOT_DIR/state/progress.json"
if [[ ! -f "$STATE_FILE" ]]; then
  cat > "$STATE_FILE" << 'EOF'
{
  "current_index": 0,
  "total_steps": 0,
  "catalog_visit_counts": {},
  "last_checkpoint": null
}
EOF
  echo "Created: $STATE_FILE"
else
  echo "Exists, not overwriting: $STATE_FILE"
fi

# ---------------------------------------------------------------------
# Touch log files
# ---------------------------------------------------------------------
TRAIN_LOG="$ROOT_DIR/logs/training.log"
FETCH_LOG="$ROOT_DIR/logs/fetch.log"
ENCODE_LOG="$ROOT_DIR/logs/encode.log"
SCOPE_LOG="$ROOT_DIR/logs/scope.log"

touch "$TRAIN_LOG" "$FETCH_LOG" "$ENCODE_LOG" "$SCOPE_LOG"

echo "Ensured log files:"
echo "  $TRAIN_LOG"
echo "  $FETCH_LOG"
echo "  $ENCODE_LOG"
echo "  $SCOPE_LOG"

# ---------------------------------------------------------------------
# Create a stub stream_train.py (only if it doesn't exist)
# ---------------------------------------------------------------------
STREAM_TRAIN="$ROOT_DIR/stream_train.py"
if [[ ! -f "$STREAM_TRAIN" ]]; then
  cat > "$STREAM_TRAIN" << 'EOF'
#!/usr/bin/env python3
"""
stream_train.py — CLI entrypoint for perceptrome streaming trainer.

Design-only stub for now. Typical future subcommands:
  - init
  - catalog-show
  - fetch-one
  - encode-one
  - train-one
  - stream
  - validate
"""

def main() -> int:
    print("stream_train.py stub — implement CLI here.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
EOF
  chmod +x "$STREAM_TRAIN"
  echo "Created stub: $STREAM_TRAIN"
else
  echo "Exists, not overwriting: $STREAM_TRAIN"
fi

# ---------------------------------------------------------------------
# Documentation scaffolding (only if missing)
# ---------------------------------------------------------------------

# README.md
README="$ROOT_DIR/README.md"
if [[ ! -f "$README" ]]; then
  cat > "$README" << EOF
# $PROJECT_NAME

A streaming genome/proteome learning playground , built for iterative experiments:
fetch → encode → train → visualize ("scope") → validate.

## Quick start

\`\`\`bash
# from repo root
./build.sh

# (optional) activate your venv
# source venv/bin/activate

# run the streaming trainer (current stub / evolving)
python3 stream_train.py
\`\`\`

## Project layout (high level)

- \`scripts/\` — core Python modules (fetch, encoding, training, scope)
- \`config/\` — YAML and catalog files
- \`cache/\` — downloaded + encoded artifacts
- \`model/\` — checkpoints and model outputs
- \`state/\` — progress / resume metadata
- \`docs/\` — project documentation
- \`VALIDATE_PROTEOME_MODE.md\` — validation notes for proteome mode

## Docs

- \`docs/INSTALL.md\`
- \`docs/CONFIG.md\`
- \`docs/PIPELINE.md\`
- \`docs/TROUBLESHOOTING.md\`
- \`ARCHITECTURE.md\`

## License

See \`LICENSE\`.
EOF
  echo "Created: $README"
else
  echo "Exists, not overwriting: $README"
fi

# LICENSE (MIT default)
LICENSE="$ROOT_DIR/LICENSE"
if [[ ! -f "$LICENSE" ]]; then
  cat > "$LICENSE" << 'EOF'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
  echo "Created: $LICENSE"
else
  echo "Exists, not overwriting: $LICENSE"
fi

# CHANGELOG.md
CHANGELOG="$ROOT_DIR/CHANGELOG.md"
if [[ ! -f "$CHANGELOG" ]]; then
  cat > "$CHANGELOG" << 'EOF'
# Changelog

## Unreleased
- Initial scaffolding for streaming genome/proteome experiments.
EOF
  echo "Created: $CHANGELOG"
else
  echo "Exists, not overwriting: $CHANGELOG"
fi

# CONTRIBUTING.md
CONTRIB="$ROOT_DIR/CONTRIBUTING.md"
if [[ ! -f "$CONTRIB" ]]; then
  cat > "$CONTRIB" << 'EOF'
# Contributing

This project is currently fast-moving and experimental.

## Dev setup
- Python 3.10+ recommended
- Create/activate a venv
- Install deps (when requirements exist)

## Style
- Keep scripts small and composable
- Prefer explicit CLI flags over hidden globals
- Write logs to `logs/` (avoid noisy stdout in library code)

## Pull requests
- Describe what you changed and why
- Include reproduction commands and expected output
EOF
  echo "Created: $CONTRIB"
else
  echo "Exists, not overwriting: $CONTRIB"
fi

# ARCHITECTURE.md
ARCH="$ROOT_DIR/ARCHITECTURE.md"
if [[ ! -f "$ARCH" ]]; then
  cat > "$ARCH" << 'EOF'
# Architecture

## Goal
A streaming pipeline that repeatedly:
1) fetches sequences (FASTA/GenBank),
2) encodes them into tokens/windows,
3) trains an online model,
4) visualizes training dynamics ("scope"),
5) validates modes (e.g., proteome mode).

## Key modules (current)
- `scripts/ncbi_fetch.py` — data acquisition + caching
- `scripts/encoding.py` — tokenization/encoding (e.g., codon mode)
- `scripts/training.py` — streaming trainer / checkpointing
- `scripts/scope.py` — live diagnostics visualizer
- `config/stream_config.yaml` — runtime parameters

## State
- `state/progress.json` tracks resume state and counters
- `model/checkpoints/` stores checkpoints

## Notes
This file is intentionally minimal for now; expand as components stabilize.
EOF
  echo "Created: $ARCH"
else
  echo "Exists, not overwriting: $ARCH"
fi

# SECURITY.md
SECURITY="$ROOT_DIR/SECURITY.md"
if [[ ! -f "$SECURITY" ]]; then
  cat > "$SECURITY" << 'EOF'
# Security

This project is not currently hardened for untrusted inputs.

## Reporting
If you discover a security issue:
- Do not publish details publicly (yet).
- Open an issue with minimal details, or contact the maintainer directly.

## Notes
Be careful when processing external sequence files:
- Validate file sizes
- Avoid executing any downloaded content
EOF
  echo "Created: $SECURITY"
else
  echo "Exists, not overwriting: $SECURITY"
fi

# CITATION.cff (minimal)
CITATION="$ROOT_DIR/CITATION.cff"
if [[ ! -f "$CITATION" ]]; then
  cat > "$CITATION" << EOF
cff-version: 1.2.0
message: "If you use this software, please cite it."
title: "$PROJECT_NAME"
type: software
authors:
  - family-names: "Selby"
    given-names: "William"
EOF
  echo "Created: $CITATION"
else
  echo "Exists, not overwriting: $CITATION"
fi

# docs/INSTALL.md
DOC_INSTALL="$ROOT_DIR/docs/INSTALL.md"
if [[ ! -f "$DOC_INSTALL" ]]; then
  cat > "$DOC_INSTALL" << 'EOF'
# Install

## Requirements
- Ubuntu (WSL2 is fine)
- Python 3.10+ recommended

## Setup (minimal)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
# pip install -r requirements.txt   # when added

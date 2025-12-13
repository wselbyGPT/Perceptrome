#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PERCEPTROME_ENV=dev

exec uvicorn server.main:app \
  --host 127.0.0.1 --port 9000 \
  --reload --log-level info

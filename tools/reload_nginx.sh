#!/usr/bin/env bash
set -euo pipefail
sudo nginx -t
sudo systemctl reload nginx 2>/dev/null || sudo service nginx reload
echo "[ok] nginx reloaded"

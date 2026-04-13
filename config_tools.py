"""Config loading, overrides, and lightweight validation for the TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_base_config(path: str = "config/stream_config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists() or yaml is None:
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def coerce_value(raw: str) -> Any:
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_override(config: dict[str, Any], item: str) -> tuple[bool, str]:
    if "=" not in item:
        return False, "missing '='"
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key:
        return False, "empty key"
    parts = [part for part in key.split(".") if part]
    if not parts:
        return False, "invalid key"
    cur = config
    for segment in parts[:-1]:
        nxt = cur.get(segment)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[segment] = nxt
        cur = nxt
    cur[parts[-1]] = coerce_value(raw.strip())
    return True, "ok"


def validate_effective(config: dict[str, Any]) -> list[str]:
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    io_cfg = config.get("io", {}) if isinstance(config.get("io"), dict) else {}

    issues: list[str] = []
    window = int(training.get("window_size", 0) or 0)
    stride = int(training.get("stride", 0) or 0)
    batch = int(training.get("batch_size", 0) or 0)
    tokenizer = str(training.get("tokenizer", "base") or "base")

    issues.append("PASS: window/stride positive" if window > 0 and stride > 0 else "FAIL: window/stride must be positive")
    if tokenizer == "codon":
        issues.append("PASS: codon divisibility" if window % 3 == 0 and stride % 3 == 0 else "FAIL: codon tokenizer requires window/stride divisible by 3")
    else:
        issues.append("PASS: tokenizer compatibility")
    issues.append("PASS: batch size > 0" if batch > 0 else "FAIL: batch_size must be > 0")
    issues.append("PASS: checkpoints dir configured" if bool(io_cfg.get("checkpoints_dir")) else "FAIL: io.checkpoints_dir missing")
    return issues

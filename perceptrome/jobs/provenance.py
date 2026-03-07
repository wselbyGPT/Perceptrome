from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from perceptrome.run_layout import RunLayout, path_in_run, update_run_manifest


DEFAULT_SEED = 1337


def resolve_seed(seed: Optional[int], default_seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    if seed is None:
        return {"value": int(default_seed), "source": "default"}
    return {"value": int(seed), "source": "provided"}


def set_global_seeds(seed: int) -> Dict[str, int]:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    if torch is not None:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    return {"python_random": s, "numpy": s, "torch": s}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        info["commit"] = commit or None
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8")
        info["dirty"] = bool(status.strip())
    except Exception:
        pass
    return info


def _package_snapshot() -> Dict[str, Any]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT)
        lines = [ln.strip() for ln in out.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        return {"tool": "pip freeze", "packages": sorted(lines)}
    except Exception as e:
        return {"tool": "pip freeze", "error": str(e), "packages": []}


def _device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if torch is None:
        info["torch"] = {"available": False}
        return info

    tinfo: Dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        count = int(torch.cuda.device_count())
        tinfo["cuda_device_count"] = count
        tinfo["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
    info["torch"] = tinfo
    return info


def _checksums(paths: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for label, raw_path in paths.items():
        p = str(raw_path)
        row: Dict[str, Any] = {"path": p, "exists": os.path.exists(p)}
        if os.path.isfile(p):
            row["sha256"] = _sha256_file(p)
            row["size_bytes"] = os.path.getsize(p)
        out[str(label)] = row
    return out


def collect_and_write_provenance(
    *,
    layout: RunLayout,
    run_kind: str,
    seed_info: Mapping[str, Any],
    input_paths: Optional[Mapping[str, str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": layout.run_id,
        "run_kind": str(run_kind),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "packages": _package_snapshot(),
        },
        "git": _git_info(),
        "system": _device_info(),
        "rng": {
            "seed_source": seed_info.get("source"),
            "seed": int(seed_info["value"]),
            "libraries": {
                "python_random": int(seed_info["value"]),
                "numpy": int(seed_info["value"]),
                "torch": int(seed_info["value"]),
            },
        },
        "input_checksums": _checksums(input_paths or {}),
    }
    if extra:
        payload["extra"] = dict(extra)

    out_path = path_in_run(layout, "provenance", "provenance.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    out_sha = _sha256_file(out_path)

    update_run_manifest(
        layout,
        paths={"provenance": {"provenance_json": out_path, "provenance_sha256": out_sha}},
        provenance={"collector": {"path": out_path, "sha256": out_sha, "run_kind": str(run_kind)}},
    )
    return {"path": out_path, "sha256": out_sha, "payload": payload}

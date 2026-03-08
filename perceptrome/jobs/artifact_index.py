from __future__ import annotations

import hashlib
import mimetypes
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def compute_file_digest(path: str) -> tuple[Optional[str], Optional[int]]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None, None
    digest = hashlib.sha256()
    size_bytes = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def build_artifact_entry(
    *,
    artifact_id: str,
    role: str,
    path: str,
    created_at: Optional[str] = None,
    artifact_type: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parents: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sha256, size_bytes = compute_file_digest(path)
    inferred_mime = mime_type or mimetypes.guess_type(path)[0]
    entry: Dict[str, Any] = {
        "id": str(artifact_id),
        "role": str(role),
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "created_at": created_at or utc_now_iso(),
    }
    if artifact_type:
        entry["type"] = str(artifact_type)
    if inferred_mime:
        entry["mime_type"] = str(inferred_mime)
    if metadata:
        entry["metadata"] = dict(metadata)
    if parents:
        entry["parents"] = [dict(parent) for parent in parents if isinstance(parent, dict)]
    return entry


def append_artifact_entries(payload: Dict[str, Any], entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    artifacts = payload.setdefault("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
        payload["artifacts"] = artifacts

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        artifact_id = str(entry.get("id") or "")
        path = str(entry.get("path") or "")
        if not artifact_id and not path:
            continue
        idx = next((i for i, old in enumerate(artifacts) if (old.get("id") == artifact_id and artifact_id) or (old.get("path") == path and path)), None)
        if idx is None:
            artifacts.append(entry)
        else:
            artifacts[idx] = entry
    return payload


def normalize_artifact_list(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(dict(item))
    return out

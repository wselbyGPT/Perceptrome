from __future__ import annotations

import json
from typing import Any, Dict

from perceptrome.jobs.manifest_schema import normalize_run_manifest


def load_run_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_run_manifest(payload)


def build_lineage_adjacency(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_run_manifest(manifest)
    run = normalized.get("run") if isinstance(normalized.get("run"), dict) else {}
    run_id = str(run.get("id") or "run")

    nodes: Dict[str, Dict[str, Any]] = {
        f"run:{run_id}": {"id": f"run:{run_id}", "kind": "run", "payload": {"run_id": run_id}}
    }
    edges: list[Dict[str, Any]] = []

    def _ensure_node(ref: Dict[str, Any]) -> str:
        artifact_id = str(ref.get("artifact_id") or "").strip()
        path = str(ref.get("path") or "").strip()
        if artifact_id:
            node_id = f"artifact:{artifact_id}"
        elif path:
            node_id = f"path:{path}"
        else:
            node_id = "unknown:lineage_ref"
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "kind": "artifact_ref", "payload": dict(ref)}
        return node_id

    for parent in run.get("parents") or []:
        if not isinstance(parent, dict):
            continue
        parent_id = _ensure_node(parent)
        edges.append({"source": parent_id, "target": f"run:{run_id}", "relation": str(parent.get("relation") or "parent")})

    for child in run.get("children") or []:
        if not isinstance(child, dict):
            continue
        child_id = _ensure_node(child)
        edges.append({"source": f"run:{run_id}", "target": child_id, "relation": str(child.get("relation") or "child")})

    for artifact in normalized.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("id") or artifact.get("path") or "")
        if not artifact_id:
            continue
        artifact_node_id = f"artifact:{artifact_id}"
        nodes.setdefault(
            artifact_node_id,
            {
                "id": artifact_node_id,
                "kind": "artifact",
                "payload": {
                    "artifact_id": artifact.get("id"),
                    "path": artifact.get("path"),
                    "role": artifact.get("role"),
                    "type": artifact.get("type"),
                },
            },
        )
        edges.append({"source": f"run:{run_id}", "target": artifact_node_id, "relation": "emits.artifact"})

        for parent in artifact.get("parents") or []:
            if not isinstance(parent, dict):
                continue
            parent_id = _ensure_node(parent)
            edges.append({"source": parent_id, "target": artifact_node_id, "relation": str(parent.get("relation") or "artifact_parent")})

    return {"run_id": run_id, "nodes": list(nodes.values()), "edges": edges}

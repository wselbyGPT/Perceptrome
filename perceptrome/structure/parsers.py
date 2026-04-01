from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_RANK_PATTERN = re.compile(r"rank[_-]?(\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveredFoldArtifacts:
    structures_pdb: List[str]
    structures_cif: List[str]
    json_metadata: List[str]
    msa_files: List[str]
    image_files: List[str]
    ranking_json: Optional[str]
    result_jsons: List[str]


def _find_rank_index(path: str) -> Optional[int]:
    m = _RANK_PATTERN.search(os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _sorted_with_rank(paths: Iterable[str]) -> List[str]:
    def _key(path: str) -> tuple[int, str]:
        idx = _find_rank_index(path)
        return (idx if idx is not None else 9999, os.path.basename(path))

    return sorted((str(p) for p in paths), key=_key)


def discover_colabfold_outputs(output_dir: str) -> DiscoveredFoldArtifacts:
    pdbs: List[str] = []
    cifs: List[str] = []
    jsons: List[str] = []
    msas: List[str] = []
    images: List[str] = []
    ranking: Optional[str] = None
    result_jsons: List[str] = []

    for root, _, files in os.walk(output_dir):
        for filename in files:
            path = os.path.join(root, filename)
            low = filename.lower()
            if low.endswith(".pdb"):
                pdbs.append(path)
            elif low.endswith(".cif"):
                cifs.append(path)
            elif low.endswith(".a3m"):
                msas.append(path)
            elif low.endswith((".png", ".jpg", ".jpeg", ".svg")):
                images.append(path)
            elif low.endswith(".json"):
                jsons.append(path)
                if low == "ranking_debug.json":
                    ranking = path
                if "result_" in low:
                    result_jsons.append(path)

    return DiscoveredFoldArtifacts(
        structures_pdb=_sorted_with_rank(pdbs),
        structures_cif=_sorted_with_rank(cifs),
        json_metadata=sorted(jsons),
        msa_files=sorted(msas),
        image_files=sorted(images),
        ranking_json=ranking,
        result_jsons=_sorted_with_rank(result_jsons),
    )


def read_json_if_exists(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_plddt_values(result_json_path: str) -> List[float]:
    payload = read_json_if_exists(result_json_path)
    raw = payload.get("plddt")
    if not isinstance(raw, list):
        return []
    vals: List[float] = []
    for item in raw:
        try:
            vals.append(float(item))
        except Exception:
            continue
    return vals

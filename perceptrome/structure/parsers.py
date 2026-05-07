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


def discover_alphafold3_outputs(output_dir: str) -> DiscoveredFoldArtifacts:
    cifs: List[str] = []
    pdbs: List[str] = []
    jsons: List[str] = []
    msas: List[str] = []
    images: List[str] = []
    ranking: Optional[str] = None
    summary_confidences: List[str] = []
    top_level_summary: Optional[str] = None
    top_level_model_cif: Optional[str] = None

    if not os.path.isdir(output_dir):
        return DiscoveredFoldArtifacts(
            structures_pdb=[],
            structures_cif=[],
            json_metadata=[],
            msa_files=[],
            image_files=[],
            ranking_json=None,
            result_jsons=[],
        )

    norm_root = os.path.normpath(output_dir)
    for root, _, files in os.walk(output_dir):
        for filename in files:
            path = os.path.join(root, filename)
            low = filename.lower()
            at_root = os.path.normpath(os.path.dirname(path)) == norm_root
            if low.endswith(".cif"):
                cifs.append(path)
                if at_root and low.endswith("_model.cif"):
                    top_level_model_cif = path
            elif low.endswith(".pdb"):
                pdbs.append(path)
            elif low.endswith(".a3m") or low.endswith(".sto"):
                msas.append(path)
            elif low.endswith((".png", ".jpg", ".jpeg", ".svg")):
                images.append(path)
            elif low.endswith(".json"):
                jsons.append(path)
                if low == "summary_confidences.json" or low.endswith("_summary_confidences.json"):
                    summary_confidences.append(path)
                    if at_root and low.endswith("_summary_confidences.json"):
                        top_level_summary = path
            elif low == "ranking_scores.csv":
                ranking = path

    ordered_cifs: List[str] = []
    if top_level_model_cif and top_level_model_cif in cifs:
        ordered_cifs.append(top_level_model_cif)
        ordered_cifs.extend(sorted(p for p in cifs if p != top_level_model_cif))
    else:
        ordered_cifs = sorted(cifs)

    ordered_result_jsons: List[str] = []
    if top_level_summary and top_level_summary in summary_confidences:
        ordered_result_jsons.append(top_level_summary)
        ordered_result_jsons.extend(sorted(p for p in summary_confidences if p != top_level_summary))
    else:
        ordered_result_jsons = sorted(summary_confidences)

    return DiscoveredFoldArtifacts(
        structures_pdb=sorted(pdbs),
        structures_cif=ordered_cifs,
        json_metadata=sorted(jsons),
        msa_files=sorted(msas),
        image_files=sorted(images),
        ranking_json=ranking,
        result_jsons=ordered_result_jsons,
    )


def _sibling_confidences_json(summary_path: str) -> Optional[str]:
    """Given .../<prefix>summary_confidences.json, return .../<prefix>confidences.json if present."""
    if not summary_path:
        return None
    base = os.path.basename(summary_path)
    directory = os.path.dirname(summary_path)
    # top-level: <job>_summary_confidences.json -> <job>_confidences.json
    if base.endswith("_summary_confidences.json"):
        prefix = base[: -len("_summary_confidences.json")]
        candidate = os.path.join(directory, f"{prefix}_confidences.json")
        if os.path.exists(candidate):
            return candidate
    # per-sample: summary_confidences.json -> confidences.json
    if base == "summary_confidences.json":
        candidate = os.path.join(directory, "confidences.json")
        if os.path.exists(candidate):
            return candidate
    return None


def read_alphafold3_plddt_values(summary_confidences_path: str) -> List[float]:
    """Extract per-token plDDT values from an AlphaFold 3 confidences.json sibling."""
    sibling = _sibling_confidences_json(summary_confidences_path)
    if not sibling:
        return []
    payload = read_json_if_exists(sibling)
    for key in ("token_plddts", "atom_plddts"):
        raw = payload.get(key)
        if isinstance(raw, list):
            vals: List[float] = []
            for item in raw:
                try:
                    vals.append(float(item))
                except Exception:
                    continue
            if vals:
                return vals
    return []

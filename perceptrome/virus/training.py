from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VirusTrainingInput:
    catalog_path: str
    sequence_source: str
    record_source: str
    provenance: Dict[str, Any]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _catalog_path_from_catalog_manifest(path: str) -> str:
    payload = _load_json(path)
    catalog = payload.get("catalog") or {}
    catalog_path = str(catalog.get("path") or "").strip()
    if not catalog_path:
        raise ValueError(f"catalog.path missing in manifest: {path}")
    return catalog_path


def _catalog_path_from_fetch_manifest(path: str) -> tuple[Optional[str], Optional[str]]:
    payload = _load_json(path)
    source = payload.get("source") or {}
    source_type = str(source.get("type") or "").strip().lower()
    if source_type == "catalog":
        return str(source.get("catalog") or "").strip() or None, None
    if source_type == "manifest":
        catalog_manifest_path = str(source.get("manifest") or "").strip()
        if not catalog_manifest_path:
            return None, None
        return _catalog_path_from_catalog_manifest(catalog_manifest_path), catalog_manifest_path
    return None, None


def normalize_virus_training_input(
    *,
    catalog: Optional[str],
    catalog_manifest: Optional[str],
    fetch_manifest: Optional[str],
    sequence_source: Optional[str],
    segmented_policy: Optional[str],
    dedupe: Optional[str],
    metadata_path: Optional[str],
    complete_only: bool,
    refseq_only: bool,
) -> VirusTrainingInput:
    sequence_source_value = str(sequence_source or "genome").strip().lower()
    if sequence_source_value not in {"genome", "cds", "protein"}:
        raise ValueError(f"Unsupported --sequence-source: {sequence_source}")
    record_source = "fasta" if sequence_source_value == "genome" else "genbank"

    catalog_path = str(catalog or "").strip()
    resolved_catalog_manifest = str(catalog_manifest or "").strip() or None
    resolved_fetch_manifest = str(fetch_manifest or "").strip() or None

    if not catalog_path and resolved_catalog_manifest:
        catalog_path = _catalog_path_from_catalog_manifest(resolved_catalog_manifest)
    if not catalog_path and resolved_fetch_manifest:
        catalog_path, nested_manifest = _catalog_path_from_fetch_manifest(resolved_fetch_manifest)
        if nested_manifest and not resolved_catalog_manifest:
            resolved_catalog_manifest = nested_manifest

    if not catalog_path:
        raise ValueError("Unable to resolve virus catalog; provide --catalog, --catalog-manifest, or --fetch-manifest")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Virus catalog not found: {catalog_path}")

    provenance: Dict[str, Any] = {
        "catalog_manifest_path": resolved_catalog_manifest,
        "fetch_manifest_path": resolved_fetch_manifest,
        "sequence_source": sequence_source_value,
        "segmented_policy": str(segmented_policy or "none"),
        "dedupe_mode": str(dedupe or "none"),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "complete_only": bool(complete_only),
        "refseq_only": bool(refseq_only),
    }
    return VirusTrainingInput(
        catalog_path=catalog_path,
        sequence_source=sequence_source_value,
        record_source=record_source,
        provenance=provenance,
    )

from __future__ import annotations

from typing import Any, Dict

RUN_MANIFEST_SCHEMA_VERSION = 2
RUN_MANIFEST_TYPE = "run_manifest"

RUN_MANIFEST_SECTIONS = (
    "dataset_catalog_manifest",
    "tokenizer_encoding_config",
    "model_objective_config",
    "checkpoints",
    "metrics",
    "generated_sequences",
    "validation_results",
    "provenance_metadata",
)


def empty_run_manifest(*, run_kind: str, config_path: str, config_hash: str | None, run_id: str, created_at: str, git_sha: str | None) -> Dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "manifest_type": RUN_MANIFEST_TYPE,
        "run": {
            "id": run_id,
            "kind": str(run_kind),
            "created_at": created_at,
        },
        "dataset_catalog_manifest": None,
        "tokenizer_encoding_config": None,
        "model_objective_config": None,
        "checkpoints": None,
        "metrics": None,
        "generated_sequences": None,
        "validation_results": None,
        "provenance_metadata": {
            "software": {"git_sha": git_sha},
            "config": {
                "path": config_path,
                "sha256": config_hash,
            },
        },
    }


def _ensure_section_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for key in RUN_MANIFEST_SECTIONS:
        out.setdefault(key, None)
    return out


def _upgrade_legacy_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifact_type = str(payload.get("artifact_type") or "legacy_artifact")
    run_id = str(payload.get("accession") or payload.get("artifact_path") or "legacy")
    created_at = str(payload.get("encoded_at") or payload.get("fetched_at") or "")
    base = empty_run_manifest(
        run_kind=artifact_type,
        config_path=str((payload.get("config") or {}).get("path") or ""),
        config_hash=(payload.get("config") or {}).get("sha256"),
        run_id=run_id,
        created_at=created_at,
        git_sha=(payload.get("software") or {}).get("git_sha"),
    )

    base["dataset_catalog_manifest"] = {
        "accession": payload.get("accession"),
        "source": payload.get("source"),
        "artifact_path": payload.get("artifact_path"),
    }

    if artifact_type == "fetched_record":
        base["provenance_metadata"]["fetch"] = payload.get("fetch")
    if artifact_type == "encoded_windows":
        base["tokenizer_encoding_config"] = payload.get("encoding")
        base["metrics"] = {"encoded_shape": payload.get("shape")}

    base["provenance_metadata"]["legacy"] = {
        "schema_version": payload.get("schema_version"),
        "artifact_type": artifact_type,
    }
    return base


def normalize_run_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _ensure_section_defaults({})

    if payload.get("manifest_type") == RUN_MANIFEST_TYPE:
        return _ensure_section_defaults(payload)

    if int(payload.get("schema_version") or 0) == 1:
        return _upgrade_legacy_v1(payload)

    return _ensure_section_defaults(payload)


def extract_tokenizer_encoding_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_run_manifest(payload)
    section = normalized.get("tokenizer_encoding_config")
    if isinstance(section, dict):
        return section

    legacy = payload.get("encoding") if isinstance(payload, dict) else None
    if isinstance(legacy, dict):
        return legacy

    return {}

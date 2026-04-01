from __future__ import annotations

from typing import Any, Dict, List

from perceptrome.jobs.artifact_index import build_artifact_entry
from perceptrome.structure.summary import FoldSummaryRecord


def _artifact(artifact_id: str, role: str, path: str, artifact_type: str | None = None, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return build_artifact_entry(artifact_id=str(artifact_id), role=str(role), path=str(path), artifact_type=artifact_type, metadata=metadata)


def build_fold_manifest_update(
    *,
    command_name: str,
    run_id: str,
    summary_json_path: str,
    summary_tsv_path: str,
    stdout_log_path: str,
    stderr_log_path: str,
    records: List[FoldSummaryRecord],
) -> Dict[str, Any]:
    artifacts = [
        _artifact(f"fold_{command_name}_summary_json", "structure.summary", summary_json_path, artifact_type="json"),
        _artifact(f"fold_{command_name}_summary_tsv", "structure.summary", summary_tsv_path, artifact_type="tsv"),
        _artifact(f"fold_{command_name}_stdout", "provenance.fold", stdout_log_path, artifact_type="log"),
        _artifact(f"fold_{command_name}_stderr", "provenance.fold", stderr_log_path, artifact_type="log"),
    ]

    fold_paths = {
        "run_id": str(run_id),
        "summary_json": str(summary_json_path),
        "summary_tsv": str(summary_tsv_path),
    }

    for row in records:
        if row.rank_1_structure_path:
            artifacts.append(
                _artifact(
                    f"fold_{row.protein_id}_rank1",
                    "structure.rank1",
                    row.rank_1_structure_path,
                    artifact_type="structure",
                    metadata={"protein_id": row.protein_id, "engine_status": row.engine_status},
                )
            )

    metrics = {
        "fold": {
            "records": [
                {
                    "protein_id": row.protein_id,
                    "engine_status": row.engine_status,
                    "aa_length": row.aa_length,
                    "mean_plddt": row.mean_plddt,
                    "ptm": row.ptm,
                }
                for row in records
            ]
        }
    }

    provenance = {
        "fold": {
            "command": str(command_name),
            "records": len(records),
            "stdout_log": str(stdout_log_path),
            "stderr_log": str(stderr_log_path),
        }
    }

    return {
        "paths": {"fold": fold_paths},
        "metrics": metrics,
        "provenance": provenance,
        "artifacts": artifacts,
    }

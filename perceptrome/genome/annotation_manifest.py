from __future__ import annotations

from typing import Any, Dict, List, Optional

from perceptrome.jobs.artifact_index import build_artifact_entry
from perceptrome.genome.summary import GenomeAnnotationRecord


def _artifact(
    artifact_id: str,
    role: str,
    path: str,
    artifact_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_artifact_entry(
        artifact_id=str(artifact_id),
        role=str(role),
        path=str(path),
        artifact_type=artifact_type,
        metadata=metadata,
    )


def build_annotation_manifest_update(
    *,
    command_name: str,
    run_id: str,
    summary_json_path: str,
    summary_tsv_path: str,
    stdout_log_path: str,
    stderr_log_path: str,
    records: List[GenomeAnnotationRecord],
) -> Dict[str, Any]:
    artifacts: List[Dict[str, Any]] = [
        _artifact(
            f"genome_{command_name}_summary_json",
            "genome.summary",
            summary_json_path,
            artifact_type="json",
        ),
        _artifact(
            f"genome_{command_name}_summary_tsv",
            "genome.summary",
            summary_tsv_path,
            artifact_type="tsv",
        ),
        _artifact(
            f"genome_{command_name}_stdout",
            "provenance.genome",
            stdout_log_path,
            artifact_type="log",
        ),
        _artifact(
            f"genome_{command_name}_stderr",
            "provenance.genome",
            stderr_log_path,
            artifact_type="log",
        ),
    ]

    for row in records:
        if row.annotation_json_path:
            artifacts.append(
                _artifact(
                    f"genome_{row.accession}_annotation",
                    "genome.annotation",
                    row.annotation_json_path,
                    artifact_type="json",
                    metadata={
                        "accession": row.accession,
                        "annotation_status": row.annotation_status,
                        "source_format": row.source_format,
                    },
                )
            )

    genome_paths = {
        "run_id": str(run_id),
        "summary_json": str(summary_json_path),
        "summary_tsv": str(summary_tsv_path),
    }

    metrics = {
        "genome": {
            "records": [
                {
                    "accession": row.accession,
                    "annotation_status": row.annotation_status,
                    "sequence_length": row.sequence_length,
                    "gene_count": row.gene_count,
                    "cds_count": row.cds_count,
                    "promoter_count": row.promoter_count,
                    "operator_count": row.operator_count,
                    "total_nodes": row.total_nodes,
                    "total_edges": row.total_edges,
                }
                for row in records
            ]
        }
    }

    provenance = {
        "genome": {
            "command": str(command_name),
            "records": len(records),
            "stdout_log": str(stdout_log_path),
            "stderr_log": str(stderr_log_path),
        }
    }

    return {
        "paths": {"genome": genome_paths},
        "metrics": metrics,
        "provenance": provenance,
        "artifacts": artifacts,
    }

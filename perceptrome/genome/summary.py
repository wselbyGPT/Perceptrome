from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from perceptrome.genome.annotator import GenomeAnnotationResult


@dataclass(frozen=True)
class GenomeAnnotationRecord:
    accession: str
    source_input_path: str
    source_format: str
    cds_source: str
    annotation_status: str
    annotation_json_path: Optional[str]
    sequence_length: int
    cds_count: int
    gene_count: int
    promoter_count: int
    operator_count: int
    rbs_count: int
    terminator_count: int
    total_nodes: int
    total_edges: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenomeBatchSummary:
    run_id: str
    total_inputs: int
    annotated_count: int
    failed_count: int
    skipped_count: int
    started_at: str
    completed_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_genome_annotation_record(
    *,
    accession: str,
    source_input_path: str,
    annotation_status: str,
    annotation_json_path: Optional[str],
    result: Optional[GenomeAnnotationResult],
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> GenomeAnnotationRecord:
    if result is not None:
        counts = result.counts
        return GenomeAnnotationRecord(
            accession=str(accession),
            source_input_path=str(source_input_path),
            source_format=str(result.source_format),
            cds_source=str(result.cds_source),
            annotation_status=str(annotation_status),
            annotation_json_path=annotation_json_path,
            sequence_length=int(result.sequence_length),
            cds_count=int(counts.cds_count),
            gene_count=int(counts.gene_count),
            promoter_count=int(counts.promoter_count),
            operator_count=int(counts.operator_count),
            rbs_count=int(counts.rbs_count),
            terminator_count=int(counts.terminator_count),
            total_nodes=int(counts.total_nodes),
            total_edges=int(counts.total_edges),
            warnings=list(warnings or []),
            errors=list(errors or []),
            started_at=started_at or _utc_now(),
            completed_at=completed_at or _utc_now(),
        )
    return GenomeAnnotationRecord(
        accession=str(accession),
        source_input_path=str(source_input_path),
        source_format="",
        cds_source="",
        annotation_status=str(annotation_status),
        annotation_json_path=annotation_json_path,
        sequence_length=0,
        cds_count=0,
        gene_count=0,
        promoter_count=0,
        operator_count=0,
        rbs_count=0,
        terminator_count=0,
        total_nodes=0,
        total_edges=0,
        warnings=list(warnings or []),
        errors=list(errors or []),
        started_at=started_at or _utc_now(),
        completed_at=completed_at or _utc_now(),
    )


_TSV_COLUMNS = [
    "accession",
    "source_input_path",
    "source_format",
    "cds_source",
    "annotation_status",
    "annotation_json_path",
    "sequence_length",
    "cds_count",
    "gene_count",
    "promoter_count",
    "operator_count",
    "rbs_count",
    "terminator_count",
    "total_nodes",
    "total_edges",
    "warnings",
    "errors",
    "started_at",
    "completed_at",
]


def write_summary_json(path: str, rows: List[GenomeAnnotationRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"records": [row.to_dict() for row in rows]}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_summary_tsv(path: str, rows: List[GenomeAnnotationRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TSV_COLUMNS, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            payload["warnings"] = ";".join(row.warnings)
            payload["errors"] = ";".join(row.errors)
            writer.writerow({k: payload.get(k) for k in _TSV_COLUMNS})
    return path


def build_batch_summary(
    run_id: str,
    rows: List[GenomeAnnotationRecord],
    started_at: str,
    *,
    total_inputs: Optional[int] = None,
    skipped_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GenomeBatchSummary:
    annotated = sum(1 for row in rows if row.annotation_status == "ok")
    failed = sum(1 for row in rows if row.annotation_status != "ok")
    resolved_total = int(total_inputs) if total_inputs is not None else len(rows)
    resolved_skipped = (
        int(skipped_count) if skipped_count is not None else max(resolved_total - len(rows), 0)
    )
    return GenomeBatchSummary(
        run_id=str(run_id),
        total_inputs=resolved_total,
        annotated_count=annotated,
        failed_count=failed,
        skipped_count=resolved_skipped,
        started_at=str(started_at),
        completed_at=_utc_now(),
        metadata=dict(metadata or {}),
    )


def write_batch_summary_json(
    path: str, batch_summary: GenomeBatchSummary, rows: List[GenomeAnnotationRecord]
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = asdict(batch_summary)
    payload["records"] = [row.to_dict() for row in rows]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_batch_summary_tsv(path: str, batch_summary: GenomeBatchSummary) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(batch_summary).keys()), dialect="excel-tab")
        writer.writeheader()
        writer.writerow(asdict(batch_summary))
    return path

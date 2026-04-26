from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from perceptrome.encoding.parse import parse_fasta_sequence
from perceptrome.structure.features import FoldFeatures, build_fold_features
from perceptrome.structure.parsers import (
    DiscoveredFoldArtifacts,
    read_alphafold3_plddt_values,
    read_json_if_exists,
    read_plddt_values,
)


ALPHAFOLD3_ENGINE = "alphafold3"


@dataclass(frozen=True)
class FoldSummaryRecord:
    protein_id: str
    source_input_path: str
    aa_length: int
    fold_engine: str
    engine_status: str
    rank_1_structure_path: Optional[str]
    mean_plddt: Optional[float]
    min_plddt: Optional[float]
    max_plddt: Optional[float]
    ptm: Optional[float]
    rank_index: Optional[int]
    discovered_artifact_paths: Dict[str, List[str]]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FoldBatchSummary:
    run_id: str
    engine: str
    total_inputs: int
    folded_count: int
    failed_count: int
    skipped_count: int
    started_at: str
    completed_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ptm_from_json(result_json_path: Optional[str]) -> Optional[float]:
    payload = read_json_if_exists(result_json_path)
    for key in ("ptm", "predicted_tm_score", "ranking_score"):
        val = payload.get(key)
        try:
            if val is not None:
                return float(val)
        except Exception:
            pass
    return None


def _rank1_structure(artifacts: DiscoveredFoldArtifacts) -> Optional[str]:
    if artifacts.structures_pdb:
        return artifacts.structures_pdb[0]
    if artifacts.structures_cif:
        return artifacts.structures_cif[0]
    return None


def build_fold_summary_record(
    *,
    protein_id: str,
    source_input_path: str,
    engine: str,
    engine_status: str,
    artifacts: DiscoveredFoldArtifacts,
    warnings: Optional[Iterable[str]] = None,
    errors: Optional[Iterable[str]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> FoldSummaryRecord:
    seq = parse_fasta_sequence(source_input_path)
    first_result_json = artifacts.result_jsons[0] if artifacts.result_jsons else None
    if str(engine).lower() == ALPHAFOLD3_ENGINE:
        plddt = read_alphafold3_plddt_values(first_result_json) if first_result_json else []
    else:
        plddt = read_plddt_values(first_result_json) if first_result_json else []
    ptm = _ptm_from_json(first_result_json)

    feature: FoldFeatures = build_fold_features(
        aa_length=len(seq),
        plddt_values=plddt,
        ptm=ptm,
        rank_index=1 if (artifacts.structures_pdb or artifacts.structures_cif) else None,
        has_pdb=bool(artifacts.structures_pdb),
        has_cif=bool(artifacts.structures_cif),
        has_msa=bool(artifacts.msa_files),
        has_plots=bool(artifacts.image_files),
        completion_state="completed" if engine_status == "ok" else "error",
    )

    return FoldSummaryRecord(
        protein_id=str(protein_id),
        source_input_path=str(source_input_path),
        aa_length=int(feature.aa_length),
        fold_engine=str(engine),
        engine_status=str(engine_status),
        rank_1_structure_path=_rank1_structure(artifacts),
        mean_plddt=feature.mean_plddt,
        min_plddt=feature.min_plddt,
        max_plddt=feature.max_plddt,
        ptm=feature.ptm,
        rank_index=feature.rank_index,
        discovered_artifact_paths={
            "structures_pdb": list(artifacts.structures_pdb),
            "structures_cif": list(artifacts.structures_cif),
            "json_metadata": list(artifacts.json_metadata),
            "msa_files": list(artifacts.msa_files),
            "image_files": list(artifacts.image_files),
        },
        warnings=list(warnings or []),
        errors=list(errors or []),
        started_at=started_at or _utc_now(),
        completed_at=completed_at or _utc_now(),
    )


def write_summary_json(path: str, rows: List[FoldSummaryRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"records": [row.to_dict() for row in rows]}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_summary_tsv(path: str, rows: List[FoldSummaryRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [
        "protein_id",
        "source_input_path",
        "aa_length",
        "fold_engine",
        "engine_status",
        "rank_1_structure_path",
        "mean_plddt",
        "min_plddt",
        "max_plddt",
        "ptm",
        "rank_index",
        "warnings",
        "errors",
        "started_at",
        "completed_at",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            payload["warnings"] = ";".join(row.warnings)
            payload["errors"] = ";".join(row.errors)
            writer.writerow({k: payload.get(k) for k in keys})
    return path


def build_batch_summary(
    run_id: str,
    engine: str,
    rows: List[FoldSummaryRecord],
    started_at: str,
    *,
    total_inputs: Optional[int] = None,
    skipped_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FoldBatchSummary:
    failed = sum(1 for row in rows if row.engine_status != "ok")
    folded = sum(1 for row in rows if row.engine_status == "ok")
    resolved_total_inputs = int(total_inputs) if total_inputs is not None else len(rows)
    resolved_skipped = int(skipped_count) if skipped_count is not None else max(resolved_total_inputs - len(rows), 0)
    return FoldBatchSummary(
        run_id=str(run_id),
        engine=str(engine),
        total_inputs=resolved_total_inputs,
        folded_count=folded,
        failed_count=failed,
        skipped_count=resolved_skipped,
        started_at=str(started_at),
        completed_at=_utc_now(),
        metadata=dict(metadata or {}),
    )


def write_batch_summary_json(path: str, batch_summary: FoldBatchSummary, rows: List[FoldSummaryRecord]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = asdict(batch_summary)
    payload["records"] = [row.to_dict() for row in rows]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_batch_summary_tsv(path: str, batch_summary: FoldBatchSummary) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(batch_summary).keys()), dialect="excel-tab")
        writer.writeheader()
        writer.writerow(asdict(batch_summary))
    return path

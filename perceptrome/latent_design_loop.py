"""End-to-end protein design pipeline orchestrator.

Drives latent-seed → strategy → decode → score → ELBO → rank for protein
candidates. Folding is intentionally not invoked here so this module remains
unit-testable without AlphaFold3 binaries on PATH; the CLI command takes the
ranked top-K and folds them via ``_run_fold_one_internal``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from perceptrome.elbo_score import ELBORecord
from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
from perceptrome.scorecard import build_protein_scorecard


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    seed_accession: str
    cluster_id: int
    strategy: str
    latent_vector: List[float]
    sequence: str
    scorecard: Dict[str, Any]
    elbo_record: Dict[str, Any]
    combined_score: float
    fold_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_combined_score(
    scorecard: Dict[str, Any],
    elbo_record: Dict[str, Any],
    alpha: float,
    error: Optional[str],
) -> float:
    if error is not None:
        return -math.inf
    elbo_status = str((elbo_record or {}).get("status") or "ok")
    metrics = (scorecard or {}).get("metrics") or {}
    sc_score = metrics.get("score")
    if sc_score is None:
        return -math.inf
    if elbo_status != "ok":
        return float(sc_score)
    elbo_value = float((elbo_record or {}).get("elbo", 0.0))
    return float(sc_score) + float(alpha) * elbo_value


def combine_candidate_record(
    *,
    candidate_id: str,
    seed_accession: str,
    cluster_id: int,
    strategy: str,
    z_vec: Any,
    sequence: str,
    scorecard: Dict[str, Any],
    elbo_record: Dict[str, Any],
    alpha: float,
    fold_summary: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> CandidateRecord:
    """Pure constructor — no torch required if z_vec is already a list of floats."""
    if hasattr(z_vec, "detach"):
        latent_list = z_vec.detach().cpu().reshape(-1).tolist()
    elif hasattr(z_vec, "tolist"):
        latent_list = list(z_vec.reshape(-1).tolist()) if hasattr(z_vec, "reshape") else list(z_vec.tolist())
    else:
        latent_list = [float(v) for v in z_vec]
    combined = _compute_combined_score(scorecard, elbo_record, alpha, error)
    return CandidateRecord(
        candidate_id=candidate_id,
        seed_accession=seed_accession,
        cluster_id=int(cluster_id),
        strategy=str(strategy),
        latent_vector=[float(v) for v in latent_list],
        sequence=str(sequence),
        scorecard=dict(scorecard) if scorecard else {},
        elbo_record=dict(elbo_record) if elbo_record else {},
        combined_score=float(combined),
        fold_summary=dict(fold_summary) if fold_summary else None,
        error=error,
    )


def attach_fold_summary(candidate: CandidateRecord, fold_summary: Optional[Dict[str, Any]]) -> CandidateRecord:
    """Return a new CandidateRecord with ``fold_summary`` replaced."""
    return replace(candidate, fold_summary=dict(fold_summary) if fold_summary else None)


def rank_and_select_for_folding(
    candidates: List[CandidateRecord],
    fold_top_k: int,
) -> List[str]:
    """Stable-sort by combined_score DESC, return ids of top-K eligible candidates.

    Errored candidates and those with -inf scores are excluded from the result.
    """
    eligible = [c for c in candidates if c.error is None and math.isfinite(c.combined_score)]
    eligible.sort(key=lambda c: c.combined_score, reverse=True)
    k = max(0, int(fold_top_k))
    return [c.candidate_id for c in eligible[:k]]


def _fasta_lines(seq: str, width: int = 60) -> List[str]:
    return [seq[i : i + width] for i in range(0, len(seq), width)]


def write_candidate_outputs(
    out_dir: str,
    candidates: List[CandidateRecord],
) -> List[Tuple[str, str]]:
    """Write cand_NNNN.json and cand_NNNN.fasta for each candidate.

    Returns a list of ``(fasta_path, json_path)`` in candidate order.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[Tuple[str, str]] = []
    for cand in candidates:
        json_path = os.path.join(out_dir, f"{cand.candidate_id}.json")
        fasta_path = os.path.join(out_dir, f"{cand.candidate_id}.fasta")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cand.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        if cand.sequence:
            header = (
                f">{cand.candidate_id}|seed={cand.seed_accession}"
                f"|cluster={cand.cluster_id}|strategy={cand.strategy}"
            )
            with open(fasta_path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                for line in _fasta_lines(cand.sequence):
                    f.write(line + "\n")
        else:
            with open(fasta_path, "w", encoding="utf-8") as f:
                f.write(f">{cand.candidate_id}|empty\n")
        written.append((fasta_path, json_path))
    return written


def _leaderboard_row(cand: CandidateRecord) -> Dict[str, Any]:
    sc_metrics = (cand.scorecard or {}).get("metrics") or {}
    fs = cand.fold_summary or {}
    return {
        "candidate_id": cand.candidate_id,
        "seed_accession": cand.seed_accession,
        "cluster_id": int(cand.cluster_id),
        "strategy": cand.strategy,
        "combined_score": (None if not math.isfinite(cand.combined_score) else float(cand.combined_score)),
        "scorecard_score": (float(sc_metrics["score"]) if "score" in sc_metrics else None),
        "elbo": (
            float((cand.elbo_record or {}).get("elbo"))
            if (cand.elbo_record or {}).get("elbo") is not None and (cand.elbo_record or {}).get("status") == "ok"
            else None
        ),
        "mean_plddt": (float(fs["mean_plddt"]) if fs.get("mean_plddt") is not None else None),
        "ptm": (float(fs["ptm"]) if fs.get("ptm") is not None else None),
        "folded": cand.fold_summary is not None,
        "error": cand.error,
    }


def write_leaderboard_and_manifest(
    out_dir: str,
    candidates: List[CandidateRecord],
    *,
    run_id: str,
    seeds_source: str,
    strategy: str,
    n_per_seed: int,
    fold_top_k: int,
    fold_enabled: bool,
    alpha: float,
    started_at: str,
    completed_at: str,
    model_meta: Mapping[str, Any],
    extra_counts: Optional[Mapping[str, int]] = None,
) -> Tuple[str, str]:
    """Write leaderboard.json (ranked DESC) and manifest.json. Returns (lb_path, mf_path)."""
    os.makedirs(out_dir, exist_ok=True)
    leaderboard_path = os.path.join(out_dir, "leaderboard.json")
    manifest_path = os.path.join(out_dir, "manifest.json")

    rows = [_leaderboard_row(c) for c in candidates]
    rows.sort(
        key=lambda r: (r["combined_score"] if r["combined_score"] is not None else -math.inf),
        reverse=True,
    )
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "ranked": rows}, f, indent=2, sort_keys=True)
        f.write("\n")

    folded = sum(1 for c in candidates if c.fold_summary is not None)
    fold_failed = sum(
        1 for c in candidates
        if c.fold_summary is not None and str(c.fold_summary.get("engine_status") or "ok") != "ok"
    )
    scored_ok = sum(1 for c in candidates if c.error is None and (c.scorecard or {}).get("metrics", {}).get("score") is not None)
    elbo_failed = sum(
        1 for c in candidates
        if c.error is None and str((c.elbo_record or {}).get("status") or "ok") != "ok"
    )
    counts = {
        "candidates": len(candidates),
        "scored_ok": scored_ok,
        "folded": folded,
        "fold_failed": fold_failed,
        "elbo_failed": elbo_failed,
        "errored": sum(1 for c in candidates if c.error is not None),
    }
    if extra_counts:
        counts.update({str(k): int(v) for k, v in extra_counts.items()})

    manifest = {
        "run_id": run_id,
        "model": dict(model_meta),
        "seeds_source": seeds_source,
        "strategy": strategy,
        "n_per_seed": int(n_per_seed),
        "fold_top_k": int(fold_top_k),
        "fold_enabled": bool(fold_enabled),
        "alpha_elbo": float(alpha),
        "counts": counts,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return leaderboard_path, manifest_path


# ---------------------------------------------------------------------------
# Torch-touching helpers (stubs — filled in by task #5)
# ---------------------------------------------------------------------------

def decode_latent_to_protein_sequence(
    model: Any,
    z: Any,
    seq_len: int,
    vocab_size: int,
    tok: str = "aa",
    sampling_cfg: Any = None,
) -> str:
    raise NotImplementedError("decode_latent_to_protein_sequence: torch path not yet implemented")


def score_candidate(
    model: Any,
    sequence: str,
    *,
    tok: str,
    window_size: int,
    stride: int,
    seq_len: int,
    vocab_size: int,
    device: Any,
    scorecard_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raise NotImplementedError("score_candidate: torch path not yet implemented")


def run_latent_design_loop(
    *,
    model: Any,
    device: Any,
    layout: Any,
    seeds_payload: Mapping[str, Any],
    encoded_by_accession: Mapping[str, np.ndarray],
    latent_cfg: LatentConfig,
    sampling_cfg: Any,
    n_per_seed: int,
    alpha: float,
    tok: str,
    seq_len: int,
    vocab_size: int,
    window_size: int,
    stride: int,
    scorecard_context: Optional[Mapping[str, Any]] = None,
) -> List[CandidateRecord]:
    raise NotImplementedError("run_latent_design_loop: torch path not yet implemented")

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class FoldFeatures:
    aa_length: int
    mean_plddt: Optional[float]
    min_plddt: Optional[float]
    max_plddt: Optional[float]
    ptm: Optional[float]
    rank_index: Optional[int]
    has_pdb: bool
    has_cif: bool
    has_msa: bool
    has_plots: bool
    completion_state: str


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _stats(values: Iterable[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    collected: List[float] = [float(v) for v in values]
    if not collected:
        return None, None, None
    return sum(collected) / len(collected), min(collected), max(collected)


def build_fold_features(
    *,
    aa_length: int,
    plddt_values: Optional[Iterable[float]],
    ptm: Optional[float],
    rank_index: Optional[int],
    has_pdb: bool,
    has_cif: bool,
    has_msa: bool,
    has_plots: bool,
    completion_state: str,
) -> FoldFeatures:
    mean_plddt, min_plddt, max_plddt = _stats(plddt_values or [])
    return FoldFeatures(
        aa_length=int(aa_length),
        mean_plddt=mean_plddt,
        min_plddt=min_plddt,
        max_plddt=max_plddt,
        ptm=_safe_float(ptm),
        rank_index=(None if rank_index is None else int(rank_index)),
        has_pdb=bool(has_pdb),
        has_cif=bool(has_cif),
        has_msa=bool(has_msa),
        has_plots=bool(has_plots),
        completion_state=str(completion_state),
    )

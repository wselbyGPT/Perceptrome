from __future__ import annotations

import difflib
import math
import re
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .encoding.constants import START_CODON, STOP_CODONS
from .encoding.orf import find_orfs_proteins, translate_orf
from .encoding.parse import reverse_complement

SCORECARD_VERSION = "v2"


@dataclass(frozen=True)
class ScorecardMetadata:
    sequence_id: Optional[str]
    sequence_type: str
    sequence_length: int
    tokenizer: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None


@dataclass(frozen=True)
class PerSequenceMetrics:
    score: float
    gc_fraction: Optional[float] = None
    gc_deviation: Optional[float] = None
    max_homopolymer: Optional[int] = None
    homopolymer_penalty: Optional[float] = None
    x_fraction: Optional[float] = None
    invalid_fraction: Optional[float] = None
    stop_count: Optional[int] = None
    roundtrip_recon: Optional[float] = None
    recon_weight: Optional[float] = None
    seq_similarity: Optional[float] = None
    kmer_jaccard: Optional[float] = None
    gc_delta: Optional[float] = None
    length_ratio: Optional[float] = None
    gc_percent: Optional[float] = None
    repeat_density: Optional[float] = None
    repeat_burden: Optional[float] = None
    motif_hit_count: Optional[int] = None
    restriction_site_count: Optional[int] = None
    orf_count: Optional[int] = None
    longest_orf_aa: Optional[int] = None
    hydrophobic_fraction: Optional[float] = None
    charge_balance: Optional[float] = None
    aromaticity: Optional[float] = None
    instability_proxy: Optional[float] = None
    low_complexity_fraction: Optional[float] = None
    low_complexity_longest: Optional[int] = None


@dataclass(frozen=True)
class ReferenceNeighborMatch:
    reference_id: str
    reference_length: int
    score: float
    seq_similarity: float
    kmer_jaccard: float
    gc_delta: float
    length_ratio: float


@dataclass(frozen=True)
class RiskFlag:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class HumanReadableSummary:
    title: str
    highlights: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SequenceScorecard:
    scorecard_version: str
    sequence_kind: str
    metadata: ScorecardMetadata
    metrics: PerSequenceMetrics
    reference_neighbors: List[ReferenceNeighborMatch] = field(default_factory=list)
    risk_flags: List[RiskFlag] = field(default_factory=list)
    summary: Optional[HumanReadableSummary] = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    run = 1
    best = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def gc_fraction(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def kmer_set(seq: str, k: int) -> set[str]:
    if k <= 0 or len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard_kmers(a: str, b: str, k: int = 9) -> float:
    ka = kmer_set(a, k)
    kb = kmer_set(b, k)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    return float(len(ka & kb) / len(ka | kb))


def sequence_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _reference_neighbor(generated_seq: str, ref_id: str, ref_seq: str) -> ReferenceNeighborMatch:
    seq_sim = sequence_similarity(generated_seq, ref_seq)
    kmer_sim = jaccard_kmers(generated_seq, ref_seq, k=9)
    gc_delta = abs(gc_fraction(generated_seq) - gc_fraction(ref_seq))
    length_ratio = min(len(generated_seq), len(ref_seq)) / max(len(generated_seq), len(ref_seq)) if generated_seq and ref_seq else 0.0
    score = (0.55 * seq_sim) + (0.30 * kmer_sim) + (0.10 * length_ratio) + (0.05 * (1.0 - gc_delta))
    return ReferenceNeighborMatch(
        reference_id=str(ref_id),
        reference_length=len(ref_seq),
        score=float(score),
        seq_similarity=float(seq_sim),
        kmer_jaccard=float(kmer_sim),
        gc_delta=float(gc_delta),
        length_ratio=float(length_ratio),
    )


def _build_reference_neighbors(sequence: str, context: Mapping[str, Any]) -> List[ReferenceNeighborMatch]:
    matches: List[ReferenceNeighborMatch] = []
    reference_sequence = context.get("reference_sequence")
    if isinstance(reference_sequence, str):
        matches.append(_reference_neighbor(sequence, str(context.get("reference_id") or "reference"), reference_sequence))

    for item in context.get("reference_neighbors", []) or []:
        if not isinstance(item, Mapping):
            continue
        ref_seq = item.get("sequence")
        if not isinstance(ref_seq, str):
            continue
        matches.append(_reference_neighbor(sequence, str(item.get("reference_id") or item.get("id") or "reference"), ref_seq))

    matches.sort(key=lambda row: row.score, reverse=True)
    top_n = int(context.get("reference_top_n", 5))
    return matches[: max(1, top_n)]


def _normalize_motif_set(context: Mapping[str, Any]) -> Dict[str, str]:
    motifs = context.get("motifs")
    if isinstance(motifs, Mapping):
        return {str(k): str(v).upper() for k, v in motifs.items() if isinstance(v, str) and str(v).strip()}
    if isinstance(motifs, Sequence) and not isinstance(motifs, (str, bytes)):
        out: Dict[str, str] = {}
        for i, item in enumerate(motifs):
            if isinstance(item, str) and item.strip():
                out[f"motif_{i}"] = item.upper()
            elif isinstance(item, Mapping) and isinstance(item.get("sequence"), str):
                name = str(item.get("name") or f"motif_{i}")
                out[name] = str(item["sequence"]).upper()
        if out:
            return out
    return {
        "t7_promoter": "TAATACGACTCACTATAGGG",
        "lac_operator": "AATTGTGAGCGGATAACAATT",
        "bgh_polyA_signal": "AATAAA",
    }


def _restriction_enzymes(context: Mapping[str, Any]) -> Dict[str, str]:
    enzymes = context.get("restriction_enzymes")
    if isinstance(enzymes, Mapping):
        return {str(k): str(v).upper() for k, v in enzymes.items() if isinstance(v, str) and str(v).strip()}
    return {
        "EcoRI": "GAATTC",
        "BamHI": "GGATCC",
        "HindIII": "AAGCTT",
        "NotI": "GCGGCCGC",
        "XhoI": "CTCGAG",
    }


def _find_pattern_positions(seq: str, pattern: str) -> List[int]:
    hits: List[int] = []
    i = seq.find(pattern)
    while i >= 0:
        hits.append(i)
        i = seq.find(pattern, i + 1)
    return hits


def _aa_composition(sequence: str) -> Dict[str, Dict[str, float]]:
    seq = (sequence or "").upper()
    canonical = list("ACDEFGHIKLMNPQRSTVWY")
    non_standard = sorted({ch for ch in seq if ch not in canonical})
    acids = canonical + non_standard
    length = max(1, len(seq))
    out: Dict[str, Dict[str, float]] = {}
    for aa in acids:
        count = int(seq.count(aa))
        out[aa] = {"count": count, "fraction": float(count / length)}
    return out


def _window_entropy(window: str) -> float:
    if not window:
        return 0.0
    length = len(window)
    freqs = [window.count(ch) / float(length) for ch in set(window)]
    return float(-sum(p * math.log2(max(p, 1e-12)) for p in freqs))


def _low_complexity_regions(sequence: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    seq = (sequence or "").upper()
    window = max(6, int(context.get("low_complexity_window", 12)))
    entropy_threshold = float(context.get("low_complexity_entropy_threshold", 2.2))
    if len(seq) < window:
        return {
            "window": window,
            "entropy_threshold": entropy_threshold,
            "windows": [],
            "regions": [],
            "covered_fraction": 0.0,
            "longest_region": 0,
            "region_count": 0,
        }

    low_windows: List[Dict[str, Any]] = []
    for start in range(0, len(seq) - window + 1):
        chunk = seq[start : start + window]
        entropy = _window_entropy(chunk)
        if entropy <= entropy_threshold:
            low_windows.append({"start": int(start), "end": int(start + window), "entropy": float(entropy)})

    if not low_windows:
        return {
            "window": window,
            "entropy_threshold": entropy_threshold,
            "windows": [],
            "regions": [],
            "covered_fraction": 0.0,
            "longest_region": 0,
            "region_count": 0,
        }

    regions: List[Dict[str, Any]] = []
    current = dict(low_windows[0])
    for row in low_windows[1:]:
        if int(row["start"]) <= int(current["end"]):
            current["end"] = max(int(current["end"]), int(row["end"]))
            current["entropy"] = min(float(current["entropy"]), float(row["entropy"]))
            continue
        regions.append({
            "start": int(current["start"]),
            "end": int(current["end"]),
            "length": int(current["end"] - current["start"]),
            "min_entropy": float(current["entropy"]),
        })
        current = dict(row)
    regions.append({
        "start": int(current["start"]),
        "end": int(current["end"]),
        "length": int(current["end"] - current["start"]),
        "min_entropy": float(current["entropy"]),
    })

    covered = sum(int(r["length"]) for r in regions)
    longest = max((int(r["length"]) for r in regions), default=0)
    return {
        "window": window,
        "entropy_threshold": entropy_threshold,
        "windows": low_windows,
        "regions": regions,
        "covered_fraction": float(covered / float(max(1, len(seq)))),
        "longest_region": int(longest),
        "region_count": int(len(regions)),
    }


def _protein_pattern_library(context: Mapping[str, Any]) -> Dict[str, str]:
    library = context.get("protein_pattern_library")
    if isinstance(library, Mapping):
        out = {str(k): str(v) for k, v in library.items() if isinstance(v, str) and str(v).strip()}
        if out:
            return out
    return {
        "n_glycosylation": r"N[^P][ST][^P]",
        "p_loop_ntp_binding": r"[AGX]{4}GK[ST]",
        "acidic_patch": r"[DE]{4,}",
    }


def _protein_pattern_hits(sequence: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    seq = (sequence or "").upper()
    mode = str(context.get("protein_pattern_mode", "regex")).strip().lower()
    library = _protein_pattern_library(context)
    hits: Dict[str, Any] = {}
    for name, pattern in library.items():
        if mode == "literal":
            positions = _find_pattern_positions(seq, pattern.upper())
            hits[name] = {
                "pattern": pattern,
                "mode": "literal",
                "count": len(positions),
                "hits": [{"start": int(p), "end": int(p + len(pattern)), "match": pattern} for p in positions],
            }
            continue
        rows = []
        for m in re.finditer(pattern, seq):
            rows.append({"start": int(m.start()), "end": int(m.end()), "match": str(m.group(0))})
        hits[name] = {"pattern": pattern, "mode": "regex", "count": len(rows), "hits": rows}
    return hits


def _protein_proxies(sequence: str, composition: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    length = max(1, len(sequence))
    hydrophobic = set("AVILMFWYC")
    aromatic = set("FYW")
    positive = set("KRH")
    negative = set("DE")
    instability = set("PEDSTNQG")
    frac = lambda aa: float((composition.get(aa) or {}).get("fraction", 0.0))

    hydrophobic_fraction = sum(frac(aa) for aa in hydrophobic)
    aromaticity = sum(frac(aa) for aa in aromatic)
    positive_frac = sum(frac(aa) for aa in positive)
    negative_frac = sum(frac(aa) for aa in negative)
    charge_balance = positive_frac - negative_frac
    instability_proxy = (sum(frac(aa) for aa in instability) + max(0.0, frac("G") - 0.08)) * 100.0

    penalty = 0.0
    penalty += max(0.0, hydrophobic_fraction - 0.55) * 2.2
    penalty += max(0.0, aromaticity - 0.12) * 2.0
    penalty += max(0.0, 0.05 - abs(charge_balance)) * 1.5
    penalty += max(0.0, instability_proxy - 45.0) / 35.0
    solubility_proxy = max(0.0, 1.0 - penalty)
    return {
        "hydrophobic_fraction": float(hydrophobic_fraction),
        "aromaticity": float(aromaticity),
        "charge_balance": float(charge_balance),
        "instability_proxy": float(instability_proxy),
        "solubility_proxy": float(solubility_proxy),
        "length_aa": int(length),
    }


def _repeat_metrics(seq: str, min_k: int = 3, max_k: int = 6) -> Dict[str, Any]:
    L = max(1, len(seq))
    run = 1
    homopolymer_bp = 0
    long_homopolymers: List[Dict[str, int]] = []
    for i in range(1, len(seq) + 1):
        if i < len(seq) and seq[i] == seq[i - 1]:
            run += 1
            continue
        if run >= 4:
            homopolymer_bp += run
            long_homopolymers.append({"start": i - run, "length": run, "base": seq[i - 1]})
        run = 1

    kmer_components: List[float] = []
    kmer_breakdown: Dict[str, float] = {}
    for k in range(max(1, min_k), max(1, max_k) + 1):
        total = max(0, len(seq) - k + 1)
        if total == 0:
            continue
        uniq = len({seq[i : i + k] for i in range(total)})
        burden = max(0.0, (total - uniq) / float(total))
        kmer_breakdown[f"k{k}"] = float(burden)
        kmer_components.append(float(burden))

    tandem_bp = 0
    for motif_len in (2, 3):
        i = 0
        while i + motif_len * 3 <= len(seq):
            motif = seq[i : i + motif_len]
            reps = 1
            j = i + motif_len
            while j + motif_len <= len(seq) and seq[j : j + motif_len] == motif:
                reps += 1
                j += motif_len
            if reps >= 3:
                tandem_bp += reps * motif_len
                i = j
            else:
                i += 1

    homopolymer_density = homopolymer_bp / float(L)
    tandem_density = tandem_bp / float(L)
    kmer_density = mean(kmer_components) if kmer_components else 0.0
    repeat_density = float((homopolymer_density + tandem_density + kmer_density) / 3.0)
    return {
        "repeat_density": repeat_density,
        "repeat_burden": repeat_density,
        "homopolymer_density": float(homopolymer_density),
        "tandem_repeat_density": float(tandem_density),
        "kmer_repeat_density": float(kmer_density),
        "kmer_repeat_breakdown": kmer_breakdown,
        "long_homopolymers": long_homopolymers,
    }


def _orf_summary(seq: str, min_orf_aa: int) -> Dict[str, Any]:
    proteins = find_orfs_proteins(seq, min_orf_aa=min_orf_aa)
    orf_records: List[Dict[str, Any]] = []
    frame_counts: Dict[str, int] = {f"+{i}": 0 for i in (1, 2, 3)}
    frame_counts.update({f"-{i}": 0 for i in (1, 2, 3)})

    strands = [("+", seq), ("-", reverse_complement(seq))]
    for strand, strand_seq in strands:
        L = len(strand_seq)
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < L:
                cod = strand_seq[i : i + 3]
                if cod != START_CODON:
                    i += 3
                    continue
                j = i + 3
                while j + 2 < L:
                    stop = strand_seq[j : j + 3]
                    if stop in STOP_CODONS:
                        orf_nt = strand_seq[i:j]
                        aa = translate_orf(orf_nt)
                        if len(aa) >= min_orf_aa:
                            key = f"{strand}{frame + 1}"
                            frame_counts[key] = int(frame_counts.get(key, 0) + 1)
                            orf_records.append(
                                {
                                    "strand": strand,
                                    "frame": int(frame + 1),
                                    "start": int(i),
                                    "end": int(j),
                                    "length_nt": int(len(orf_nt)),
                                    "length_aa": int(len(aa)),
                                }
                            )
                        i = j + 3
                        break
                    j += 3
                else:
                    i += 3

    longest = max((int(row["length_aa"]) for row in orf_records), default=0)
    return {
        "count": len(orf_records),
        "longest_orf_aa": int(longest),
        "frame_counts": frame_counts,
        "orfs": orf_records,
        "protein_count": len(proteins),
    }


def build_plasmid_scorecard(sequence: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    target_gc = float(context.get("target_gc", 0.5))
    max_homopolymer = context.get("max_homopolymer")
    recon = context.get("roundtrip_recon")
    recon_weight = float(context.get("recon_weight", 0.0))

    gc = gc_fraction(sequence)
    run = max_homopolymer_run(sequence)
    gc_percent = gc * 100.0
    gc_dev = abs(gc - target_gc)
    run_pen = 0.0 if max_homopolymer is None else max(0, run - int(max_homopolymer)) / max(1.0, float(max_homopolymer))
    score = -float(gc_dev) - float(run_pen) - (recon_weight * float(recon) if recon is not None else 0.0)

    repeat_metrics = _repeat_metrics(
        sequence,
        min_k=int(context.get("repeat_min_k", 3)),
        max_k=int(context.get("repeat_max_k", 6)),
    )
    motifs = _normalize_motif_set(context)
    motif_hits = {
        name: _find_pattern_positions(sequence.upper(), motif)
        for name, motif in motifs.items()
    }
    enzymes = _restriction_enzymes(context)
    restriction_map = {
        name: {
            "site": site,
            "count": len(_find_pattern_positions(sequence.upper(), site)),
            "positions": _find_pattern_positions(sequence.upper(), site),
        }
        for name, site in enzymes.items()
    }
    min_orf_aa = int(context.get("min_orf_aa", 90))
    orf_summary = _orf_summary(sequence.upper(), min_orf_aa=min_orf_aa)

    reference_neighbors = _build_reference_neighbors(sequence, context)
    risk_flags: List[RiskFlag] = []
    if max_homopolymer is not None and run > int(max_homopolymer):
        risk_flags.append(RiskFlag(code="homopolymer", severity="warning", message=f"Homopolymer run {run} exceeds limit {max_homopolymer}."))
    if gc < float(context.get("min_gc_fraction", 0.30)):
        risk_flags.append(RiskFlag(code="low_gc", severity="warning", message=f"GC fraction {gc:.3f} is below threshold."))
    if gc > float(context.get("max_gc_fraction", 0.70)):
        risk_flags.append(RiskFlag(code="high_gc", severity="warning", message=f"GC fraction {gc:.3f} is above threshold."))
    if repeat_metrics["repeat_density"] > float(context.get("repeat_density_warn", 0.35)):
        risk_flags.append(RiskFlag(code="repeat_burden", severity="warning", message=f"Repeat density {repeat_metrics['repeat_density']:.3f} exceeds threshold."))
    if any(int(row.get("length", 0)) >= int(context.get("long_homopolymer_warn", 12)) for row in repeat_metrics["long_homopolymers"]):
        risk_flags.append(RiskFlag(code="long_homopolymer", severity="warning", message="Detected long homopolymer run(s)."))

    summary = HumanReadableSummary(
        title="Plasmid scorecard",
        highlights=[
            f"Length={len(sequence)} bp",
            f"GC={gc:.3f} ({gc_percent:.2f}%) target {target_gc:.3f}",
            f"Repeat density={repeat_metrics['repeat_density']:.3f}",
            f"ORFs={orf_summary['count']} (longest {orf_summary['longest_orf_aa']} aa)",
            f"Heuristic score={score:.4f}",
        ],
    )
    if reference_neighbors:
        summary.highlights.append(f"Best reference match score={reference_neighbors[0].score:.4f} ({reference_neighbors[0].reference_id})")

    card = SequenceScorecard(
        scorecard_version=SCORECARD_VERSION,
        sequence_kind="plasmid",
        metadata=ScorecardMetadata(
            sequence_id=context.get("sequence_id"),
            sequence_type="dna",
            sequence_length=len(sequence),
            tokenizer=context.get("tokenizer"),
            model_name=context.get("model_name"),
            model_type=context.get("model_type"),
        ),
        metrics=PerSequenceMetrics(
            score=float(score),
            gc_fraction=float(gc),
            gc_deviation=float(gc_dev),
            max_homopolymer=int(run),
            homopolymer_penalty=float(run_pen),
            roundtrip_recon=float(recon) if recon is not None else None,
            recon_weight=float(recon_weight),
            seq_similarity=reference_neighbors[0].seq_similarity if reference_neighbors else None,
            kmer_jaccard=reference_neighbors[0].kmer_jaccard if reference_neighbors else None,
            gc_delta=reference_neighbors[0].gc_delta if reference_neighbors else None,
            length_ratio=reference_neighbors[0].length_ratio if reference_neighbors else None,
            gc_percent=float(gc_percent),
            repeat_density=float(repeat_metrics["repeat_density"]),
            repeat_burden=float(repeat_metrics["repeat_burden"]),
            motif_hit_count=int(sum(len(pos) for pos in motif_hits.values())),
            restriction_site_count=int(sum(int(row["count"]) for row in restriction_map.values())),
            orf_count=int(orf_summary["count"]),
            longest_orf_aa=int(orf_summary["longest_orf_aa"]),
        ),
        reference_neighbors=reference_neighbors,
        risk_flags=risk_flags,
        summary=summary,
    )
    payload = card.to_payload()
    payload["details"] = {
        "motif_hits": {name: {"count": len(pos), "positions": pos} for name, pos in motif_hits.items()},
        "restriction_site_map": restriction_map,
        "repeat": repeat_metrics,
        "orf_summary": orf_summary,
    }
    return payload


def build_protein_scorecard(sequence: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    max_homopolymer = int(context.get("max_homopolymer", 6))
    max_x_frac = float(context.get("max_x_frac", 0.05))
    max_internal_stops = int(context.get("max_internal_stops", 0))
    recon = context.get("roundtrip_recon")
    recon_weight = float(context.get("recon_weight", 0.0))
    allowed = set(context.get("allowed", [])) if context.get("allowed") is not None else None

    run = max_homopolymer_run(sequence)
    x_frac = sequence.count("X") / float(max(1, len(sequence)))
    allowed_chars = allowed if allowed is not None else set(sequence) | {"*", "X"}
    invalid_frac = sum(1 for ch in sequence if ch not in allowed_chars) / float(max(1, len(sequence)))
    stop_count = sequence.count("*")
    pen_run = max(0, run - max_homopolymer) / max(1.0, float(max_homopolymer))
    pen_x = max(0.0, x_frac - max_x_frac)
    pen_stop = max(0, stop_count - max_internal_stops)
    pen_invalid = invalid_frac * 2.0
    composition = _aa_composition(sequence)
    low_complexity = _low_complexity_regions(sequence, context)
    motif_hits = _protein_pattern_hits(sequence, context)
    proxies = _protein_proxies(sequence, composition)
    pen_lc = max(0.0, float(low_complexity.get("covered_fraction", 0.0)) - float(context.get("low_complexity_warn_fraction", 0.25))) * 1.5
    pen_instability = max(0.0, (float(proxies["instability_proxy"]) - float(context.get("instability_warn_threshold", 45.0))) / 35.0)
    pen_solubility = max(0.0, float(context.get("min_solubility_proxy", 0.35)) - float(proxies["solubility_proxy"]))
    score = -(pen_run + pen_x + pen_stop + pen_invalid + pen_lc + pen_instability + pen_solubility) - (recon_weight * float(recon) if recon is not None else 0.0)

    risk_flags: List[RiskFlag] = []
    if run > max_homopolymer:
        risk_flags.append(RiskFlag(code="homopolymer", severity="warning", message=f"Homopolymer run {run} exceeds limit {max_homopolymer}."))
    if x_frac > max_x_frac:
        risk_flags.append(RiskFlag(code="ambiguous_x", severity="warning", message=f"X fraction {x_frac:.3f} exceeds limit {max_x_frac:.3f}."))
    if stop_count > max_internal_stops:
        risk_flags.append(RiskFlag(code="internal_stop", severity="warning", message=f"Stop count {stop_count} exceeds limit {max_internal_stops}."))
    if float(proxies["hydrophobic_fraction"]) > float(context.get("hydrophobic_warn_threshold", 0.58)):
        risk_flags.append(RiskFlag(code="poor_solubility_hydrophobic", severity="warning", message=f"Hydrophobic fraction {proxies['hydrophobic_fraction']:.3f} suggests poor solubility."))
    if float(proxies["instability_proxy"]) > float(context.get("instability_warn_threshold", 45.0)):
        risk_flags.append(RiskFlag(code="instability_proxy", severity="warning", message=f"Instability proxy {proxies['instability_proxy']:.2f} exceeds threshold."))
    if float(low_complexity.get("covered_fraction", 0.0)) > float(context.get("low_complexity_warn_fraction", 0.25)):
        risk_flags.append(RiskFlag(code="low_complexity_excess", severity="warning", message=f"Low-complexity coverage {low_complexity['covered_fraction']:.3f} exceeds threshold."))
    if int(low_complexity.get("longest_region", 0)) >= int(context.get("low_complexity_longest_warn", 40)):
        risk_flags.append(RiskFlag(code="low_complexity_long_span", severity="warning", message=f"Longest low-complexity span {low_complexity['longest_region']} aa is high."))

    summary = HumanReadableSummary(
        title="Protein scorecard",
        highlights=[
            f"Length={len(sequence)} aa",
            f"Max run={run}",
            f"Hydrophobic={proxies['hydrophobic_fraction']:.3f} Aromaticity={proxies['aromaticity']:.3f}",
            f"Low-complexity coverage={low_complexity['covered_fraction']:.3f}",
            f"Heuristic score={score:.4f}",
        ],
    )

    card = SequenceScorecard(
        scorecard_version=SCORECARD_VERSION,
        sequence_kind="protein",
        metadata=ScorecardMetadata(
            sequence_id=context.get("sequence_id"),
            sequence_type="protein",
            sequence_length=len(sequence),
            tokenizer=context.get("tokenizer"),
            model_name=context.get("model_name"),
            model_type=context.get("model_type"),
        ),
        metrics=PerSequenceMetrics(
            score=float(score),
            max_homopolymer=int(run),
            x_fraction=float(x_frac),
            invalid_fraction=float(invalid_frac),
            stop_count=int(stop_count),
            hydrophobic_fraction=float(proxies["hydrophobic_fraction"]),
            charge_balance=float(proxies["charge_balance"]),
            aromaticity=float(proxies["aromaticity"]),
            instability_proxy=float(proxies["instability_proxy"]),
            low_complexity_fraction=float(low_complexity.get("covered_fraction", 0.0)),
            low_complexity_longest=int(low_complexity.get("longest_region", 0)),
            roundtrip_recon=float(recon) if recon is not None else None,
            recon_weight=float(recon_weight),
        ),
        risk_flags=risk_flags,
        summary=summary,
    )
    payload = card.to_payload()
    payload["details"] = {
        "protein_diagnostics": {
            "amino_acid_composition": composition,
            "low_complexity": low_complexity,
            "pattern_hits": motif_hits,
            "proxies": proxies,
        }
    }
    return payload

from __future__ import annotations

from dataclasses import dataclass
from math import log2, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from perceptrome.encoding.parse import parse_fasta_sequence

ALLOWED_DNA = frozenset({"A", "C", "G", "T", "N"})


@dataclass(frozen=True)
class ValidationThresholds:
    min_allowed_alphabet_ratio: float = 0.98
    min_length: int = 100
    max_length: int = 500_000
    min_gc_fraction: float = 0.25
    max_gc_fraction: float = 0.75
    max_gc_window_variance: Optional[float] = 0.02
    gc_window_size: int = 200
    max_homopolymer_run: int = 14
    min_kmer_diversity: float = 0.10
    kmer_size: int = 4
    max_kmer_distance: float = 0.25
    min_approx_identity: float = 0.60
    distance_metric: str = "cosine"


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper().replace("U", "T")


def parse_plasmid_fasta(path: str) -> str:
    """Parse one FASTA file using the shared parser and normalize into DNA alphabet space."""
    return normalize_sequence(parse_fasta_sequence(path))


def parse_reference_fastas(paths: Sequence[str]) -> List[str]:
    return [parse_plasmid_fasta(path) for path in paths]


def _base_fractions(sequence: str, alphabet: Iterable[str] = ALLOWED_DNA) -> Dict[str, float]:
    seq = normalize_sequence(sequence)
    total = max(len(seq), 1)
    return {base: seq.count(base) / total for base in alphabet}


def allowed_alphabet_ratio(sequence: str, alphabet: Iterable[str] = ALLOWED_DNA) -> float:
    seq = normalize_sequence(sequence)
    if not seq:
        return 0.0
    allowed = set(alphabet)
    good = sum(1 for base in seq if base in allowed)
    return good / len(seq)


def gc_fraction(sequence: str) -> float:
    frac = _base_fractions(sequence)
    return frac.get("G", 0.0) + frac.get("C", 0.0)


def windowed_gc_variance(sequence: str, window_size: int = 200) -> Optional[float]:
    seq = normalize_sequence(sequence)
    if len(seq) < max(1, window_size):
        return None
    values = [gc_fraction(seq[i:i + window_size]) for i in range(0, len(seq) - window_size + 1)]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def max_homopolymer_run(sequence: str) -> int:
    seq = normalize_sequence(sequence)
    if not seq:
        return 0
    best = run = 1
    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def shannon_entropy(sequence: str, alphabet: Iterable[str] = ALLOWED_DNA) -> float:
    fractions = _base_fractions(sequence, alphabet)
    return -sum(p * log2(p) for p in fractions.values() if p > 0)


def kmer_frequency_vector(sequence: str, k: int = 4) -> Dict[str, float]:
    seq = normalize_sequence(sequence)
    if k <= 0 or len(seq) < k:
        return {}
    counts: Dict[str, int] = {}
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
        total += 1
    return {kmer: count / total for kmer, count in counts.items()}


def kmer_diversity(sequence: str, k: int = 4) -> float:
    seq = normalize_sequence(sequence)
    denom = max(1, len(seq) - k + 1)
    return len(kmer_frequency_vector(seq, k)) / denom


def _vector_union_keys(a: Dict[str, float], b: Dict[str, float]) -> List[str]:
    return sorted(set(a) | set(b))


def cosine_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = _vector_union_keys(a, b)
    if not keys:
        return 1.0
    dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in keys)
    na = sqrt(sum(a.get(key, 0.0) ** 2 for key in keys))
    nb = sqrt(sum(b.get(key, 0.0) ** 2 for key in keys))
    if na == 0.0 or nb == 0.0:
        return 1.0
    similarity = max(min(dot / (na * nb), 1.0), -1.0)
    return 1.0 - similarity


def jensen_shannon_distance(a: Dict[str, float], b: Dict[str, float], smoothing: float = 1e-9) -> float:
    keys = _vector_union_keys(a, b)
    if not keys:
        return 1.0
    n = len(keys)
    pa = {key: (a.get(key, 0.0) + smoothing) for key in keys}
    pb = {key: (b.get(key, 0.0) + smoothing) for key in keys}
    za = sum(pa.values())
    zb = sum(pb.values())
    pa = {key: val / za for key, val in pa.items()}
    pb = {key: val / zb for key, val in pb.items()}
    m = {key: 0.5 * (pa[key] + pb[key]) for key in keys}

    def kl(p: Dict[str, float], q: Dict[str, float]) -> float:
        return sum(p[key] * log2(p[key] / q[key]) for key in keys)

    jsd = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return sqrt(max(jsd, 0.0))


def approximate_identity_shared_kmers(sequence: str, reference: str, k: int = 5) -> float:
    query_freq = kmer_frequency_vector(sequence, k)
    ref_freq = kmer_frequency_vector(reference, k)
    keys = set(query_freq) | set(ref_freq)
    if not keys:
        return 0.0
    shared = sum(min(query_freq.get(key, 0.0), ref_freq.get(key, 0.0)) for key in keys)
    return max(0.0, min(1.0, shared))


def _closest_reference_metrics(
    sequence: str,
    references: Sequence[str],
    k: int,
    distance_metric: str,
) -> Tuple[float, Optional[int], float]:
    if not references:
        return 1.0, None, 0.0

    query_vec = kmer_frequency_vector(sequence, k)
    best_dist = float("inf")
    best_identity = 0.0
    best_idx: Optional[int] = None

    for idx, ref in enumerate(references):
        ref_vec = kmer_frequency_vector(ref, k)
        if distance_metric == "jensen-shannon":
            dist = jensen_shannon_distance(query_vec, ref_vec)
        else:
            dist = cosine_distance(query_vec, ref_vec)
        ident = approximate_identity_shared_kmers(sequence, ref, max(3, k))
        if dist < best_dist:
            best_dist = dist
            best_identity = ident
            best_idx = idx

    return best_dist, best_idx, best_identity


def validate_plasmid_sequence(
    sequence: str,
    references: Optional[Sequence[str]] = None,
    thresholds: Optional[ValidationThresholds] = None,
) -> Dict[str, object]:
    cfg = thresholds or ValidationThresholds()
    seq = normalize_sequence(sequence)
    refs = [normalize_sequence(r) for r in (references or []) if normalize_sequence(r)]

    alpha_ratio = allowed_alphabet_ratio(seq)
    length = len(seq)
    gc = gc_fraction(seq)
    gc_var = windowed_gc_variance(seq, cfg.gc_window_size)
    homopolymer = max_homopolymer_run(seq)
    entropy = shannon_entropy(seq)
    diversity = kmer_diversity(seq, cfg.kmer_size)

    kmer_dist, closest_idx, approx_identity = _closest_reference_metrics(
        seq, refs, cfg.kmer_size, cfg.distance_metric
    )

    checks = {
        "allowed_alphabet_ratio": {
            "value": alpha_ratio,
            "threshold": cfg.min_allowed_alphabet_ratio,
            "pass": alpha_ratio >= cfg.min_allowed_alphabet_ratio,
        },
        "length_bounds": {
            "value": length,
            "min": cfg.min_length,
            "max": cfg.max_length,
            "pass": cfg.min_length <= length <= cfg.max_length,
        },
        "gc_fraction": {
            "value": gc,
            "min": cfg.min_gc_fraction,
            "max": cfg.max_gc_fraction,
            "pass": cfg.min_gc_fraction <= gc <= cfg.max_gc_fraction,
        },
        "gc_window_variance": {
            "value": gc_var,
            "max": cfg.max_gc_window_variance,
            "pass": (gc_var is None) or (cfg.max_gc_window_variance is None) or (gc_var <= cfg.max_gc_window_variance),
        },
        "max_homopolymer_run": {
            "value": homopolymer,
            "max": cfg.max_homopolymer_run,
            "pass": homopolymer <= cfg.max_homopolymer_run,
        },
        "kmer_diversity": {
            "value": diversity,
            "min": cfg.min_kmer_diversity,
            "pass": diversity >= cfg.min_kmer_diversity,
        },
        "kmer_distance": {
            "value": kmer_dist,
            "metric": cfg.distance_metric,
            "max": cfg.max_kmer_distance,
            "pass": (not refs) or (kmer_dist <= cfg.max_kmer_distance),
            "closest_reference_index": closest_idx,
        },
        "approx_identity": {
            "value": approx_identity,
            "min": cfg.min_approx_identity,
            "pass": (not refs) or (approx_identity >= cfg.min_approx_identity),
            "closest_reference_index": closest_idx,
        },
    }

    overall = all(metric["pass"] for metric in checks.values())

    return {
        "sequence_length": length,
        "base_entropy": entropy,
        "references_compared": len(refs),
        "checks": checks,
        "overall_pass": overall,
    }

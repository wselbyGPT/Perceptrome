from __future__ import annotations

import difflib
from typing import Dict


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


def gc_fraction(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def reference_score(generated_seq: str, ref_seq: str) -> Dict[str, float]:
    seq_sim = sequence_similarity(generated_seq, ref_seq)
    kmer_sim = jaccard_kmers(generated_seq, ref_seq, k=9)
    gc_delta = abs(gc_fraction(generated_seq) - gc_fraction(ref_seq))
    length_ratio = (
        min(len(generated_seq), len(ref_seq)) / max(len(generated_seq), len(ref_seq)) if generated_seq and ref_seq else 0.0
    )
    score = (0.55 * seq_sim) + (0.30 * kmer_sim) + (0.10 * length_ratio) + (0.05 * (1.0 - gc_delta))
    return {
        "score": float(score),
        "seq_similarity": float(seq_sim),
        "kmer_jaccard": float(kmer_sim),
        "gc_delta": float(gc_delta),
        "length_ratio": float(length_ratio),
    }


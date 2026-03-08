from __future__ import annotations

import difflib
from typing import Any, Dict, List, Mapping


def gc_fraction(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def _kmer_set(seq: str, k: int) -> set[str]:
    if k <= 0 or len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard_kmers(a: str, b: str, k: int = 9) -> float:
    ka = _kmer_set(a, k)
    kb = _kmer_set(b, k)
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


def score_neighbors(
    sequence: str,
    references: List[Mapping[str, Any]],
    *,
    sequence_kind: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    k = 9 if str(sequence_kind).lower() == "plasmid" else 3
    include_gc = str(sequence_kind).lower() == "plasmid"
    for item in references:
        ref_seq = str(item.get("sequence") or "").strip().upper()
        if not ref_seq:
            continue
        ref_id = str(item.get("reference_id") or item.get("id") or "reference")
        source = str(item.get("source") or "catalog")
        seq_sim = sequence_similarity(sequence, ref_seq)
        kmer_sim = jaccard_kmers(sequence, ref_seq, k=k)
        length_ratio = min(len(sequence), len(ref_seq)) / max(len(sequence), len(ref_seq)) if sequence and ref_seq else 0.0
        gc_delta = abs(gc_fraction(sequence) - gc_fraction(ref_seq)) if include_gc else 0.0
        if include_gc:
            score = (0.55 * seq_sim) + (0.30 * kmer_sim) + (0.10 * length_ratio) + (0.05 * (1.0 - gc_delta))
        else:
            score = (0.65 * seq_sim) + (0.25 * kmer_sim) + (0.10 * length_ratio)
        rows.append(
            {
                "reference_id": ref_id,
                "source": source,
                "reference_length": len(ref_seq),
                "length_delta": int(len(sequence) - len(ref_seq)),
                "score": float(score),
                "seq_similarity": float(seq_sim),
                "kmer_jaccard": float(kmer_sim),
                "gc_delta": float(gc_delta),
                "length_ratio": float(length_ratio),
            }
        )
    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return rows[: max(1, int(top_n))]

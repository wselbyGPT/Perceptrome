from __future__ import annotations

from typing import Dict

from .scorecard import build_plasmid_scorecard, gc_fraction, jaccard_kmers, kmer_set, sequence_similarity


def reference_score(generated_seq: str, ref_seq: str) -> Dict[str, float]:
    payload = build_plasmid_scorecard(
        generated_seq,
        {
            "reference_id": "reference",
            "reference_sequence": ref_seq,
        },
    )
    metrics = payload["metrics"]
    best = (payload.get("reference_neighbors") or [{}])[0]
    return {
        "score": float(metrics.get("score", 0.0)),
        "seq_similarity": float(best.get("seq_similarity", 0.0)),
        "kmer_jaccard": float(best.get("kmer_jaccard", 0.0)),
        "gc_delta": float(best.get("gc_delta", 0.0)),
        "length_ratio": float(best.get("length_ratio", 0.0)),
    }

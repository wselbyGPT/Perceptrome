from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from perceptrome.generate import generate_plasmid_sequence


@dataclass(slots=True)
class Candidate:
    sequence: str
    source: str
    parent_ids: list[int]
    score: float = 0.0
    metrics: Dict[str, float] | None = None


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    return float((seq.count("G") + seq.count("C")) / len(seq))


def _max_homopolymer(seq: str) -> int:
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


def _sequence_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _score_candidate(seq: str, references: Sequence[str], target_gc: float, max_homopolymer: int) -> Dict[str, float]:
    gc = _gc_fraction(seq)
    gc_dev = abs(gc - target_gc)
    best_ref = 0.0
    for ref in references:
        best_ref = max(best_ref, _sequence_similarity(seq, ref))
    max_run = _max_homopolymer(seq)
    run_penalty = max(0.0, (max_run - max_homopolymer) / float(max_homopolymer)) if max_homopolymer > 0 else 0.0
    score = (0.65 * best_ref) + (0.25 * (1.0 - gc_dev)) + (0.10 * (1.0 - run_penalty))
    return {
        "score": float(score),
        "best_ref_similarity": float(best_ref),
        "gc_fraction": float(gc),
        "gc_deviation": float(gc_dev),
        "max_homopolymer": float(max_run),
        "run_penalty": float(run_penalty),
    }


def _mutate_sequence(seq: str, mutation_rate: float, mutation_scale: float, rng: random.Random) -> str:
    if not seq:
        return seq
    alphabet = "ACGT"
    n_mut = int(max(1, round(len(seq) * max(0.0, mutation_rate) * max(0.0, mutation_scale))))
    chars = list(seq)
    for _ in range(n_mut):
        idx = rng.randrange(0, len(chars))
        current = chars[idx]
        choices = [ch for ch in alphabet if ch != current]
        chars[idx] = rng.choice(choices)
    return "".join(chars)


def _crossover(a: str, b: str, rng: random.Random) -> str:
    if not a:
        return b
    if not b:
        return a
    n = min(len(a), len(b))
    if n < 2:
        return a
    cut = rng.randrange(1, n)
    merged = a[:cut] + b[cut:]
    if len(a) > n:
        merged += a[n:]
    elif len(b) > n:
        merged += b[n:]
    return merged


def run_design_loop(
    *,
    train_cfg: Any,
    io_cfg: Any,
    rounds: int,
    population_size: int,
    survivor_count: int,
    mutation_rate: float,
    mutation_scale: float,
    crossover_rate: float,
    early_stop_threshold: Optional[float],
    target_gc: float,
    max_homopolymer: int,
    length_bp: int,
    references: Sequence[str],
    seed: Optional[int],
    emit: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    population_size = max(2, int(population_size))
    survivor_count = max(1, min(int(survivor_count), population_size))

    population: List[Candidate] = []
    for idx in range(population_size):
        seq = generate_plasmid_sequence(
            train_cfg=train_cfg,
            io_cfg=io_cfg,
            length_bp=length_bp,
            num_windows=None,
            window_size_bp=int(getattr(train_cfg, "window_size", 256)),
            seed=rng.randint(0, 2**31 - 1),
            latent_scale=1.0,
            temperature=1.0,
            gc_bias=1.0,
            name=f"design_seed_{idx}",
            output_path=f"design_seed_{idx}.fasta",
            tokenizer="base",
            num_candidates=1,
            top_k=1,
            target_gc=target_gc,
            max_homopolymer=max_homopolymer,
        )
        population.append(Candidate(sequence=seq, source="initial", parent_ids=[]))
    if emit:
        emit("design_loop", f"initialized population={len(population)}")

    round_summaries: List[Dict[str, Any]] = []
    best: Optional[Candidate] = None

    for round_idx in range(1, max(1, int(rounds)) + 1):
        scored: List[Candidate] = []
        for cand in population:
            metrics = _score_candidate(cand.sequence, references, target_gc=target_gc, max_homopolymer=max_homopolymer)
            cand.score = float(metrics["score"])
            cand.metrics = metrics
            scored.append(cand)

        scored.sort(key=lambda c: c.score, reverse=True)
        survivors = scored[:survivor_count]
        best = survivors[0] if best is None or survivors[0].score > best.score else best

        round_summary = {
            "round": round_idx,
            "best_score": float(survivors[0].score),
            "mean_score": float(sum(c.score for c in scored) / max(1, len(scored))),
            "best_metrics": dict(survivors[0].metrics or {}),
            "survivors": [
                {
                    "score": float(c.score),
                    "source": c.source,
                    "parents": list(c.parent_ids),
                    "metrics": dict(c.metrics or {}),
                    "sequence": c.sequence,
                }
                for c in survivors
            ],
        }
        round_summaries.append(round_summary)

        if emit:
            emit("design_round", f"round={round_idx} best={survivors[0].score:.4f}")
        if early_stop_threshold is not None and survivors[0].score >= float(early_stop_threshold):
            if emit:
                emit("design_loop", f"early stop at round={round_idx} threshold={early_stop_threshold}")
            break

        next_population: List[Candidate] = [
            Candidate(sequence=s.sequence, source="elite", parent_ids=[]) for s in survivors
        ]

        while len(next_population) < population_size:
            p1_idx = rng.randrange(0, len(survivors))
            p2_idx = rng.randrange(0, len(survivors))
            p1 = survivors[p1_idx]
            p2 = survivors[p2_idx]
            child_seq = p1.sequence
            source = "mutate"
            parent_ids = [p1_idx]
            if rng.random() < float(crossover_rate):
                child_seq = _crossover(p1.sequence, p2.sequence, rng)
                source = "crossover"
                parent_ids = [p1_idx, p2_idx]
            child_seq = _mutate_sequence(child_seq, mutation_rate=float(mutation_rate), mutation_scale=float(mutation_scale), rng=rng)
            next_population.append(Candidate(sequence=child_seq, source=source, parent_ids=parent_ids))

        population = next_population

    ranked_best = sorted(
        [
            {
                "sequence": c["sequence"],
                "score": float(c["score"]),
                "metrics": c["metrics"],
                "source": c["source"],
                "parents": c["parents"],
            }
            for c in (round_summaries[-1]["survivors"] if round_summaries else [])
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "best_candidate": {
            "sequence": best.sequence if best else "",
            "score": float(best.score) if best else 0.0,
            "metrics": dict(best.metrics or {}) if best else {},
            "source": best.source if best else "",
        },
        "best_candidates": ranked_best,
        "rounds_completed": len(round_summaries),
        "round_summaries": round_summaries,
    }

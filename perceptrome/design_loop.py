from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from perceptrome.evolution_adapters import (
    default_registry_and_spec,
    genome_to_candidate_metadata,
    seed_metadata_from_defaults,
)
from perceptrome.generate import generate_plasmid_sequence
from perceptrome.genome_evolution import crossover_genomes, initialize_genome, mutate_genome, repair, validate


@dataclass(slots=True)
class Candidate:
    sequence: str
    source: str
    parent_ids: list[int]
    genome: Dict[str, Any] | None = None
    metadata: Dict[str, Any] | None = None
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
    enable_sequence_operators: bool = False,
    sequence_operator_top_k: int = 3,
    emit: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    population_size = max(2, int(population_size))
    survivor_count = max(1, min(int(survivor_count), population_size))

    registry, spec = default_registry_and_spec()
    base_metadata = seed_metadata_from_defaults(target_gc=target_gc, max_homopolymer=max_homopolymer)

    def _normalize_genome(genome: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        pre_validation = validate(genome, registry, spec)
        repaired_genome = genome
        repair_actions: list[Dict[str, Any]] = []
        if not pre_validation["valid"]:
            repaired_genome, repair_actions = repair(genome, pre_validation["violations"], registry, spec)
        post_validation = validate(repaired_genome, registry, spec)
        return repaired_genome, {
            "pre_repair_validation": pre_validation,
            "repair_actions": repair_actions,
            "post_repair_validation": post_validation,
        }

    def _generate_candidate(genome: Dict[str, Any], source: str, parent_ids: list[int], sequence_overrides: Sequence[str] | None = None) -> Candidate:
        normalized_genome, validation_info = _normalize_genome(genome)
        metadata = genome_to_candidate_metadata(normalized_genome, base_metadata=base_metadata)
        seq = generate_plasmid_sequence(
            train_cfg=train_cfg,
            io_cfg=io_cfg,
            length_bp=length_bp,
            num_windows=None,
            window_size_bp=int(getattr(train_cfg, "window_size", 256)),
            seed=rng.randint(0, 2**31 - 1),
            latent_scale=float(metadata["latent_scale"]),
            temperature=float(metadata["temperature"]),
            gc_bias=float(metadata["gc_bias"]),
            name=f"design_{source}_{len(parent_ids)}",
            output_path=f"design_{source}_{rng.randint(0, 2**31 - 1)}.fasta",
            tokenizer="base",
            num_candidates=max(1, int(metadata.get("top_k", 1))),
            top_k=max(1, int(metadata.get("top_k", 1))),
            target_gc=float(metadata["target_gc"]),
            max_homopolymer=int(metadata["max_homopolymer"]),
        )

        if enable_sequence_operators and sequence_overrides:
            donor = rng.choice(list(sequence_overrides))
            if rng.random() < float(crossover_rate):
                seq = _crossover(seq, donor, rng)
            seq = _mutate_sequence(seq, mutation_rate=float(mutation_rate), mutation_scale=float(mutation_scale), rng=rng)

        return Candidate(
            sequence=seq,
            source=source,
            parent_ids=parent_ids,
            genome=normalized_genome,
            metadata={**metadata, "validation": validation_info},
        )

    population: List[Candidate] = []
    for _ in range(population_size):
        seeded = initialize_genome(registry, rng=rng)
        population.append(_generate_candidate(seeded, source="initial", parent_ids=[]))
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
            Candidate(sequence=s.sequence, source="elite", parent_ids=[], genome=dict(s.genome or {}), metadata=dict(s.metadata or {}))
            for s in survivors
        ]

        top_k_sequences: List[str] = []
        if enable_sequence_operators:
            top_cap = max(1, int(sequence_operator_top_k))
            top_k_sequences = [str(item["sequence"]) for item in round_summary["survivors"][:top_cap] if item.get("sequence")]

        while len(next_population) < population_size:
            p1_idx = rng.randrange(0, len(survivors))
            p2_idx = rng.randrange(0, len(survivors))
            p1 = survivors[p1_idx]
            p2 = survivors[p2_idx]

            p1_genome = dict(p1.genome or {})
            p2_genome = dict(p2.genome or {})
            child_genome = dict(p1_genome)
            source = "mutate"
            parent_ids = [p1_idx]
            if rng.random() < float(crossover_rate):
                child_genome = crossover_genomes(p1_genome, p2_genome, registry, rng=rng)
                source = "crossover"
                parent_ids = [p1_idx, p2_idx]

            child_genome = mutate_genome(
                child_genome,
                registry,
                mutation_rate=float(mutation_rate),
                mutation_scale=float(mutation_scale),
                rng=rng,
            )

            next_population.append(
                _generate_candidate(
                    child_genome,
                    source=source,
                    parent_ids=parent_ids,
                    sequence_overrides=top_k_sequences,
                )
            )

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

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, MutableMapping, Tuple


def default_evolution_registry() -> Dict[str, Dict[str, float]]:
    return {
        "temperature": {"min": 0.4, "max": 2.0},
        "gc_bias": {"min": 0.5, "max": 1.7},
        "latent_scale": {"min": 0.4, "max": 2.0},
        "target_gc": {"min": 0.25, "max": 0.75},
        "max_homopolymer": {"min": 3.0, "max": 16.0},
        "top_k": {"min": 1.0, "max": 8.0},
        "motif_toggle_promoter": {"min": 0.0, "max": 1.0},
        "motif_toggle_terminator": {"min": 0.0, "max": 1.0},
    }


def default_evolution_spec() -> Dict[str, Any]:
    return {
        "conflicts": [("motif_toggle_promoter", "motif_toggle_terminator")],
        "conflict_priority": {"motif_toggle_promoter": 1.0, "motif_toggle_terminator": 1.0},
        "budgets": {
            "motif_toggle_budget": {
                "limit": 1.0,
                "genes": ["motif_toggle_promoter", "motif_toggle_terminator"],
            }
        },
    }


def candidate_metadata_to_genome(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    motif_toggles = dict(metadata.get("motif_toggles") or {})
    return {
        "temperature": float(metadata.get("temperature", 1.0)),
        "gc_bias": float(metadata.get("gc_bias", 1.0)),
        "latent_scale": float(metadata.get("latent_scale", 1.0)),
        "target_gc": float(metadata.get("target_gc", 0.5)),
        "max_homopolymer": float(metadata.get("max_homopolymer", 8.0)),
        "top_k": float(metadata.get("top_k", 1.0)),
        "motif_toggle_promoter": 1.0 if bool(motif_toggles.get("promoter", False)) else 0.0,
        "motif_toggle_terminator": 1.0 if bool(motif_toggles.get("terminator", False)) else 0.0,
    }


def genome_to_candidate_metadata(genome: Mapping[str, Any], base_metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload: MutableMapping[str, Any] = copy.deepcopy(dict(base_metadata or {}))

    payload["temperature"] = float(genome.get("temperature", payload.get("temperature", 1.0)))
    payload["gc_bias"] = float(genome.get("gc_bias", payload.get("gc_bias", 1.0)))
    payload["latent_scale"] = float(genome.get("latent_scale", payload.get("latent_scale", 1.0)))
    payload["target_gc"] = float(genome.get("target_gc", payload.get("target_gc", 0.5)))
    payload["max_homopolymer"] = int(round(float(genome.get("max_homopolymer", payload.get("max_homopolymer", 8)))))
    payload["top_k"] = max(1, int(round(float(genome.get("top_k", payload.get("top_k", 1))))))

    motif_toggles = dict(payload.get("motif_toggles") or {})
    motif_toggles["promoter"] = float(genome.get("motif_toggle_promoter", 0.0)) >= 0.5
    motif_toggles["terminator"] = float(genome.get("motif_toggle_terminator", 0.0)) >= 0.5
    payload["motif_toggles"] = motif_toggles

    return dict(payload)


def seed_metadata_from_defaults(*, target_gc: float, max_homopolymer: int) -> Dict[str, Any]:
    return {
        "temperature": 1.0,
        "gc_bias": 1.0,
        "latent_scale": 1.0,
        "target_gc": float(target_gc),
        "max_homopolymer": int(max_homopolymer),
        "top_k": 1,
        "motif_toggles": {"promoter": False, "terminator": False},
    }


def clamp_genome_to_registry(genome: Mapping[str, Any], registry: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    clamped = copy.deepcopy(dict(genome))
    for gene, bounds in registry.items():
        if gene not in clamped:
            continue
        value = float(clamped[gene])
        clamped[gene] = min(max(value, float(bounds.get("min", value))), float(bounds.get("max", value)))
    return clamped


def default_registry_and_spec() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    return default_evolution_registry(), default_evolution_spec()

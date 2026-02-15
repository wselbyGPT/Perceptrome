from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence


NumericValue = float
CategoricalValue = Any
SamplerFn = Callable[[Any, "GeneDefinition", "SamplerContext"], Any]


@dataclass
class GeneDefinition:
    name: str
    gene_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[Sequence[CategoricalValue]] = None
    sampler_key: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class GenomeSpec:
    genes: Sequence[GeneDefinition]
    sampler_intensity: Optional[float] = None


@dataclass
class SamplerContext:
    rng: random.Random = field(default_factory=random.Random)
    sampler_intensity: Optional[float] = None
    mode: Optional[str] = None
    runtime_config: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_SAMPLER_KEY_BY_TYPE: Dict[str, str] = {
    "numeric": "numeric_baseline",
    "categorical": "categorical_baseline",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mode_default_intensity(mode: Optional[str]) -> float:
    mode_key = (mode or "").strip().lower()
    if mode_key == "exploration":
        return 1.6
    if mode_key == "exploitation":
        return 0.6
    return 1.0


def resolve_sampler_intensity(
    context: Optional[SamplerContext] = None,
    genome_spec: Optional[GenomeSpec] = None,
    gene_def: Optional[GeneDefinition] = None,
) -> float:
    """Resolve sampler intensity with runtime override support.

    Priority:
      1) `context.runtime_config["sampler_intensity"]`
      2) `context.sampler_intensity`
      3) `genome_spec.sampler_intensity`
      4) `gene_def.metadata["sampler_intensity"]`
      5) default inferred from mode (`exploration`/`exploitation`)
    """
    runtime_value = None
    if context is not None:
        runtime_value = context.runtime_config.get("sampler_intensity")

    for candidate in (
        runtime_value,
        None if context is None else context.sampler_intensity,
        None if genome_spec is None else genome_spec.sampler_intensity,
        None if gene_def is None else gene_def.metadata.get("sampler_intensity"),
    ):
        if candidate is not None:
            return max(0.0, float(candidate))

    return _mode_default_intensity(None if context is None else context.mode)


def resolve_sampler_key(gene_def: GeneDefinition) -> str:
    if gene_def.sampler_key:
        return gene_def.sampler_key
    key = DEFAULT_SAMPLER_KEY_BY_TYPE.get(str(gene_def.gene_type).lower())
    if key is None:
        raise KeyError(f"No sampler key mapping for gene type '{gene_def.gene_type}'.")
    return key


def _sample_numeric_initial(_: Any, gene_def: GeneDefinition, context: SamplerContext) -> NumericValue:
    if gene_def.min_value is None or gene_def.max_value is None:
        raise ValueError(f"Numeric gene '{gene_def.name}' requires min_value and max_value.")
    lo = float(gene_def.min_value)
    hi = float(gene_def.max_value)
    if hi < lo:
        raise ValueError(f"Numeric gene '{gene_def.name}' has max_value < min_value.")
    return context.rng.uniform(lo, hi)


def _mutate_numeric_value(old_value: NumericValue, gene_def: GeneDefinition, context: SamplerContext) -> NumericValue:
    if gene_def.min_value is None or gene_def.max_value is None:
        raise ValueError(f"Numeric gene '{gene_def.name}' requires min_value and max_value.")
    lo = float(gene_def.min_value)
    hi = float(gene_def.max_value)
    span = max(1e-12, hi - lo)
    intensity = resolve_sampler_intensity(context=context, gene_def=gene_def)
    sigma = span * 0.12 * max(0.0, intensity)
    mutated = float(old_value) + context.rng.gauss(0.0, sigma)
    return _clamp(mutated, lo, hi)


def _sample_categorical_initial(_: Any, gene_def: GeneDefinition, context: SamplerContext) -> CategoricalValue:
    cats = list(gene_def.categories or [])
    if not cats:
        raise ValueError(f"Categorical gene '{gene_def.name}' requires categories.")
    return context.rng.choice(cats)


def _mutate_categorical_value(old_value: CategoricalValue, gene_def: GeneDefinition, context: SamplerContext) -> CategoricalValue:
    cats = list(gene_def.categories or [])
    if not cats:
        raise ValueError(f"Categorical gene '{gene_def.name}' requires categories.")
    if len(cats) == 1:
        return cats[0]

    intensity = resolve_sampler_intensity(context=context, gene_def=gene_def)
    switch_prob = _clamp(0.25 * intensity, 0.0, 1.0)

    if old_value in cats and context.rng.random() > switch_prob:
        return old_value

    alternatives = [v for v in cats if v != old_value]
    return context.rng.choice(alternatives or cats)


INITIAL_SAMPLERS: Dict[str, SamplerFn] = {
    "numeric_baseline": _sample_numeric_initial,
    "categorical_baseline": _sample_categorical_initial,
}

MUTATION_SAMPLERS: Dict[str, SamplerFn] = {
    "numeric_baseline": _mutate_numeric_value,
    "categorical_baseline": _mutate_categorical_value,
}


def sample_initial(gene_def: GeneDefinition, context: SamplerContext) -> Any:
    sampler_key = resolve_sampler_key(gene_def)
    sampler = INITIAL_SAMPLERS.get(sampler_key)
    if sampler is None:
        raise KeyError(f"Unknown initial sampler '{sampler_key}' for gene '{gene_def.name}'.")
    return sampler(None, gene_def, context)


def mutate_value(old_value: Any, gene_def: GeneDefinition, context: SamplerContext) -> Any:
    sampler_key = resolve_sampler_key(gene_def)
    sampler = MUTATION_SAMPLERS.get(sampler_key)
    if sampler is None:
        raise KeyError(f"Unknown mutation sampler '{sampler_key}' for gene '{gene_def.name}'.")
    return sampler(old_value, gene_def, context)

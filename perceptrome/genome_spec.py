from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Gene:
    """A simple gene model used by the evolutionary utilities."""

    gene_id: str
    group: str
    weight: float = 0.0
    complexity: float = 0.0


@dataclass(frozen=True)
class Genome:
    """A genome represented as an ordered list of genes."""

    genes: Tuple[Gene, ...]

    @classmethod
    def from_genes(cls, genes: Iterable[Gene]) -> "Genome":
        return cls(tuple(genes))


@dataclass(frozen=True)
class GenomeValidationError(ValueError):
    """Validation failure annotated with machine-readable metadata."""

    rule_name: str
    gene_ids: Tuple[str, ...]
    details: str

    def __str__(self) -> str:
        ids = ", ".join(self.gene_ids) if self.gene_ids else "<none>"
        return f"GenomeSpec violation '{self.rule_name}' for genes [{ids}]: {self.details}"


@dataclass(frozen=True)
class GenomeSpec:
    """Declarative constraints for a valid genome."""

    min_gene_count: int = 0
    max_gene_count: Optional[int] = None
    max_total_weight: Optional[float] = None
    max_total_complexity: Optional[float] = None
    required_gene_groups: Set[str] = field(default_factory=set)
    optional_gene_groups: Set[str] = field(default_factory=set)
    incompatible_gene_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    def normalized_incompatibilities(self) -> Set[Tuple[str, str]]:
        return {tuple(sorted(pair)) for pair in self.incompatible_gene_pairs}


def validate_genome(genome: Genome, spec: GenomeSpec) -> None:
    genes = list(genome.genes)
    gene_count = len(genes)

    if gene_count < spec.min_gene_count:
        raise GenomeValidationError(
            rule_name="min_gene_count",
            gene_ids=tuple(g.gene_id for g in genes),
            details=f"gene count {gene_count} is below minimum {spec.min_gene_count}",
        )

    if spec.max_gene_count is not None and gene_count > spec.max_gene_count:
        raise GenomeValidationError(
            rule_name="max_gene_count",
            gene_ids=tuple(g.gene_id for g in genes),
            details=f"gene count {gene_count} exceeds maximum {spec.max_gene_count}",
        )

    total_weight = sum(float(g.weight) for g in genes)
    if spec.max_total_weight is not None and total_weight > spec.max_total_weight:
        raise GenomeValidationError(
            rule_name="max_total_weight",
            gene_ids=tuple(g.gene_id for g in genes),
            details=f"total weight {total_weight:.4f} exceeds budget {spec.max_total_weight:.4f}",
        )

    total_complexity = sum(float(g.complexity) for g in genes)
    if spec.max_total_complexity is not None and total_complexity > spec.max_total_complexity:
        raise GenomeValidationError(
            rule_name="max_total_complexity",
            gene_ids=tuple(g.gene_id for g in genes),
            details=(
                f"total complexity {total_complexity:.4f} exceeds budget "
                f"{spec.max_total_complexity:.4f}"
            ),
        )

    groups = {g.group for g in genes}
    missing_groups = sorted(spec.required_gene_groups - groups)
    if missing_groups:
        raise GenomeValidationError(
            rule_name="required_gene_groups",
            gene_ids=tuple(g.gene_id for g in genes),
            details=f"missing required group(s): {', '.join(missing_groups)}",
        )

    if spec.optional_gene_groups:
        allowed_groups = set(spec.required_gene_groups) | set(spec.optional_gene_groups)
        disallowed = [g for g in genes if g.group not in allowed_groups]
        if disallowed:
            raise GenomeValidationError(
                rule_name="optional_gene_groups",
                gene_ids=tuple(g.gene_id for g in disallowed),
                details=(
                    "gene group outside required/optional sets: "
                    + ", ".join(sorted({g.group for g in disallowed}))
                ),
            )

    present_ids = {g.gene_id for g in genes}
    for left, right in sorted(spec.normalized_incompatibilities()):
        if left in present_ids and right in present_ids:
            raise GenomeValidationError(
                rule_name="incompatible_gene_pair",
                gene_ids=(left, right),
                details=f"incompatible pair present together: {left} and {right}",
            )


def create_genome(genes: Sequence[Gene], spec: GenomeSpec) -> Genome:
    genome = Genome.from_genes(genes)
    validate_genome(genome, spec)
    return genome


def mutate_genome(
    genome: Genome,
    spec: GenomeSpec,
    *,
    add_gene: Optional[Gene] = None,
    remove_gene_id: Optional[str] = None,
    replace_gene: Optional[Gene] = None,
) -> Genome:
    """Apply mutation(s) and validate immediately after the operation."""
    genes: List[Gene] = list(genome.genes)

    if remove_gene_id is not None:
        genes = [g for g in genes if g.gene_id != remove_gene_id]

    if replace_gene is not None:
        replaced = False
        for i, gene in enumerate(genes):
            if gene.gene_id == replace_gene.gene_id:
                genes[i] = replace_gene
                replaced = True
                break
        if not replaced:
            genes.append(replace_gene)

    if add_gene is not None:
        genes.append(add_gene)

    mutated = Genome.from_genes(genes)
    validate_genome(mutated, spec)
    return mutated


def crossover_genomes(parent_a: Genome, parent_b: Genome, spec: GenomeSpec, split_at: Optional[int] = None) -> Genome:
    """Single-point crossover with post-crossover validation."""
    a = list(parent_a.genes)
    b = list(parent_b.genes)
    if split_at is None:
        split_at = len(a) // 2
    child = Genome.from_genes(a[:split_at] + b[split_at:])
    validate_genome(child, spec)
    return child


def evaluate_genome(
    genome: Genome,
    spec: GenomeSpec,
    evaluator: Callable[[Genome], float],
) -> float:
    """Validate before evaluation so scoring assumes a legal genome."""
    validate_genome(genome, spec)
    return float(evaluator(genome))

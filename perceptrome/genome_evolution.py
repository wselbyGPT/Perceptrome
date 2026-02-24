import copy
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from perceptrome.genome_ast import evaluate_relationship_constraints, parse_relationship_spec


Violation = Dict[str, Any]
ValidationResult = Dict[str, Any]
RepairAction = Dict[str, Any]


def _as_weighted_value(value: Any) -> float:
    """Convert a gene payload to its numeric value for budget math."""
    if isinstance(value, dict):
        return float(value.get("value", 0.0))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _is_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        if "enabled" in value:
            return bool(value.get("enabled"))
        return bool(value.get("value", 0.0))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return False


def _set_enabled(genome: Dict[str, Any], gene: str, enabled: bool) -> None:
    value = genome.get(gene)
    if isinstance(value, dict):
        value["enabled"] = bool(enabled)
        if not enabled and "value" in value and isinstance(value["value"], (int, float)):
            value["value"] = 0.0
    elif isinstance(value, bool):
        genome[gene] = bool(enabled)
    elif isinstance(value, (int, float)):
        genome[gene] = float(value) if enabled else 0.0
    else:
        genome[gene] = bool(enabled)


def validate(genome: Dict[str, Any], registry: Dict[str, Any], spec: Dict[str, Any]) -> ValidationResult:
    """Validate genome values, conflicts, and global budgets.

    registry format:
      {
        "gene_name": {"min": 0.0, "max": 1.0}
      }

    spec format (optional):
      {
        "conflicts": [("geneA", "geneB"), ...],
        "budgets": {
          "name": {"limit": 1.0, "genes": ["geneA", "geneB"], "weights": {"geneA": 2.0}}
        }
      }
    """
    violations: List[Violation] = []

    for gene, value in genome.items():
        if gene not in registry:
            violations.append(
                {
                    "type": "unknown_gene",
                    "gene": gene,
                    "path": "initialization/mutation/crossover",
                    "message": f"Gene '{gene}' is not in registry.",
                }
            )
            continue

        bounds = registry.get(gene, {})
        min_v = bounds.get("min")
        max_v = bounds.get("max")
        numeric_value = _as_weighted_value(value)

        if min_v is not None and numeric_value < float(min_v):
            violations.append(
                {
                    "type": "out_of_range",
                    "gene": gene,
                    "message": f"{gene} below min ({numeric_value} < {min_v}).",
                    "details": {"value": numeric_value, "min": float(min_v), "max": max_v},
                }
            )
        if max_v is not None and numeric_value > float(max_v):
            violations.append(
                {
                    "type": "out_of_range",
                    "gene": gene,
                    "message": f"{gene} above max ({numeric_value} > {max_v}).",
                    "details": {"value": numeric_value, "min": min_v, "max": float(max_v)},
                }
            )

    relationship_schema = parse_relationship_spec(spec)
    relationship_violations = evaluate_relationship_constraints(
        values=genome,
        enabled_genes=[g for g in genome.keys() if _is_enabled(genome.get(g))],
        weighted_values={g: _as_weighted_value(v) for g, v in genome.items()},
        schema=relationship_schema,
    )
    for violation in relationship_violations:
        if violation.get("type") == "mutual_exclusion" and str(violation.get("gene", "")).startswith("conflict:"):
            details = violation.get("details", {})
            genes = details.get("genes", [])
            violations.append(
                {
                    "type": "conflict",
                    "gene": ",".join(genes),
                    "message": f"Conflicting genes are simultaneously enabled: {', '.join(genes)}.",
                    "details": {"genes": genes},
                }
            )
        else:
            violations.append(violation)

    summary: Dict[str, int] = {}
    for v in violations:
        summary[v["type"]] = summary.get(v["type"], 0) + 1

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "summary": summary,
    }


def repair(
    genome: Dict[str, Any],
    violations: List[Violation],
    registry: Dict[str, Any],
    spec: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[RepairAction]]:
    """Repair invalid genomes by clamping ranges, disabling conflicts, and rebalancing budgets."""
    repaired = copy.deepcopy(genome)
    actions: List[RepairAction] = []

    # 1) Clamp out-of-range values.
    for gene, bounds in registry.items():
        if gene not in repaired:
            continue
        if isinstance(repaired[gene], bool):
            continue

        current = _as_weighted_value(repaired[gene])
        min_v = bounds.get("min")
        max_v = bounds.get("max")
        clamped = current
        if min_v is not None:
            clamped = max(clamped, float(min_v))
        if max_v is not None:
            clamped = min(clamped, float(max_v))
        if clamped != current:
            if isinstance(repaired[gene], dict):
                repaired[gene]["value"] = clamped
            else:
                repaired[gene] = clamped
            action = {
                "action": "clamp",
                "gene": gene,
                "before": current,
                "after": clamped,
            }
            actions.append(action)
            logging.debug("[repair] %s", action)

    # 2) Disable conflicting genes.
    priorities = spec.get("conflict_priority", {}) or {}
    for pair in spec.get("conflicts", []):
        if len(pair) != 2:
            continue
        left, right = pair
        if not (_is_enabled(repaired.get(left)) and _is_enabled(repaired.get(right))):
            continue

        left_priority = float(priorities.get(left, _as_weighted_value(repaired.get(left))))
        right_priority = float(priorities.get(right, _as_weighted_value(repaired.get(right))))
        disable_gene = right if left_priority >= right_priority else left
        _set_enabled(repaired, disable_gene, False)
        action = {
            "action": "disable_conflict",
            "gene": disable_gene,
            "reason": f"Conflict between {left} and {right}",
        }
        actions.append(action)
        logging.debug("[repair] %s", action)

    # 3) Rebalance budgets globally.
    budgets = spec.get("budgets", {})
    for budget_name, budget_cfg in budgets.items():
        limit = float(budget_cfg.get("limit", 0.0))
        genes = list(budget_cfg.get("genes", []))
        weights = budget_cfg.get("weights", {}) or {}

        current_total = 0.0
        active_genes: List[Tuple[str, float, float]] = []
        for gene in genes:
            if gene not in repaired:
                continue
            val = _as_weighted_value(repaired[gene])
            weight = float(weights.get(gene, 1.0))
            contribution = val * weight
            current_total += contribution
            if val > 0:
                active_genes.append((gene, val, weight))

        if current_total <= limit or not active_genes:
            continue

        scale = limit / current_total if current_total > 0 else 0.0
        for gene, val, _weight in active_genes:
            new_val = val * scale
            if isinstance(repaired[gene], dict):
                repaired[gene]["value"] = new_val
                if new_val <= 0:
                    repaired[gene]["enabled"] = False
            else:
                repaired[gene] = new_val
            action = {
                "action": "rebalance_budget",
                "budget": budget_name,
                "gene": gene,
                "before": val,
                "after": new_val,
                "scale": scale,
            }
            actions.append(action)
            logging.debug("[repair] %s", action)

    # 4) Remove unknown genes that cannot be repaired safely.
    unknown = {v.get("gene") for v in violations if v.get("type") == "unknown_gene"}
    for gene in unknown:
        if gene in repaired:
            before = repaired.pop(gene)
            action = {
                "action": "drop_unknown_gene",
                "gene": gene,
                "before": before,
            }
            actions.append(action)
            logging.debug("[repair] %s", action)

    return repaired, actions


def initialize_genome(registry: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, float]:
    rng = rng or random
    genome: Dict[str, float] = {}
    for gene, bounds in registry.items():
        min_v = float(bounds.get("min", 0.0))
        max_v = float(bounds.get("max", 1.0))
        genome[gene] = rng.uniform(min_v, max_v)
    return genome


def mutate_genome(genome: Dict[str, Any], registry: Dict[str, Any], mutation_rate: float = 0.1, mutation_scale: float = 0.2, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    rng = rng or random
    child = copy.deepcopy(genome)
    for gene in registry.keys():
        if gene not in child or rng.random() >= mutation_rate:
            continue
        if isinstance(child[gene], bool):
            child[gene] = not child[gene]
            continue
        delta = rng.uniform(-mutation_scale, mutation_scale)
        if isinstance(child[gene], dict):
            child[gene]["value"] = _as_weighted_value(child[gene]) + delta
        else:
            child[gene] = _as_weighted_value(child[gene]) + delta
    return child


def crossover_genomes(parent_a: Dict[str, Any], parent_b: Dict[str, Any], registry: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    rng = rng or random
    child: Dict[str, Any] = {}
    for gene in registry.keys():
        source = parent_a if rng.random() < 0.5 else parent_b
        if gene in source:
            child[gene] = copy.deepcopy(source[gene])
    return child


def run_generation_flow(parent_a: Dict[str, Any], parent_b: Dict[str, Any], registry: Dict[str, Any], spec: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Run initialization/mutation/crossover and enforce validate->repair->validate."""
    rng = rng or random

    init_genome = initialize_genome(registry, rng=rng)
    mutation_genome = mutate_genome(init_genome, registry, rng=rng)
    crossover_genome = crossover_genomes(mutation_genome, parent_b, registry, rng=rng)

    init_validation = validate(init_genome, registry, spec)
    mutation_validation = validate(mutation_genome, registry, spec)
    crossover_validation = validate(crossover_genome, registry, spec)

    invalid_paths = []
    if not init_validation["valid"]:
        invalid_paths.append("initialization")
    if not mutation_validation["valid"]:
        invalid_paths.append("mutation")
    if not crossover_validation["valid"]:
        invalid_paths.append("crossover")
    if invalid_paths:
        logging.warning("Invalid genomes can occur in generation paths: %s", ", ".join(invalid_paths))

    pre_validation = validate(crossover_genome, registry, spec)
    repaired = crossover_genome
    actions: List[RepairAction] = []
    if not pre_validation["valid"]:
        repaired, actions = repair(crossover_genome, pre_validation["violations"], registry, spec)
    post_validation = validate(repaired, registry, spec)

    for action in actions:
        logging.debug("[generation-flow][repair] %s", action)

    return {
        "invalid_paths": invalid_paths,
        "initialization_validation": init_validation,
        "mutation_validation": mutation_validation,
        "crossover_validation": crossover_validation,
        "pre_repair_validation": pre_validation,
        "repair_actions": actions,
        "post_repair_validation": post_validation,
        "genome": repaired,
    }

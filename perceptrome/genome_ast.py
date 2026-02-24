from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class RelationshipEdge:
    type: str
    source: str
    target: str
    weight: float = 1.0
    condition: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class GroupCardinalityConstraint:
    group: str
    min_count: int = 0
    max_count: Optional[int] = None


@dataclass(frozen=True)
class WeightedBudgetConstraint:
    name: str
    limit: float
    genes: Tuple[str, ...]
    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MutualExclusionConstraint:
    name: str
    genes: Tuple[str, ...]
    max_enabled: int = 1


@dataclass(frozen=True)
class ConditionalEnablementConstraint:
    name: str
    trigger_gene: str
    trigger_value: Any = True
    required_genes: Tuple[str, ...] = ()
    forbidden_genes: Tuple[str, ...] = ()


GlobalConstraint = Union[
    GroupCardinalityConstraint,
    WeightedBudgetConstraint,
    MutualExclusionConstraint,
    ConditionalEnablementConstraint,
]


@dataclass(frozen=True)
class RelationshipConstraintSet:
    edges: Tuple[RelationshipEdge, ...] = ()
    constraints: Tuple[GlobalConstraint, ...] = ()


def _condition_matches(condition: Optional[Mapping[str, Any]], values: Mapping[str, Any]) -> bool:
    if not condition:
        return True
    gene = str(condition.get("gene", ""))
    if not gene:
        return True
    op = str(condition.get("op", "eq"))
    actual = values.get(gene)
    expected = condition.get("value", True)
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "gt":
        return float(actual or 0.0) > float(expected)
    if op == "lt":
        return float(actual or 0.0) < float(expected)
    return False


def evaluate_relationship_constraints(
    *,
    values: Mapping[str, Any],
    enabled_genes: Iterable[str],
    gene_groups: Optional[Mapping[str, str]] = None,
    weighted_values: Optional[Mapping[str, float]] = None,
    schema: Optional[RelationshipConstraintSet] = None,
) -> List[Dict[str, Any]]:
    schema = schema or RelationshipConstraintSet()
    enabled = set(enabled_genes)
    groups = gene_groups or {}
    wvalues = weighted_values or {}
    violations: List[Dict[str, Any]] = []

    for edge in schema.edges:
        if edge.source not in enabled:
            continue
        if not _condition_matches(edge.condition, values):
            continue
        if edge.type == "requires" and edge.target not in enabled:
            violations.append(
                {
                    "type": "relationship_requires",
                    "gene": edge.source,
                    "message": f"{edge.source} requires {edge.target} to be enabled.",
                    "details": {"edge": edge},
                }
            )
        elif edge.type == "excludes" and edge.target in enabled:
            violations.append(
                {
                    "type": "relationship_excludes",
                    "gene": edge.source,
                    "message": f"{edge.source} excludes {edge.target}.",
                    "details": {"edge": edge},
                }
            )

    for constraint in schema.constraints:
        if isinstance(constraint, GroupCardinalityConstraint):
            count = sum(1 for g in enabled if groups.get(g) == constraint.group)
            if count < constraint.min_count:
                violations.append(
                    {
                        "type": "group_cardinality",
                        "gene": constraint.group,
                        "message": f"Group '{constraint.group}' has {count} genes, below minimum {constraint.min_count}.",
                        "details": {"group": constraint.group, "count": count, "min": constraint.min_count},
                    }
                )
            if constraint.max_count is not None and count > constraint.max_count:
                violations.append(
                    {
                        "type": "group_cardinality",
                        "gene": constraint.group,
                        "message": f"Group '{constraint.group}' has {count} genes, above maximum {constraint.max_count}.",
                        "details": {"group": constraint.group, "count": count, "max": constraint.max_count},
                    }
                )
        elif isinstance(constraint, WeightedBudgetConstraint):
            total = 0.0
            used: Dict[str, float] = {}
            for gene in constraint.genes:
                if gene not in enabled:
                    continue
                val = float(wvalues.get(gene, 0.0))
                weight = float(constraint.weights.get(gene, 1.0))
                contribution = val * weight
                used[gene] = contribution
                total += contribution
            if total > float(constraint.limit):
                violations.append(
                    {
                        "type": "budget_exceeded",
                        "gene": constraint.name,
                        "message": f"Budget '{constraint.name}' exceeded ({total} > {constraint.limit}).",
                        "details": {"budget": constraint.name, "total": total, "limit": float(constraint.limit), "contributions": used},
                    }
                )
        elif isinstance(constraint, MutualExclusionConstraint):
            active = [gene for gene in constraint.genes if gene in enabled]
            if len(active) > constraint.max_enabled:
                violations.append(
                    {
                        "type": "mutual_exclusion",
                        "gene": constraint.name,
                        "message": f"Mutual exclusion '{constraint.name}' violated by: {', '.join(active)}.",
                        "details": {"set": constraint.name, "genes": active, "max_enabled": constraint.max_enabled},
                    }
                )
        elif isinstance(constraint, ConditionalEnablementConstraint):
            if values.get(constraint.trigger_gene) != constraint.trigger_value:
                continue
            missing = [g for g in constraint.required_genes if g not in enabled]
            forbidden = [g for g in constraint.forbidden_genes if g in enabled]
            if missing or forbidden:
                violations.append(
                    {
                        "type": "conditional_enablement",
                        "gene": constraint.name,
                        "message": f"Conditional constraint '{constraint.name}' violated.",
                        "details": {"missing_required": missing, "forbidden_enabled": forbidden},
                    }
                )

    return violations


def parse_relationship_spec(spec: Mapping[str, Any]) -> RelationshipConstraintSet:
    edges: List[RelationshipEdge] = []
    constraints: List[GlobalConstraint] = []

    for edge in spec.get("relationship_edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        edges.append(
            RelationshipEdge(
                type=str(edge.get("type", "requires")),
                source=str(edge.get("source", "")),
                target=str(edge.get("target", "")),
                weight=float(edge.get("weight", 1.0)),
                condition=edge.get("condition"),
            )
        )

    for pair in spec.get("conflicts", []) or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            constraints.append(MutualExclusionConstraint(name=f"conflict:{pair[0]}:{pair[1]}", genes=(str(pair[0]), str(pair[1]))))

    budgets = spec.get("budgets", {}) or {}
    for budget_name, budget_cfg in budgets.items():
        if not isinstance(budget_cfg, Mapping):
            continue
        constraints.append(
            WeightedBudgetConstraint(
                name=str(budget_name),
                limit=float(budget_cfg.get("limit", 0.0)),
                genes=tuple(str(g) for g in budget_cfg.get("genes", [])),
                weights={str(k): float(v) for k, v in (budget_cfg.get("weights", {}) or {}).items()},
            )
        )

    for item in spec.get("global_constraints", []) or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind", "")).lower()
        if kind == "group_cardinality":
            constraints.append(
                GroupCardinalityConstraint(
                    group=str(item.get("group", "")),
                    min_count=int(item.get("min_count", 0)),
                    max_count=int(item["max_count"]) if item.get("max_count") is not None else None,
                )
            )
        elif kind == "weighted_budget":
            constraints.append(
                WeightedBudgetConstraint(
                    name=str(item.get("name", "budget")),
                    limit=float(item.get("limit", 0.0)),
                    genes=tuple(str(g) for g in item.get("genes", [])),
                    weights={str(k): float(v) for k, v in (item.get("weights", {}) or {}).items()},
                )
            )
        elif kind == "mutual_exclusion":
            constraints.append(
                MutualExclusionConstraint(
                    name=str(item.get("name", "mutex")),
                    genes=tuple(str(g) for g in item.get("genes", [])),
                    max_enabled=int(item.get("max_enabled", 1)),
                )
            )
        elif kind == "conditional_enablement":
            constraints.append(
                ConditionalEnablementConstraint(
                    name=str(item.get("name", "conditional")),
                    trigger_gene=str(item.get("trigger_gene", "")),
                    trigger_value=item.get("trigger_value", True),
                    required_genes=tuple(str(g) for g in item.get("required_genes", [])),
                    forbidden_genes=tuple(str(g) for g in item.get("forbidden_genes", [])),
                )
            )

    return RelationshipConstraintSet(edges=tuple(edges), constraints=tuple(constraints))

import random
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perceptrome.genome_evolution import validate, repair, run_generation_flow


def _registry():
    return {
        "gene_a": {"min": 0.0, "max": 1.0},
        "gene_b": {"min": 0.0, "max": 1.0},
        "gene_c": {"min": 0.0, "max": 2.0},
    }


def _spec():
    return {
        "conflicts": [("gene_a", "gene_b")],
        "conflict_priority": {"gene_a": 10, "gene_b": 1},
        "budgets": {
            "total_expression": {
                "limit": 1.5,
                "genes": ["gene_a", "gene_b", "gene_c"],
                "weights": {"gene_c": 1.0},
            }
        },
    }


def test_validate_reports_structured_violations():
    genome = {"gene_a": 2.0, "gene_b": 0.8, "gene_c": 1.1, "unknown": 1.0}
    result = validate(genome, _registry(), _spec())

    assert result["valid"] is False
    kinds = {v["type"] for v in result["violations"]}
    assert "out_of_range" in kinds
    assert "conflict" in kinds
    assert "budget_exceeded" in kinds
    assert "unknown_gene" in kinds
    assert result["summary"]["out_of_range"] >= 1


def test_repair_clamps_disables_and_rebalances():
    genome = {"gene_a": 2.0, "gene_b": 0.8, "gene_c": 1.4, "unknown": 5.0}
    first = validate(genome, _registry(), _spec())

    repaired, actions = repair(genome, first["violations"], _registry(), _spec())
    second = validate(repaired, _registry(), _spec())

    assert second["valid"] is True
    assert repaired["gene_a"] <= 1.0
    assert repaired["gene_b"] == 0.0
    assert "unknown" not in repaired
    action_types = {a["action"] for a in actions}
    assert "clamp" in action_types
    assert "disable_conflict" in action_types
    assert "rebalance_budget" in action_types


def test_generation_flow_runs_validate_repair_validate():
    parent_a = {"gene_a": 1.0, "gene_b": 1.0, "gene_c": 1.5}
    parent_b = {"gene_a": 1.0, "gene_b": 1.0, "gene_c": 1.2}

    result = run_generation_flow(parent_a, parent_b, _registry(), _spec(), rng=random.Random(7))

    assert "pre_repair_validation" in result
    assert "post_repair_validation" in result
    assert result["post_repair_validation"]["valid"] is True
    assert isinstance(result["repair_actions"], list)

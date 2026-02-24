import unittest

from perceptrome.genome_ast import (
    ConditionalEnablementConstraint,
    GroupCardinalityConstraint,
    RelationshipConstraintSet,
    RelationshipEdge,
)
from perceptrome.genome_spec import (
    Gene,
    GenomeSpec,
    GenomeValidationError,
    create_genome,
    mutate_genome,
    crossover_genomes,
    evaluate_genome,
)


class GenomeSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = GenomeSpec(
            min_gene_count=1,
            max_gene_count=4,
            max_total_weight=10.0,
            max_total_complexity=8.0,
            required_gene_groups={"core"},
            optional_gene_groups={"aux"},
            incompatible_gene_pairs={("g1", "gX")},
        )

    def test_create_genome_valid(self):
        genome = create_genome(
            [Gene("g1", "core", weight=2.0, complexity=1.0), Gene("g2", "aux", weight=1.0, complexity=1.0)],
            self.spec,
        )
        self.assertEqual(len(genome.genes), 2)

    def test_create_genome_reports_rule_and_gene_ids(self):
        with self.assertRaises(GenomeValidationError) as ctx:
            create_genome([Gene("gX", "aux", weight=1.0, complexity=1.0)], self.spec)
        self.assertEqual(ctx.exception.rule_name, "required_gene_groups")
        self.assertIn("gX", ctx.exception.gene_ids)

    def test_mutation_is_validated(self):
        genome = create_genome([Gene("g1", "core", weight=1.0, complexity=1.0)], self.spec)
        with self.assertRaises(GenomeValidationError) as ctx:
            mutate_genome(genome, self.spec, add_gene=Gene("gX", "core", weight=1.0, complexity=1.0))
        self.assertEqual(ctx.exception.rule_name, "incompatible_gene_pair")
        self.assertEqual(ctx.exception.gene_ids, ("g1", "gX"))

    def test_crossover_is_validated(self):
        parent_a = create_genome([Gene("g1", "core", 1.0, 1.0), Gene("g2", "aux", 1.0, 1.0)], self.spec)
        parent_b = create_genome([Gene("g3", "core", 1.0, 1.0), Gene("g4", "aux", 1.0, 1.0)], self.spec)
        child = crossover_genomes(parent_a, parent_b, self.spec, split_at=1)
        self.assertEqual(tuple(g.gene_id for g in child.genes), ("g1", "g4"))

    def test_evaluation_runs_on_valid_genome(self):
        genome = create_genome([Gene("g1", "core", 1.0, 1.0)], self.spec)
        score = evaluate_genome(genome, self.spec, lambda g: len(g.genes) * 2)
        self.assertEqual(score, 2.0)

    def test_relationship_requires_target_gene(self):
        spec = GenomeSpec(
            min_gene_count=1,
            required_gene_groups={"core"},
            optional_gene_groups={"aux"},
            relationship_edges=(RelationshipEdge(type="requires", source="g1", target="g2"),),
        )
        with self.assertRaises(GenomeValidationError) as ctx:
            create_genome([Gene("g1", "core", weight=1.0, complexity=1.0)], spec)
        self.assertEqual(ctx.exception.rule_name, "relationship_requires")

    def test_global_constraints_group_and_conditional(self):
        spec = GenomeSpec(
            min_gene_count=1,
            required_gene_groups={"core"},
            optional_gene_groups={"aux"},
            global_constraints=RelationshipConstraintSet(
                constraints=(
                    GroupCardinalityConstraint(group="aux", min_count=1),
                    ConditionalEnablementConstraint(
                        name="if_g1_then_g2",
                        trigger_gene="g1",
                        trigger_value=True,
                        required_genes=("g2",),
                    ),
                )
            ),
        )
        with self.assertRaises(GenomeValidationError):
            create_genome([Gene("g1", "core", 1.0, 1.0)], spec)


if __name__ == "__main__":
    unittest.main()

import unittest

from perceptrome.genome_schema import (
    CURRENT_GENOME_SCHEMA_VERSION,
    AST_EDGE_TYPE_CONFLICTS_WITH,
    AST_EDGE_TYPE_DEPENDS_ON,
    downgrade_genome_payload,
    extract_genes_from_payload,
    migrate_genome_payload,
)


class GenomeSchemaTests(unittest.TestCase):
    def _defaults(self):
        return dict(
            tokenizer="base",
            seq_len=256,
            vocab_size=5,
            hidden_dim=128,
            loss_type="mse",
            model_type="transformer",
            transformer_d_model=128,
            transformer_nhead=8,
            transformer_layers=4,
            transformer_dropout=0.1,
            learning_rate=1e-3,
            beta_kl=1e-3,
        )

    def test_migrate_flat_genes_to_ast(self):
        payload = {
            "schema_version": 2,
            "genes": {"tokenizer": "AA", "learning_rate_norm": 0.5, "latent_dim": 64},
            "dependencies": [["transformer_layers", "model_type"]],
            "conflicts": [["hidden_dim", "transformer_d_model"]],
            "constraints": {"max_active": 5},
        }

        migrated = migrate_genome_payload(payload, **self._defaults())

        self.assertEqual(migrated["schema_version"], CURRENT_GENOME_SCHEMA_VERSION)
        self.assertIn("nodes", migrated)
        self.assertIn("edges", migrated)
        self.assertEqual(migrated["constraints"], {"max_active": 5})

        genes = extract_genes_from_payload(migrated)
        self.assertEqual(genes["tokenizer"], "aa")
        self.assertEqual(genes["hidden_dim"], 64)
        self.assertIn("learning_rate", genes)

        edges = {(e["source"], e["target"], e["type"]) for e in migrated["edges"]}
        self.assertIn(("transformer_layers", "model_type", AST_EDGE_TYPE_DEPENDS_ON), edges)
        self.assertIn(("hidden_dim", "transformer_d_model", AST_EDGE_TYPE_CONFLICTS_WITH), edges)
        self.assertIn(("transformer_d_model", "hidden_dim", AST_EDGE_TYPE_CONFLICTS_WITH), edges)

    def test_extract_and_downgrade_from_ast(self):
        ast_payload = {
            "schema_version": 3,
            "nodes": [
                {"id": "tokenizer", "kind": "gene", "value": "base"},
                {"id": "lr", "kind": "gene", "value": 0.01},
            ],
            "edges": [],
        }

        genes = extract_genes_from_payload(ast_payload)
        self.assertEqual(genes["learning_rate"], 0.01)

        downgraded = downgrade_genome_payload(ast_payload)
        self.assertEqual(downgraded["schema_version"], 2)
        self.assertEqual(downgraded["genes"]["learning_rate"], 0.01)


if __name__ == "__main__":
    unittest.main()

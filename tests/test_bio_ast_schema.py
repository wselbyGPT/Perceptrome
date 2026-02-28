import unittest

from perceptrome.bio_ast import BioAST, DomainNode, GeneNode, SMENode, build_sme_node
from tests.fixtures.bio_ast_regression_fixtures import SME_MOTIF_FIXTURES


class BioASTSchemaTests(unittest.TestCase):
    def test_serialization_deserialization_round_trip(self):
        motif = SME_MOTIF_FIXTURES[0]
        nodes = (
            GeneNode(canonical_id="gene:motif:1", gene_id="motif1", child_ids=("domain:motif:1",), start=1, end=120, strand="+"),
            DomainNode(canonical_id="domain:motif:1", parent_id="gene:motif:1", child_ids=("sme:motif:1",), start=20, end=90),
            SMENode(
                canonical_id="sme:motif:1",
                parent_id="domain:motif:1",
                start=25,
                end=55,
                secondary_tag=motif["secondary_tag"],
                motif_family=motif["motif_family"],
                motif_subtype=motif["motif_subtype"],
                energetic_evolutionary=motif["energetic_evolutionary"],
                metadata={"motif_name": motif["motif_name"]},
            ),
        )
        ast = BioAST(nodes=nodes)

        restored = BioAST.from_dict(ast.to_dict())

        self.assertEqual(restored.schema_version, 2)
        self.assertEqual(len(restored.nodes), 3)
        restored_sme = next(node for node in restored.nodes if isinstance(node, SMENode))
        self.assertEqual(restored_sme.metadata.get("motif_name"), "helix-entry")
        self.assertEqual(restored_sme.secondary_tag, "H")

    def test_schema_version_migration_from_v1(self):
        payload = {
            "schema_version": 1,
            "genes": [
                {"gene_id": "gA", "value": 1.0},
                {"gene_id": "gB", "value": 2.0},
            ],
        }

        ast = BioAST.from_dict(payload)

        self.assertEqual(ast.schema_version, 2)
        self.assertEqual(tuple(g.gene_id for g in ast.genes), ("gA", "gB"))
        self.assertEqual(len(ast.nodes), 2)

    def test_parent_child_integrity_for_sme_subtree(self):
        parent = DomainNode(canonical_id="domain:int:1", child_ids=("existing",), start=10, end=80)
        updated_parent, sme, children = build_sme_node(
            parent=parent,
            sme_id="sme:int:1",
            residue_window=[(20, 20), 21],
            kmer_window=[(22, 24)],
            secondary_tag=SME_MOTIF_FIXTURES[1]["secondary_tag"],
            motif_family=SME_MOTIF_FIXTURES[1]["motif_family"],
            motif_subtype=SME_MOTIF_FIXTURES[1]["motif_subtype"],
            energetic_evolutionary=SME_MOTIF_FIXTURES[1]["energetic_evolutionary"],
            metadata={"motif_name": SME_MOTIF_FIXTURES[1]["motif_name"]},
        )

        self.assertIn(sme.canonical_id, updated_parent.child_ids)
        self.assertEqual(sme.parent_id, updated_parent.canonical_id)
        self.assertTrue(all(child.parent_id == sme.canonical_id for child in children))


if __name__ == "__main__":
    unittest.main()

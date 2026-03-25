import unittest

from perceptrome.bio_ast import BioAST, GeneNode, PlasmidNode, RelationshipEdge
from perceptrome.bio_reg_graph import build_bio_regulatory_graph


class BioRegGraphTests(unittest.TestCase):
    def _synthetic_ast(self) -> BioAST:
        root = PlasmidNode(canonical_id="plasmid:SYN", start=1, end=5000)
        gene_a = GeneNode(canonical_id="gene:SYN:repA", gene_id="repA", parent_id=root.canonical_id, start=100, end=900, strand="+")
        gene_b = GeneNode(canonical_id="gene:SYN:kanR", gene_id="kanR", parent_id=root.canonical_id, start=980, end=1700, strand="+")
        gene_c = GeneNode(canonical_id="gene:SYN:cargo_gfp", gene_id="cargo_gfp", parent_id=root.canonical_id, start=1900, end=2600, strand="+")
        return BioAST(nodes=(root, gene_a, gene_b, gene_c), relationships=(RelationshipEdge(source_id=gene_a.canonical_id, target_id=gene_b.canonical_id, kind="adjacent_to"),))

    def test_brg_inference_adds_regulatory_and_module_nodes(self):
        sequence = "A" * 80 + "AGGAGG" + "A" * 5000
        brg = build_bio_regulatory_graph(self._synthetic_ast(), sequence=sequence)

        node_types = {node.node_type for node in brg.nodes}
        self.assertIn("promoter", node_types)
        self.assertIn("rbs", node_types)
        self.assertIn("transcript_unit", node_types)
        self.assertIn("operon", node_types)
        self.assertIn("replication_module", node_types)
        self.assertIn("selection_module", node_types)
        self.assertIn("cargo_module", node_types)

        edge_kinds = {edge.kind for edge in brg.semantic_edges}
        self.assertIn("promoter_of", edge_kinds)
        self.assertIn("rbs_for", edge_kinds)
        self.assertIn("part_of_transcript_unit", edge_kinds)
        self.assertIn("part_of_operon", edge_kinds)
        self.assertIn("part_of_module", edge_kinds)
        self.assertIn("encodes", edge_kinds)

    def test_roundtrip_with_extended_schema(self):
        brg = build_bio_regulatory_graph(self._synthetic_ast(), sequence="A" * 6000)
        restored = BioAST.from_dict(brg.to_dict())

        self.assertEqual(restored.schema_version, 4)
        self.assertEqual(len(restored.nodes), len(brg.nodes))
        self.assertEqual({e.kind for e in restored.semantic_edges}, {e.kind for e in brg.semantic_edges})


if __name__ == "__main__":
    unittest.main()

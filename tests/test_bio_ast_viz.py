import unittest

from perceptrome.bio_ast import BioAST, CDSNode, GeneNode, GenomeNode, ORFNode, RelationshipEdge, SMENode
from perceptrome.encoding.bio_ast_edges import derive_semantic_edges
from perceptrome.encoding.bio_ast_viz import ast_to_graph_json, ast_to_tree_json


class BioASTVisualizationTests(unittest.TestCase):
    def _sample_ast(self) -> BioAST:
        root = GenomeNode(canonical_id="genome:ACC", start=1, end=120)
        gene_b = GeneNode(canonical_id="gene:ACC:2", gene_id="g_b", parent_id=root.canonical_id, start=70, end=100)
        orf_b = ORFNode(canonical_id="orf:ACC:2", parent_id=gene_b.canonical_id, start=70, end=100)
        cds_b = CDSNode(canonical_id="cds:ACC:2", parent_id=orf_b.canonical_id, start=70, end=100)

        gene_a = GeneNode(canonical_id="gene:ACC:1", gene_id="g_a", parent_id=root.canonical_id, start=10, end=50)
        orf_a = ORFNode(canonical_id="orf:ACC:1", parent_id=gene_a.canonical_id, start=10, end=50)
        cds_a = CDSNode(canonical_id="cds:ACC:1", parent_id=orf_a.canonical_id, start=10, end=50)

        rel = RelationshipEdge(source_id="gene:ACC:2", target_id="gene:ACC:1", kind="regulates", metadata={"weight": 1})

        # Intentionally scrambled node order to verify deterministic sorting.
        return BioAST(nodes=(gene_b, cds_b, root, cds_a, orf_b, gene_a, orf_a), relationships=(rel,))

    def test_tree_json_schema(self):
        payload = ast_to_tree_json(self._sample_ast(), accession="ACC")

        self.assertEqual(payload["schema"], "bio_ast_tree_v1")
        self.assertEqual(payload["accession"], "ACC")
        self.assertIn("hierarchy", payload)
        self.assertEqual(payload["roots"], ["genome:ACC"])

        root = payload["hierarchy"][0]
        self.assertEqual(root["id"], "genome:ACC")
        self.assertEqual(root["node_type"], "genome")
        self.assertIn("children", root)
        self.assertEqual(root["children"][0]["id"], "gene:ACC:1")
        self.assertEqual(root["children"][1]["id"], "gene:ACC:2")

    def test_graph_json_schema_and_deterministic_node_order(self):
        payload = ast_to_graph_json(self._sample_ast(), accession="ACC")

        self.assertEqual(payload["schema"], "bio_ast_graph_v1")
        self.assertEqual(payload["accession"], "ACC")
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)

        node_ids = [node["id"] for node in payload["nodes"]]
        self.assertEqual(
            node_ids,
            [
                "genome:ACC",
                "gene:ACC:1",
                "orf:ACC:1",
                "cds:ACC:1",
                "gene:ACC:2",
                "orf:ACC:2",
                "cds:ACC:2",
            ],
        )

        semantic_edges = [edge for edge in payload["edges"] if edge["relation_type"] == "semantic"]
        self.assertEqual(len(semantic_edges), 1)
        self.assertEqual(semantic_edges[0]["source"], "gene:ACC:2")
        self.assertEqual(semantic_edges[0]["target"], "gene:ACC:1")
        self.assertEqual(semantic_edges[0]["relation"], "regulates")
        self.assertEqual(semantic_edges[0]["id"], "semantic:regulates:gene:ACC:2->gene:ACC:1")

    def test_derived_coordinate_edges_label_evidence_and_directionality(self):
        root = GenomeNode(canonical_id="genome:EDGE", start=1, end=120)
        gene_a = GeneNode(canonical_id="gene:EDGE:1", gene_id="g1", parent_id=root.canonical_id, start=10, end=20)
        gene_b = GeneNode(canonical_id="gene:EDGE:2", gene_id="g2", parent_id=root.canonical_id, start=18, end=30)
        gene_c = GeneNode(canonical_id="gene:EDGE:3", gene_id="g3", parent_id=root.canonical_id, start=31, end=40)
        ast = BioAST(nodes=(root, gene_c, gene_b, gene_a))

        derived = derive_semantic_edges(ast)
        edges = {(edge.source_id, edge.target_id, edge.kind): edge.metadata for edge in derived}

        self.assertIn(("gene:EDGE:1", "gene:EDGE:2", "overlaps"), edges)
        self.assertIn(("gene:EDGE:2", "gene:EDGE:3", "adjacent_to"), edges)
        self.assertNotIn(("gene:EDGE:2", "gene:EDGE:1", "overlaps"), edges)
        self.assertEqual(edges[("gene:EDGE:1", "gene:EDGE:2", "overlaps")]["evidence"], "inferred")
        self.assertTrue(edges[("gene:EDGE:2", "gene:EDGE:3", "adjacent_to")]["inferred"])

    def test_coordinate_derivation_suppresses_duplicates_against_existing_edges(self):
        root = GenomeNode(canonical_id="genome:DUP", start=1, end=80)
        gene_a = GeneNode(canonical_id="gene:DUP:1", gene_id="ga", parent_id=root.canonical_id, start=10, end=20)
        gene_b = GeneNode(canonical_id="gene:DUP:2", gene_id="gb", parent_id=root.canonical_id, start=18, end=30)
        existing = RelationshipEdge(source_id=gene_a.canonical_id, target_id=gene_b.canonical_id, kind="overlaps", metadata={"evidence": "curated"})
        ast = BioAST(nodes=(root, gene_a, gene_b), relationships=(existing,))

        derived = derive_semantic_edges(ast)
        self.assertEqual(derived, ())

    def test_regulatory_edges_require_annotation_or_regulatory_feature_support(self):
        root = GenomeNode(canonical_id="genome:REG", start=1, end=150)
        regulator = SMENode(
            canonical_id="sme:REG:1",
            parent_id=root.canonical_id,
            start=10,
            end=15,
            motif_family="REGULATORY",
            metadata={"feature_type": "promoter"},
        )
        gene = GeneNode(canonical_id="gene:REG:1", gene_id="target", parent_id=root.canonical_id, start=20, end=50)
        ast = BioAST(nodes=(root, regulator, gene))

        derived = derive_semantic_edges(ast, feature_annotations={"sme:REG:1": {"feature_type": "promoter", "regulates": ["gene:REG:1"]}})
        payload = ast_to_graph_json(BioAST(nodes=ast.nodes, relationships=derived), accession="REG")
        semantic_edges = [edge for edge in payload["semantic_edges"] if edge["relation"] == "regulates"]

        self.assertEqual(len(semantic_edges), 1)
        self.assertEqual(semantic_edges[0]["evidence"], "curated")
        self.assertEqual(semantic_edges[0]["metadata"]["provenance"]["source"], "annotation")


if __name__ == "__main__":
    unittest.main()

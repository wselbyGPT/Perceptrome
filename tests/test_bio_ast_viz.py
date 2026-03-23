import unittest

from perceptrome.bio_ast import BioAST, CDSNode, GeneNode, GenomeNode, ORFNode, RelationshipEdge
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


if __name__ == "__main__":
    unittest.main()

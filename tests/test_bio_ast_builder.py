import unittest

from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.encoding.genbank_features import CDSFeature


class BioASTBuilderTests(unittest.TestCase):
    def test_build_with_cds_features_and_adapters(self):
        seq = "ATG" + ("GCT" * 80) + "TAA"
        features = [
            CDSFeature(
                start=1,
                end=len(seq),
                strand=1,
                gene_or_locus_tag="geneA",
                product="proteinA",
                protein_length=80,
                translation_source="provided",
            )
        ]

        built = BioASTBuilder().build(
            sequence=seq,
            cds_features=features,
            top_level_type="plasmid",
            accession="ACC1",
        )

        node_types = [node.node_type for node in built.ast.nodes]
        self.assertIn("plasmid", node_types)
        self.assertIn("gene", node_types)
        self.assertIn("cds", node_types)
        self.assertIn("sme", node_types)

        paths = built.to_serialized_paths()
        self.assertTrue(paths)

        tensors = built.to_tree_message_passing_tensors()
        self.assertEqual(set(tensors.keys()), {"node_type_ids", "coords", "strand", "edge_index"})
        self.assertEqual(tensors["coords"].shape[0], len(built.ast.nodes))

        windows = built.to_local_windows(window_size=32, stride=16)
        self.assertEqual(windows.shape[1:], (32, 4))

    def test_fallback_orf_path_when_features_absent(self):
        seq = "ATG" + ("GCT" * 35) + "TAA"
        built = BioASTBuilder().build(sequence=seq, accession="ACC2")
        self.assertGreaterEqual(len(built.ast.genes), 1)


if __name__ == "__main__":
    unittest.main()

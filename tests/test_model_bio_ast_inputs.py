import unittest

import numpy as np

from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.encoding.genbank_features import CDSFeature
from tests.fixtures.bio_ast_regression_fixtures import SYNTHETIC_FASTA_SEQUENCE


class ModelBioASTInputAdapterTests(unittest.TestCase):
    def test_tree_cnn_transformer_adapter_shapes(self):
        feature = CDSFeature(
            start=1,
            end=len(SYNTHETIC_FASTA_SEQUENCE),
            strand=1,
            gene_or_locus_tag="adapter_gene",
            product="adapter protein",
            protein_length=max(0, len(SYNTHETIC_FASTA_SEQUENCE) // 3 - 1),
            translation_source="provided",
        )
        built = BioASTBuilder().build(sequence=SYNTHETIC_FASTA_SEQUENCE, cds_features=[feature], accession="ADAPT1")

        tree_tensors = built.to_tree_message_passing_tensors()
        self.assertEqual(tree_tensors["node_type_ids"].shape[0], len(built.ast.nodes))
        self.assertEqual(tree_tensors["coords"].shape, (len(built.ast.nodes), 2))
        self.assertEqual(tree_tensors["edge_index"].shape[0], 2)

        cnn_windows = built.to_local_windows(window_size=24, stride=12)
        self.assertEqual(cnn_windows.shape[1:], (24, 4))
        self.assertGreaterEqual(cnn_windows.shape[0], 1)

        transformer_paths = built.to_serialized_paths()
        transformer_lengths = np.asarray([len(path["path"]) for path in transformer_paths], dtype=np.int64)
        self.assertGreaterEqual(transformer_lengths.shape[0], 1)
        self.assertEqual(transformer_lengths.ndim, 1)
        self.assertTrue(all(len(item["path"]) == len(item["types"]) for item in transformer_paths))


if __name__ == "__main__":
    unittest.main()

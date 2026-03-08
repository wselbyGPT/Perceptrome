import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.encoding.genbank_features import CDSFeature
from tests.fixtures.bio_ast_regression_fixtures import SYNTHETIC_FASTA_SEQUENCE
from perceptrome.model import BioASTEmbeddingAPI
from perceptrome.cli.commands import _export_bio_ast_embeddings_for_accession


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


class BioASTEmbeddingAPITests(unittest.TestCase):
    def test_embedding_api_shapes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        feature_len = len(SYNTHETIC_FASTA_SEQUENCE)
        seq = np.zeros((1, feature_len, 4), dtype=np.float32)
        seq[:, :, 0] = 1.0
        node_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        coords = np.array([[[1, 5], [6, 10], [11, 15], [16, 20], [21, 25]]], dtype=np.float32)
        strand = np.array([[0, 1, -1, 0, 1]], dtype=np.int64)

        model = BioASTEmbeddingAPI(seq_vocab_size=4, hidden_dim=32, ast_tree_layers=2, motif_kernel_size=3, motif_channels=16)
        out = model(
            torch.from_numpy(seq),
            torch.from_numpy(node_ids),
            ast_coords=torch.from_numpy(coords),
            ast_strand=torch.from_numpy(strand),
        )

        self.assertEqual(tuple(out.fixed.shape), (1, 32))
        self.assertEqual(tuple(out.token.shape), (1, feature_len, 32))
        self.assertEqual(tuple(out.node.shape), (1, 5, 32))

    def test_embedding_export_metadata_schema(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")

        built = BioASTBuilder().build(sequence=SYNTHETIC_FASTA_SEQUENCE, accession="META1")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "latest.pt")
            with open(ckpt_path, "wb") as handle:
                handle.write(b"checkpoint-bytes")

            with patch("perceptrome.cli.commands._build_bio_ast", return_value=built), patch(
                "perceptrome.cli.commands.ensure_run_layout", return_value=tmp
            ), patch("perceptrome.cli.commands.path_in_run", side_effect=lambda layout, *parts: os.path.join(layout, *parts)), patch(
                "perceptrome.cli.commands.update_run_manifest", return_value=None
            ):
                outputs = _export_bio_ast_embeddings_for_accession(
                    accession="META1",
                    source="genbank",
                    io_cfg=None,
                    ckpt_path=ckpt_path,
                    hidden_dim=24,
                    ast_tree_layers=3,
                    ast_motif_kernel_size=5,
                    ast_motif_channels=12,
                )

            with open(outputs["metadata"], "r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            self.assertEqual(metadata["schema_version"], "bio_ast_embedding_v1")
            self.assertIn("sha256", metadata["checkpoint"])
            self.assertEqual(metadata["ast_config"]["ast_tree_layers"], 3)
            self.assertEqual(metadata["ast_config"]["ast_motif_kernel_size"], 5)
            self.assertEqual(metadata["ast_config"]["ast_motif_channels"], 12)
            self.assertEqual(metadata["accession"], "META1")
            self.assertEqual(metadata["sequence_length"], len(SYNTHETIC_FASTA_SEQUENCE))
            self.assertEqual(metadata["node_count"], len(built.ast.nodes))
            self.assertIn("fixed", metadata["embedding_shapes"])
            self.assertIn("token", metadata["embedding_shapes"])
            self.assertIn("node", metadata["embedding_shapes"])
            self.assertTrue(os.path.exists(outputs["fixed"]))
            self.assertTrue(os.path.exists(outputs["token"]))
            self.assertTrue(os.path.exists(outputs["node"]))


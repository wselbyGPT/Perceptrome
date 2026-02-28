import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.encoding.genbank_features import CDSFeature, parse_cds_features_from_genbank
from perceptrome.encoding.parse import parse_fasta_sequence, parse_genbank_dna
from tests.fixtures.bio_ast_regression_fixtures import SYNTHETIC_FASTA_SEQUENCE, render_fasta, render_genbank


class BioASTBuilderTests(unittest.TestCase):
    def _assert_coordinate_consistency(self, built):
        seq_len = len(built.sequence)
        for node in built.ast.nodes:
            if node.start is None or node.end is None:
                continue
            self.assertGreaterEqual(node.start, 1)
            self.assertLessEqual(node.end, seq_len)
            self.assertLessEqual(node.start, node.end)

    def test_build_from_fasta_sequence(self):
        with TemporaryDirectory() as tmp:
            fasta_path = Path(tmp) / "synthetic.fasta"
            fasta_path.write_text(render_fasta(), encoding="utf-8")
            sequence = parse_fasta_sequence(str(fasta_path))

        feature = CDSFeature(
            start=1,
            end=len(sequence),
            strand=1,
            gene_or_locus_tag="fasta_gene",
            product="synthetic fasta protein",
            protein_length=max(0, len(sequence) // 3 - 1),
            translation_source="provided",
        )
        built = BioASTBuilder().build(sequence=sequence, cds_features=[feature], accession="FASTA1")

        self.assertEqual(built.sequence, SYNTHETIC_FASTA_SEQUENCE)
        self.assertEqual(built.ast.nodes[0].start, 1)
        self.assertEqual(built.ast.nodes[0].end, len(SYNTHETIC_FASTA_SEQUENCE))
        self._assert_coordinate_consistency(built)

    def test_build_from_genbank_features(self):
        with TemporaryDirectory() as tmp:
            gb_path = Path(tmp) / "synthetic.gbk"
            gb_path.write_text(render_genbank(), encoding="utf-8")
            sequence = parse_genbank_dna(str(gb_path))
            features = parse_cds_features_from_genbank(str(gb_path))

        self.assertEqual(len(features), 1)
        built = BioASTBuilder().build(sequence=sequence, cds_features=features, top_level_type="plasmid", accession="GB1")

        node_types = [node.node_type for node in built.ast.nodes]
        self.assertIn("plasmid", node_types)
        self.assertIn("gene", node_types)
        self.assertIn("cds", node_types)
        self.assertIn("sme", node_types)
        self._assert_coordinate_consistency(built)


if __name__ == "__main__":
    unittest.main()

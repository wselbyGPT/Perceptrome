import os
import tempfile
import unittest

import numpy as np

from perceptrome.config import IOConfig
from perceptrome.encoding_main import encode_accession, tokenizer_meta


class TokenizerMetaTests(unittest.TestCase):
    def test_codon_window_requires_multiple_of_three(self):
        with self.assertRaises(ValueError):
            tokenizer_meta("codon", 10)

    def test_tokenizer_meta_shapes(self):
        self.assertEqual(tokenizer_meta("base", 12), (12, 4))
        self.assertEqual(tokenizer_meta("codon", 12), (4, 65))
        self.assertEqual(tokenizer_meta("aa", 7), (7, 21))


class EncodeAccessionTests(unittest.TestCase):
    def _write_fasta(self, path: str, seq: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(">test\n")
            handle.write(seq)
            handle.write("\n")

    def test_encode_accession_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_fasta = os.path.join(tmp, "fasta")
            cache_genbank = os.path.join(tmp, "genbank")
            cache_encoded = os.path.join(tmp, "encoded")
            os.makedirs(cache_fasta, exist_ok=True)
            os.makedirs(cache_genbank, exist_ok=True)
            os.makedirs(cache_encoded, exist_ok=True)

            accession = "TEST_BASE"
            fasta_path = os.path.join(cache_fasta, f"{accession}.fasta")
            self._write_fasta(fasta_path, "ACGTACGTAC")

            io_cfg = IOConfig(
                cache_fasta_dir=cache_fasta,
                cache_genbank_dir=cache_genbank,
                cache_encoded_dir=cache_encoded,
                model_dir=os.path.join(tmp, "model"),
                checkpoints_dir=os.path.join(tmp, "model", "checkpoints"),
                logs_dir=os.path.join(tmp, "logs"),
                state_file=os.path.join(tmp, "state.json"),
            )

            encoded = encode_accession(
                accession,
                io_cfg,
                window_size=4,
                stride=2,
                tokenizer="base",
                source="fasta",
                save_to_disk=False,
            )

            self.assertEqual(encoded.shape, (4, 4, 4))
            self.assertTrue(np.all((encoded == 0) | (encoded == 1)))

    def test_encode_accession_aa_from_fasta_orf(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_fasta = os.path.join(tmp, "fasta")
            cache_genbank = os.path.join(tmp, "genbank")
            cache_encoded = os.path.join(tmp, "encoded")
            os.makedirs(cache_fasta, exist_ok=True)
            os.makedirs(cache_genbank, exist_ok=True)
            os.makedirs(cache_encoded, exist_ok=True)

            accession = "TEST_AA"
            fasta_path = os.path.join(cache_fasta, f"{accession}.fasta")
            # ATG AAA AAA TAA -> ORF length 9bp = 3aa
            self._write_fasta(fasta_path, "ATGAAAAAATAA")

            io_cfg = IOConfig(
                cache_fasta_dir=cache_fasta,
                cache_genbank_dir=cache_genbank,
                cache_encoded_dir=cache_encoded,
                model_dir=os.path.join(tmp, "model"),
                checkpoints_dir=os.path.join(tmp, "model", "checkpoints"),
                logs_dir=os.path.join(tmp, "logs"),
                state_file=os.path.join(tmp, "state.json"),
            )

            encoded = encode_accession(
                accession,
                io_cfg,
                window_size=4,
                stride=2,
                tokenizer="aa",
                source="fasta",
                min_orf_aa=2,
                save_to_disk=False,
            )

            self.assertEqual(encoded.shape, (1, 4, 21))
            self.assertTrue(np.all((encoded == 0) | (encoded == 1)))


if __name__ == "__main__":
    unittest.main()

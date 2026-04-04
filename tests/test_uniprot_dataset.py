import gzip
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Session = object
    requests_stub.Response = object
    requests_stub.request = lambda *a, **k: None
    sys.modules["requests"] = requests_stub

from perceptrome.io_utils import read_catalog
from perceptrome import uniprot_dataset


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


class UniProtDatasetTests(unittest.TestCase):
    def test_iter_fasta_records_and_malformed(self):
        records = list(
            uniprot_dataset.iter_fasta_records(
                [">sp|P1|a", "ACD", "EF", ">tr|Q2|b", "MN*"]
            )
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].sequence, "ACDEF")

        with self.assertRaises(uniprot_dataset.FastaValidationError):
            list(uniprot_dataset.iter_fasta_records(["ACD", ">sp|P1|a", "AAA"]))

        with self.assertRaises(uniprot_dataset.FastaValidationError):
            list(uniprot_dataset.iter_fasta_records([">sp|P1|a", "AA1"]))

    def test_shard_split_boundary_and_off_by_one(self):
        with tempfile.TemporaryDirectory() as td:
            writer = uniprot_dataset.ShardWriter(prefix_path=os.path.join(td, "out"), records_per_shard=2, use_gzip=False)
            writer.add_record(uniprot_dataset.FastaRecord(header=">a", sequence="AA"))
            writer.add_record(uniprot_dataset.FastaRecord(header=">b", sequence="BB"))
            writer.add_record(uniprot_dataset.FastaRecord(header=">c", sequence="CC"))
            writer.close()

            self.assertEqual(writer.total_records, 3)
            self.assertEqual(len(writer.shards), 2)
            self.assertEqual(writer.shards[0].record_count, 2)
            self.assertEqual(writer.shards[1].record_count, 1)

    def test_gzip_and_non_gzip_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "dataset")
            w_plain = uniprot_dataset.ShardWriter(prefix_path=prefix, records_per_shard=10, use_gzip=False)
            w_plain.add_record(uniprot_dataset.FastaRecord(header=">a", sequence="AAAA"))
            w_plain.close()
            plain_path = w_plain.shards[0].path
            self.assertTrue(plain_path.endswith(".fasta"))
            with open(plain_path, "r", encoding="utf-8") as f:
                self.assertIn(">a", f.read())

            w_gz = uniprot_dataset.ShardWriter(prefix_path=prefix + "_gz", records_per_shard=10, use_gzip=True)
            w_gz.add_record(uniprot_dataset.FastaRecord(header=">b", sequence="BBBB"))
            w_gz.close()
            gz_path = w_gz.shards[0].path
            self.assertTrue(gz_path.endswith(".fasta.gz"))
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                self.assertIn(">b", f.read())

    def test_fetch_dataset_manifest_and_catalog_generation(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "uniprot")
            fake_stream = FakeResponse([
                ">sp|P11111|A", "AAAA", ">tr|Q22222|B", "CC", ">sp|R33333|C", "DDDD",
            ])

            with patch("perceptrome.uniprot_dataset.fetch_uniprot_count", return_value={"count": 3, "count_source": "header:x-total-results"}), patch(
                "perceptrome.uniprot_dataset.stream_uniprot_fasta", return_value=fake_stream.iter_lines()
            ):
                result = uniprot_dataset.fetch_uniprot_dataset(
                    query="taxonomy_id:2",
                    include_isoforms=False,
                    prefix_path=prefix,
                    records_per_shard=2,
                    use_gzip=False,
                    resume=False,
                    timeout=1,
                    max_retries=1,
                    backoff_seconds=0,
                )

            manifest_path = result["manifest_path"]
            catalog_path = result["catalog_path"]
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["total_records"], 3)
            self.assertEqual(manifest["total_residues"], 10)
            self.assertAlmostEqual(manifest["average_length"], 10 / 3)
            self.assertEqual(manifest["accession_preview"], ["P11111", "Q22222", "R33333"])
            self.assertEqual(len(manifest["shards"]), 2)

            for shard in manifest["shards"]:
                with open(shard["path"], "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                self.assertEqual(shard["sha256"], digest)

            with open(catalog_path, "r", encoding="utf-8") as f:
                raw = f.read()
            self.assertIn("# Perceptrome UniProt accession catalog", raw)
            self.assertEqual(read_catalog(catalog_path), ["P11111", "Q22222", "R33333"])


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest

from perceptrome.virus.training import normalize_virus_training_input


class VirusTrainingInputTests(unittest.TestCase):
    def test_resolves_catalog_from_catalog_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            catalog_path = os.path.join(td, "catalog.txt")
            with open(catalog_path, "w", encoding="utf-8") as handle:
                handle.write("NC_1\n")
            manifest_path = os.path.join(td, "catalog.manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"catalog": {"path": catalog_path}}, handle)

            normalized = normalize_virus_training_input(
                catalog=None,
                catalog_manifest=manifest_path,
                fetch_manifest=None,
                sequence_source="cds",
                segmented_policy="split",
                dedupe="accession",
                metadata_path=None,
                complete_only=True,
                refseq_only=False,
            )

            self.assertEqual(normalized.catalog_path, catalog_path)
            self.assertEqual(normalized.record_source, "genbank")
            self.assertTrue(normalized.provenance["complete_only"])

    def test_resolves_catalog_from_fetch_manifest_with_nested_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            catalog_path = os.path.join(td, "catalog.txt")
            with open(catalog_path, "w", encoding="utf-8") as handle:
                handle.write("NC_1\n")
            catalog_manifest_path = os.path.join(td, "catalog.manifest.json")
            with open(catalog_manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"catalog": {"path": catalog_path}}, handle)
            fetch_manifest_path = os.path.join(td, "fetch.manifest.json")
            with open(fetch_manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"source": {"type": "manifest", "manifest": catalog_manifest_path}}, handle)

            normalized = normalize_virus_training_input(
                catalog=None,
                catalog_manifest=None,
                fetch_manifest=fetch_manifest_path,
                sequence_source="genome",
                segmented_policy="none",
                dedupe="none",
                metadata_path=None,
                complete_only=False,
                refseq_only=True,
            )

            self.assertEqual(normalized.catalog_path, catalog_path)
            self.assertEqual(normalized.record_source, "fasta")
            self.assertEqual(normalized.provenance["catalog_manifest_path"], catalog_manifest_path)
            self.assertTrue(normalized.provenance["refseq_only"])


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from perceptrome.virus import catalog


class VirusCatalogTests(unittest.TestCase):
    def test_build_catalog_accession_mode_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "catalog.txt")
            args = types.SimpleNamespace(
                taxon=None,
                accession=["NC_1", "NC_2", "NC_2"],
                inputfile=None,
                include=["host"],
                filter=["--assembly-level", "complete"],
                datasets_bin=None,
                output=out,
                snapshot=False,
                snapshot_dir=None,
                snapshot_metadata=None,
            )

            fake_json = {
                "reports": [
                    {"accession": "NC_1"},
                    {"accession": "NC_2"},
                ]
            }
            with mock.patch.object(catalog, "resolve_datasets_binary", return_value="/bin/datasets"), mock.patch.object(
                catalog, "_run_datasets_json", return_value={"parsed": fake_json, "stdout": "{}", "stderr": "", "return_code": 0}
            ):
                result = catalog.build_catalog_from_args(args)

            self.assertTrue(os.path.exists(out))
            manifest_path = os.path.join(td, "catalog.manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            manifest = catalog.load_manifest(manifest_path)
            self.assertEqual(manifest["source"], "ncbi_datasets_virus")
            self.assertEqual(manifest["query"]["mode"], "accession")
            self.assertEqual(manifest["query"]["includes"], ["host"])
            self.assertEqual(manifest["catalog"]["accession_count"], 2)
            self.assertEqual(result["accession_count"], 2)

    def test_rebuild_reports_mismatch_when_hash_changed(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "catalog.manifest.json")
            manifest = {
                "catalog": {
                    "path": os.path.join(td, "catalog.txt"),
                    "accession_count": 1,
                    "accession_sha256": "not-the-right-hash",
                },
                "datasets": {"argv": ["/usr/bin/datasets", "summary", "virus", "genome", "accession", "NC_1", "--as-json"]},
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            fake_json = {"reports": [{"accession": "NC_1"}]}
            with mock.patch.object(catalog, "resolve_datasets_binary", return_value="/bin/datasets"), mock.patch.object(
                catalog, "_run_datasets_json", return_value={"parsed": fake_json, "stdout": "{}", "stderr": "", "return_code": 0}
            ):
                result = catalog.rebuild_from_manifest(manifest_path=manifest_path)

            self.assertFalse(result["match"])
            self.assertEqual(result["actual_count"], 1)
            self.assertTrue(os.path.exists(manifest["catalog"]["path"]))

    def test_snapshot_bundle_copies_catalog_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            catalog_path = os.path.join(td, "catalog.txt")
            manifest_path = os.path.join(td, "catalog.manifest.json")
            with open(catalog_path, "w", encoding="utf-8") as handle:
                handle.write("NC_1\nNC_2\n")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"catalog": {"path": catalog_path}}, handle)

            result = catalog.create_snapshot_bundle(catalog_path=catalog_path, manifest_path=manifest_path)

            self.assertTrue(os.path.isdir(result["bundle_dir"]))
            self.assertTrue(os.path.exists(result["bundle_manifest_path"]))
            self.assertTrue(os.path.exists(os.path.join(result["bundle_dir"], "catalog.txt")))
            self.assertTrue(os.path.exists(os.path.join(result["bundle_dir"], "catalog.manifest.json")))


if __name__ == "__main__":
    unittest.main()

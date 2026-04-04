import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = object
    np_stub.float32 = float
    np_stub.int64 = int
    np_stub.array = lambda v, dtype=None: list(v)
    np_stub.random = types.SimpleNamespace(Generator=object)
    sys.modules["numpy"] = np_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Module = object
    optim_stub = types.ModuleType("torch.optim")
    fn_stub = types.ModuleType("torch.nn.functional")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")
    data_stub.DataLoader = object
    data_stub.TensorDataset = object

    torch_stub.nn = nn_stub
    torch_stub.optim = optim_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.optim"] = optim_stub
    sys.modules["torch.nn.functional"] = fn_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.data"] = data_stub

import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Session = object
    requests_stub.Response = object
    requests_stub.request = lambda *a, **k: None
    sys.modules["requests"] = requests_stub

from perceptrome.cli.commands import cmd_uniprot_count, cmd_uniprot_fetch


class UniProtCliTests(unittest.TestCase):
    def _base_patches(self, td):
        io_cfg = SimpleNamespace(cache_fasta_dir=td)
        uniprot_cfg = SimpleNamespace(
            base_url="https://rest.uniprot.org",
            default_query="taxonomy_id:2",
            records_per_shard=100,
            gzip_output=False,
            include_isoforms=False,
            request_timeout=5,
            retries=2,
            backoff_seconds=0,
        )
        return [
            patch("perceptrome.cli.commands.load_full_config", return_value={}),
            patch("perceptrome.cli.commands.extract_configs", return_value=(None, None, io_cfg)),
            patch("perceptrome.cli.commands.extract_uniprot_config", return_value=uniprot_cfg),
            patch("perceptrome.cli.commands._run_local_io_cfg", side_effect=lambda c: c),
            patch("perceptrome.cli.commands.ensure_dirs", return_value=None),
        ]

    def test_cmd_uniprot_fetch_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                config="config/stream_config.yaml",
                query="taxonomy_id:2",
                mode="all",
                records_per_shard=2,
                output_dir=td,
                prefix="u",
                gzip_output=False,
                include_isoforms=False,
                count_only=False,
                resume=False,
                json=True,
            )
            fake_result = {
                "resumed": False,
                "manifest_path": os.path.join(td, "u.manifest.json"),
                "catalog_path": os.path.join(td, "u.catalog.txt"),
                "manifest": {"total_records": 2, "shards": [{"path": "a.fasta"}]},
                "live_count": {"count": 99, "count_source": "header:x-total-results"},
            }
            out = io.StringIO()
            patches = self._base_patches(td) + [patch("perceptrome.cli.commands.fetch_uniprot_dataset", return_value=fake_result)]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], redirect_stdout(out):
                rc = cmd_uniprot_fetch(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["count"], 99)
            self.assertEqual(payload["downloaded_records"], 2)

    def test_cmd_uniprot_fetch_count_only(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                config="config/stream_config.yaml",
                query="taxonomy_id:2",
                mode="reviewed",
                records_per_shard=2,
                output_dir=td,
                prefix="u",
                gzip_output=False,
                include_isoforms=False,
                count_only=True,
                resume=False,
                json=True,
            )
            fake_result = {
                "count_only": True,
                "query": "x",
                "include_isoforms": False,
                "live_count": {"count": 17, "count_source": "body:total"},
            }
            out = io.StringIO()
            patches = self._base_patches(td) + [patch("perceptrome.cli.commands.fetch_uniprot_dataset", return_value=fake_result)]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], redirect_stdout(out):
                rc = cmd_uniprot_fetch(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["count_only"])
            self.assertEqual(payload["count"], 17)

    def test_cmd_uniprot_fetch_normal_print(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                config="config/stream_config.yaml",
                query="taxonomy_id:2",
                mode="unreviewed",
                records_per_shard=2,
                output_dir=td,
                prefix="u",
                gzip_output=False,
                include_isoforms=False,
                count_only=False,
                resume=True,
                json=False,
            )
            fake_result = {
                "resumed": True,
                "manifest_path": os.path.join(td, "u.manifest.json"),
                "catalog_path": os.path.join(td, "u.catalog.txt"),
                "manifest": {"total_records": 1, "shards": [{"path": "x.fasta"}]},
                "live_count": {"count": 1, "count_source": "header:x-total-results"},
            }
            out = io.StringIO()
            patches = self._base_patches(td) + [patch("perceptrome.cli.commands.fetch_uniprot_dataset", return_value=fake_result)]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], redirect_stdout(out):
                rc = cmd_uniprot_fetch(args)

            self.assertEqual(rc, 0)
            printed = out.getvalue()
            self.assertIn("[uniprot-fetch] query count=1 downloaded=1", printed)
            self.assertIn("manifest:", printed)

    def test_cmd_uniprot_count_falls_back_to_config_query_plaintext(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                config="config/stream_config.yaml",
                query=None,
                mode="reviewed",
                json=False,
            )
            out = io.StringIO()
            patches = self._base_patches(td) + [
                patch("perceptrome.cli.commands.fetch_uniprot_count", return_value={"count": 23}),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as fetch_count, redirect_stdout(out):
                rc = cmd_uniprot_count(args)

            self.assertEqual(rc, 0)
            fetch_count.assert_called_once_with(
                "taxonomy_id:2",
                timeout=5.0,
                max_retries=2,
                backoff_seconds=0.0,
                base_url="https://rest.uniprot.org",
            )
            self.assertIn("[uniprot-count] mode=reviewed count=23", out.getvalue())

    def test_cmd_uniprot_count_falls_back_to_config_query_json(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(
                config="config/stream_config.yaml",
                query=None,
                mode="reviewed",
                json=True,
            )
            out = io.StringIO()
            patches = self._base_patches(td) + [
                patch("perceptrome.cli.commands.fetch_uniprot_count", return_value={"count": 31}),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], redirect_stdout(out):
                rc = cmd_uniprot_count(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["query"], "taxonomy_id:2")
            self.assertEqual(payload["mode"], "reviewed")
            self.assertEqual(payload["effective_query"], "taxonomy_id:2")
            self.assertEqual(payload["count"], 31)


if __name__ == "__main__":
    unittest.main()

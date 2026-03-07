import json
import sys
import tempfile
import types
import unittest
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

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *a, **k: None
    sys.modules["requests"] = requests_stub

from perceptrome.cli.commands import _reference_score, cmd_validate_plasmid


class ValidatePlasmidTests(unittest.TestCase):
    def test_reference_score_prefers_identical(self):
        same = _reference_score("ACGTACGT", "ACGTACGT")
        diff = _reference_score("ACGTACGT", "TTTTTTTT")
        self.assertGreater(same["score"], diff["score"])
        self.assertAlmostEqual(same["seq_similarity"], 1.0)

    def test_cmd_validate_plasmid_writes_ranked_json(self):
        with tempfile.TemporaryDirectory() as td:
            generated = f"{td}/gen.fasta"
            ref1 = f"{td}/REF1.fasta"
            ref2 = f"{td}/REF2.fasta"
            out_json = f"{td}/report.json"

            with open(generated, "w", encoding="utf-8") as f:
                f.write(">gen\nACGTACGTACGT\n")
            with open(ref1, "w", encoding="utf-8") as f:
                f.write(">ref1\nACGTACGTACGT\n")
            with open(ref2, "w", encoding="utf-8") as f:
                f.write(">ref2\nTTTTTTTTTTTT\n")

            catalog_path = f"{td}/catalog.txt"
            with open(catalog_path, "w", encoding="utf-8") as f:
                f.write("REF1\nREF2\n")

            args = SimpleNamespace(
                config="config/stream_config.yaml",
                generated_fasta=generated,
                catalog=catalog_path,
                top_n=2,
                output_json=out_json,
                force_fetch=False,
            )

            with patch("perceptrome.jobs.engine.load_full_config", return_value={}), patch(
                "perceptrome.jobs.engine.extract_configs",
                return_value=(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(logs_dir=td)),
            ), patch("perceptrome.jobs.engine.ensure_dirs", return_value=None), patch(
                "perceptrome.jobs.engine.setup_logging", return_value=None
            ), patch("perceptrome.jobs.engine.read_catalog", return_value=["REF1", "REF2"]), patch(
                "perceptrome.jobs.engine._ensure_record",
                side_effect=lambda accession, src, io_cfg, ncbi_cfg, force: ref1 if accession == "REF1" else ref2,
            ):
                rc = cmd_validate_plasmid(args)

            self.assertEqual(rc, 0)
            with open(out_json, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["results"][0]["accession"], "REF1")
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(str(report["generated_fasta"]).endswith("gen.fasta"))
            self.assertTrue(str(report["catalog"]).endswith("catalog.txt"))


if __name__ == "__main__":
    unittest.main()

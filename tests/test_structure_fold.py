import json
import os
import stat
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
    np_stub.random = types.SimpleNamespace(Generator=object, seed=lambda *a, **k: None)
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

from perceptrome.cli_main import build_parser
from perceptrome.cli import commands
from perceptrome.structure.colabfold_runner import ENV_COLABFOLD_BIN, resolve_colabfold_binary
from perceptrome.structure.fold_manifest import build_fold_manifest_update
from perceptrome.structure.parsers import discover_colabfold_outputs
from perceptrome.structure.summary import build_fold_summary_record, write_summary_json, write_summary_tsv


class StructureFoldTests(unittest.TestCase):
    def test_cli_parser_fold_one(self):
        args = build_parser().parse_args(["fold-one", "protein.fasta", "--num-recycle", "4"])
        self.assertEqual(args.command, "fold-one")
        self.assertEqual(args.fasta, "protein.fasta")
        self.assertEqual(args.num_recycle, 4)

    def test_resolve_colabfold_binary_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            exe = os.path.join(td, "colabfold_batch")
            with open(exe, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(exe, stat.S_IRWXU)
            with patch.dict(os.environ, {ENV_COLABFOLD_BIN: exe}, clear=False):
                resolved = resolve_colabfold_binary(None)
            self.assertEqual(resolved, exe)

    def test_discover_and_summary_export(self):
        fixture = os.path.join("tests", "fixtures", "colabfold", "sample_job")
        fasta = os.path.join(fixture, "input.fasta")

        artifacts = discover_colabfold_outputs(fixture)
        self.assertTrue(artifacts.structures_pdb)

        record = build_fold_summary_record(
            protein_id="p1",
            source_input_path=fasta,
            engine="colabfold",
            engine_status="ok",
            artifacts=artifacts,
        )
        self.assertAlmostEqual(record.mean_plddt or 0.0, 85.0, places=2)
        self.assertAlmostEqual(record.ptm or 0.0, 0.71, places=2)

        with tempfile.TemporaryDirectory() as td:
            jpath = write_summary_json(os.path.join(td, "summary.json"), [record])
            tpath = write_summary_tsv(os.path.join(td, "summary.tsv"), [record])
            self.assertTrue(os.path.exists(jpath))
            self.assertTrue(os.path.exists(tpath))

    def test_manifest_update_payload(self):
        rec = SimpleNamespace(
            protein_id="p1",
            engine_status="ok",
            aa_length=24,
            mean_plddt=85.0,
            ptm=0.71,
            rank_1_structure_path=os.path.join("tests", "fixtures", "colabfold", "sample_job", "rank_001_model_1.pdb"),
        )
        payload = build_fold_manifest_update(
            command_name="fold_one",
            run_id="run_test",
            summary_json_path="runs/run_test/outputs/summary.json",
            summary_tsv_path="runs/run_test/outputs/summary.tsv",
            stdout_log_path="runs/run_test/provenance/stdout.log",
            stderr_log_path="runs/run_test/provenance/stderr.log",
            records=[rec],
        )
        self.assertIn("paths", payload)
        self.assertIn("artifacts", payload)
        self.assertGreaterEqual(len(payload["artifacts"]), 4)

    def test_fold_batch_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            os.makedirs(in_dir, exist_ok=True)
            with open(os.path.join(in_dir, "short.fasta"), "w", encoding="utf-8") as f:
                f.write(">s\nMK\n")
            with open(os.path.join(in_dir, "long.fasta"), "w", encoding="utf-8") as f:
                f.write(">l\n" + ("M" * 40) + "\n")

            out_run = commands.ensure_run_layout(run_id="fold_batch_test", base_dir=os.path.join(td, "runs"))
            sample_fasta = os.path.join("tests", "fixtures", "colabfold", "sample_job", "input2.fasta")
            sample_artifacts = discover_colabfold_outputs(os.path.join("tests", "fixtures", "colabfold", "sample_job"))
            sample_record = build_fold_summary_record(
                protein_id="long",
                source_input_path=sample_fasta,
                engine="colabfold",
                engine_status="ok",
                artifacts=sample_artifacts,
            )

            args = SimpleNamespace(
                input_dir=in_dir,
                run_id="fold_batch_test",
                engine="colabfold",
                num_recycle=3,
                num_models=5,
                colabfold_bin=None,
                min_protein_aa=10,
                max_protein_aa=50,
                keep_going=True,
            )
            with patch("perceptrome.cli.commands.ensure_run_layout", return_value=out_run), patch(
                "perceptrome.cli.commands._run_fold_one_internal", return_value=(sample_record, os.path.join(td, "o.log"), os.path.join(td, "e.log"))
            ):
                rc = commands.cmd_fold_batch(args)
            self.assertEqual(rc, 0)
            batch_path = os.path.join(out_run.outputs_dir, "batch_summary.json")
            with open(batch_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload.get("total_inputs"), 1)


if __name__ == "__main__":
    unittest.main()

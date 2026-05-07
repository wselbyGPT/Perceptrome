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
    requests_stub.Session = object
    requests_stub.Response = object
    sys.modules["requests"] = requests_stub

from perceptrome.cli_main import build_parser
from perceptrome.cli import commands
from perceptrome.structure.alphafold3_runner import (
    ENV_ALPHAFOLD3_BIN,
    ENV_ALPHAFOLD3_DB_DIR,
    ENV_ALPHAFOLD3_MODEL_DIR,
    build_alphafold3_protein_job,
    resolve_alphafold3_binary,
    resolve_alphafold3_db_dir,
    resolve_alphafold3_model_dir,
    sanitize_job_name,
)
from perceptrome.structure.colabfold_runner import ENV_COLABFOLD_BIN, resolve_colabfold_binary
from perceptrome.structure.fold_manifest import build_fold_manifest_update
from perceptrome.structure.parsers import (
    discover_alphafold3_outputs,
    discover_colabfold_outputs,
    read_alphafold3_plddt_values,
)
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


class AlphaFold3BackendTests(unittest.TestCase):
    FIXTURE_DIR = os.path.join("tests", "fixtures", "alphafold3", "sample_job")

    def test_cli_parser_fold_one_alphafold3(self):
        args = build_parser().parse_args(
            [
                "fold-one",
                "protein.fasta",
                "--engine",
                "alphafold3",
                "--alphafold3-bin",
                "/opt/af3/run_alphafold.py",
                "--alphafold3-model-dir",
                "/opt/af3/models",
                "--alphafold3-db-dir",
                "/opt/af3/databases",
                "--num-seeds",
                "2",
                "--num-diffusion-samples",
                "3",
            ]
        )
        self.assertEqual(args.engine, "alphafold3")
        self.assertEqual(args.alphafold3_bin, "/opt/af3/run_alphafold.py")
        self.assertEqual(args.alphafold3_model_dir, "/opt/af3/models")
        self.assertEqual(args.alphafold3_db_dir, "/opt/af3/databases")
        self.assertEqual(args.num_seeds, 2)
        self.assertEqual(args.num_diffusion_samples, 3)

    def test_cli_parser_fold_batch_engine_choices(self):
        args = build_parser().parse_args(
            [
                "fold-batch",
                "proteins/",
                "--engine",
                "alphafold3",
            ]
        )
        self.assertEqual(args.engine, "alphafold3")

    def test_resolve_alphafold3_binary_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "run_alphafold.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env python\nprint('stub')\n")
            with patch.dict(os.environ, {ENV_ALPHAFOLD3_BIN: script}, clear=False):
                resolved = resolve_alphafold3_binary(None)
            self.assertEqual(resolved, script)

    def test_resolve_alphafold3_model_and_db_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "models")
            db_dir = os.path.join(td, "db")
            os.makedirs(model_dir)
            os.makedirs(db_dir)
            with patch.dict(
                os.environ,
                {ENV_ALPHAFOLD3_MODEL_DIR: model_dir, ENV_ALPHAFOLD3_DB_DIR: db_dir},
                clear=False,
            ):
                self.assertEqual(resolve_alphafold3_model_dir(None), model_dir)
                self.assertEqual(resolve_alphafold3_db_dir(None), db_dir)

    def test_sanitize_job_name(self):
        self.assertEqual(sanitize_job_name("a b / c"), "a_b_c")
        self.assertEqual(sanitize_job_name(""), "job")

    def test_build_alphafold3_protein_job(self):
        fasta = os.path.join(self.FIXTURE_DIR, "input.fasta")
        payload = build_alphafold3_protein_job(fasta_path=fasta, job_name="p1", model_seeds=[7])
        self.assertEqual(payload["name"], "p1")
        self.assertEqual(payload["modelSeeds"], [7])
        self.assertEqual(payload["dialect"], "alphafold3")
        self.assertEqual(payload["version"], 1)
        seq_entry = payload["sequences"][0]["protein"]
        self.assertEqual(seq_entry["id"], "A")
        self.assertTrue(seq_entry["sequence"].startswith("MKTIIAL"))

    def test_discover_alphafold3_outputs(self):
        artifacts = discover_alphafold3_outputs(self.FIXTURE_DIR)
        self.assertTrue(artifacts.structures_cif)
        top = artifacts.structures_cif[0]
        self.assertTrue(top.endswith("p1_model.cif"))
        self.assertTrue(artifacts.result_jsons)
        self.assertTrue(artifacts.result_jsons[0].endswith("p1_summary_confidences.json"))
        self.assertIsNotNone(artifacts.ranking_json)
        self.assertTrue(artifacts.ranking_json.endswith("ranking_scores.csv"))

    def test_read_alphafold3_plddt_values(self):
        summary = os.path.join(self.FIXTURE_DIR, "p1_summary_confidences.json")
        vals = read_alphafold3_plddt_values(summary)
        self.assertEqual(len(vals), 5)
        self.assertAlmostEqual(sum(vals) / len(vals), 84.5, places=2)

    def test_alphafold3_summary_record_plddt_and_ptm(self):
        artifacts = discover_alphafold3_outputs(self.FIXTURE_DIR)
        record = build_fold_summary_record(
            protein_id="p1",
            source_input_path=os.path.join(self.FIXTURE_DIR, "input.fasta"),
            engine="alphafold3",
            engine_status="ok",
            artifacts=artifacts,
        )
        self.assertEqual(record.fold_engine, "alphafold3")
        self.assertIsNotNone(record.mean_plddt)
        self.assertAlmostEqual(record.mean_plddt or 0.0, 84.5, places=2)
        self.assertAlmostEqual(record.ptm or 0.0, 0.82, places=2)
        self.assertTrue(record.rank_1_structure_path.endswith("p1_model.cif"))

    def test_fold_internal_dispatch_alphafold3(self):
        with tempfile.TemporaryDirectory() as td:
            out_run = commands.ensure_run_layout(
                run_id="af3_dispatch_test", base_dir=os.path.join(td, "runs")
            )
            source_fasta = os.path.join(self.FIXTURE_DIR, "input.fasta")

            def _fake_runner(**kwargs):
                fold_out = kwargs["output_dir"]
                os.makedirs(fold_out, exist_ok=True)
                for name in (
                    "p1_model.cif",
                    "p1_summary_confidences.json",
                    "p1_confidences.json",
                    "ranking_scores.csv",
                ):
                    src = os.path.join(self.FIXTURE_DIR, name)
                    with open(src, "rb") as rf, open(os.path.join(fold_out, name), "wb") as wf:
                        wf.write(rf.read())
                stdout_log = kwargs["stdout_log_path"]
                stderr_log = kwargs["stderr_log_path"]
                os.makedirs(os.path.dirname(stdout_log), exist_ok=True)
                open(stdout_log, "w").close()
                open(stderr_log, "w").close()
                return SimpleNamespace(
                    return_code=0,
                    command=["run_alphafold.py"],
                    stdout_log_path=stdout_log,
                    stderr_log_path=stderr_log,
                    json_input_path=os.path.join(fold_out, "p1.input.json"),
                    job_name="p1",
                )

            with patch(
                "perceptrome.cli.commands.resolve_alphafold3_binary", return_value="/stub/run_alphafold.py"
            ), patch(
                "perceptrome.cli.commands.resolve_alphafold3_model_dir", return_value="/stub/models"
            ), patch(
                "perceptrome.cli.commands.resolve_alphafold3_db_dir", return_value="/stub/db"
            ), patch(
                "perceptrome.cli.commands.run_alphafold3_monomer", side_effect=_fake_runner
            ):
                record, stdout_log, stderr_log = commands._run_fold_one_internal(
                    fasta_path=source_fasta,
                    layout=out_run,
                    engine="alphafold3",
                    num_recycle=3,
                    num_models=5,
                    colabfold_bin=None,
                    alphafold3_bin=None,
                    alphafold3_model_dir=None,
                    alphafold3_db_dir=None,
                    num_seeds=1,
                    num_diffusion_samples=5,
                )
            self.assertEqual(record.engine_status, "ok")
            self.assertEqual(record.fold_engine, "alphafold3")
            self.assertTrue(record.rank_1_structure_path.endswith("p1_model.cif"))
            self.assertTrue(os.path.basename(stdout_log).endswith("alphafold3.stdout.log"))
            self.assertTrue(os.path.basename(stderr_log).endswith("alphafold3.stderr.log"))


if __name__ == "__main__":
    unittest.main()

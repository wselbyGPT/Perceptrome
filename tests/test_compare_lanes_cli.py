import json
import os
import tempfile
import unittest
import sys
import types

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

from types import SimpleNamespace
from unittest.mock import Mock, patch

from perceptrome.cli import commands


class CompareLanesBookkeepingTests(unittest.TestCase):
    def test_config_parity_check_raises(self):
        baseline = {"training": {"window_size": 512, "stride": 256, "tokenizer": "base", "frame_offset": 0, "min_orf_aa": 90, "batch_size": 8, "steps_per_plasmid": 2}}
        ast = {"training": {"window_size": 256, "stride": 256, "tokenizer": "base", "frame_offset": 0, "min_orf_aa": 90, "batch_size": 8, "steps_per_plasmid": 2}}
        with self.assertRaisesRegex(ValueError, "Config parity check failed"):
            commands._assert_benchmark_config_parity(baseline, ast)

    def test_split_parity_mismatch_raises(self):
        left = {"splits": {"train": ["A"], "val": ["B"], "test": ["C"]}}
        right = {"splits": {"train": ["A", "D"], "val": ["B"], "test": ["C"]}}
        with self.assertRaisesRegex(ValueError, "Split mismatch"):
            commands._assert_split_parity(left, right, "baseline.json", "ast.json")

    def test_cmd_compare_lanes_writes_metrics_files(self):
        with tempfile.TemporaryDirectory() as td:
            split_path = os.path.join(td, "split.json")
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump({"splits": {"train": ["A1", "A2"], "val": ["V1"], "test": ["T1"]}}, f)

            logs_dir = os.path.join(td, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            baseline_manifest = os.path.join(td, "baseline_manifest.json")
            ast_manifest = os.path.join(td, "ast_manifest.json")
            with open(baseline_manifest, "w", encoding="utf-8") as f:
                json.dump({"metrics": {"processed_accessions": 2, "last_total_loss": 0.9}}, f)
            with open(ast_manifest, "w", encoding="utf-8") as f:
                json.dump({"metrics": {"processed_accessions": 2, "last_total_loss": 0.7}}, f)

            run_root = os.path.join(td, "runs", "cmp")
            layout = commands.ensure_run_layout(run_id="cmp", base_dir=os.path.join(td, "runs"))

            engine = Mock()
            engine.run.side_effect = [
                SimpleNamespace(ok=True, message="ok", data={"manifest_path": baseline_manifest}),
                SimpleNamespace(ok=True, message="ok", data={"manifest_path": ast_manifest}),
            ]

            args = SimpleNamespace(
                config="stream_config.yaml",
                baseline_config=None,
                ast_config=None,
                split_name="default",
                baseline_split_path=split_path,
                ast_split_path=split_path,
                max_epochs=1,
                baseline_model_type="mlp",
                ast_model_type="hybrid",
                seed=123,
                experiment_id="cmp",
                steps_per_plasmid=1,
                batch_size=2,
                window_size=64,
                stride=32,
                delete_cache=False,
                tokenizer="base",
                frame_offset=0,
                min_orf_aa=90,
                source="fasta",
                max_windows_per_protein=None,
                protein_len_min=None,
                protein_len_max=None,
                translation_only=False,
                loss_type="mse",
                tb_run_id=None,
                tb_log_every=None,
            )

            cfg = {"training": {"window_size": 64, "stride": 32, "tokenizer": "base", "frame_offset": 0, "min_orf_aa": 90, "batch_size": 2, "steps_per_plasmid": 1}}
            with patch("perceptrome.cli.commands.load_full_config", side_effect=[cfg, cfg]), \
                 patch("perceptrome.cli.commands.extract_configs", return_value=(None, None, SimpleNamespace(state_file=os.path.join(td, "state", "progress.json"), logs_dir=logs_dir))), \
                 patch("perceptrome.cli.commands.ensure_dirs"), \
                 patch("perceptrome.cli.commands.setup_logging"), \
                 patch("perceptrome.cli.commands.ensure_run_layout", return_value=layout), \
                 patch("perceptrome.cli.commands.JobEngine", return_value=engine), \
                 patch("perceptrome.cli.commands._count_model_parameters", side_effect=[101, 202]), \
                 patch("perceptrome.cli.commands.update_run_manifest"):
                rc = commands.cmd_compare_lanes(args)

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(run_root, "metrics", "compare_lanes.json")))
            self.assertTrue(os.path.exists(os.path.join(run_root, "metrics", "compare_lanes.csv")))

    def test_cmd_compare_lanes_errors_when_metric_manifest_missing(self):
        with tempfile.TemporaryDirectory() as td:
            split_path = os.path.join(td, "split.json")
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump({"splits": {"train": ["A1"], "val": ["V1"], "test": ["T1"]}}, f)

            layout = commands.ensure_run_layout(run_id="cmp_missing", base_dir=os.path.join(td, "runs"))
            engine = Mock()
            engine.run.side_effect = [
                SimpleNamespace(ok=True, message="ok", data={"manifest_path": os.path.join(td, "missing_baseline.json")}),
                SimpleNamespace(ok=True, message="ok", data={"manifest_path": os.path.join(td, "missing_ast.json")}),
            ]
            args = SimpleNamespace(
                config="stream_config.yaml",
                baseline_config=None,
                ast_config=None,
                split_name="default",
                baseline_split_path=split_path,
                ast_split_path=split_path,
                max_epochs=1,
                baseline_model_type="mlp",
                ast_model_type="hybrid",
                seed=123,
                experiment_id="cmp_missing",
                steps_per_plasmid=1,
                batch_size=2,
                window_size=64,
                stride=32,
                delete_cache=False,
                tokenizer="base",
                frame_offset=0,
                min_orf_aa=90,
                source="fasta",
                max_windows_per_protein=None,
                protein_len_min=None,
                protein_len_max=None,
                translation_only=False,
                loss_type="mse",
                tb_run_id=None,
                tb_log_every=None,
            )
            cfg = {"training": {"window_size": 64, "stride": 32, "tokenizer": "base", "frame_offset": 0, "min_orf_aa": 90, "batch_size": 2, "steps_per_plasmid": 1}}
            with patch("perceptrome.cli.commands.load_full_config", side_effect=[cfg, cfg]), \
                 patch("perceptrome.cli.commands.extract_configs", return_value=(None, None, SimpleNamespace(state_file=os.path.join(td, "state", "progress.json"), logs_dir=os.path.join(td, "logs")))), \
                 patch("perceptrome.cli.commands.ensure_dirs"), \
                 patch("perceptrome.cli.commands.setup_logging"), \
                 patch("perceptrome.cli.commands.ensure_run_layout", return_value=layout), \
                 patch("perceptrome.cli.commands.JobEngine", return_value=engine):
                with self.assertRaisesRegex(RuntimeError, "metrics manifest missing"):
                    commands.cmd_compare_lanes(args)


if __name__ == "__main__":
    unittest.main()

import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = object
    np_stub.float64 = float
    np_stub.float32 = float
    np_stub.array = lambda v, dtype=None: list(v)
    np_stub.ones = lambda shape, dtype=None: [1.0] * int(shape[0])
    np_stub.clip = lambda a, amin, amax: a
    np_stub.log = lambda x: x
    np_stub.exp = lambda x: x
    np_stub.isfinite = lambda x: True
    np_stub.max = max
    np_stub.random = types.SimpleNamespace(Generator=object, choice=lambda n, p=None: 0, randint=lambda a, b=None: 0)
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
from perceptrome.generate import ast_conditioning_metadata, parse_ast_conditioning_config
from perceptrome.jobs.engine import JobEngine, JobSpec


class AstConditioningCliTests(unittest.TestCase):
    def test_generate_entrypoints_accept_ast_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "generate-plasmid",
                "--ast-artifact",
                "ast.json",
                "--ast-node-type-prompt",
                "gene",
                "--ast-region-span",
                "10:200",
                "--ast-graph-mask",
                "neighbors",
                "--ast-graph-hop-limit",
                "2",
                "--ast-mask-strength",
                "0.5",
            ]
        )
        self.assertEqual(args.ast_artifact, "ast.json")
        self.assertEqual(args.ast_node_type_prompt, ["gene"])
        self.assertEqual(args.ast_region_span, ["10:200"])
        self.assertEqual(args.ast_graph_mask, "neighbors")
        self.assertEqual(args.ast_graph_hop_limit, 2)
        self.assertEqual(args.ast_mask_strength, 0.5)

        protein_args = parser.parse_args(["generate-protein", "--ast-node-type-prompt", "motif"])
        self.assertEqual(protein_args.ast_node_type_prompt, ["motif"])

        design_args = parser.parse_args(["design-loop", "--catalog", "catalog.txt", "--ast-region-span", "1:9"])
        self.assertEqual(design_args.ast_region_span, ["1:9"])


class AstConditioningMetadataTests(unittest.TestCase):
    def test_metadata_is_deterministic(self):
        cfg_a = parse_ast_conditioning_config(
            ast_artifact="/tmp/ast.json",
            ast_node_type_prompt=["gene", "motif", "gene"],
            ast_region_span=["5:10", "10:15"],
            ast_graph_mask="neighbors",
            ast_graph_hop_limit=2,
            ast_mask_strength=0.4,
        )
        cfg_b = parse_ast_conditioning_config(
            ast_artifact="/tmp/ast.json",
            ast_node_type_prompt=["motif", "gene"],
            ast_region_span=["5:10", "10:15"],
            ast_graph_mask="neighbors",
            ast_graph_hop_limit=2,
            ast_mask_strength=0.4,
        )
        details = {"edge_count": 7, "node_type_counts": {"gene": 2, "motif": 3}}
        self.assertEqual(ast_conditioning_metadata(cfg_a, details), ast_conditioning_metadata(cfg_b, details))


class AstConditioningJobEngineTests(unittest.TestCase):
    def test_generate_plasmid_job_passes_ast_conditioning(self):
        with tempfile.TemporaryDirectory() as td:
            ast_path = os.path.join(td, "ast.json")
            with open(ast_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": [{"node_type": "gene"}], "edges": []}, f)

            io_cfg = SimpleNamespace(logs_dir=td)
            train_cfg = SimpleNamespace(window_size=16, tokenizer="base", model_type="mlp")

            captured = {}

            def _fake_generate_plasmid_sequence(**kwargs):
                captured["ast_conditioning"] = kwargs.get("ast_conditioning")
                return "ACGT"

            with patch("perceptrome.jobs.engine.load_full_config", return_value={}), \
                patch("perceptrome.jobs.engine.extract_configs", return_value=(None, train_cfg, io_cfg)), \
                patch("perceptrome.jobs.engine.ensure_dirs"), \
                patch("perceptrome.jobs.engine.setup_logging"), \
                patch("perceptrome.jobs.engine.ensure_run_layout", return_value=SimpleNamespace(run_id="r1")), \
                patch("perceptrome.jobs.engine.path_in_run", side_effect=lambda _layout, _kind, name: os.path.join(td, name)), \
                patch("perceptrome.jobs.engine.generate_plasmid_sequence", side_effect=_fake_generate_plasmid_sequence), \
                patch.object(JobEngine, "_write_run_manifest", return_value=os.path.join(td, "manifest.json")), \
                patch("perceptrome.jobs.engine.update_run_manifest"):
                engine = JobEngine()
                out = engine._run_generate_plasmid(
                    JobSpec(
                        kind="generate_plasmid",
                        config_path="config.yml",
                        params={
                            "ast_artifact": ast_path,
                            "ast_node_type_prompt": ["gene"],
                            "ast_region_span": ["4:12"],
                            "ast_graph_mask": "neighbors",
                            "ast_graph_hop_limit": 2,
                            "ast_mask_strength": 0.3,
                        },
                    )
                )

            self.assertEqual(out["length"], 4)
            self.assertIsNotNone(captured["ast_conditioning"])
            self.assertEqual(captured["ast_conditioning"].artifact_path, ast_path)
            self.assertEqual(captured["ast_conditioning"].node_type_prompts, ("gene",))


if __name__ == "__main__":
    unittest.main()

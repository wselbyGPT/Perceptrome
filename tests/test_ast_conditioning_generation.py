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
from perceptrome.generate import ast_conditioning_metadata, ast_template_validation_metadata, parse_ast_conditioning_config, parse_ast_template_validation_config
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
                "--ast-template-artifact",
                "template.json",
                "--ast-template-mode",
                "reject",
                "--ast-template-span-tolerance",
                "12",
            ]
        )
        self.assertEqual(args.ast_artifact, "ast.json")
        self.assertEqual(args.ast_node_type_prompt, ["gene"])
        self.assertEqual(args.ast_region_span, ["10:200"])
        self.assertEqual(args.ast_graph_mask, "neighbors")
        self.assertEqual(args.ast_graph_hop_limit, 2)
        self.assertEqual(args.ast_mask_strength, 0.5)
        self.assertEqual(args.ast_template_artifact, "template.json")
        self.assertEqual(args.ast_template_mode, "reject")
        self.assertEqual(args.ast_template_span_tolerance, 12)

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




class AstTemplateValidationMetadataTests(unittest.TestCase):
    def test_metadata_serializes_template_policy(self):
        cfg = parse_ast_template_validation_config(
            template_artifact="/tmp/template.json",
            template_mode="reject",
            template_span_tolerance=15,
            template_min_score=0.8,
            template_max_mismatches=2,
            template_include_semantic_edges=True,
        )
        self.assertEqual(
            ast_template_validation_metadata(cfg),
            {
                "enabled": True,
                "artifact_path": "/tmp/template.json",
                "mode": "reject",
                "span_tolerance": 15,
                "min_score": 0.8,
                "max_mismatches": 2,
                "include_semantic_edges": True,
            },
        )

class AstConditioningJobEngineTests(unittest.TestCase):
    def test_generate_plasmid_job_passes_ast_conditioning(self):
        with tempfile.TemporaryDirectory() as td:
            ast_path = os.path.join(td, "ast.json")
            with open(ast_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": [{"node_type": "gene"}], "edges": []}, f)

            ckpt_dir = os.path.join(td, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "latest.pt"), "wb") as f:
                f.write(b"checkpoint")
            io_cfg = SimpleNamespace(logs_dir=td, checkpoints_dir=ckpt_dir)
            train_cfg = SimpleNamespace(window_size=16, tokenizer="base", model_type="mlp")

            captured = {}

            def _fake_generate_plasmid_sequence(**kwargs):
                captured["ast_conditioning"] = kwargs.get("ast_conditioning")
                return "ACGT"

            manifest_kwargs = {}

            def _capture_manifest(*_args, **kwargs):
                manifest_kwargs.update(kwargs)
                return os.path.join(td, "manifest.json")

            with patch("perceptrome.jobs.engine.load_full_config", return_value={}), \
                patch("perceptrome.jobs.engine.extract_configs", return_value=(None, train_cfg, io_cfg)), \
                patch("perceptrome.jobs.engine.ensure_dirs"), \
                patch("perceptrome.jobs.engine.setup_logging"), \
                patch("perceptrome.jobs.engine.ensure_run_layout", return_value=SimpleNamespace(run_id="r1")), \
                patch("perceptrome.jobs.engine.path_in_run", side_effect=lambda _layout, _kind, name: os.path.join(td, name)), \
                patch("perceptrome.jobs.engine.generate_plasmid_sequence", side_effect=_fake_generate_plasmid_sequence), \
                patch.object(JobEngine, "_write_run_manifest", side_effect=_capture_manifest), \
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
                            "ast_template_artifact": ast_path,
                            "ast_template_mode": "reject",
                            "ast_template_span_tolerance": 10,
                            "ast_template_min_score": 0.7,
                            "ast_template_max_mismatches": 1,
                            "ast_template_include_semantic_edges": True,
                        },
                    )
                )

            self.assertEqual(out["length"], 4)
            self.assertIsNotNone(captured["ast_conditioning"])
            self.assertEqual(captured["ast_conditioning"].artifact_path, ast_path)
            self.assertEqual(captured["ast_conditioning"].node_type_prompts, ("gene",))
            self.assertTrue(manifest_kwargs.get("run_parents"))
            self.assertTrue(manifest_kwargs.get("run_children"))
            generated_artifacts = manifest_kwargs.get("artifacts") or []
            generated = generated_artifacts[-1]
            self.assertTrue(generated.get("parents"))
            manifest_entry = ((manifest_kwargs.get("generated_sequences") or {}).get("entries") or [])[0]
            self.assertTrue(manifest_entry.get("ast_template_validation", {}).get("enabled"))
            self.assertEqual(manifest_entry.get("ast_template_validation", {}).get("mode"), "reject")


if __name__ == "__main__":
    unittest.main()

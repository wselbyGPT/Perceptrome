import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from perceptrome.bio_ast_template import compare_bio_ast_to_template, derive_bio_ast_template, load_bio_ast_template
from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.generate import AstTemplateValidationConfig, generate_plasmid_sequence


class BioASTTemplateComparisonTests(unittest.TestCase):
    def test_load_template_from_canonical_ast_and_compare_reports_mismatches(self):
        builder = BioASTBuilder()
        seq = "ATG" * 60
        built = builder.build(sequence=seq, accession="TMP1", top_level_type="plasmid")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "canonical_ast.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(built.ast.to_dict(), f)
            template = load_bio_ast_template(path, span_tolerance=0, include_semantic_edges=True)
        report_ok = compare_bio_ast_to_template(built.ast, template)
        self.assertTrue(report_ok["topology_match"])
        self.assertEqual(report_ok["summary"]["mismatch_count"], 0)

        shifted = builder.build(sequence=seq + "ATG", accession="TMP2", top_level_type="plasmid")
        report_bad = compare_bio_ast_to_template(shifted.ast, template)
        self.assertGreaterEqual(report_bad["summary"]["mismatch_count"], 1)
        self.assertTrue(report_bad["node_failures"] or report_bad["missing_nodes"] or report_bad["extra_nodes"])


class GeneratePlasmidTemplatePolicyTests(unittest.TestCase):
    def _fake_torch(self):
        class _NoGrad:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return SimpleNamespace(no_grad=lambda: _NoGrad(), randn=lambda *a, **k: 0.0)


    def _fake_np(self):
        class _NP:
            float64 = float
            def ones(self, shape, dtype=None):
                return [1.0] * int(shape[0])
            def exp(self, x):
                return x
            def log(self, x):
                return x
            def clip(self, a, amin, amax):
                return a
            def mean(self, values):
                values = list(values)
                return sum(float(v) for v in values) / float(len(values) or 1)
        return _NP()

    def _fake_model(self):
        class _Vec(list):
            @property
            def shape(self):
                return (len(self),)
            def __neg__(self):
                return _Vec([-float(v) for v in self])
            def __radd__(self, other):
                return _Vec([float(other) + float(v) for v in self])
            def __rtruediv__(self, other):
                return _Vec([float(other) / float(v) for v in self])
            def __mul__(self, other):
                if isinstance(other, (list, tuple)):
                    return _Vec([float(a) * float(b) for a, b in zip(self, other)])
                return _Vec([float(other) * float(v) for v in self])
        class _Matrix(list):
            def __getitem__(self, idx):
                row = super().__getitem__(idx)
                return _Vec(row)
        class _Tensor:
            def __init__(self):
                self.arr = _Matrix([[0.0, 0.0, 0.0, 0.0] for _ in range(4)])
            def view(self, *shape):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.arr
        class _Model:
            def eval(self):
                return None
            def decode(self, _z):
                return _Tensor()
        return _Model()

    def test_generate_plasmid_rescores_and_writes_template_report(self):
        with tempfile.TemporaryDirectory() as td:
            template_path = os.path.join(td, "template.json")
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump({"source_kind": "bio_ast_template", "nodes": []}, f)
            output_path = os.path.join(td, "out.fasta")
            summary_path = os.path.join(td, "summary.json")
            train_cfg = SimpleNamespace(hidden_dim=4, model_type="mlp", transformer_d_model=4, transformer_nhead=1, transformer_layers=1, transformer_dropout=0.0, learning_rate=0.001, beta_kl=0.0, min_orf_aa=1)
            io_cfg = SimpleNamespace(cache_fasta_dir=td, cache_genbank_dir=td, cache_encoded_dir=td, model_dir=td, checkpoints_dir=td, logs_dir=td, state_file=os.path.join(td, "state.json"))
            fake_model = self._fake_model()
            scorecards = {
                "ACGT": {"metrics": {"gc_fraction": 0.5, "gc_deviation": 0.0, "max_homopolymer": 1, "homopolymer_penalty": 0.0, "score": 0.1, "repeat_density": 0.0, "orf_count": 1}, "scorecard_version": "v1", "risk_flags": [], "summary": {}},
                "TGCA": {"metrics": {"gc_fraction": 0.5, "gc_deviation": 0.0, "max_homopolymer": 1, "homopolymer_penalty": 0.0, "score": 0.9, "repeat_density": 0.0, "orf_count": 1}, "scorecard_version": "v1", "risk_flags": [], "summary": {}},
            }
            reports = {
                "ACGT": {"accepted": True, "score": 1.0, "mismatch_count": 0, "report": {"summary": {"score": 1.0, "mismatch_count": 0}}},
                "TGCA": {"accepted": False, "score": 0.0, "mismatch_count": 3, "report": {"summary": {"score": 0.0, "mismatch_count": 3}}},
            }
            with patch("perceptrome.generate.torch", self._fake_torch()), \
                patch("perceptrome.generate.np", self._fake_np()), \
                patch("perceptrome.generate.get_device", return_value="cpu"), \
                patch("perceptrome.generate.tokenizer_meta", return_value=(4, 4)), \
                patch("perceptrome.generate.load_or_init_model", return_value=(fake_model, None, 0, None)), \
                patch("perceptrome.generate.ensure_run_layout", return_value=SimpleNamespace(artifacts_dir=td)), \
                patch("perceptrome.generate.collect_and_write_provenance"), \
                patch("perceptrome.generate.path_in_run", side_effect=lambda _layout, _kind, name: os.path.join(td, name)), \
                patch("perceptrome.generate.resolve_seed", return_value={"value": 7, "source": "test"}), \
                patch("perceptrome.generate.set_global_seeds"), \
                patch("perceptrome.generate.update_run_manifest"), \
                patch("perceptrome.generate._sample_from_logits", side_effect=[0, 1, 2, 3, 3, 2, 1, 0]), \
                patch("perceptrome.generate.build_plasmid_scorecard", side_effect=lambda seq, ctx: scorecards[seq]), \
                patch("perceptrome.generate._rebuild_bio_ast_roundtrip", side_effect=lambda sequence, template_cfg: reports[sequence]):
                seq = generate_plasmid_sequence(
                    train_cfg=train_cfg,
                    io_cfg=io_cfg,
                    length_bp=4,
                    num_windows=1,
                    window_size_bp=4,
                    seed=7,
                    latent_scale=1.0,
                    temperature=1.0,
                    gc_bias=1.0,
                    name="tmpl",
                    output_path=output_path,
                    tokenizer="base",
                    num_candidates=2,
                    top_k=2,
                    summary_path=summary_path,
                    ast_template_validation=AstTemplateValidationConfig(artifact_path=template_path, mode="rescore", min_score=0.5, max_mismatches=0),
                )
            self.assertEqual(seq, "ACGT")
            summary = json.load(open(summary_path, "r", encoding="utf-8"))
            self.assertTrue(summary["ast_template_validation"]["enabled"])
            self.assertEqual(summary["winner"]["template_validation"]["mismatch_count"], 0)
            self.assertEqual(summary["top_candidates"][0]["candidate"], 0)

    def test_generate_plasmid_reject_mode_raises_when_every_candidate_fails(self):
        with tempfile.TemporaryDirectory() as td:
            template_path = os.path.join(td, "template.json")
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump({"source_kind": "bio_ast_template", "nodes": []}, f)
            train_cfg = SimpleNamespace(hidden_dim=4, model_type="mlp", transformer_d_model=4, transformer_nhead=1, transformer_layers=1, transformer_dropout=0.0, learning_rate=0.001, beta_kl=0.0, min_orf_aa=1)
            io_cfg = SimpleNamespace(cache_fasta_dir=td, cache_genbank_dir=td, cache_encoded_dir=td, model_dir=td, checkpoints_dir=td, logs_dir=td, state_file=os.path.join(td, "state.json"))
            fake_model = self._fake_model()
            scorecard = {"metrics": {"gc_fraction": 0.5, "gc_deviation": 0.0, "max_homopolymer": 1, "homopolymer_penalty": 0.0, "score": 0.1, "repeat_density": 0.0, "orf_count": 1}, "scorecard_version": "v1", "risk_flags": [], "summary": {}}
            with patch("perceptrome.generate.torch", self._fake_torch()), \
                patch("perceptrome.generate.np", self._fake_np()), \
                patch("perceptrome.generate.get_device", return_value="cpu"), \
                patch("perceptrome.generate.tokenizer_meta", return_value=(4, 4)), \
                patch("perceptrome.generate.load_or_init_model", return_value=(fake_model, None, 0, None)), \
                patch("perceptrome.generate.ensure_run_layout", return_value=SimpleNamespace(artifacts_dir=td)), \
                patch("perceptrome.generate.collect_and_write_provenance"), \
                patch("perceptrome.generate.path_in_run", side_effect=lambda _layout, _kind, name: os.path.join(td, name)), \
                patch("perceptrome.generate.resolve_seed", return_value={"value": 7, "source": "test"}), \
                patch("perceptrome.generate.set_global_seeds"), \
                patch("perceptrome.generate.update_run_manifest"), \
                patch("perceptrome.generate._sample_from_logits", side_effect=[0, 1, 2, 3]), \
                patch("perceptrome.generate.build_plasmid_scorecard", return_value=scorecard), \
                patch("perceptrome.generate._rebuild_bio_ast_roundtrip", return_value={"accepted": False, "score": 0.0, "mismatch_count": 2, "report": {"summary": {"score": 0.0, "mismatch_count": 2}}}):
                with self.assertRaisesRegex(RuntimeError, "No generated plasmid candidates satisfied"):
                    generate_plasmid_sequence(
                        train_cfg=train_cfg,
                        io_cfg=io_cfg,
                        length_bp=4,
                        num_windows=1,
                        window_size_bp=4,
                        seed=7,
                        latent_scale=1.0,
                        temperature=1.0,
                        gc_bias=1.0,
                        name="tmpl",
                        output_path=os.path.join(td, "out.fasta"),
                        tokenizer="base",
                        num_candidates=1,
                        ast_template_validation=AstTemplateValidationConfig(artifact_path=template_path, mode="reject", min_score=1.0, max_mismatches=0),
                    )


if __name__ == "__main__":
    unittest.main()

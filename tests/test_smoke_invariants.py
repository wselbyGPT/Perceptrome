import sys
import types
import unittest
import tempfile
from types import SimpleNamespace


# Local test environment may not have numpy; provide a tiny stub so modules import.
if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = object
    np_stub.float32 = float
    np_stub.int64 = int

    def _array(v, dtype=None):
        return list(v)

    np_stub.array = _array
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

from perceptrome.cli.common import _resolve_proteome_params, _validate_tok_params, _get_grounded
from perceptrome.config import deep_update, extract_configs
from perceptrome.encoding_main import tokenizer_meta
from perceptrome.generate import (
    _gc_fraction,
    _max_homopolymer_run,
    _plasmid_candidate_score,
    _protein_candidate_score,
    _write_top_k_fasta,
)
from perceptrome.scope.stream import ScopeStreamContext


class TokenizerInvariantTests(unittest.TestCase):
    def test_tokenizer_meta_base(self):
        seq_len, vocab = tokenizer_meta("base", 12)
        self.assertEqual(seq_len, 12)
        self.assertEqual(vocab, 4)

    def test_tokenizer_meta_codon(self):
        seq_len, vocab = tokenizer_meta("codon", 12)
        self.assertEqual(seq_len, 4)
        self.assertEqual(vocab, 65)

    def test_tokenizer_meta_aa(self):
        seq_len, vocab = tokenizer_meta("aa", 31)
        self.assertEqual(seq_len, 31)
        self.assertEqual(vocab, 21)

    def test_validate_tok_params_codon_constraints(self):
        with self.assertRaises(ValueError):
            _validate_tok_params("codon", 10, 3, 0)
        with self.assertRaises(ValueError):
            _validate_tok_params("codon", 12, 5, 0)
        with self.assertRaises(ValueError):
            _validate_tok_params("codon", 12, 3, 7)

    def test_validate_tok_params_valid_cases(self):
        _validate_tok_params("base", 10, 5, 0)
        _validate_tok_params("aa", 10, 5, 0)
        _validate_tok_params("codon", 12, 3, 2)


class ProteomeResolutionTests(unittest.TestCase):
    def _train_cfg(self):
        return SimpleNamespace(
            protein_len_min=100,
            protein_len_max=800,
            translation_only=True,
            max_windows_per_protein=4,
            aa_mask_prob=0.10,
            aa_span_mask_prob=0.05,
            aa_span_mask_len=12,
            curriculum_enabled=True,
            curriculum_steps=[0],
            curriculum_phases=[
                {
                    "protein_len_min": 60,
                    "protein_len_max": 400,
                    "translation_only": False,
                    "max_windows_per_protein": 2,
                    "aa_mask_prob": 0.2,
                    "aa_span_mask_prob": 0.3,
                    "aa_span_mask_len": 9,
                }
            ],
        )

    def test_resolve_precedence_cli_over_config_over_curriculum(self):
        cfg = self._train_cfg()
        args = SimpleNamespace(
            no_curriculum=False,
            protein_len_min=120,
            protein_len_max=None,
            max_windows_per_protein=None,
            translation_only=None,
            allow_translated=None,
            mask_prob=0.44,
            span_mask_prob=None,
            span_mask_len=None,
        )
        out = _resolve_proteome_params(args, cfg, {"total_steps": 5000}, tok="aa", src="genbank")

        # CLI wins.
        self.assertEqual(out["protein_len_min"], 120)
        self.assertAlmostEqual(out["mask_prob"], 0.44)
        # Config beats curriculum.
        self.assertEqual(out["protein_len_max"], 800)
        self.assertEqual(out["max_windows_per_protein"], 4)
        self.assertTrue(out["translation_only"])
        self.assertAlmostEqual(out["span_mask_prob"], 0.05)
        self.assertEqual(out["span_mask_len"], 12)

    def test_resolve_uses_curriculum_when_config_missing(self):
        cfg = self._train_cfg()
        cfg.protein_len_min = None
        cfg.aa_mask_prob = None
        args = SimpleNamespace(
            no_curriculum=False,
            protein_len_min=None,
            protein_len_max=None,
            max_windows_per_protein=None,
            translation_only=None,
            allow_translated=None,
            mask_prob=None,
            span_mask_prob=None,
            span_mask_len=None,
        )
        out = _resolve_proteome_params(args, cfg, {"total_steps": 0}, tok="aa", src="genbank")
        self.assertEqual(out["protein_len_min"], 60)
        self.assertAlmostEqual(out["mask_prob"], 0.2)
        self.assertEqual(out["curriculum_tag"], "cur0")

    def test_aa_profile_balanced_sets_defaults(self):
        cfg = self._train_cfg()
        args = SimpleNamespace(
            aa_profile="balanced",
            no_curriculum=False,
            protein_len_min=None,
            protein_len_max=None,
            max_windows_per_protein=None,
            translation_only=None,
            allow_translated=None,
            mask_prob=None,
            span_mask_prob=None,
            span_mask_len=None,
        )
        out = _resolve_proteome_params(args, cfg, {"total_steps": 0}, tok="aa", src="genbank")
        self.assertEqual(out["protein_len_min"], 60)
        self.assertEqual(out["protein_len_max"], 800)
        self.assertTrue(out["translation_only"])
        self.assertEqual(out["max_windows_per_protein"], 4)

    def test_aa_profile_conservative_no_curriculum(self):
        cfg = self._train_cfg()
        cfg.protein_len_min = None
        args = SimpleNamespace(
            aa_profile="conservative",
            no_curriculum=False,
            protein_len_min=None,
            protein_len_max=None,
            max_windows_per_protein=None,
            translation_only=None,
            allow_translated=None,
            mask_prob=None,
            span_mask_prob=None,
            span_mask_len=None,
            strict_cds=False,
            require_translation=False,
            x_free=False,
            require_start_m=False,
            reject_partial_cds=False,
            max_protein_aa=None,
        )
        out = _resolve_proteome_params(args, cfg, {"total_steps": 99999}, tok="aa", src="genbank")
        self.assertEqual(out["curriculum_tag"], None)
        self.assertEqual(out["protein_len_min"], 100)
        grounded = _get_grounded(args, cfg, tok="aa", src="genbank")
        self.assertTrue(grounded["strict_cds"])
        self.assertTrue(grounded["require_translation"])


class ConfigMergeTests(unittest.TestCase):
    def test_deep_update_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 5}
        updates = {"a": {"y": 9, "z": 10}}
        out = deep_update(base, updates)
        self.assertEqual(out["a"]["x"], 1)
        self.assertEqual(out["a"]["y"], 9)
        self.assertEqual(out["a"]["z"], 10)

    def test_extract_configs_reads_overrides(self):
        cfg = {
            "ncbi": {"email": "a@b.com"},
            "training": {"tokenizer": "aa", "window_size": 256, "stride": 64, "model_type": "ssm"},
            "io": {"state_file": "state/custom.json"},
        }
        _, train_cfg, io_cfg = extract_configs(cfg)
        self.assertEqual(train_cfg.tokenizer, "aa")
        self.assertEqual(train_cfg.window_size, 256)
        self.assertEqual(train_cfg.stride, 64)
        self.assertEqual(train_cfg.model_type, "ssm")
        self.assertEqual(io_cfg.state_file, "state/custom.json")


class GenerateHeuristicTests(unittest.TestCase):
    def test_gc_fraction(self):
        self.assertAlmostEqual(_gc_fraction("GGCCAA"), 4.0 / 6.0)
        self.assertEqual(_gc_fraction(""), 0.0)

    def test_max_homopolymer_run(self):
        self.assertEqual(_max_homopolymer_run("AAABBBCCCCA"), 4)
        self.assertEqual(_max_homopolymer_run("ATGC"), 1)
        self.assertEqual(_max_homopolymer_run(""), 0)


class ScopeStreamArchitectureAgnosticTests(unittest.TestCase):
    def test_scope_context_accepts_non_plasmid_model(self):
        dummy_model = object()
        ctx = ScopeStreamContext(
            model=dummy_model,
            optimizer=object(),
            device=object(),
            dataloader=object(),
            dataloader_iter=iter(()),
            global_step=0,
            last_total=0.0,
            steps_target=1,
            steps_done=0,
            beta_kl=0.0,
            kl_warmup_steps=0,
            max_grad_norm=0.0,
            loss_type="mse",
            seq_len=8,
            vocab_size=4,
        )
        self.assertIs(ctx.model, dummy_model)


class GenerationTopKOutputTests(unittest.TestCase):
    def test_write_top_k_fasta_writes_ranked_multifasta(self):
        ranked = [
            {"candidate": 3, "sequence": "ACGT", "score": 1.5},
            {"candidate": 1, "sequence": "TTAA", "score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = f"{td}/topk.fasta"
            _write_top_k_fasta(out, "demo", ranked, top_k=2)
            txt = open(out, "r", encoding="utf-8").read()
        self.assertIn(">demo|rank=1|candidate=3|score=1.500000", txt)
        self.assertIn("ACGT", txt)
        self.assertIn(">demo|rank=2|candidate=1|score=0.500000", txt)
        self.assertIn("TTAA", txt)


if __name__ == "__main__":
    unittest.main()


class CandidateScoringTests(unittest.TestCase):
    def test_plasmid_score_includes_optional_reconstruction_penalty(self):
        score_no_recon, run_pen = _plasmid_candidate_score(
            gc_dev=0.05,
            homopolymer_run=8,
            max_homopolymer=10,
            recon=None,
            recon_weight=0.1,
        )
        self.assertAlmostEqual(run_pen, 0.0)
        self.assertAlmostEqual(score_no_recon, -0.05)

        score_recon, _ = _plasmid_candidate_score(
            gc_dev=0.05,
            homopolymer_run=8,
            max_homopolymer=10,
            recon=2.0,
            recon_weight=0.2,
        )
        self.assertAlmostEqual(score_recon, -0.45)

    def test_protein_score_tracks_penalties(self):
        metrics = _protein_candidate_score(
            seq="MXX**",
            max_homopolymer=1,
            max_x_frac=0.2,
            max_internal_stops=0,
            recon=1.0,
            recon_weight=0.1,
            allowed=set("ACDEFGHIKLMNPQRSTVWY*X"),
        )
        self.assertEqual(metrics["max_homopolymer"], 2.0)
        self.assertEqual(metrics["stop_count"], 2.0)
        self.assertAlmostEqual(metrics["score"], -3.3, places=6)

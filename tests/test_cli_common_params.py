import argparse
import unittest
from types import SimpleNamespace

try:
    from perceptrome.cli.common import _resolve_proteome_params, _validate_tok_params
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency/environment gate
    _resolve_proteome_params = _validate_tok_params = None
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"Perceptrome deps unavailable: {IMPORT_ERROR}")
class ValidateTokParamsTests(unittest.TestCase):
    def test_codon_requires_divisible_window_and_stride(self):
        with self.assertRaisesRegex(ValueError, "window_size divisible by 3"):
            _validate_tok_params("codon", window_size=10, stride=3, frame=0)
        with self.assertRaisesRegex(ValueError, "stride divisible by 3"):
            _validate_tok_params("codon", window_size=12, stride=5, frame=0)

    def test_codon_requires_valid_frame_offset(self):
        with self.assertRaisesRegex(ValueError, "frame-offset"):
            _validate_tok_params("codon", window_size=12, stride=3, frame=4)

    def test_aa_and_base_require_positive_dimensions(self):
        with self.assertRaisesRegex(ValueError, "tokenizer aa requires positive"):
            _validate_tok_params("aa", window_size=0, stride=3, frame=0)
        with self.assertRaisesRegex(ValueError, "window_size/stride must be positive"):
            _validate_tok_params("base", window_size=10, stride=0, frame=0)

    def test_valid_values_do_not_raise(self):
        _validate_tok_params("codon", window_size=12, stride=3, frame=2)
        _validate_tok_params("aa", window_size=12, stride=2, frame=0)
        _validate_tok_params("base", window_size=12, stride=2, frame=0)


@unittest.skipIf(IMPORT_ERROR is not None, f"Perceptrome deps unavailable: {IMPORT_ERROR}")
class ResolveProteomeParamsTests(unittest.TestCase):
    def _args(self, **kwargs):
        defaults = {
            "no_curriculum": False,
            "protein_len_min": None,
            "protein_len_max": None,
            "max_windows_per_protein": None,
            "translation_only": None,
            "allow_translated": None,
            "mask_prob": None,
            "span_mask_prob": None,
            "span_mask_len": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_precedence_cli_over_config_over_curriculum(self):
        train_cfg = SimpleNamespace(
            protein_len_min=80,
            protein_len_max=900,
            translation_only=False,
            max_windows_per_protein=5,
            aa_mask_prob=0.12,
            aa_span_mask_prob=0.07,
            aa_span_mask_len=14,
            curriculum_enabled=True,
            curriculum_steps=[0],
            curriculum_phases=[
                {
                    "protein_len_min": 40,
                    "protein_len_max": 300,
                    "translation_only": True,
                    "max_windows_per_protein": 2,
                    "mask_prob": 0.5,
                    "span_mask_prob": 0.3,
                    "span_mask_len": 21,
                }
            ],
        )
        args = self._args(
            protein_len_min=120,
            translation_only=True,
            mask_prob=0.2,
        )

        pol = _resolve_proteome_params(args, train_cfg, {"total_steps": 10}, tok="aa", src="genbank")

        self.assertEqual(pol["protein_len_min"], 120)  # CLI
        self.assertEqual(pol["protein_len_max"], 900)  # config over curriculum
        self.assertEqual(pol["translation_only"], True)  # CLI
        self.assertEqual(pol["max_windows_per_protein"], 5)  # config over curriculum
        self.assertEqual(pol["mask_prob"], 0.2)  # CLI
        self.assertEqual(pol["span_mask_prob"], 0.07)  # config over curriculum
        self.assertEqual(pol["span_mask_len"], 14)  # config over curriculum
        self.assertEqual(pol["curriculum_tag"], "cur0")

    def test_allow_translated_disables_translation_only(self):
        train_cfg = SimpleNamespace(
            protein_len_min=None,
            protein_len_max=None,
            translation_only=True,
            max_windows_per_protein=3,
            aa_mask_prob=0.1,
            aa_span_mask_prob=0.02,
            aa_span_mask_len=6,
            curriculum_enabled=False,
            curriculum_steps=[],
            curriculum_phases=[],
        )
        args = self._args(allow_translated=True)
        pol = _resolve_proteome_params(args, train_cfg, {}, tok="aa", src="genbank")
        self.assertFalse(pol["translation_only"])


if __name__ == "__main__":
    unittest.main()

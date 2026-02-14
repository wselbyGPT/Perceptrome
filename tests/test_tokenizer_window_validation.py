from __future__ import annotations

import argparse

import pytest


def test_pick_window_stride_prefers_aa_defaults(commands_module) -> None:
    args = argparse.Namespace(window_size=None, stride=None)
    cfg = argparse.Namespace(window_size=512, stride=256, protein_window_aa=128, protein_stride_aa=64)

    window, stride = commands_module._pick_window_stride(args, cfg, "aa")

    assert (window, stride) == (128, 64)


def test_validate_tok_params_rejects_bad_codon_values(commands_module) -> None:
    with pytest.raises(ValueError, match="window_size divisible by 3"):
        commands_module._validate_tok_params("codon", 10, 6, 0)

    with pytest.raises(ValueError, match="frame_offset"):
        commands_module._validate_tok_params("codon", 12, 6, 4)


def test_validate_tok_params_rejects_non_positive_values(commands_module) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        commands_module._validate_tok_params("base", 0, 1, 0)

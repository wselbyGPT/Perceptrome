import numpy as np

from perceptrome.cli_main import build_parser
from perceptrome.pretrain.transforms import MaskSMETransform, MaskedLanguageModelTransform


def test_cli_parser_accepts_pretrain_command():
    parser = build_parser()
    args = parser.parse_args(["pretrain", "--dataset", "mock.npz", "--vocab-size", "32"])
    assert args.command == "pretrain"
    assert args.dataset == "mock.npz"
    assert args.vocab_size == 32


def test_mlm_transform_creates_labels_and_masked_ids():
    tr = MaskedLanguageModelTransform(mask_prob=1.0, mask_token_id=99, vocab_size=128, seed=7)
    out = tr(np.array([1, 2, 3, 4], dtype=np.int64))
    assert out["input_ids"].shape == (4,)
    assert out["mlm_labels"].tolist() == [1, 2, 3, 4]
    assert set(out["input_ids"].tolist()) <= set(range(128)) | {99}


def test_sme_transform_masks_values_and_keeps_labels():
    tr = MaskSMETransform(mask_prob_s=1.0, mask_prob_m=1.0, mask_prob_e=1.0, seed=1)
    sme_s = np.array([0, 1, 2], dtype=np.int64)
    sme_m = np.array([4, 5, 6], dtype=np.int64)
    sme_e = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = tr(sme_s, sme_m, sme_e)

    assert out["sme_s_labels"].tolist() == [0, 1, 2]
    assert out["sme_m_labels"].tolist() == [4, 5, 6]
    assert out["sme_e_labels"].tolist() == [0.1, 0.2, 0.3]
    assert out["sme_s"].tolist() == [-1, -1, -1]
    assert out["sme_m"].tolist() == [-1, -1, -1]
    assert out["sme_e"].tolist() == [0.0, 0.0, 0.0]

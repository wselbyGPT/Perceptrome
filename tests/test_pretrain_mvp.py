import tempfile

import numpy as np

from perceptrome.cli_main import build_parser
from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from perceptrome.pretrain.datasets import DatasetSpec, NPZPretrainDataset, pretrain_collate
from perceptrome.pretrain.models import BackboneConfig, SequenceBackbone
from perceptrome.pretrain.transforms import MaskSMETransform, MaskedLanguageModelTransform
from tests.fixtures.bio_ast_regression_fixtures import SYNTHETIC_FASTA_SEQUENCE


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


def test_pretrain_dataset_and_backbone_support_optional_bio_ast_payload():
    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    built = BioASTBuilder().build(sequence=SYNTHETIC_FASTA_SEQUENCE, accession="PRE1")
    canonical = built.ast.to_dict()

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/dataset.npz"
        np.savez(
            path,
            input_ids=np.array([[1, 2, 3, 0], [2, 3, 4, 0]], dtype=np.int64),
            bio_ast=np.array([canonical, None], dtype=object),
            sample_ids=np.array(["a", "b"], dtype=object),
        )
        ds = NPZPretrainDataset(DatasetSpec(path=path))
        batch = pretrain_collate([ds[0], ds[1]])
        assert "ast_node_type_ids" in batch
        assert tuple(batch["ast_node_type_ids"].shape[:2]) == (2, len(built.ast.nodes))
        assert tuple(batch["ast_span_coords"].shape) == (2, len(built.ast.nodes), 2)

        model = SequenceBackbone(BackboneConfig(vocab_size=16, hidden_size=12))
        fused = model.encode(batch)
        seq_only = model.encode({"input_ids": batch["input_ids"], "input_ids_mask": batch["input_ids_mask"]})

        assert tuple(fused.token_embeddings.shape) == (2, 4, 12)
        assert tuple(fused.pooled_embedding.shape) == (2, 12)
        assert fused.aux["ast_used"] is True
        assert seq_only.aux["ast_used"] is False

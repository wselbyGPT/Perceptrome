import json
import math
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Torch stub — keep tests runnable without a GPU install
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _nn = types.ModuleType("torch.nn")
    _nn.Module = object
    _optim = types.ModuleType("torch.optim")
    _fn = types.ModuleType("torch.nn.functional")
    _utils = types.ModuleType("torch.utils")
    _data = types.ModuleType("torch.utils.data")
    _data.DataLoader = object
    _data.TensorDataset = object
    _torch.nn = _nn
    _torch.optim = _optim
    sys.modules.update({
        "torch": _torch, "torch.nn": _nn, "torch.optim": _optim,
        "torch.nn.functional": _fn, "torch.utils": _utils,
        "torch.utils.data": _data,
    })

if "requests" not in sys.modules:
    _req = types.ModuleType("requests")
    _req.get = lambda *a, **k: None
    _req.Session = object
    sys.modules["requests"] = _req

import numpy as np

from perceptrome.latent_design_loop import (
    CandidateRecord,
    attach_fold_summary,
    combine_candidate_record,
    rank_and_select_for_folding,
    write_candidate_outputs,
    write_leaderboard_and_manifest,
)


def _scorecard_payload(score: float) -> dict:
    return {
        "scorecard_version": "v2",
        "sequence_kind": "protein",
        "metadata": {"sequence_id": None, "sequence_type": "protein", "sequence_length": 256},
        "metrics": {"score": float(score), "max_homopolymer": 4, "x_fraction": 0.0,
                    "invalid_fraction": 0.0, "stop_count": 0,
                    "hydrophobic_fraction": 0.4, "charge_balance": 0.0, "aromaticity": 0.08,
                    "instability_proxy": 35.0, "low_complexity_fraction": 0.0,
                    "low_complexity_longest": 0},
        "reference_neighbors": [],
        "risk_flags": [],
        "summary": {"title": "Protein scorecard", "highlights": []},
        "details": {},
    }


def _elbo_payload(elbo: float, status: str = "ok") -> dict:
    return {
        "accession": "cand_0001",
        "status": status,
        "elbo": float(elbo),
        "recon_loss": -float(elbo) - 0.3,
        "kl": 0.3,
        "n_windows": 2,
        "sequence_length": 512,
        "error": None if status == "ok" else "stub_error",
    }


def _fold_payload(mean_plddt: float = 78.4, ptm: float = 0.71, status: str = "ok") -> dict:
    return {
        "protein_id": "cand_0001",
        "source_input_path": "/tmp/cand_0001.fasta",
        "aa_length": 256,
        "fold_engine": "alphafold3",
        "engine_status": status,
        "rank_1_structure_path": "/tmp/runs/x/artifacts/fold/cand_0001/model.cif",
        "mean_plddt": float(mean_plddt),
        "min_plddt": 30.0,
        "max_plddt": 95.0,
        "ptm": float(ptm),
        "rank_index": 1,
        "discovered_artifact_paths": {},
        "warnings": [],
        "errors": [],
        "started_at": "2026-04-26T12:00:00Z",
        "completed_at": "2026-04-26T12:18:42Z",
    }


def _make_candidate(
    *,
    candidate_id: str = "cand_0001",
    seed: str = "P12345",
    cluster_id: int = 0,
    strategy: str = "random",
    sequence: str = "MKTAYIAK",
    sc_score: float = -0.21,
    elbo: float = -1.72,
    elbo_status: str = "ok",
    fold_summary: dict = None,
    error: str = None,
    alpha: float = 1.0,
    z_dim: int = 4,
) -> CandidateRecord:
    z = np.zeros(z_dim, dtype=np.float32)
    return combine_candidate_record(
        candidate_id=candidate_id,
        seed_accession=seed,
        cluster_id=cluster_id,
        strategy=strategy,
        z_vec=z,
        sequence=sequence,
        scorecard=_scorecard_payload(sc_score),
        elbo_record=_elbo_payload(elbo, elbo_status),
        alpha=alpha,
        fold_summary=fold_summary,
        error=error,
    )


# ---------------------------------------------------------------------------
# CombinedScoreFormulaTests
# ---------------------------------------------------------------------------

class CombinedScoreFormulaTests(unittest.TestCase):
    def test_score_combines_additively(self):
        c = _make_candidate(sc_score=-0.5, elbo=-1.0, alpha=1.0)
        self.assertAlmostEqual(c.combined_score, -1.5, places=6)

    def test_alpha_zero_drops_elbo(self):
        c = _make_candidate(sc_score=-0.4, elbo=-2.0, alpha=0.0)
        self.assertAlmostEqual(c.combined_score, -0.4, places=6)

    def test_errored_candidate_gets_neg_inf(self):
        c = _make_candidate(error="empty_sequence_after_decode")
        self.assertEqual(c.combined_score, -math.inf)

    def test_elbo_status_error_drops_elbo_term(self):
        c = _make_candidate(sc_score=-0.4, elbo=0.0, elbo_status="error", alpha=1.0)
        self.assertAlmostEqual(c.combined_score, -0.4, places=6)


# ---------------------------------------------------------------------------
# CandidateRecordSchemaTests
# ---------------------------------------------------------------------------

class CandidateRecordSchemaTests(unittest.TestCase):
    def test_required_keys_present(self):
        cand = _make_candidate()
        d = cand.to_dict()
        for key in (
            "candidate_id", "seed_accession", "cluster_id", "strategy",
            "latent_vector", "sequence", "scorecard", "elbo_record",
            "combined_score", "fold_summary", "error",
        ):
            self.assertIn(key, d)

    def test_fold_summary_optional(self):
        cand = _make_candidate(fold_summary=None)
        self.assertIsNone(cand.fold_summary)
        cand2 = attach_fold_summary(cand, _fold_payload())
        self.assertIsNotNone(cand2.fold_summary)
        self.assertEqual(cand2.fold_summary["fold_engine"], "alphafold3")

    def test_json_serializable(self):
        cand = _make_candidate(fold_summary=_fold_payload())
        s = json.dumps(cand.to_dict())
        roundtripped = json.loads(s)
        self.assertEqual(roundtripped["candidate_id"], cand.candidate_id)
        self.assertEqual(roundtripped["fold_summary"]["mean_plddt"], 78.4)

    def test_latent_vector_serializes_as_floats(self):
        z = np.array([1.5, -0.25, 0.0, 3.14], dtype=np.float32)
        cand = combine_candidate_record(
            candidate_id="cand_0007", seed_accession="P12345", cluster_id=2,
            strategy="walk", z_vec=z, sequence="MK",
            scorecard=_scorecard_payload(-0.1), elbo_record=_elbo_payload(-0.5),
            alpha=1.0,
        )
        self.assertEqual(len(cand.latent_vector), 4)
        self.assertAlmostEqual(cand.latent_vector[0], 1.5, places=4)
        self.assertAlmostEqual(cand.latent_vector[3], 3.14, places=4)


# ---------------------------------------------------------------------------
# RankAndSelectTests
# ---------------------------------------------------------------------------

class RankAndSelectTests(unittest.TestCase):
    def test_top_k_truncates(self):
        cands = [
            _make_candidate(candidate_id=f"cand_{i:04d}", sc_score=-float(i), elbo=0.0)
            for i in range(5)
        ]
        ids = rank_and_select_for_folding(cands, fold_top_k=2)
        self.assertEqual(ids, ["cand_0000", "cand_0001"])

    def test_stable_sort_descending(self):
        cands = [
            _make_candidate(candidate_id="cand_0001", sc_score=-0.5, elbo=0.0),
            _make_candidate(candidate_id="cand_0002", sc_score=-0.1, elbo=0.0),
            _make_candidate(candidate_id="cand_0003", sc_score=-0.3, elbo=0.0),
        ]
        ids = rank_and_select_for_folding(cands, fold_top_k=3)
        self.assertEqual(ids, ["cand_0002", "cand_0003", "cand_0001"])

    def test_errored_excluded_from_top_k(self):
        cands = [
            _make_candidate(candidate_id="cand_0001", sc_score=-0.1, elbo=0.0),
            _make_candidate(candidate_id="cand_0002", error="oops"),
            _make_candidate(candidate_id="cand_0003", sc_score=-0.3, elbo=0.0),
        ]
        ids = rank_and_select_for_folding(cands, fold_top_k=5)
        self.assertEqual(ids, ["cand_0001", "cand_0003"])

    def test_top_k_larger_than_pool_returns_all_eligible(self):
        cands = [
            _make_candidate(candidate_id=f"cand_{i:04d}", sc_score=-float(i), elbo=0.0)
            for i in range(3)
        ]
        ids = rank_and_select_for_folding(cands, fold_top_k=99)
        self.assertEqual(set(ids), {"cand_0000", "cand_0001", "cand_0002"})

    def test_top_k_zero_returns_empty(self):
        cands = [_make_candidate()]
        self.assertEqual(rank_and_select_for_folding(cands, fold_top_k=0), [])


# ---------------------------------------------------------------------------
# WriteOutputsTests
# ---------------------------------------------------------------------------

class WriteOutputsTests(unittest.TestCase):
    def test_per_candidate_files_written(self):
        cands = [
            _make_candidate(candidate_id="cand_0001", sequence="MKTA"),
            _make_candidate(candidate_id="cand_0002", sequence="GHRP"),
            _make_candidate(candidate_id="cand_0003", sequence="WLYS"),
        ]
        with tempfile.TemporaryDirectory() as td:
            written = write_candidate_outputs(td, cands)
            self.assertEqual(len(written), 3)
            for fasta_path, json_path in written:
                self.assertTrue(os.path.isfile(fasta_path))
                self.assertTrue(os.path.isfile(json_path))
            with open(written[0][1], "r") as f:
                payload = json.load(f)
            self.assertEqual(payload["candidate_id"], "cand_0001")
            with open(written[0][0], "r") as f:
                fasta_text = f.read()
            self.assertIn(">cand_0001", fasta_text)
            self.assertIn("MKTA", fasta_text)

    def test_leaderboard_contains_expected_fields(self):
        cands = [
            _make_candidate(candidate_id="cand_0001", sc_score=-0.5, elbo=-1.0,
                            fold_summary=_fold_payload(mean_plddt=80.0, ptm=0.65)),
            _make_candidate(candidate_id="cand_0002", sc_score=-0.1, elbo=-2.0),
            _make_candidate(candidate_id="cand_0003", sc_score=-0.3, elbo=-0.5),
        ]
        with tempfile.TemporaryDirectory() as td:
            lb_path, _ = write_leaderboard_and_manifest(
                td, cands, run_id="r1", seeds_source="seeds.json", strategy="walk",
                n_per_seed=2, fold_top_k=1, fold_enabled=True, alpha=1.0,
                started_at="2026-04-26T12:00:00Z", completed_at="2026-04-26T12:01:00Z",
                model_meta={"tokenizer": "aa", "window_aa": 256},
            )
            with open(lb_path, "r") as f:
                payload = json.load(f)
        self.assertEqual(payload["run_id"], "r1")
        self.assertEqual(len(payload["ranked"]), 3)
        first = payload["ranked"][0]
        for key in ("candidate_id", "combined_score", "scorecard_score", "elbo",
                    "mean_plddt", "ptm", "folded"):
            self.assertIn(key, first)
        # Ranked descending by combined_score; cand_0003 has -0.3 + (-0.5) = -0.8 (highest).
        self.assertEqual(payload["ranked"][0]["candidate_id"], "cand_0003")

    def test_manifest_counts(self):
        cands = [
            _make_candidate(candidate_id="cand_0001", fold_summary=_fold_payload()),
            _make_candidate(candidate_id="cand_0002", fold_summary=_fold_payload(status="error")),
            _make_candidate(candidate_id="cand_0003", error="empty"),
            _make_candidate(candidate_id="cand_0004", elbo_status="error"),
        ]
        with tempfile.TemporaryDirectory() as td:
            _, mf_path = write_leaderboard_and_manifest(
                td, cands, run_id="r1", seeds_source="seeds.json", strategy="walk",
                n_per_seed=2, fold_top_k=1, fold_enabled=True, alpha=1.0,
                started_at="t0", completed_at="t1",
                model_meta={"tokenizer": "aa"},
                extra_counts={"seeds_skipped_missing_artifact": 2},
            )
            with open(mf_path, "r") as f:
                manifest = json.load(f)
        counts = manifest["counts"]
        self.assertEqual(counts["candidates"], 4)
        self.assertEqual(counts["folded"], 2)
        self.assertEqual(counts["fold_failed"], 1)
        self.assertEqual(counts["errored"], 1)
        self.assertEqual(counts["elbo_failed"], 1)
        self.assertEqual(counts["seeds_skipped_missing_artifact"], 2)

    def test_no_fold_leaves_fold_summary_null_in_leaderboard(self):
        cands = [_make_candidate(fold_summary=None)]
        with tempfile.TemporaryDirectory() as td:
            lb_path, _ = write_leaderboard_and_manifest(
                td, cands, run_id="r1", seeds_source="x", strategy="random",
                n_per_seed=1, fold_top_k=1, fold_enabled=False, alpha=1.0,
                started_at="t0", completed_at="t1", model_meta={},
            )
            with open(lb_path, "r") as f:
                payload = json.load(f)
        row = payload["ranked"][0]
        self.assertFalse(row["folded"])
        self.assertIsNone(row["mean_plddt"])
        self.assertIsNone(row["ptm"])


# ---------------------------------------------------------------------------
# LoadOrEncodeAccessionTests
# ---------------------------------------------------------------------------

class LoadOrEncodeAccessionTests(unittest.TestCase):
    """Tests for perceptrome.io_utils.load_or_encode_accession."""

    def _io_cfg(self, td: str):
        return SimpleNamespace(cache_encoded_dir=td)

    def test_uses_cached_npy_when_present(self):
        from perceptrome.io_utils import load_or_encode_accession, encoded_cache_path
        with tempfile.TemporaryDirectory() as td:
            io_cfg = self._io_cfg(td)
            cache_kw = {
                "min_orf_aa": 90, "max_windows_per_protein": None,
                "protein_len_min": None, "protein_len_max": None,
                "translation_only": False, "curriculum_tag": None,
            }
            enc_path = encoded_cache_path(io_cfg, "ACC1", "aa", 256, 128, 0,
                                          source="fasta", **cache_kw)
            os.makedirs(os.path.dirname(enc_path), exist_ok=True)
            arr = np.zeros((3, 256 * 21), dtype=np.float32)
            np.save(enc_path, arr)
            with patch("perceptrome.encoding_main.encode_accession") as mock_enc:
                out = load_or_encode_accession(
                    "ACC1", io_cfg,
                    tokenizer="aa", window_size=256, stride=128, frame_offset=0,
                    source="fasta", cache_kw=cache_kw,
                )
            self.assertEqual(out.shape, (3, 256 * 21))
            mock_enc.assert_not_called()

    def test_calls_encode_accession_on_cache_miss(self):
        from perceptrome.io_utils import load_or_encode_accession
        with tempfile.TemporaryDirectory() as td:
            io_cfg = self._io_cfg(td)
            cache_kw = {
                "min_orf_aa": 90, "max_windows_per_protein": None,
                "protein_len_min": None, "protein_len_max": None,
                "translation_only": False, "curriculum_tag": None,
            }
            fake_arr = np.zeros((1, 256 * 21), dtype=np.float32)
            with patch("perceptrome.encoding_main.encode_accession", return_value=fake_arr) as mock_enc:
                out = load_or_encode_accession(
                    "ACC2", io_cfg,
                    tokenizer="aa", window_size=256, stride=128, frame_offset=0,
                    source="fasta", cache_kw=cache_kw,
                    protein_opts={"strict_cds": False},
                )
            self.assertIs(out, fake_arr)
            mock_enc.assert_called_once()
            kwargs = mock_enc.call_args.kwargs
            self.assertEqual(kwargs["tokenizer"], "aa")
            self.assertEqual(kwargs["min_orf_aa"], 90)
            self.assertEqual(kwargs["protein_opts"], {"strict_cds": False})


if __name__ == "__main__":
    unittest.main()

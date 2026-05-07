import io
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
# Torch stub — keeps the test runnable without a GPU install
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

from perceptrome.elbo_score import (
    ELBORecord,
    ELBOSummary,
    score_windows,
    write_elbo_json,
    write_elbo_tsv,
)
from perceptrome.cli_main import build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(accession="acc1", status="ok", elbo=-1.5, recon=1.2, kl=0.3):
    if status == "ok":
        return ELBORecord(
            accession=accession, status=status,
            elbo=elbo, recon_loss=recon, kl=kl,
            n_windows=2, sequence_length=256,
        )
    return ELBORecord(
        accession=accession, status="error",
        elbo=0.0, recon_loss=0.0, kl=0.0,
        n_windows=0, sequence_length=0,
        error="encoding failed",
    )


def _make_summary(run_id="r1"):
    return ELBOSummary(
        run_id=run_id,
        catalog_path="catalog.txt",
        n_scored=3,
        n_failed=1,
        tokenizer="base",
        model_type="mlp",
        window_size=128,
        loss_type="mse",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:01:00+00:00",
        mean_elbo=-1.5,
        mean_recon_loss=1.2,
        mean_kl=0.3,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class ScoreOneParsserTests(unittest.TestCase):
    def test_score_one_positional(self):
        args = build_parser().parse_args(["score-one", "ACC001"])
        self.assertEqual(args.command, "score-one")
        self.assertEqual(args.accession, "ACC001")

    def test_score_one_run_id(self):
        args = build_parser().parse_args(["score-one", "ACC001", "--run-id", "myrun"])
        self.assertEqual(args.run_id, "myrun")

    def test_score_one_tokenizer_flag(self):
        args = build_parser().parse_args(["score-one", "ACC001", "--tokenizer", "aa"])
        self.assertEqual(args.tokenizer, "aa")

    def test_score_one_window_size(self):
        args = build_parser().parse_args(["score-one", "ACC001", "--window-size", "64"])
        self.assertEqual(args.window_size, 64)


class ScoreBatchParserTests(unittest.TestCase):
    def test_score_batch_catalog_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["score-batch"])

    def test_score_batch_catalog(self):
        args = build_parser().parse_args(["score-batch", "--catalog", "catalog.txt"])
        self.assertEqual(args.catalog, "catalog.txt")

    def test_score_batch_keep_going(self):
        args = build_parser().parse_args(["score-batch", "--catalog", "c.txt", "--keep-going"])
        self.assertTrue(args.keep_going)

    def test_score_batch_run_id(self):
        args = build_parser().parse_args(["score-batch", "--catalog", "c.txt", "--run-id", "r1"])
        self.assertEqual(args.run_id, "r1")


# ---------------------------------------------------------------------------
# ELBORecord / ELBOSummary dataclass tests
# ---------------------------------------------------------------------------

class ELBORecordTests(unittest.TestCase):
    def test_ok_record_fields(self):
        r = _make_record()
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.elbo, -1.5)
        self.assertAlmostEqual(r.recon_loss, 1.2)
        self.assertAlmostEqual(r.kl, 0.3)
        self.assertIsNone(r.error)

    def test_error_record_has_message(self):
        r = _make_record(status="error")
        self.assertEqual(r.status, "error")
        self.assertEqual(r.elbo, 0.0)
        self.assertIsNotNone(r.error)

    def test_elbo_equals_negative_sum(self):
        elbo, recon, kl = -1.8, 1.5, 0.3
        r = ELBORecord(
            accession="x", status="ok",
            elbo=elbo, recon_loss=recon, kl=kl,
            n_windows=1, sequence_length=128,
        )
        self.assertAlmostEqual(r.elbo, -(r.recon_loss + r.kl), places=6)


# ---------------------------------------------------------------------------
# score_windows computation tests (use real numpy, stub torch)
# ---------------------------------------------------------------------------

class _TorchTensor:
    """Minimal tensor façade backed by numpy so score_windows can run without real torch."""

    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32)

    @property
    def shape(self):
        return self._arr.shape

    def size(self, dim=None):
        if dim is None:
            return self._arr.shape
        return self._arr.shape[dim]

    def view(self, *shape):
        return _TorchTensor(self._arr.reshape(shape))

    def pow(self, n):
        return _TorchTensor(self._arr ** n)

    def exp(self):
        return _TorchTensor(np.exp(self._arr))

    def mean(self, dim=None):
        if dim is None:
            return _TorchTensor(np.array(self._arr.mean()))
        return _TorchTensor(self._arr.mean(axis=dim))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __add__(self, other):
        v = other._arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self._arr + v)

    def __radd__(self, other):
        return _TorchTensor(other + self._arr)

    def __sub__(self, other):
        v = other._arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self._arr - v)

    def __mul__(self, other):
        v = other._arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self._arr * v)

    def __rmul__(self, other):
        return _TorchTensor(other * self._arr)

    def __neg__(self):
        return _TorchTensor(-self._arr)

    def argmax(self, dim=None):
        return _TorchTensor(np.argmax(self._arr, axis=dim))


def _build_fake_torch(seq_len, vocab_size):
    """Return a minimal torch-like module that satisfies score_windows."""
    import types as _types

    fake_torch = _types.ModuleType("torch")
    fake_F = _types.ModuleType("torch.nn.functional")

    def from_numpy(arr):
        return _TorchTensor(arr)

    def no_grad():
        class _ctx:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _ctx()

    # cross_entropy: returns a 1-D tensor of per-element loss values
    def cross_entropy(logits_2d, targets_1d, reduction="mean"):
        # logits_2d: (N*L, V), targets: (N*L,)
        lg = logits_2d._arr
        tg = targets_1d._arr.astype(int)
        # softmax + NLL
        shifted = lg - lg.max(axis=1, keepdims=True)
        log_sm = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        nll = -log_sm[np.arange(len(tg)), tg]
        return _TorchTensor(nll)

    fake_F.cross_entropy = cross_entropy
    fake_torch.no_grad = no_grad
    fake_torch.from_numpy = from_numpy
    return fake_torch, fake_F


class ScoreWindowsTests(unittest.TestCase):
    """
    Patch torch and verify that score_windows returns the right shapes and
    that elbo = -(recon_loss + kl).
    """

    def _run_score(self, seq_len=4, vocab_size=4, n_windows=2, loss_type="ce"):
        fake_torch, fake_F = _build_fake_torch(seq_len, vocab_size)

        # Build minimal model stub
        # encode returns (mu, logvar); decode returns logits
        rng = np.random.default_rng(0)
        hidden = 8
        mu_arr = rng.standard_normal((n_windows, hidden)).astype(np.float32)
        logvar_arr = np.zeros((n_windows, hidden), dtype=np.float32)
        logits_arr = rng.standard_normal((n_windows, seq_len * vocab_size)).astype(np.float32)

        model = MagicMock()
        model.encode.return_value = (_TorchTensor(mu_arr), _TorchTensor(logvar_arr))
        model.decode.return_value = _TorchTensor(logits_arr)

        encoded = rng.integers(0, vocab_size, size=(n_windows, seq_len, vocab_size)).astype(np.float32)
        # make one-hot-ish (doesn't need to be exact for unit test)
        encoded = np.eye(vocab_size)[rng.integers(0, vocab_size, size=(n_windows, seq_len))].astype(np.float32)

        with patch.dict(sys.modules, {"torch": fake_torch, "torch.nn.functional": fake_F}):
            elbo, recon_loss, kl = score_windows(model, encoded, seq_len, vocab_size, loss_type, device="cpu")

        return elbo, recon_loss, kl

    def test_elbo_equals_negative_sum_ce(self):
        elbo, recon, kl = self._run_score(loss_type="ce")
        self.assertAlmostEqual(elbo, -(recon + kl), places=5)

    def test_all_finite(self):
        elbo, recon, kl = self._run_score(loss_type="ce")
        self.assertTrue(math.isfinite(elbo))
        self.assertTrue(math.isfinite(recon))
        self.assertTrue(math.isfinite(kl))

    def test_kl_non_negative_for_zero_logvar(self):
        # logvar=0 → std=1, so KL = 0.5 * (mu^2) ≥ 0 for any mu
        _, _, kl = self._run_score(loss_type="ce")
        self.assertGreaterEqual(kl, 0.0)

    def test_kl_zero_for_zero_mu_zero_logvar(self):
        """When mu=0 and logvar=0, KL = -0.5*(1+0-0-1) = 0."""
        fake_torch, fake_F = _build_fake_torch(4, 4)
        seq_len, vocab_size, n_windows = 4, 4, 3
        model = MagicMock()
        model.encode.return_value = (
            _TorchTensor(np.zeros((n_windows, 8), dtype=np.float32)),
            _TorchTensor(np.zeros((n_windows, 8), dtype=np.float32)),
        )
        rng = np.random.default_rng(1)
        model.decode.return_value = _TorchTensor(
            rng.standard_normal((n_windows, seq_len * vocab_size)).astype(np.float32)
        )
        encoded = np.eye(vocab_size)[rng.integers(0, vocab_size, (n_windows, seq_len))].astype(np.float32)
        with patch.dict(sys.modules, {"torch": fake_torch, "torch.nn.functional": fake_F}):
            _, _, kl = score_windows(model, encoded, seq_len, vocab_size, "ce", "cpu")
        self.assertAlmostEqual(kl, 0.0, places=5)


# ---------------------------------------------------------------------------
# write_elbo_json tests
# ---------------------------------------------------------------------------

class WriteELBOJsonTests(unittest.TestCase):
    def test_round_trip_records(self):
        records = [_make_record("a1"), _make_record("a2", status="error")]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.json")
            write_elbo_json(path, records)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(data["records"][0]["accession"], "a1")
        self.assertEqual(data["records"][0]["status"], "ok")
        self.assertEqual(data["records"][1]["status"], "error")

    def test_summary_included_when_provided(self):
        records = [_make_record()]
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.json")
            write_elbo_json(path, records, summary=summary)
            with open(path) as f:
                data = json.load(f)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["run_id"], "r1")
        self.assertAlmostEqual(data["summary"]["mean_elbo"], -1.5)

    def test_summary_absent_when_not_provided(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.json")
            write_elbo_json(path, [_make_record()])
            with open(path) as f:
                data = json.load(f)
        self.assertNotIn("summary", data)

    def test_elbo_values_preserved(self):
        r = ELBORecord(
            accession="x", status="ok",
            elbo=-2.345, recon_loss=2.0, kl=0.345,
            n_windows=4, sequence_length=512,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "e.json")
            write_elbo_json(path, [r])
            with open(path) as f:
                data = json.load(f)
        row = data["records"][0]
        self.assertAlmostEqual(row["elbo"], -2.345, places=5)
        self.assertAlmostEqual(row["recon_loss"], 2.0, places=5)
        self.assertAlmostEqual(row["kl"], 0.345, places=5)


# ---------------------------------------------------------------------------
# write_elbo_tsv tests
# ---------------------------------------------------------------------------

class WriteELBOTsvTests(unittest.TestCase):
    def test_has_header_and_data_row(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.tsv")
            write_elbo_tsv(path, [_make_record()])
            with open(path) as f:
                lines = f.readlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("accession", lines[0])
        self.assertIn("elbo", lines[0])

    def test_key_columns_present(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.tsv")
            write_elbo_tsv(path, [_make_record()])
            with open(path) as f:
                header = f.readline()
        for col in ("accession", "status", "elbo", "recon_loss", "kl", "n_windows"):
            self.assertIn(col, header)

    def test_data_row_contains_accession(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.tsv")
            write_elbo_tsv(path, [_make_record("myacc")])
            with open(path) as f:
                lines = f.readlines()
        self.assertIn("myacc", lines[1])

    def test_error_row_has_error_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.tsv")
            write_elbo_tsv(path, [_make_record(status="error")])
            with open(path) as f:
                lines = f.readlines()
        self.assertIn("encoding failed", lines[1])

    def test_multiple_records(self):
        records = [_make_record(f"acc{i}") for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "elbo.tsv")
            write_elbo_tsv(path, records)
            with open(path) as f:
                lines = f.readlines()
        self.assertEqual(len(lines), 6)  # header + 5 rows


# ---------------------------------------------------------------------------
# ELBOSummary content tests
# ---------------------------------------------------------------------------

class ELBOSummaryTests(unittest.TestCase):
    def test_summary_fields(self):
        s = _make_summary()
        self.assertEqual(s.n_scored, 3)
        self.assertEqual(s.n_failed, 1)
        self.assertAlmostEqual(s.mean_elbo, -1.5)
        self.assertEqual(s.tokenizer, "base")
        self.assertEqual(s.loss_type, "mse")

    def test_summary_json_serialisable(self):
        s = _make_summary()
        from dataclasses import asdict
        d = asdict(s)
        # should not raise
        json.dumps(d)

    def test_summary_mean_elbo_consistent(self):
        records = [_make_record(elbo=-1.0), _make_record(elbo=-2.0), _make_record(elbo=-3.0)]
        mean_elbo = float(np.mean([r.elbo for r in records]))
        self.assertAlmostEqual(mean_elbo, -2.0, places=6)


if __name__ == "__main__":
    unittest.main()

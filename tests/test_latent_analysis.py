import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

from perceptrome.cli_main import build_parser
from perceptrome.cli import commands
from perceptrome.latent_cluster import (
    LatentClusterRecord,
    LatentClusterSummary,
    write_cluster_records_json,
    write_cluster_records_tsv,
    write_cluster_summary_json,
)
from perceptrome.latent_interp import (
    InterpStep,
    InterpSummary,
    tokens_to_sequence,
    write_interp_fasta,
    write_interp_summary_json,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class LatentClusterParserTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["latent-cluster", "--catalog", "cat.txt"])
        self.assertEqual(args.command, "latent-cluster")
        self.assertEqual(args.catalog, "cat.txt")
        self.assertEqual(args.n_clusters, 8)
        self.assertEqual(args.umap_n_neighbors, 15)
        self.assertAlmostEqual(args.umap_min_dist, 0.1)

    def test_overrides(self):
        args = build_parser().parse_args([
            "latent-cluster", "--catalog", "cat.txt",
            "--n-clusters", "4", "--umap-n-neighbors", "10", "--umap-min-dist", "0.05",
        ])
        self.assertEqual(args.n_clusters, 4)
        self.assertEqual(args.umap_n_neighbors, 10)
        self.assertAlmostEqual(args.umap_min_dist, 0.05)

    def test_run_id_arg(self):
        args = build_parser().parse_args(
            ["latent-cluster", "--catalog", "cat.txt", "--run-id", "r42"]
        )
        self.assertEqual(args.run_id, "r42")


class LatentInterpolateParserTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["latent-interpolate", "ACC_A", "ACC_B"])
        self.assertEqual(args.command, "latent-interpolate")
        self.assertEqual(args.acc_a, "ACC_A")
        self.assertEqual(args.acc_b, "ACC_B")
        self.assertEqual(args.steps, 8)
        self.assertEqual(args.n_windows, 1)
        self.assertAlmostEqual(args.temperature, 0.0)

    def test_overrides(self):
        args = build_parser().parse_args([
            "latent-interpolate", "A", "B",
            "--steps", "12", "--n-windows", "3", "--temperature", "0.7",
            "--output", "my_path.fasta",
        ])
        self.assertEqual(args.steps, 12)
        self.assertEqual(args.n_windows, 3)
        self.assertAlmostEqual(args.temperature, 0.7)
        self.assertEqual(args.output, "my_path.fasta")


class LatentSeedsParserTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["latent-seeds", "run_abc"])
        self.assertEqual(args.command, "latent-seeds")
        self.assertEqual(args.run_or_path, "run_abc")
        self.assertFalse(args.outliers)
        self.assertIsNone(args.vectors)
        self.assertIsNone(args.output_dir)

    def test_outliers_flag(self):
        args = build_parser().parse_args(["latent-seeds", "run_abc", "--outliers"])
        self.assertTrue(args.outliers)

    def test_explicit_vectors(self):
        args = build_parser().parse_args(
            ["latent-seeds", "run_abc", "--vectors", "/tmp/vecs.npy"]
        )
        self.assertEqual(args.vectors, "/tmp/vecs.npy")


# ---------------------------------------------------------------------------
# tokens_to_sequence
# ---------------------------------------------------------------------------

class TokensToSequenceTests(unittest.TestCase):
    def test_base_canonical(self):
        self.assertEqual(tokens_to_sequence([0, 1, 2, 3], "base"), "ACGT")

    def test_base_wraps_modulo(self):
        # index 4 → 4%4=0 → A
        self.assertEqual(tokens_to_sequence([4], "base"), "A")

    def test_aa_round_trip(self):
        from perceptrome.encoding_main import IDX_TO_AA
        tokens = list(range(min(5, len(IDX_TO_AA))))
        seq = tokens_to_sequence(tokens, "aa")
        expected = "".join(IDX_TO_AA[t] for t in tokens)
        self.assertEqual(seq, expected)

    def test_codon_round_trip(self):
        from perceptrome.encoding_main import IDX_TO_CODON
        tokens = [0, 1, 2]
        seq = tokens_to_sequence(tokens, "codon")
        expected = "".join(IDX_TO_CODON[t] for t in tokens)
        self.assertEqual(seq, expected)

    def test_unknown_tokenizer_falls_back_to_base(self):
        seq = tokens_to_sequence([0, 1], "unknown_tok")
        self.assertEqual(seq, "AC")


# ---------------------------------------------------------------------------
# InterpStep math
# ---------------------------------------------------------------------------

class InterpStepMathTests(unittest.TestCase):
    def _make_steps(self, n):
        return [
            InterpStep(step=i, t=(0.0 if n == 1 else float(i) / float(n - 1)), sequence="X")
            for i in range(n)
        ]

    def test_step_count(self):
        self.assertEqual(len(self._make_steps(8)), 8)

    def test_endpoints(self):
        steps = self._make_steps(8)
        self.assertAlmostEqual(steps[0].t, 0.0)
        self.assertAlmostEqual(steps[-1].t, 1.0)

    def test_midpoint(self):
        steps = self._make_steps(5)
        self.assertAlmostEqual(steps[2].t, 0.5)

    def test_single_step(self):
        steps = self._make_steps(1)
        self.assertAlmostEqual(steps[0].t, 0.0)


# ---------------------------------------------------------------------------
# write_interp_fasta
# ---------------------------------------------------------------------------

class WriteInterpFastaTests(unittest.TestCase):
    def _steps(self):
        return [
            InterpStep(step=0, t=0.0, sequence="ACGT"),
            InterpStep(step=1, t=0.5, sequence="TTTT"),
            InterpStep(step=2, t=1.0, sequence="GGGG"),
        ]

    def test_header_format(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_interp_fasta(os.path.join(td, "out.fasta"), "ACCA", "ACCB", self._steps())
            with open(path) as f:
                content = f.read()
        self.assertIn(">interp_step0_of3", content)
        self.assertIn(">interp_step2_of3", content)
        self.assertIn("t=0.0000", content)
        self.assertIn("t=1.0000", content)
        self.assertIn("a=ACCA", content)
        self.assertIn("b=ACCB", content)

    def test_sequences_present(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_interp_fasta(os.path.join(td, "out.fasta"), "A", "B", self._steps())
            with open(path) as f:
                content = f.read()
        self.assertIn("ACGT", content)
        self.assertIn("TTTT", content)
        self.assertIn("GGGG", content)

    def test_fasta_entry_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_interp_fasta(os.path.join(td, "out.fasta"), "A", "B", self._steps())
            with open(path) as f:
                headers = [l for l in f if l.startswith(">")]
        self.assertEqual(len(headers), 3)


class WriteInterpSummaryTests(unittest.TestCase):
    def test_round_trip(self):
        steps = [InterpStep(step=0, t=0.0, sequence="ACGT")]
        summary = InterpSummary(
            run_id="r1", acc_a="A", acc_b="B", steps=1, latent_dim=64,
            tokenizer="base", n_windows=1, window_size=4, output_length=4,
            output_fasta="out.fasta", started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z", metadata={"temperature": 0.0},
        )
        with tempfile.TemporaryDirectory() as td:
            path = write_interp_summary_json(os.path.join(td, "s.json"), summary, steps)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(data["acc_a"], "A")
        self.assertEqual(data["latent_dim"], 64)
        self.assertEqual(len(data["steps"]), 1)
        self.assertAlmostEqual(data["steps"][0]["t"], 0.0)
        self.assertEqual(data["steps"][0]["sequence"], "ACGT")


# ---------------------------------------------------------------------------
# LatentClusterRecord writers
# ---------------------------------------------------------------------------

class ClusterRecordWriterTests(unittest.TestCase):
    def _records(self):
        return [
            LatentClusterRecord(accession="a1", cluster_id=0, umap_x=1.0, umap_y=2.0,
                                mu_norm=0.5, mu_mean=0.1, n_windows=3),
            LatentClusterRecord(accession="a2", cluster_id=1, umap_x=-1.0, umap_y=0.5,
                                mu_norm=0.8, mu_mean=0.2, n_windows=2),
        ]

    def test_json_record_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_cluster_records_json(os.path.join(td, "r.json"), self._records())
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(len(data["records"]), 2)

    def test_json_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_cluster_records_json(os.path.join(td, "r.json"), self._records())
            with open(path) as f:
                data = json.load(f)
        row = data["records"][0]
        self.assertEqual(row["accession"], "a1")
        self.assertEqual(row["cluster_id"], 0)
        self.assertAlmostEqual(row["umap_x"], 1.0)

    def test_tsv_header_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_cluster_records_tsv(os.path.join(td, "r.tsv"), self._records())
            with open(path) as f:
                lines = f.readlines()
        self.assertGreaterEqual(len(lines), 3)
        self.assertIn("accession", lines[0])
        self.assertIn("cluster_id", lines[0])
        self.assertIn("a1", lines[1])
        self.assertIn("a2", lines[2])


# ---------------------------------------------------------------------------
# latent-seeds: resolver, path derivation, and archetype/outlier selection
# ---------------------------------------------------------------------------

class LatentSeedsResolverTests(unittest.TestCase):
    def test_resolve_direct_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "latent_cluster_summary.json")
            with open(path, "w") as f:
                json.dump({"records": [], "centroids": []}, f)
            self.assertEqual(commands._resolve_cluster_summary_json(path), path)

    def test_resolve_by_run_id(self):
        orig = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                out_dir = os.path.join(td, "runs", "run42", "outputs")
                os.makedirs(out_dir)
                summary = os.path.join(out_dir, "latent_cluster_summary.json")
                with open(summary, "w") as f:
                    json.dump({"records": [], "centroids": []}, f)
                resolved = commands._resolve_cluster_summary_json("run42")
                self.assertEqual(os.path.normpath(resolved), os.path.normpath(summary))
            finally:
                os.chdir(orig)

    def test_resolve_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            commands._resolve_cluster_summary_json("no_such_latent_run_xyz")

    def test_derive_vectors_npy_path(self):
        summary = os.path.join("runs", "myrun", "outputs", "latent_cluster_summary.json")
        derived = commands._derive_vectors_npy(summary)
        self.assertIn("myrun", derived)
        self.assertTrue(derived.endswith(os.path.join("artifacts", "latent_vectors.npy")))


class LatentSeedsComputationTests(unittest.TestCase):
    """
    Two clusters of three points each in 4-D latent space.

    Cluster 0  centroid = [1, 0, 0, 0]
      a0 = [1.0, 0, 0, 0]  dist = 0.0   ← nearest
      a1 = [1.1, 0, 0, 0]  dist = 0.1
      a2 = [1.5, 0, 0, 0]  dist = 0.5   ← farthest (outlier)

    Cluster 1  centroid = [0, 1, 0, 0]
      b0 = [0, 1.0, 0, 0]  dist = 0.0   ← nearest
      b1 = [0, 1.1, 0, 0]  dist = 0.1
      b2 = [0, 2.0, 0, 0]  dist = 1.0   ← farthest (outlier)
    """

    def _setup(self, td):
        mu = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ], dtype=np.float32)
        centroids = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        records = [
            {"accession": "a0", "cluster_id": 0},
            {"accession": "a1", "cluster_id": 0},
            {"accession": "a2", "cluster_id": 0},
            {"accession": "b0", "cluster_id": 1},
            {"accession": "b1", "cluster_id": 1},
            {"accession": "b2", "cluster_id": 1},
        ]
        vec_path = os.path.join(td, "artifacts", "latent_vectors.npy")
        os.makedirs(os.path.dirname(vec_path), exist_ok=True)
        np.save(vec_path, mu)
        sum_path = os.path.join(td, "outputs", "latent_cluster_summary.json")
        os.makedirs(os.path.dirname(sum_path), exist_ok=True)
        with open(sum_path, "w") as f:
            json.dump({"records": records, "centroids": centroids}, f)
        return sum_path, vec_path

    def _run(self, td, outliers=False):
        sum_path, vec_path = self._setup(td)
        out_dir = os.path.join(td, "out")
        args = SimpleNamespace(
            run_or_path=sum_path, outliers=outliers,
            vectors=vec_path, output_dir=out_dir,
        )
        rc = commands.cmd_latent_seeds(args)
        self.assertEqual(rc, 0)
        with open(os.path.join(out_dir, "latent_seeds.json")) as f:
            seeds = json.load(f)
        with open(os.path.join(out_dir, "latent_seeds.catalog.txt")) as f:
            catalog = [l.strip() for l in f if l.strip()]
        return seeds, catalog

    def test_archetypes_are_nearest(self):
        with tempfile.TemporaryDirectory() as td:
            seeds, _ = self._run(td)
        self.assertEqual(seeds["clusters"]["0"]["archetype"], "a0")
        self.assertEqual(seeds["clusters"]["1"]["archetype"], "b0")

    def test_archetype_distances(self):
        with tempfile.TemporaryDirectory() as td:
            seeds, _ = self._run(td)
        self.assertAlmostEqual(seeds["clusters"]["0"]["archetype_dist"], 0.0, places=5)
        self.assertAlmostEqual(seeds["clusters"]["1"]["archetype_dist"], 0.0, places=5)

    def test_outliers_are_farthest(self):
        with tempfile.TemporaryDirectory() as td:
            seeds, _ = self._run(td, outliers=True)
        self.assertEqual(seeds["clusters"]["0"]["outlier"], "a2")
        self.assertEqual(seeds["clusters"]["1"]["outlier"], "b2")

    def test_catalog_archetypes_only(self):
        with tempfile.TemporaryDirectory() as td:
            _, catalog = self._run(td, outliers=False)
        self.assertEqual(sorted(catalog), sorted(["a0", "b0"]))

    def test_catalog_with_outliers(self):
        with tempfile.TemporaryDirectory() as td:
            _, catalog = self._run(td, outliers=True)
        self.assertEqual(len(catalog), 4)
        for acc in ("a0", "b0", "a2", "b2"):
            self.assertIn(acc, catalog)

    def test_seeds_json_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            seeds, _ = self._run(td)
        self.assertEqual(seeds["n_clusters"], 2)
        self.assertEqual(seeds["n_selected"], 2)
        self.assertFalse(seeds["include_outliers"])

    def test_missing_vectors_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            sum_path, _ = self._setup(td)
            args = SimpleNamespace(
                run_or_path=sum_path, outliers=False,
                vectors="/nonexistent/latent_vectors.npy", output_dir=os.path.join(td, "out"),
            )
            with self.assertRaises(FileNotFoundError):
                commands.cmd_latent_seeds(args)


if __name__ == "__main__":
    unittest.main()

import io
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

from perceptrome.cli_main import build_parser
from perceptrome.cli import commands
from perceptrome.genome.summary import (
    build_genome_annotation_record,
    write_summary_json,
    write_summary_tsv,
)
from perceptrome.genome.annotation_manifest import build_annotation_manifest_update


def _make_result(accession="seq1"):
    from perceptrome.genome.annotator import GenomeAnnotationCounts, GenomeAnnotationResult
    return GenomeAnnotationResult(
        accession=accession,
        sequence_length=1000,
        source_format="fasta",
        cds_source="orf_scan",
        counts=GenomeAnnotationCounts(
            node_counts={"cds": 3, "gene": 3, "promoter": 2},
            edge_counts={"activates": 1},
            cds_count=3,
            gene_count=3,
            promoter_count=2,
            operator_count=0,
            rbs_count=1,
            terminator_count=1,
            total_nodes=10,
            total_edges=2,
        ),
    )


def _make_record(accession="seq1", status="ok"):
    result = _make_result(accession) if status == "ok" else None
    return build_genome_annotation_record(
        accession=accession,
        source_input_path=f"cache/{accession}.fasta",
        annotation_status=status,
        annotation_json_path=f"runs/r1/artifacts/genome/{accession}/annotation.json" if status == "ok" else None,
        result=result,
    )


class GenomeAnnotateCLIParserTests(unittest.TestCase):
    def test_parser_genome_annotate_one(self):
        args = build_parser().parse_args(["genome-annotate-one", "genome.gb"])
        self.assertEqual(args.command, "genome-annotate-one")
        self.assertEqual(args.input, "genome.gb")
        self.assertIsNone(args.run_id)

    def test_parser_genome_annotate_one_run_id(self):
        args = build_parser().parse_args(["genome-annotate-one", "genome.gb", "--run-id", "abc123"])
        self.assertEqual(args.run_id, "abc123")

    def test_parser_genome_annotate_inspect(self):
        args = build_parser().parse_args(["genome-annotate-inspect", "run_xyz"])
        self.assertEqual(args.command, "genome-annotate-inspect")
        self.assertEqual(args.run_or_path, "run_xyz")

    def test_parser_genome_annotate_export(self):
        args = build_parser().parse_args(["genome-annotate-export", "run_xyz"])
        self.assertEqual(args.command, "genome-annotate-export")
        self.assertIsNone(args.output_dir)

    def test_parser_genome_annotate_export_output_dir(self):
        args = build_parser().parse_args(
            ["genome-annotate-export", "run_xyz", "--output-dir", "/tmp/out"]
        )
        self.assertEqual(args.output_dir, "/tmp/out")

    def test_parser_genome_annotate_batch_flags(self):
        args = build_parser().parse_args(
            ["genome-annotate-batch", "genomes/", "--keep-going", "--resume"]
        )
        self.assertTrue(args.keep_going)
        self.assertTrue(args.resume)


class GenomeAnnotateSummaryTests(unittest.TestCase):
    def test_summary_json_round_trip(self):
        record = _make_record()
        with tempfile.TemporaryDirectory() as td:
            path = write_summary_json(os.path.join(td, "summary.json"), [record])
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                payload = json.load(f)
        rows = payload["records"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accession"], "seq1")
        self.assertEqual(rows[0]["annotation_status"], "ok")
        self.assertEqual(rows[0]["gene_count"], 3)
        self.assertEqual(rows[0]["total_nodes"], 10)

    def test_summary_tsv_columns(self):
        record = _make_record()
        with tempfile.TemporaryDirectory() as td:
            path = write_summary_tsv(os.path.join(td, "summary.tsv"), [record])
            with open(path) as f:
                lines = f.readlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("accession", lines[0])
        self.assertIn("annotation_status", lines[0])
        self.assertIn("seq1", lines[1])

    def test_failed_record_zero_counts(self):
        record = _make_record(status="error")
        self.assertEqual(record.annotation_status, "error")
        self.assertEqual(record.gene_count, 0)
        self.assertEqual(record.total_nodes, 0)

    def test_annotation_manifest_payload(self):
        record = _make_record()
        payload = build_annotation_manifest_update(
            command_name="genome_annotate_one",
            run_id="r1",
            summary_json_path="runs/r1/outputs/genome_summary.json",
            summary_tsv_path="runs/r1/outputs/genome_summary.tsv",
            stdout_log_path="runs/r1/provenance/stdout.log",
            stderr_log_path="runs/r1/provenance/stderr.log",
            records=[record],
        )
        self.assertIn("paths", payload)
        self.assertIn("artifacts", payload)
        self.assertGreaterEqual(len(payload["artifacts"]), 4)
        roles = {a["role"] for a in payload["artifacts"]}
        self.assertIn("genome.annotation", roles)
        self.assertIn("genome.summary", roles)


class GenomeAnnotateResolverTests(unittest.TestCase):
    def test_resolve_direct_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "genome_summary.json")
            with open(path, "w") as f:
                json.dump({"records": []}, f)
            self.assertEqual(commands._resolve_genome_summary_json(path), path)

    def test_resolve_by_run_id(self):
        orig = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                run_out = os.path.join(td, "runs", "myrun", "outputs")
                os.makedirs(run_out)
                summary = os.path.join(run_out, "genome_summary.json")
                with open(summary, "w") as f:
                    json.dump({"records": []}, f)
                resolved = commands._resolve_genome_summary_json("myrun")
                self.assertEqual(os.path.normpath(resolved), os.path.normpath(summary))
            finally:
                os.chdir(orig)

    def test_resolve_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            commands._resolve_genome_summary_json("nonexistent_genome_run_xyz")


class GenomeAnnotateInspectTests(unittest.TestCase):
    def _write_summary(self, td, records):
        path = os.path.join(td, "genome_summary.json")
        with open(path, "w") as f:
            json.dump({"records": records}, f)
        return path

    def test_inspect_counts(self):
        with tempfile.TemporaryDirectory() as td:
            summary = self._write_summary(td, [
                {"accession": "a1", "annotation_status": "ok",
                 "sequence_length": 1000, "gene_count": 5, "cds_count": 5,
                 "promoter_count": 2, "total_nodes": 12, "total_edges": 3},
                {"accession": "a2", "annotation_status": "error",
                 "sequence_length": 0, "gene_count": 0, "cds_count": 0,
                 "promoter_count": 0, "total_nodes": 0, "total_edges": 0},
            ])
            args = SimpleNamespace(run_or_path=summary)
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                rc = commands.cmd_genome_annotate_inspect(args)
            out = mock_out.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("records=2", out)
        self.assertIn("ok=1", out)
        self.assertIn("failed=1", out)
        self.assertIn("a1", out)

    def test_inspect_empty_summary(self):
        with tempfile.TemporaryDirectory() as td:
            summary = self._write_summary(td, [])
            args = SimpleNamespace(run_or_path=summary)
            with patch("sys.stdout", new_callable=io.StringIO):
                rc = commands.cmd_genome_annotate_inspect(args)
        self.assertEqual(rc, 0)


class GenomeAnnotateExportTests(unittest.TestCase):
    def _row(self, accession="a1"):
        return {
            "accession": accession, "annotation_status": "ok",
            "source_format": "fasta", "cds_source": "orf_scan",
            "sequence_length": 1000, "gene_count": 3, "cds_count": 3,
            "promoter_count": 2, "operator_count": 0, "rbs_count": 1,
            "terminator_count": 1, "total_nodes": 10, "total_edges": 2,
            "annotation_json_path": None,
        }

    def test_export_creates_json_and_tsv(self):
        with tempfile.TemporaryDirectory() as td:
            summary = os.path.join(td, "genome_summary.json")
            with open(summary, "w") as f:
                json.dump({"records": [self._row()]}, f)
            out_dir = os.path.join(td, "export")
            args = SimpleNamespace(run_or_path=summary, output_dir=out_dir)
            rc = commands.cmd_genome_annotate_export(args)
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "genome_export.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "genome_export.tsv")))

    def test_export_json_content(self):
        with tempfile.TemporaryDirectory() as td:
            summary = os.path.join(td, "genome_summary.json")
            with open(summary, "w") as f:
                json.dump({"records": [self._row("acc1"), self._row("acc2")]}, f)
            out_dir = os.path.join(td, "export")
            args = SimpleNamespace(run_or_path=summary, output_dir=out_dir)
            commands.cmd_genome_annotate_export(args)
            with open(os.path.join(out_dir, "genome_export.json")) as f:
                data = json.load(f)
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(data["records"][0]["accession"], "acc1")

    def test_export_tsv_has_key_columns(self):
        with tempfile.TemporaryDirectory() as td:
            summary = os.path.join(td, "genome_summary.json")
            with open(summary, "w") as f:
                json.dump({"records": [self._row()]}, f)
            out_dir = os.path.join(td, "export")
            args = SimpleNamespace(run_or_path=summary, output_dir=out_dir)
            commands.cmd_genome_annotate_export(args)
            with open(os.path.join(out_dir, "genome_export.tsv")) as f:
                header = f.readline()
        for col in ("accession", "annotation_status", "gene_count", "total_nodes"):
            self.assertIn(col, header)


if __name__ == "__main__":
    unittest.main()

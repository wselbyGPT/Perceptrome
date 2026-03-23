import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from perceptrome.cli.commands import _build_and_write_bio_ast, _collect_bio_ast_transforms
from perceptrome.scope.ui import load_bio_ast_visualization
from perceptrome.encoding.bio_ast_export import export_filenames, stable_json_dumps
from perceptrome.cli_main import build_parser
from tests.fixtures.bio_ast_regression_fixtures import render_fasta


def test_cli_parser_accepts_bio_ast_subcommands():
    parser = build_parser()
    args = parser.parse_args(["bio-ast", "visualize", "--accession", "ACC1", "--source", "fasta"])
    assert args.command == "bio-ast"
    assert args.bio_ast_command == "visualize"
    assert args.accession == "ACC1"


def test_bio_ast_workbench_emits_all_transforms_and_consistent_node_ids():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta_dir = tmp_path / "cache_fasta"
        genbank_dir = tmp_path / "cache_genbank"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        genbank_dir.mkdir(parents=True, exist_ok=True)

        accession = "ACC_BIO_AST"
        (fasta_dir / f"{accession}.fasta").write_text(render_fasta(), encoding="utf-8")

        io_cfg = SimpleNamespace(cache_fasta_dir=str(fasta_dir), cache_genbank_dir=str(genbank_dir))

        old_cwd = os.getcwd()
        old_run_id = os.environ.get("PERCEPTROME_RUN_ID")
        old_run_root = os.environ.get("PERCEPTROME_RUN_ROOT")
        try:
            os.chdir(tmp)
            os.environ.pop("PERCEPTROME_RUN_ID", None)
            os.environ.pop("PERCEPTROME_RUN_ROOT", None)
            outputs = _build_and_write_bio_ast(accession=accession, source="fasta", io_cfg=io_cfg)
        finally:
            os.chdir(old_cwd)
            if old_run_id is None:
                os.environ.pop("PERCEPTROME_RUN_ID", None)
            else:
                os.environ["PERCEPTROME_RUN_ID"] = old_run_id
            if old_run_root is None:
                os.environ.pop("PERCEPTROME_RUN_ROOT", None)
            else:
                os.environ["PERCEPTROME_RUN_ROOT"] = old_run_root

        assert outputs is not None
        expected_keys = {"canonical_ast", "motif_features", "tree_tensors", "graph_edges", "tree_json", "graph_json", "storage_map", "summary_json"}
        assert set(outputs.keys()) == expected_keys
        for out_path in outputs.values():
            assert Path(out_path).exists()

        canonical = json.loads(Path(outputs["canonical_ast"]).read_text(encoding="utf-8"))
        motif_features = json.loads(Path(outputs["motif_features"]).read_text(encoding="utf-8"))
        tree_tensors = json.loads(Path(outputs["tree_tensors"]).read_text(encoding="utf-8"))
        graph_edges = json.loads(Path(outputs["graph_edges"]).read_text(encoding="utf-8"))
        tree_json = json.loads(Path(outputs["tree_json"]).read_text(encoding="utf-8"))
        graph_json = json.loads(Path(outputs["graph_json"]).read_text(encoding="utf-8"))
        storage_map = json.loads(Path(outputs["storage_map"]).read_text(encoding="utf-8"))

        ast_nodes = canonical["nodes"]
        ast_node_ids = [node["canonical_id"] for node in ast_nodes]
        ast_node_id_set = set(ast_node_ids)

        assert tree_tensors["node_ids"] == ast_node_ids
        assert len(tree_tensors["node_type_ids"]) == len(ast_node_ids)
        assert len(tree_tensors["coords"]) == len(ast_node_ids)
        assert len(tree_tensors["strand"]) == len(ast_node_ids)

        edge_index = tree_tensors["edge_index"]
        assert len(edge_index) == 2
        assert len(edge_index[0]) == len(edge_index[1]) == len(graph_edges["edges"])

        for edge in graph_edges["edges"]:
            assert edge["parent_id"] in ast_node_id_set
            assert edge["child_id"] in ast_node_id_set
            assert ast_node_ids[edge["parent_index"]] == edge["parent_id"]
            assert ast_node_ids[edge["child_index"]] == edge["child_id"]

        for row in motif_features["rows"]:
            assert row["node_id"] in ast_node_id_set
            if row["parent_id"] is not None:
                assert row["parent_id"] in ast_node_id_set

        assert storage_map["sequence_length"] >= 1
        assert "tracks" in storage_map
        assert "coordinate_segments" in storage_map


def test_visualization_payloads_include_expected_schema_fields():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta_dir = tmp_path / "cache_fasta"
        genbank_dir = tmp_path / "cache_genbank"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        genbank_dir.mkdir(parents=True, exist_ok=True)

        accession = "ACC_BIO_AST_SCHEMA"
        (fasta_dir / f"{accession}.fasta").write_text(render_fasta(), encoding="utf-8")

        io_cfg = SimpleNamespace(cache_fasta_dir=str(fasta_dir), cache_genbank_dir=str(genbank_dir))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            outputs = _build_and_write_bio_ast(accession=accession, source="fasta", io_cfg=io_cfg)
        finally:
            os.chdir(old_cwd)

        assert outputs is not None
        tree_json = json.loads(Path(outputs["tree_json"]).read_text(encoding="utf-8"))
        graph_json = json.loads(Path(outputs["graph_json"]).read_text(encoding="utf-8"))
        storage_map = json.loads(Path(outputs["storage_map"]).read_text(encoding="utf-8"))

        assert tree_json["schema"] == "bio_ast_tree_v1"
        assert tree_json["export_version"] == 1
        assert "hierarchy" in tree_json
        assert graph_json["schema"] == "bio_ast_graph_v1"
        assert graph_json["export_version"] == 1
        assert "nodes" in graph_json and "edges" in graph_json
        assert storage_map["schema"] == "bio_ast_storage_map_v1"
        assert storage_map["export_version"] == 1
        assert "topology" in storage_map
        assert "tracks" in storage_map
        assert "coordinate_segments" in storage_map

        if graph_json["nodes"]:
            node = graph_json["nodes"][0]
            assert "node_type" in node
            assert "span" in node

        if graph_json["edges"]:
            edge = graph_json["edges"][0]
            assert "relation" in edge
            assert "relation_type" in edge


def test_load_bio_ast_visualization_includes_storage_map():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta_dir = tmp_path / "cache_fasta"
        genbank_dir = tmp_path / "cache_genbank"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        genbank_dir.mkdir(parents=True, exist_ok=True)

        accession = "ACC_BIO_AST_UI"
        (fasta_dir / f"{accession}.fasta").write_text(render_fasta(), encoding="utf-8")
        io_cfg = SimpleNamespace(cache_fasta_dir=str(fasta_dir), cache_genbank_dir=str(genbank_dir))

        old_cwd = os.getcwd()
        old_run_id = os.environ.get("PERCEPTROME_RUN_ID")
        old_run_root = os.environ.get("PERCEPTROME_RUN_ROOT")
        try:
            os.chdir(tmp)
            os.environ.pop("PERCEPTROME_RUN_ID", None)
            os.environ.pop("PERCEPTROME_RUN_ROOT", None)
            outputs = _build_and_write_bio_ast(accession=accession, source="fasta", io_cfg=io_cfg)
            assert outputs is not None
            payload = load_bio_ast_visualization(accession)
        finally:
            os.chdir(old_cwd)
            if old_run_id is None:
                os.environ.pop("PERCEPTROME_RUN_ID", None)
            else:
                os.environ["PERCEPTROME_RUN_ID"] = old_run_id
            if old_run_root is None:
                os.environ.pop("PERCEPTROME_RUN_ROOT", None)
            else:
                os.environ["PERCEPTROME_RUN_ROOT"] = old_run_root

        assert payload["storage_map"]["schema"] == "bio_ast_storage_map_v1"
        assert payload["storage_map"]["coordinate_segments"]
        assert payload["export_summary"]["schema"] == "bio_ast_summary_v1"


def test_bio_ast_exports_use_canonical_filenames_and_deterministic_serialization():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta_dir = tmp_path / "cache_fasta"
        genbank_dir = tmp_path / "cache_genbank"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        genbank_dir.mkdir(parents=True, exist_ok=True)

        accession = "ACC_BIO_AST_DETERMINISTIC"
        (fasta_dir / f"{accession}.fasta").write_text(render_fasta(), encoding="utf-8")
        io_cfg = SimpleNamespace(cache_fasta_dir=str(fasta_dir), cache_genbank_dir=str(genbank_dir))

        expected_filenames = export_filenames()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            outputs = _build_and_write_bio_ast(accession=accession, source="fasta", io_cfg=io_cfg)
            assert outputs is not None
            assert {key: Path(value).name for key, value in outputs.items()} == expected_filenames

            first = _collect_bio_ast_transforms(accession=accession, source="fasta", io_cfg=io_cfg)
            second = _collect_bio_ast_transforms(accession=accession, source="fasta", io_cfg=io_cfg)
        finally:
            os.chdir(old_cwd)

        assert stable_json_dumps(first) == stable_json_dumps(second)
        assert stable_json_dumps(first["canonical_ast"]) == Path(outputs["canonical_ast"]).read_text(encoding="utf-8")


def test_bio_ast_export_snapshot_structure_for_visualization_views():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta_dir = tmp_path / "cache_fasta"
        genbank_dir = tmp_path / "cache_genbank"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        genbank_dir.mkdir(parents=True, exist_ok=True)

        accession = "ACC_BIO_AST_SNAPSHOT"
        (fasta_dir / f"{accession}.fasta").write_text(render_fasta(), encoding="utf-8")
        io_cfg = SimpleNamespace(cache_fasta_dir=str(fasta_dir), cache_genbank_dir=str(genbank_dir))

        snapshot = _collect_bio_ast_transforms(accession=accession, source="fasta", io_cfg=io_cfg)

        assert snapshot["canonical_ast"]["schema"] == "bio_ast_canonical_document_v1"
        assert snapshot["canonical_ast"]["export_version"] == 1
        assert snapshot["canonical_ast"]["schema_version"] >= 3
        assert snapshot["tree_json"]["derived_from"]["schema"] == "bio_ast_canonical_document_v1"
        assert snapshot["graph_json"]["derived_from"]["canonical_sha256"] == snapshot["storage_map"]["derived_from"]["canonical_sha256"]
        assert snapshot["summary_json"] == {
            "schema": "bio_ast_summary_v1",
            "export_version": 1,
            "accession": accession,
            "sequence_length": 57,
            "canonical_sha256": snapshot["summary_json"]["canonical_sha256"],
            "node_count": snapshot["summary_json"]["node_count"],
            "edge_count": snapshot["summary_json"]["edge_count"],
            "semantic_edge_count": snapshot["summary_json"]["semantic_edge_count"],
            "root_ids": snapshot["summary_json"]["root_ids"],
            "node_type_counts": snapshot["summary_json"]["node_type_counts"],
            "relation_counts": snapshot["summary_json"]["relation_counts"],
        }

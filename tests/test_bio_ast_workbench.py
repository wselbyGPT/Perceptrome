import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from perceptrome.cli.commands import _build_and_write_bio_ast
from perceptrome.cli_main import build_parser
from tests.fixtures.bio_ast_regression_fixtures import render_fasta


def test_cli_parser_accepts_bio_ast_subcommands():
    parser = build_parser()
    args = parser.parse_args(["bio-ast", "build", "ACC1", "--source", "fasta"])
    assert args.command == "bio-ast"
    assert args.bio_ast_command == "build"
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
        expected_keys = {"canonical_ast", "motif_features", "tree_tensors", "graph_edges"}
        assert set(outputs.keys()) == expected_keys
        for out_path in outputs.values():
            assert Path(out_path).exists()

        canonical = json.loads(Path(outputs["canonical_ast"]).read_text(encoding="utf-8"))
        motif_features = json.loads(Path(outputs["motif_features"]).read_text(encoding="utf-8"))
        tree_tensors = json.loads(Path(outputs["tree_tensors"]).read_text(encoding="utf-8"))
        graph_edges = json.loads(Path(outputs["graph_edges"]).read_text(encoding="utf-8"))

        ast_nodes = canonical["nodes"]
        ast_node_ids = [node["canonical_id"] for node in ast_nodes]
        ast_node_id_set = set(ast_node_ids)

        assert tree_tensors["node_ids"] == ast_node_ids
        assert len(tree_tensors["node_type_ids"]) == len(ast_node_ids)
        assert len(tree_tensors["coords"]) == len(ast_node_ids)
        assert len(tree_tensors["strand"]) == len(ast_node_ids)

        edge_index = tree_tensors["edge_index"]
        assert len(edge_index) == 2
        assert len(edge_index[0]) == len(edge_index[1]) == len(graph_edges)

        for edge in graph_edges:
            assert edge["parent_id"] in ast_node_id_set
            assert edge["child_id"] in ast_node_id_set
            assert ast_node_ids[edge["parent_index"]] == edge["parent_id"]
            assert ast_node_ids[edge["child_index"]] == edge["child_id"]

        for row in motif_features:
            assert row["node_id"] in ast_node_id_set
            if row["parent_id"] is not None:
                assert row["parent_id"] in ast_node_id_set

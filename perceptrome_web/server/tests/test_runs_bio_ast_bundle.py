import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.deps import get_db
from app.main import app
from app.models import Run, RunArtifact, User
from app.security import hash_password
from perceptrome.bio_ast import BioAST, CDSNode, GeneNode, GenomeNode, ORFNode, RelationshipEdge
from perceptrome.encoding.bio_ast_export import export_filenames
from perceptrome.encoding.bio_ast_viz import ast_to_graph_json, ast_to_tree_json
from perceptrome.encoding.storage_map import build_storage_map_payload
from tests.db_utils import apply_migrations


def setup_client(tmp_path):
    db_path = tmp_path / "test_runs_bio_ast.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), testing_session


def sample_ast():
    root = GenomeNode(canonical_id="genome:ACC", start=1, end=120)
    gene = GeneNode(canonical_id="gene:ACC:1", gene_id="g_a", parent_id=root.canonical_id, start=10, end=50)
    orf = ORFNode(canonical_id="orf:ACC:1", parent_id=gene.canonical_id, start=10, end=50)
    cds = CDSNode(canonical_id="cds:ACC:1", parent_id=orf.canonical_id, start=10, end=50)
    rel = RelationshipEdge(source_id=gene.canonical_id, target_id=cds.canonical_id, kind="supports")
    return BioAST(nodes=(root, gene, orf, cds), relationships=(rel,))


def write_bundle(base_dir: Path, *, valid: bool = True):
    ast = sample_ast()
    filenames = export_filenames()
    canonical = ast.to_dict() | {"schema": "bio_ast_canonical_document_v1", "export_version": 1, "accession": "ACC"}
    tree = ast_to_tree_json(ast, accession="ACC") | {"export_version": 1}
    graph = ast_to_graph_json(ast, accession="ACC") | {"export_version": 1}
    storage_map = build_storage_map_payload(ast, 120, accession="ACC") | {"export_version": 1}
    if not valid:
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "cds:ACC:1"]
    summary = {"schema": "bio_ast_summary_v1", "canonical_sha256": "abc", "node_count": len(canonical["nodes"]), "edge_count": len(graph["edges"])}
    payloads = {
        filenames["canonical_ast"]: canonical,
        filenames["tree_json"]: tree,
        filenames["graph_json"]: graph,
        filenames["storage_map"]: storage_map,
        filenames["summary_json"]: summary,
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (base_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_run_bio_ast_bundle_endpoint_resolves_from_manifest_artifacts(tmp_path):
    client, db_factory = setup_client(tmp_path)
    bundle_dir = tmp_path / "artifacts" / "bio_ast" / "ACC"
    write_bundle(bundle_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"manifest_type": "run_manifest", "run": {"id": "run_test_1", "parents": [], "children": []}, "artifacts": [{"id": "acc-bundle", "path": str(bundle_dir / export_filenames()["graph_json"]), "metadata": {"accession": "ACC"}}]}), encoding="utf-8")

    with db_factory() as db:
        user = User(email="runs@example.com", password_hash=hash_password("Strongpass1234"), role="user", is_active=True, must_change_password=False, email_verified_at=datetime.utcnow())
        db.add(user)
        db.flush()
        run = Run(run_id="run_test_1", user_id=user.id, kind="generate_plasmid", state="completed", config_json="{}", result_json=json.dumps({"manifest_path": str(manifest_path)}), message="ok")
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(manifest_path), phase="manifest", label="manifest"))
        db.commit()

    assert client.post("/api/auth/login", json={"email": "runs@example.com", "password": "Strongpass1234"}).status_code == 200
    response = client.get("/api/runs/run_test_1/bio-ast?accession=ACC")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "bio_ast_visualization_bundle_v1"
    assert payload["accession"] == "ACC"
    assert payload["graph"]["node_count"] == 4
    assert payload["storage_map"]["coordinate_segments"][0]["node_id"] == "gene:ACC:1"


def test_run_bio_ast_bundle_endpoint_rejects_invalid_cross_payload_ids(tmp_path):
    client, db_factory = setup_client(tmp_path)
    bundle_dir = tmp_path / "artifacts" / "bio_ast" / "ACC"
    write_bundle(bundle_dir, valid=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"manifest_type": "run_manifest", "run": {"id": "run_test_2", "parents": [], "children": []}, "artifacts": [{"id": "acc-bundle", "path": str(bundle_dir / export_filenames()["graph_json"]), "metadata": {"accession": "ACC"}}]}), encoding="utf-8")

    with db_factory() as db:
        user = User(email="invalid@example.com", password_hash=hash_password("Strongpass1234"), role="user", is_active=True, must_change_password=False, email_verified_at=datetime.utcnow())
        db.add(user)
        db.flush()
        run = Run(run_id="run_test_2", user_id=user.id, kind="generate_plasmid", state="completed", config_json="{}", result_json=json.dumps({"manifest_path": str(manifest_path)}), message="ok")
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(manifest_path), phase="manifest", label="manifest"))
        db.commit()

    assert client.post("/api/auth/login", json={"email": "invalid@example.com", "password": "Strongpass1234"}).status_code == 200
    response = client.get("/api/runs/run_test_2/bio-ast?accession=ACC")
    assert response.status_code == 400
    assert response.json()["detail"] == "Bio-AST visualization bundle is invalid"

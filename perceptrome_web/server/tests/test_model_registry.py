from datetime import datetime
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.deps import get_db
from app.main import app
from app.models import Run, RunArtifact, User
from app.security import hash_password
from app.services import model_registry_service
from tests.db_utils import apply_migrations


def setup_db(tmp_path):
    db_path = tmp_path / "test_models.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")
    return testing_session


def setup_client(tmp_path):
    db_factory = setup_db(tmp_path)

    def override_get_db():
        db = db_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), db_factory


def _login_client(email: str, password: str = "Strongpass1234") -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return client


def test_register_completed_run_as_model_version(tmp_path):
    db_factory = setup_db(tmp_path)

    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "latest.pt"
    config_snapshot_path = tmp_path / "resolved_config.json"
    checkpoint_path.write_bytes(b"checkpoint")
    config_snapshot_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "model_objective_config": {"model_type": "mamba", "loss_type": "mse"},
                "tokenizer_encoding_config": {"tokenizer": "base", "window_size": 512},
                "metrics": {"last_total_loss": 0.25},
                "artifacts": [
                    {"id": "weights", "role": "checkpoint", "path": str(checkpoint_path)},
                    {"id": "config", "role": "provenance.config_snapshot", "path": str(config_snapshot_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    with db_factory() as db:
        user = User(
            email="models@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.flush()
        user_id = user.id
        run = Run(
            run_id="train_run_1",
            user_id=user.id,
            kind="stream",
            state="completed",
            config_json=json.dumps({"params": {"model_type": "mamba", "tokenizer": "base"}}),
            result_json=json.dumps({"manifest_path": str(manifest_path), "config_snapshot": {"path": str(config_snapshot_path)}}),
            message="ok",
        )
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(manifest_path), phase="manifest", label="Run manifest"))
        db.commit()

    with db_factory() as db:
        user = db.get(User, user_id)
        assert user is not None

        model = model_registry_service.register_from_run(
            db,
            user,
            run_id="train_run_1",
            model_id=None,
            name="Mamba DNA registry model",
            description=None,
            visibility="private",
            tags=["dna", "mamba"],
            version_label="v1",
            version_status="candidate",
        )

        assert model.name == "Mamba DNA registry model"
        assert model.tags == ["dna", "mamba"]
        assert len(model.versions) == 1
        version = model.versions[0]
        assert version.architecture == "mamba"
        assert version.tokenizer == "base"
        assert version.checkpoint_path == str(checkpoint_path)
        assert version.artifacts

        listing = model_registry_service.list_models(db, user)
        assert listing[0].id == model.id

        promoted = model_registry_service.update_version(db, user, model.id, version.id, {"promote_current": True})
        assert promoted.current_version_id == version.id
        assert promoted.versions[0].status == "stable"

        checkpoint_artifact = next(item for item in version.artifacts if item.role == "checkpoint")
        downloaded = model_registry_service.download_model_artifact(db, user, model.id, version.id, checkpoint_artifact.id)
        assert str(downloaded.path) == str(checkpoint_path)


def test_model_visibility_and_management_boundaries(tmp_path):
    db_factory = setup_db(tmp_path)

    manifest_path = tmp_path / "boundary_manifest.json"
    checkpoint_path = tmp_path / "boundary_latest.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path.write_text(
        json.dumps(
            {
                "model_objective_config": {"model_type": "transformer"},
                "tokenizer_encoding_config": {"tokenizer": "codon"},
                "artifacts": [{"id": "weights", "role": "checkpoint", "path": str(checkpoint_path)}],
            }
        ),
        encoding="utf-8",
    )

    with db_factory() as db:
        owner = User(
            email="owner@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        viewer = User(
            email="viewer@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        admin = User(
            email="admin@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="admin",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add_all([owner, viewer, admin])
        db.flush()
        run = Run(
            run_id="owner_train_run",
            user_id=owner.id,
            kind="stream",
            state="completed",
            config_json=json.dumps({"params": {"model_type": "transformer", "tokenizer": "codon"}}),
            result_json=json.dumps({"manifest_path": str(manifest_path)}),
            message="ok",
        )
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(manifest_path), phase="manifest", label="Run manifest"))
        db.commit()
        owner_id = owner.id
        viewer_id = viewer.id
        admin_id = admin.id

    with db_factory() as db:
        owner = db.get(User, owner_id)
        viewer = db.get(User, viewer_id)
        admin = db.get(User, admin_id)
        assert owner is not None and viewer is not None and admin is not None

        private_model = model_registry_service.register_from_run(
            db,
            owner,
            run_id="owner_train_run",
            model_id=None,
            name="Owner private model",
            description=None,
            visibility="private",
            tags=["boundary"],
            version_label="v1",
            version_status="candidate",
        )

        assert model_registry_service.list_models(db, owner)[0].id == private_model.id
        assert model_registry_service.list_models(db, viewer) == []

        with pytest.raises(HTTPException) as forbidden_private:
            model_registry_service.update_model(db, viewer, private_model.id, {"status": "archived"})
        assert forbidden_private.value.status_code == 403

        team_model = model_registry_service.update_model(db, owner, private_model.id, {"visibility": "team"})
        viewer_models = model_registry_service.list_models(db, viewer)
        assert [model.id for model in viewer_models] == [team_model.id]
        assert viewer_models[0].visibility == "team"

        with pytest.raises(HTTPException) as forbidden_team:
            model_registry_service.update_model(db, viewer, team_model.id, {"status": "archived"})
        assert forbidden_team.value.status_code == 403

        admin_updated = model_registry_service.update_model(db, admin, team_model.id, {"status": "archived"})
        assert admin_updated.status == "archived"


def test_model_registry_api_enforces_visibility_and_owner_management(tmp_path):
    _, db_factory = setup_client(tmp_path)

    manifest_path = tmp_path / "api_manifest.json"
    checkpoint_path = tmp_path / "api_latest.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path.write_text(
        json.dumps(
            {
                "model_objective_config": {"model_type": "mamba"},
                "tokenizer_encoding_config": {"tokenizer": "base"},
                "artifacts": [{"id": "weights", "role": "checkpoint", "path": str(checkpoint_path)}],
            }
        ),
        encoding="utf-8",
    )

    with db_factory() as db:
        owner = User(
            email="api-owner@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        viewer = User(
            email="api-viewer@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        admin = User(
            email="api-admin@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="admin",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add_all([owner, viewer, admin])
        db.flush()
        run = Run(
            run_id="api_owner_train_run",
            user_id=owner.id,
            kind="stream",
            state="completed",
            config_json=json.dumps({"params": {"model_type": "mamba", "tokenizer": "base"}}),
            result_json=json.dumps({"manifest_path": str(manifest_path)}),
            message="ok",
        )
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(manifest_path), phase="manifest", label="Run manifest"))
        db.commit()

    owner_client = _login_client("api-owner@example.com")
    viewer_client = _login_client("api-viewer@example.com")
    admin_client = _login_client("api-admin@example.com")

    created = owner_client.post(
        "/api/models/register-from-run",
        json={
            "run_id": "api_owner_train_run",
            "name": "API private model",
            "visibility": "private",
            "version_label": "v1",
        },
    )
    assert created.status_code == 200
    model_id = created.json()["id"]

    assert viewer_client.get("/api/models").json() == []
    assert viewer_client.get(f"/api/models/{model_id}").status_code == 403
    assert viewer_client.patch(f"/api/models/{model_id}", json={"status": "archived"}).status_code == 403

    owner_update = owner_client.patch(f"/api/models/{model_id}", json={"visibility": "team"})
    assert owner_update.status_code == 200
    assert owner_update.json()["visibility"] == "team"

    viewer_listing = viewer_client.get("/api/models")
    assert viewer_listing.status_code == 200
    assert [model["id"] for model in viewer_listing.json()] == [model_id]
    assert viewer_client.patch(f"/api/models/{model_id}", json={"status": "archived"}).status_code == 403

    admin_update = admin_client.patch(f"/api/models/{model_id}", json={"status": "archived"})
    assert admin_update.status_code == 200
    assert admin_update.json()["status"] == "archived"

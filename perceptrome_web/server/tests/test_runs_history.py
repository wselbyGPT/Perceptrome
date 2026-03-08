from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.deps import get_db
from app.main import app
from app.models import Run, RunArtifact, User
from app.security import hash_password


def setup_client(tmp_path):
    db_path = tmp_path / "test_runs.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    return client, testing_session


def test_run_history_and_artifact_download(tmp_path):
    client, db_factory = setup_client(tmp_path)

    with db_factory() as db:
        user = User(
            email="runs@example.com",
            password_hash=hash_password("Strongpass1234"),
            role="user",
            is_active=True,
            must_change_password=False,
            email_verified_at=datetime.utcnow(),
        )
        db.add(user)
        db.flush()

        artifact_file = tmp_path / "artifact.txt"
        artifact_file.write_text("ok", encoding="utf-8")

        run = Run(
            run_id="run_test_1",
            user_id=user.id,
            kind="generate_plasmid",
            state="completed",
            config_json="{}",
            result_json="{}",
            message="ok",
        )
        db.add(run)
        db.flush()
        db.add(RunArtifact(run_id=run.id, path=str(artifact_file), phase="manifest", label="artifact"))
        db.commit()

    login = client.post("/api/auth/login", json={"email": "runs@example.com", "password": "Strongpass1234"})
    assert login.status_code == 200

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    data = runs.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "run_test_1"
    assert data[0]["artifacts"]

    artifact_url = data[0]["artifacts"][0]["download_url"]
    dl = client.get(artifact_url)
    assert dl.status_code == 200
    assert dl.text == "ok"

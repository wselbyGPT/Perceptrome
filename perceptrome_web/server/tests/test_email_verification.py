from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.deps import get_db
from app.main import app
from tests.db_utils import apply_migrations
from app.models import AuthToken, User
from app import main as main_module


def setup_client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module.settings, "allow_self_register", True)
    monkeypatch.setattr(main_module.settings, "email_verification_resend_cooldown_seconds", 0)

    issued_tokens = []

    def fake_send_verification_email(_recipient: str, raw_token: str):
        issued_tokens.append(raw_token)

    monkeypatch.setattr(main_module, "_send_verification_email", fake_send_verification_email)

    client = TestClient(app)
    return client, TestingSessionLocal, issued_tokens


def test_token_expiry_blocks_verification(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)
    monkeypatch.setattr(main_module.settings, "email_verification_token_ttl_minutes", 0)

    r = client.post("/api/auth/register", json={"email": "expired@example.com", "password": "verysecure123"})
    assert r.status_code == 200
    assert issued_tokens

    verify = client.post("/api/auth/verify-email", json={"token": issued_tokens[-1]})
    assert verify.status_code == 400
    assert verify.json()["detail"] == "Verification token expired"


def test_replay_prevention(monkeypatch, tmp_path):
    client, _db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    r = client.post("/api/auth/register", json={"email": "replay@example.com", "password": "verysecure123"})
    assert r.status_code == 200

    token = issued_tokens[-1]
    verify1 = client.post("/api/auth/verify-email", json={"token": token})
    assert verify1.status_code == 200

    verify2 = client.post("/api/auth/verify-email", json={"token": token})
    assert verify2.status_code == 400
    assert verify2.json()["detail"] == "Verification token already used"


def test_already_verified_resend_behavior(monkeypatch, tmp_path):
    client, db_factory, issued_tokens = setup_client(monkeypatch, tmp_path)

    r = client.post("/api/auth/register", json={"email": "verified@example.com", "password": "verysecure123"})
    assert r.status_code == 200

    token = issued_tokens[-1]
    verify = client.post("/api/auth/verify-email", json={"token": token})
    assert verify.status_code == 200

    resend = client.post("/api/auth/resend-verification", json={"email": "verified@example.com"})
    assert resend.status_code == 200
    assert resend.json()["message"] == "Email already verified"

    with db_factory() as db:
        user = db.execute(select(User).where(User.email == "verified@example.com")).scalar_one()
        assert user.email_verified_at is not None
        tokens = db.execute(select(AuthToken).where(AuthToken.user_id == user.id).where(AuthToken.purpose == "email_verification")).scalars().all()
        assert len(tokens) == 1
        assert tokens[0].used_at is not None

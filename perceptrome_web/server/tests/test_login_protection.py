from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import main as main_module
from app.auth_rate_limit import LoginAttemptStore
from app.db import Base
from app.deps import get_db
from app.main import app


class FakeClock:
    def __init__(self):
        self.current = datetime(2024, 1, 1, 0, 0, 0)

    def now(self):
        return self.current

    def advance(self, seconds: int):
        self.current = self.current + timedelta(seconds=seconds)


def setup_client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module.settings, "allow_self_register", True)
    monkeypatch.setattr(main_module.settings, "login_attempt_store", "db")
    monkeypatch.setattr(main_module.settings, "redis_url", None)
    monkeypatch.setattr(main_module.settings, "login_rate_limit_window_seconds", 5)
    monkeypatch.setattr(main_module.settings, "login_rate_limit_max_attempts", 2)
    monkeypatch.setattr(main_module.settings, "login_rate_limit_ip_max_attempts", 3)
    monkeypatch.setattr(main_module.settings, "login_lockout_threshold", 3)
    monkeypatch.setattr(main_module.settings, "login_lockout_seconds", 20)
    monkeypatch.setattr(main_module.settings, "login_backoff_base_seconds", 1)
    monkeypatch.setattr(main_module.settings, "login_backoff_max_seconds", 2)
    monkeypatch.setattr(main_module.settings, "email_verification_resend_cooldown_seconds", 0)

    monkeypatch.setattr(main_module, "login_attempt_store", LoginAttemptStore())
    monkeypatch.setattr(main_module, "_send_verification_email", lambda _recipient, _token: None)

    clock = FakeClock()
    monkeypatch.setattr(main_module, "_utcnow", clock.now)

    client = TestClient(app)
    return client, clock, TestingSessionLocal


def test_window_rollover_allows_retry_after_window(monkeypatch, tmp_path):
    client, clock, _ = setup_client(monkeypatch, tmp_path)

    register = client.post("/api/auth/register", json={"email": "window@example.com", "password": "verysecure123"})
    assert register.status_code == 200

    for _ in range(2):
        bad = client.post("/api/auth/login", json={"email": "window@example.com", "password": "wrong"})
        assert bad.status_code == 401

    limited = client.post("/api/auth/login", json={"email": "window@example.com", "password": "wrong"})
    assert limited.status_code == 429
    assert limited.json()["detail"]["reason"] == "rate_limit_ip_email"

    clock.advance(6)
    allowed = client.post("/api/auth/login", json={"email": "window@example.com", "password": "wrong"})
    assert allowed.status_code == 401


def test_db_store_is_visible_across_instances(tmp_path):
    db_path = tmp_path / "shared.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=engine)

    now = datetime(2024, 1, 1, 0, 0, 0)
    store_a = LoginAttemptStore()
    store_b = LoginAttemptStore()

    from app.config import settings

    settings.login_attempt_store = "db"
    settings.login_rate_limit_window_seconds = 60
    settings.login_rate_limit_max_attempts = 2
    settings.login_rate_limit_ip_max_attempts = 10

    with TestingSessionLocal() as db:
        assert store_a.check_and_record(db=db, ip="1.2.3.4", email="a@a.com", now=now).limited is False
    with TestingSessionLocal() as db:
        assert store_b.check_and_record(db=db, ip="1.2.3.4", email="a@a.com", now=now).limited is False
    with TestingSessionLocal() as db:
        assert store_b.check_and_record(db=db, ip="1.2.3.4", email="a@a.com", now=now).limited is True


def test_unlock_behavior_and_reset_on_success(monkeypatch, tmp_path):
    client, clock, db_factory = setup_client(monkeypatch, tmp_path)

    register = client.post("/api/auth/register", json={"email": "unlock@example.com", "password": "verysecure123"})
    assert register.status_code == 200

    from app.models import User
    from sqlalchemy import select

    with db_factory() as db:
        user = db.execute(select(User).where(User.email == "unlock@example.com")).scalar_one()
        user.email_verified_at = clock.now()
        db.commit()

    for _ in range(3):
        client.post("/api/auth/login", json={"email": "unlock@example.com", "password": "wrong"})

    locked = client.post("/api/auth/login", json={"email": "unlock@example.com", "password": "verysecure123"})
    assert locked.status_code == 429
    assert locked.json()["detail"]["reason"] in {"user_locked", "rate_limit_ip_email", "rate_limit_ip"}

    clock.advance(25)
    ok = client.post("/api/auth/login", json={"email": "unlock@example.com", "password": "verysecure123"})
    assert ok.status_code == 200

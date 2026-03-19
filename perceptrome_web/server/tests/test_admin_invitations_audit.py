from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import main as main_module
from app.deps import get_db
from app.main import app
from app.models import AuditEvent, User, UserInvitation, UserSession
from app.services import audit_service
from tests.db_utils import apply_migrations


def setup_client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True)
    testing_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    apply_migrations(f"sqlite:///{db_path}")

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module.settings, "allow_self_register", True)
    monkeypatch.setattr(main_module.settings, "email_verification_resend_cooldown_seconds", 0)
    monkeypatch.setattr(main_module.settings, "invitation_ttl_hours", 48)
    monkeypatch.setattr(main_module, "_send_verification_email", lambda _recipient, _token: None)

    client = TestClient(app)
    return client, testing_session_local


def create_admin_and_login(client: TestClient, db_factory):
    register = client.post("/api/auth/register", json={"email": "admin@example.com", "password": "verysecure123", "username": "adminuser"})
    assert register.status_code == 200

    with db_factory() as db:
        admin = db.execute(select(User).where(User.email == "admin@example.com")).scalar_one()
        admin.role = "admin"
        admin.email_verified_at = datetime.utcnow()
        db.commit()

    login = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "verysecure123"})
    assert login.status_code == 200


def test_admin_invitation_lifecycle_and_audit(monkeypatch, tmp_path):
    client, db_factory = setup_client(monkeypatch, tmp_path)
    create_admin_and_login(client, db_factory)

    created = client.post("/api/admin/invitations", json={"email": "invitee@example.com", "role": "admin"})
    assert created.status_code == 200
    body = created.json()
    assert body["email"] == "invitee@example.com"
    assert body["status"] == "pending"
    assert body["token_preview"]
    invitation_id = body["id"]

    listed = client.get("/api/admin/invitations", params={"status": "pending"})
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] == 1
    assert payload["invitations"][0]["id"] == invitation_id

    reissued = client.post("/api/admin/invitations", json={"email": "invitee@example.com", "role": "user", "reissue": True})
    assert reissued.status_code == 200
    assert reissued.json()["role"] == "user"

    with db_factory() as db:
        invitations = db.execute(select(UserInvitation).where(UserInvitation.email == "invitee@example.com").order_by(UserInvitation.created_at.asc())).scalars().all()
        assert len(invitations) == 2
        assert invitations[0].revoked_at is not None
        invitations[1].accepted_at = audit_service.utcnow()
        expired = UserInvitation(email="expired@example.com", role="user", invited_by_user_id=invitations[1].invited_by_user_id, token_hash="expired-hash", expires_at=datetime.utcnow() - timedelta(hours=1))
        db.add(expired)
        db.commit()
        active_invitation = invitations[1]

    revoke = client.post(f"/api/admin/invitations/{active_invitation.id}/revoke")
    assert revoke.status_code == 409

    expired_list = client.get("/api/admin/invitations", params={"status": "expired"})
    assert expired_list.status_code == 200
    assert expired_list.json()["total"] == 1

    audit = client.get("/api/admin/audit", params={"search": "invitee@example.com"})
    assert audit.status_code == 200
    audit_payload = audit.json()
    actions = [event["action"] for event in audit_payload["events"]]
    assert audit_service.AuditActions.INVITE_CREATED in actions


def test_admin_user_actions_emit_audit_events(monkeypatch, tmp_path):
    client, db_factory = setup_client(monkeypatch, tmp_path)
    create_admin_and_login(client, db_factory)

    created = client.post("/api/admin/users", json={"email": "member@example.com", "password": "temporary1234", "username": "member1", "role": "user", "is_active": True, "must_change_password": False})
    assert created.status_code == 200
    user_id = created.json()["id"]

    with db_factory() as db:
        user = db.get(User, user_id)
        user.email_verified_at = None
        db.add(UserSession(user_id=user.id, token_hash="session-a", expires_at=datetime.utcnow() + timedelta(hours=1)))
        db.commit()

    update = client.patch(f"/api/admin/users/{user_id}", json={"role": "admin", "username": "member2", "must_change_password": True})
    assert update.status_code == 200
    suspend = client.post(f"/api/admin/users/{user_id}/suspend")
    assert suspend.status_code == 200
    activate = client.post(f"/api/admin/users/{user_id}/activate")
    assert activate.status_code == 200
    force_reset = client.post(f"/api/admin/users/{user_id}/force-reset")
    assert force_reset.status_code == 200
    resend = client.post(f"/api/admin/users/{user_id}/resend-verification")
    assert resend.status_code == 200
    revoke_sessions = client.post(f"/api/admin/users/{user_id}/revoke-sessions")
    assert revoke_sessions.status_code == 200

    audit = client.get("/api/admin/audit", params={"target": user_id})
    assert audit.status_code == 200
    actions = {event["action"] for event in audit.json()["events"]}
    assert audit_service.AuditActions.USER_CREATED in actions
    assert audit_service.AuditActions.USER_UPDATED in actions
    assert audit_service.AuditActions.ROLE_CHANGED in actions
    assert audit_service.AuditActions.USER_SUSPENDED in actions
    assert audit_service.AuditActions.USER_ACTIVATED in actions
    assert audit_service.AuditActions.PASSWORD_RESET_FORCED in actions
    assert audit_service.AuditActions.VERIFICATION_RESENT in actions
    assert audit_service.AuditActions.SESSION_REVOKED in actions

    with db_factory() as db:
        assert db.execute(select(AuditEvent)).scalars().all()

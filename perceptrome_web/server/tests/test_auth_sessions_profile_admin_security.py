from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import main as main_module
from app.deps import get_db
from app.main import app
from app.models import AuditEvent, User, UserInvitation, UserSession
from app.security import hash_session_token
from app.services import audit_service
from tests.db_utils import apply_migrations


class CapturedEmails:
    def __init__(self):
        self.verification: list[tuple[str, str]] = []

    def send_verification(self, recipient: str, token: str) -> None:
        self.verification.append((recipient, token))


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

    emails = CapturedEmails()
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module.settings, "allow_self_register", True)
    monkeypatch.setattr(main_module.settings, "email_verification_resend_cooldown_seconds", 0)
    monkeypatch.setattr(main_module.settings, "invitation_ttl_hours", 48)
    monkeypatch.setattr(main_module, "_send_verification_email", emails.send_verification)

    client = TestClient(app)
    return client, testing_session_local, emails


def register_verify_and_login(client: TestClient, db_factory, *, email: str, password: str, username: str | None = None):
    payload = {"email": email, "password": password}
    if username is not None:
        payload["username"] = username
    register = client.post("/api/auth/register", json=payload)
    assert register.status_code == 200

    with db_factory() as db:
        user = db.execute(select(User).where(User.email == email)).scalar_one()
        user.email_verified_at = datetime.utcnow()
        db.commit()
        user_id = user.id

    login = client.post("/api/auth/login", json={"email": email, "password": password}, headers={"user-agent": f"agent-{email}"})
    assert login.status_code == 200
    return user_id


def create_admin_and_login(client: TestClient, db_factory):
    admin_id = register_verify_and_login(
        client,
        db_factory,
        email="admin@example.com",
        password="verysecure123",
        username="adminuser",
    )
    with db_factory() as db:
        admin = db.get(User, admin_id)
        admin.role = "admin"
        db.commit()
    return admin_id


def test_session_listing_revocations_profile_update_and_auth_audit(monkeypatch, tmp_path):
    client, db_factory, _ = setup_client(monkeypatch, tmp_path)
    user_id = register_verify_and_login(
        client,
        db_factory,
        email="user@example.com",
        password="verysecure123",
        username="session-user",
    )

    current_raw = client.cookies.get(main_module.settings.session_cookie_name)
    assert current_raw

    with db_factory() as db:
        current_session = db.execute(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .where(UserSession.token_hash == hash_session_token(current_raw))
        ).scalar_one()
        current_session.created_at = datetime.utcnow() - timedelta(minutes=5)

        other_session = UserSession(
            user_id=user_id,
            token_hash=hash_session_token("other-session-token"),
            created_at=datetime.utcnow() - timedelta(minutes=1),
            expires_at=datetime.utcnow() + timedelta(hours=2),
            ip_address="10.0.0.2",
            user_agent="other-browser",
        )
        expired_session = UserSession(
            user_id=user_id,
            token_hash=hash_session_token("expired-session-token"),
            created_at=datetime.utcnow() - timedelta(minutes=10),
            expires_at=datetime.utcnow() - timedelta(minutes=1),
            ip_address="10.0.0.3",
            user_agent="expired-browser",
        )
        db.add_all([other_session, expired_session])
        db.commit()
        db.refresh(current_session)
        db.refresh(other_session)
        db.refresh(expired_session)
        other_session_id = other_session.id
        current_session_id = current_session.id

    listed = client.get("/api/auth/sessions")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert [session["id"] for session in listed_payload] == [other_session_id, current_session_id, expired_session.id]
    current_item = next(session for session in listed_payload if session["id"] == current_session_id)
    assert current_item["is_current"] is True
    assert current_item["user_agent"] == "agent-user@example.com"
    other_item = next(session for session in listed_payload if session["id"] == other_session_id)
    assert other_item["is_current"] is False
    assert other_item["ip_address"] == "10.0.0.2"

    revoke_one = client.delete(f"/api/auth/sessions/{other_session_id}")
    assert revoke_one.status_code == 200
    assert revoke_one.json() == {"message": "Session revoked"}

    with db_factory() as db:
        revoked_other = db.get(UserSession, other_session_id)
        still_current = db.get(UserSession, current_session_id)
        assert revoked_other.revoked_at is not None
        assert still_current.revoked_at is None

    with db_factory() as db:
        db.add(
            UserSession(
                user_id=user_id,
                token_hash=hash_session_token("fresh-other-session"),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=1),
                ip_address="10.0.0.4",
                user_agent="fresh-browser",
            )
        )
        db.commit()

    revoke_others = client.post("/api/auth/sessions/revoke-others")
    assert revoke_others.status_code == 200
    assert revoke_others.json() == {"message": "Revoked 1 other sessions"}

    updated = client.patch("/api/auth/profile", json={"username": "renamed-user"})
    assert updated.status_code == 200
    assert updated.json()["user"]["username"] == "renamed-user"

    with db_factory() as db:
        sessions = db.execute(select(UserSession).where(UserSession.user_id == user_id)).scalars().all()
        current_db_session = next(session for session in sessions if session.id == current_session_id)
        fresh_other = next(session for session in sessions if session.user_agent == "fresh-browser")
        expired_db_session = next(session for session in sessions if session.user_agent == "expired-browser")
        assert current_db_session.revoked_at is None
        assert fresh_other.revoked_at is not None
        assert expired_db_session.revoked_at is None

        events = db.execute(
            select(AuditEvent)
            .where(AuditEvent.actor_user_id == user_id)
            .order_by(AuditEvent.created_at.asc())
        ).scalars().all()
        actions = [event.action for event in events]
        assert audit_service.AuditActions.AUTH_SESSION_REVOKED in actions
        assert audit_service.AuditActions.AUTH_OTHER_SESSIONS_REVOKED in actions
        assert audit_service.AuditActions.PROFILE_UPDATED in actions

        revoke_event = next(event for event in events if event.action == audit_service.AuditActions.AUTH_SESSION_REVOKED)
        assert audit_service._loads(revoke_event.metadata_json)["session_id"] == other_session_id
        assert audit_service._loads(revoke_event.metadata_json)["revoked_current_session"] is False

        revoke_others_event = next(event for event in events if event.action == audit_service.AuditActions.AUTH_OTHER_SESSIONS_REVOKED)
        assert audit_service._loads(revoke_others_event.metadata_json)["revoked_session_count"] == 1

        profile_event = next(event for event in events if event.action == audit_service.AuditActions.PROFILE_UPDATED)
        assert audit_service._loads(profile_event.metadata_json) == {
            "previous_username": "session-user",
            "new_username": "renamed-user",
        }


def test_admin_actions_invitation_audit_and_role_enforcement(monkeypatch, tmp_path):
    client, db_factory, emails = setup_client(monkeypatch, tmp_path)
    admin_id = create_admin_and_login(client, db_factory)

    created = client.post(
        "/api/admin/users",
        json={
            "email": "member@example.com",
            "password": "temporary1234",
            "username": "member1",
            "role": "user",
            "is_active": True,
            "must_change_password": False,
        },
    )
    assert created.status_code == 200
    user_id = created.json()["id"]

    with db_factory() as db:
        user = db.get(User, user_id)
        user.email_verified_at = None
        db.add_all(
            [
                UserSession(user_id=user.id, token_hash="session-a", expires_at=datetime.utcnow() + timedelta(hours=1)),
                UserSession(user_id=user.id, token_hash="session-b", expires_at=datetime.utcnow() + timedelta(hours=1)),
            ]
        )
        db.commit()

    listed = client.get("/api/admin/users", params={"search": "member", "state": "active", "verification": "pending"})
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    edited = client.patch(
        f"/api/admin/users/{user_id}",
        json={"username": "member-renamed", "role": "admin", "must_change_password": True},
    )
    assert edited.status_code == 200
    assert edited.json()["username"] == "member-renamed"
    assert edited.json()["role"] == "admin"

    suspended = client.post(f"/api/admin/users/{user_id}/suspend")
    assert suspended.status_code == 200
    assert suspended.json()["revoked_session_count"] == 2

    activated = client.post(f"/api/admin/users/{user_id}/activate")
    assert activated.status_code == 200
    assert activated.json()["user"]["is_active"] is True

    forced = client.post(f"/api/admin/users/{user_id}/force-reset")
    assert forced.status_code == 200
    assert forced.json()["user"]["must_change_password"] is True

    resent = client.post(f"/api/admin/users/{user_id}/resend-verification")
    assert resent.status_code == 200
    assert resent.json()["message"] == "Verification email resent"
    assert emails.verification[-1][0] == "member@example.com"

    invite_created = client.post("/api/admin/invitations", json={"email": "invitee@example.com", "role": "user"})
    assert invite_created.status_code == 200
    invitation_id = invite_created.json()["id"]

    invite_list = client.get("/api/admin/invitations", params={"status": "pending", "role": "user", "search": "invitee"})
    assert invite_list.status_code == 200
    assert invite_list.json()["total"] == 1
    assert invite_list.json()["invitations"][0]["id"] == invitation_id

    invite_revoke = client.post(f"/api/admin/invitations/{invitation_id}/revoke")
    assert invite_revoke.status_code == 200
    assert invite_revoke.json()["invitation"]["status"] == "revoked"

    revoke_sessions = client.post(f"/api/admin/users/{user_id}/revoke-sessions")
    assert revoke_sessions.status_code == 200
    assert revoke_sessions.json()["revoked_session_count"] == 0

    with db_factory() as db:
        events = db.execute(
            select(AuditEvent)
            .where(AuditEvent.actor_user_id == admin_id)
            .order_by(AuditEvent.created_at.asc())
        ).scalars().all()
        event_by_action = {event.action: audit_service._loads(event.metadata_json) for event in events}

        assert audit_service.AuditActions.USER_CREATED in event_by_action
        assert audit_service.AuditActions.USER_UPDATED in event_by_action
        assert audit_service.AuditActions.ROLE_CHANGED in event_by_action
        assert audit_service.AuditActions.USER_SUSPENDED in event_by_action
        assert audit_service.AuditActions.USER_ACTIVATED in event_by_action
        assert audit_service.AuditActions.PASSWORD_RESET_FORCED in event_by_action
        assert audit_service.AuditActions.VERIFICATION_RESENT in event_by_action
        assert audit_service.AuditActions.SESSION_REVOKED in event_by_action
        assert audit_service.AuditActions.INVITE_CREATED in event_by_action
        assert audit_service.AuditActions.INVITE_REVOKED in event_by_action

        assert event_by_action[audit_service.AuditActions.USER_UPDATED]["username"] == {"from": "member1", "to": "member-renamed"}
        assert event_by_action[audit_service.AuditActions.ROLE_CHANGED] == {"from": "user", "to": "admin"}
        assert event_by_action[audit_service.AuditActions.USER_SUSPENDED]["revoked_session_count"] == 2
        assert event_by_action[audit_service.AuditActions.PASSWORD_RESET_FORCED]["revoked_session_count"] == 0
        assert event_by_action[audit_service.AuditActions.SESSION_REVOKED]["revoked_session_count"] == 0
        assert event_by_action[audit_service.AuditActions.INVITE_CREATED]["email"] == "invitee@example.com"
        assert event_by_action[audit_service.AuditActions.INVITE_REVOKED]["invitation_id"] == invitation_id

        invitation = db.get(UserInvitation, invitation_id)
        assert invitation.revoked_at is not None

    client.post("/api/auth/logout")
    unauthenticated = client.get("/api/admin/users")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["detail"] == "Not authenticated"

    user_client, user_db_factory, _ = setup_client(monkeypatch, tmp_path / "non_admin")
    register_verify_and_login(
        user_client,
        user_db_factory,
        email="plain-user@example.com",
        password="verysecure123",
        username="plainuser",
    )
    forbidden = user_client.get("/api/admin/users")
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"] == "Forbidden"

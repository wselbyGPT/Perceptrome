from datetime import datetime

from sqlalchemy import select

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import all_routers
from .auth_rate_limit import login_attempt_store
from .core.config import settings
from .core.db import SessionLocal
from .models import User
from .services import auth_service, audit_service

app = FastAPI(title=settings.app_name)

origins = [o.strip() for o in settings.cors_origins.split(',') if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], allow_headers=['*'])

_utcnow = datetime.utcnow
_send_verification_email = auth_service.send_verification_email
_send_password_reset_email = auth_service.send_password_reset_email



def _bootstrap_admin():
    if not settings.bootstrap_admin_email or not settings.bootstrap_admin_password:
        return
    db = SessionLocal()
    try:
        email = settings.bootstrap_admin_email.lower().strip()
        existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if existing:
            return
        from .core.security import hash_password
        admin = User(email=email, password_hash=hash_password(settings.bootstrap_admin_password), role='admin', is_active=True, must_change_password=True, email_verified_at=_utcnow())
        db.add(admin)
        db.commit()
        print(f'[auth] bootstrapped admin user: {admin.email}')
    finally:
        db.close()


@app.on_event('startup')
def on_startup():
    _bootstrap_admin()


@app.get('/api/health')
def health():
    return {'ok': True, 'service': 'perceptrome-api'}


for router in all_routers:
    app.include_router(router)

from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.routes import all_routers
from .auth_rate_limit import login_attempt_store
from .core.config import settings
from .core.db import SessionLocal
from .core.security import hash_password
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
        admin = User(email=email, password_hash=hash_password(settings.bootstrap_admin_password), role='admin', is_active=True, must_change_password=True, email_verified_at=_utcnow())
        db.add(admin)
        try:
            db.commit()
            print(f'[auth] bootstrapped admin user: {admin.email}')
        except IntegrityError:
            # Another worker won the race on first boot; the admin already
            # exists. Safe to ignore.
            db.rollback()
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

# In production, serve the built Vite SPA from the same process.
# The client/dist directory is baked into the Docker image.
_client_dist = Path(__file__).resolve().parent.parent.parent / "client" / "dist"
_index_html = _client_dist / "index.html"
if _client_dist.is_dir() and _index_html.is_file():
    # StaticFiles(html=True) handles static assets and root index, but does not
    # fall back to index.html for arbitrary client-side routes like /login or
    # /dashboard on reload. Catch 404s on non-API GETs and serve index.html so
    # react-router can take over.
    @app.exception_handler(StarletteHTTPException)
    async def _spa_fallback(request: Request, exc: StarletteHTTPException):
        if (
            exc.status_code == 404
            and request.method == "GET"
            and not request.url.path.startswith("/api")
            and not request.url.path.startswith("/ws")
            and "text/html" in request.headers.get("accept", "")
        ):
            return FileResponse(_index_html)
        from fastapi.exception_handlers import http_exception_handler
        return await http_exception_handler(request, exc)

    app.mount("/", StaticFiles(directory=_client_dist, html=True), name="spa")

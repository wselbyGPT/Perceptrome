# Perceptrome Web

Perceptrome Web is the repository's browser-based control plane: a Vite/React single-page app in `perceptrome_web/client/` backed by a FastAPI API in `perceptrome_web/server/`. The production-oriented architecture is **PostgreSQL-first**. Run the Alembic migration chain before starting the API, and treat SQLite as a convenience mode for local experiments or automated tests rather than the primary deployment target.

## Architecture at a glance

- **Client:** React 19 + Vite SPA in `perceptrome_web/client/`.
- **Server:** FastAPI app in `perceptrome_web/server/app/`.
- **Database:** `DATABASE_URL` defaults to PostgreSQL, and Alembic is the supported schema-management path.
- **Realtime runs:** the runs UI uses an authenticated WebSocket at `/ws`; the browser must carry the session cookie used by the REST API.

## Quick start

From the repository root, bootstrap the Python server stack and the Vite client with dedicated install paths instead of the old monolithic root requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/web.txt
npm install --prefix perceptrome_web/client
```

Or use the Makefile shortcut:

```bash
make setup-web
```

## Local development workflow

### 1) Configure the database

Production and serious local development should use PostgreSQL.

Create a database and application user, then export `DATABASE_URL`:

```bash
createdb perceptrome
createuser perceptrome --pwprompt
export DATABASE_URL='postgresql+psycopg://perceptrome:YOUR_PASSWORD@localhost:5432/perceptrome'
```

### 2) Apply Alembic migrations

Run this from the repository root:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini upgrade head
```

Or with Make:

```bash
make web-migrate
```

### 3) Start the API server

```bash
cd perceptrome_web/server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with Make:

```bash
make web-api
```

### 4) Start the Vite dev server

```bash
npm run dev --prefix perceptrome_web/client
```

Or with Make:

```bash
make web-client
```

Then open `http://127.0.0.1:5173`.

## Why the install flow changed

The repository now separates Python dependency paths so web-only work does not pull in unrelated optional stacks such as CUDA packages:

- `requirements/core.txt` → lean CLI/core install
- `requirements/web.txt` → FastAPI/Alembic/PostgreSQL stack
- `requirements/dev.txt` → combined local-development Python stack
- `requirements/gpu-cu12.txt` → optional CUDA 12 / GPU training packages

`requirements.txt` remains only as a compatibility shim for legacy GPU-oriented setups.

## Client notes

- The Vite dev server is pinned to `http://127.0.0.1:5173` / `http://localhost:5173`.
- `vite.config.ts` proxies both `/api` and `/ws` to `http://127.0.0.1:8000`, so the SPA can talk to the local FastAPI backend without changing frontend code.
- The frontend uses same-origin relative URLs such as `/api/auth/me` and `/ws`, so the proxy is the normal local-development path.

## SQLite convenience mode

SQLite is still useful for quick local smoke tests and for automated test coverage, but it is **not** the primary architecture. If you intentionally want SQLite for dev-only work, override `DATABASE_URL` yourself:

```bash
export DATABASE_URL='sqlite:///./perceptrome_dev.db'
python -m alembic -c perceptrome_web/server/alembic.ini upgrade head
```

`LOCAL_SQLITE_DATABASE_URL=sqlite:///./perceptrome_dev.db` exists as a convenience/documentation setting so local SQLite is explicit, not accidental.

## Alembic migration commands

Run these from the repository root unless noted otherwise.

Apply the full schema:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini upgrade head
```

Check the current revision:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini current
```

See migration history:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini history
```

Create a new migration after model changes:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini revision --autogenerate -m "describe change"
```

Downgrade one step if needed:

```bash
python -m alembic -c perceptrome_web/server/alembic.ini downgrade -1
```

Because production is Postgres-first, schema changes should always be captured as Alembic revisions and applied before the API is deployed. Migrations are mandatory, not optional housekeeping.

## Environment variables

The server reads `.env` from `perceptrome_web/server/.env`. The most important settings are:

### Core runtime

```env
APP_NAME=Perceptrome API
APP_ENV=dev
DATABASE_URL=postgresql+psycopg://perceptrome:perceptrome@localhost:5432/perceptrome
LOCAL_SQLITE_DATABASE_URL=sqlite:///./perceptrome_dev.db
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://perceptrome.com
```

### Session / cookie behavior

```env
SESSION_COOKIE_NAME=perceptrome_session
SESSION_TTL_HOURS=168
COOKIE_SECURE=false
COOKIE_SAMESITE=lax
COOKIE_DOMAIN=
```

Use `COOKIE_SECURE=true` when serving over HTTPS.

### Auth / bootstrap admin

```env
ALLOW_SELF_REGISTER=false
BOOTSTRAP_ADMIN_EMAIL=admin@example.com
BOOTSTRAP_ADMIN_PASSWORD=change-me-now
```

### Rate limiting and mail

```env
LOGIN_RATE_LIMIT_WINDOW_SECONDS=60
LOGIN_RATE_LIMIT_MAX_ATTEMPTS=10
LOGIN_RATE_LIMIT_IP_MAX_ATTEMPTS=30
LOGIN_ATTEMPT_STORE=auto
REDIS_URL=
MAIL_PROVIDER=console
MAIL_FROM_EMAIL=no-reply@perceptrome.local
SMTP_HOST=localhost
SMTP_PORT=1025
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_USE_TLS=false
EMAIL_VERIFICATION_BASE_URL=http://localhost:5173/verify_email.html
PASSWORD_RESET_BASE_URL=http://localhost:5173/reset_password.html
```

## Bootstrap admin flow

On API startup, the server checks `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD`. If both are set and the user does not already exist, it creates an active admin account automatically.

Recommended workflow:

1. Set `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD` in `perceptrome_web/server/.env`.
2. Run `python -m alembic -c perceptrome_web/server/alembic.ini upgrade head`.
3. Start `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` from `perceptrome_web/server/`.
4. Sign in through the SPA with the bootstrap admin credentials.
5. Immediately change the password when prompted.
6. Remove `BOOTSTRAP_ADMIN_PASSWORD` from your `.env` after the initial admin account exists.

Important behavior:

- The bootstrap admin is created with `must_change_password=True`.
- REST endpoints that require a strict authenticated user will reject privileged actions until the password is changed.
- The WebSocket endpoint also rejects users with `must_change_password=True`, so the runs page will not work until the admin completes the password-change flow.

After the first admin is established, use the admin UI and invitation/user-management APIs for ongoing account administration.

## Production-style workflow

Build the SPA and serve it from the same host as the API or through a reverse proxy that preserves `/api` and `/ws` routing:

```bash
npm run build --prefix perceptrome_web/client
```

If you serve the SPA and API from different origins, you must account for CORS, cookie scope, secure cookie settings, and WebSocket routing explicitly.

## WebSocket expectations for runs

The runs UI expects an authenticated WebSocket at `/ws`.

- Authentication is cookie-based; log in through `/api/auth/login` first so the `perceptrome_session` cookie exists.
- If no valid session cookie is present, the server closes the socket with code `4401`.
- If the user still has `must_change_password=True`, the server closes the socket with code `4403`.
- The first successful server message is a status payload confirming authentication.
- Starting a run sends `{"type":"start_run","config":{...}}`.
- The server emits status, log, phase, progress, metric, artifact, result, and cancellation-related messages as the run advances.
- The dev server proxy must forward WebSocket upgrades on `/ws`; this is already configured in `vite.config.ts`.

In practice, if the runs page looks idle, check these first:

1. Did you log in successfully through the SPA?
2. Did you complete the forced password change for a bootstrap admin?
3. Is the API reachable on port `8000`?
4. Did you start Vite with the default proxy configuration?
5. Did you run Alembic so the session, runs, and artifact tables exist?

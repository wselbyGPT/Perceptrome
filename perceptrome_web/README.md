# Perceptrome Web

## API database workflow

The API now uses Alembic for all schema management under `perceptrome_web/server/alembic/`.
Application startup no longer calls `Base.metadata.create_all(...)` or performs ad hoc `ALTER TABLE` repair steps, so you must run migrations explicitly before starting the backend.

### Default database targets

- `DATABASE_URL` now defaults to a PostgreSQL connection string intended for deployed/runtime environments.
- SQLite is still supported for local development and automated tests, but only when you opt into it deliberately by overriding `DATABASE_URL` (for example `sqlite:///./perceptrome_dev.db`).
- `LOCAL_SQLITE_DATABASE_URL` is provided as documentation/config convenience so the intended local SQLite target is explicit rather than an accidental production default.

### Local development

From `perceptrome_web/server/`:

```bash
export DATABASE_URL=sqlite:///./perceptrome_dev.db
python -m alembic -c alembic.ini upgrade head
uvicorn app.main:app --reload
```

For PostgreSQL-based local development, point `DATABASE_URL` at your database and run the same Alembic upgrade command before `uvicorn`.

### Tests

Server tests intentionally keep SQLite coverage by applying the Alembic migration chain to temporary SQLite databases before exercising the API. This makes SQLite compatibility deliberate and verifies that the migrations themselves remain runnable outside PostgreSQL.

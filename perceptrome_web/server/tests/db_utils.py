from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def apply_migrations(database_url: str) -> None:
    server_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(server_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(server_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")

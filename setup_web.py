#!/usr/bin/env python3
"""
Interactive setup for Perceptrome Web.

Walks through environment configuration for either local development
or AWS production deployment, then writes the .env file, optionally
runs migrations, and prints next-step instructions.

Usage:
    python setup_web.py
"""

from __future__ import annotations

import os
import re
import secrets
import string
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / "perceptrome_web" / "server" / ".env"
CLIENT_DIR = ROOT / "perceptrome_web" / "client"
ALEMBIC_INI = ROOT / "perceptrome_web" / "server" / "alembic.ini"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m"


def _banner() -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  Perceptrome Web  --  Interactive Setup"))
    print(_bold("=" * 60))
    print()


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{_cyan(default)}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    return answer if answer else default


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = input(f"  {prompt} [{_cyan(hint)}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    print(f"  {prompt}")
    for i, choice in enumerate(choices, 1):
        marker = _green("*") if choice == default else " "
        print(f"    {marker} {i}) {choice}")
    answer = input(f"  Choice [{_cyan(default)}]: ").strip()
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1]
    if answer in choices:
        return answer
    print(_yellow(f"    Invalid choice, using default: {default}"))
    return default


def _generate_password(length: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _section(title: str) -> None:
    print()
    print(f"  {_bold('── ' + title + ' ──')}")
    print()


# ── Environment Profiles ────────────────────────────────────────────────────

def _collect_common() -> dict[str, str]:
    """Settings shared by both environments."""
    _section("Bootstrap Admin Account")
    print("  This creates the first admin user on startup.")
    print("  You'll be prompted to change the password on first login.")
    print()

    admin_email = _ask("Admin email", "admin@perceptrome.dev")
    while not _validate_email(admin_email):
        print(_red("    Invalid email address, try again."))
        admin_email = _ask("Admin email", "admin@perceptrome.dev")

    generated_pw = _generate_password()
    admin_password = _ask("Admin password (leave blank to auto-generate)", "")
    if not admin_password:
        admin_password = generated_pw
        print(f"    Generated: {_green(admin_password)}")
        print(_yellow("    Save this password -- you'll need it for first login."))

    return {
        "BOOTSTRAP_ADMIN_EMAIL": admin_email,
        "BOOTSTRAP_ADMIN_PASSWORD": admin_password,
    }


def _collect_dev() -> dict[str, str]:
    """Local development settings."""
    _section("Database")

    db_choice = _ask_choice(
        "Database engine:",
        ["postgresql", "sqlite"],
        default="postgresql",
    )

    if db_choice == "postgresql":
        db_host = _ask("PostgreSQL host", "localhost")
        db_port = _ask("PostgreSQL port", "5432")
        db_name = _ask("Database name", "perceptrome")
        db_user = _ask("Database user", "perceptrome")
        db_pass = _ask("Database password", "perceptrome")
        database_url = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        db_path = _ask("SQLite file path", "./perceptrome_dev.db")
        database_url = f"sqlite:///{db_path}"

    _section("Server Ports")
    api_port = _ask("API server port", "8000")
    client_port = _ask("Vite dev server port", "5173")

    _section("Email (dev)")
    mail_provider = _ask_choice(
        "Mail delivery:",
        ["console", "smtp"],
        default="console",
    )
    mail_settings: dict[str, str] = {"MAIL_PROVIDER": mail_provider}
    if mail_provider == "smtp":
        mail_settings["SMTP_HOST"] = _ask("SMTP host", "localhost")
        mail_settings["SMTP_PORT"] = _ask("SMTP port", "1025")
        mail_settings["SMTP_USE_TLS"] = "false"
        print(_yellow("    Tip: use MailHog (port 1025) for local SMTP testing."))

    cors = f"http://localhost:{client_port},http://127.0.0.1:{client_port}"

    env = {
        "APP_ENV": "dev",
        "DATABASE_URL": database_url,
        "CORS_ORIGINS": cors,
        "COOKIE_SECURE": "false",
        "COOKIE_SAMESITE": "lax",
        "ALLOW_SELF_REGISTER": "false",
        "EMAIL_VERIFICATION_BASE_URL": f"http://localhost:{client_port}/verify-email",
        "PASSWORD_RESET_BASE_URL": f"http://localhost:{client_port}/reset-password",
        **mail_settings,
    }
    env["_api_port"] = api_port
    env["_client_port"] = client_port
    env["_use_docker"] = str(_ask_yes_no("Use Docker Compose for PostgreSQL?", default=(db_choice == "postgresql")))
    return env


def _collect_prod() -> dict[str, str]:
    """AWS production settings."""
    _section("Domain & HTTPS")
    domain = _ask("Production domain (e.g. app.perceptrome.com)", "app.perceptrome.com")
    use_https = _ask_yes_no("Serve over HTTPS?", default=True)
    scheme = "https" if use_https else "http"

    _section("Database (AWS)")
    print("  For AWS, use RDS PostgreSQL or Aurora.")
    print()
    db_host = _ask("RDS endpoint", f"perceptrome-db.xxxx.us-east-1.rds.amazonaws.com")
    db_port = _ask("RDS port", "5432")
    db_name = _ask("Database name", "perceptrome")
    db_user = _ask("Database user", "perceptrome")
    db_pass = _ask("Database password", "")
    while not db_pass:
        print(_red("    Database password is required for production."))
        db_pass = _ask("Database password", "")
    database_url = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    _section("Redis (optional)")
    use_redis = _ask_yes_no("Use Redis/ElastiCache for rate limiting?", default=False)
    redis_url = ""
    if use_redis:
        redis_url = _ask("Redis URL", "redis://perceptrome-cache.xxxx.cache.amazonaws.com:6379/0")

    _section("Email (production)")
    mail_provider = _ask_choice(
        "Mail delivery:",
        ["smtp", "console"],
        default="smtp",
    )
    mail_settings: dict[str, str] = {"MAIL_PROVIDER": mail_provider}
    if mail_provider == "smtp":
        print("  For AWS, use SES SMTP credentials.")
        print()
        mail_settings["SMTP_HOST"] = _ask("SMTP host (SES endpoint)", "email-smtp.us-east-1.amazonaws.com")
        mail_settings["SMTP_PORT"] = _ask("SMTP port", "587")
        mail_settings["SMTP_USERNAME"] = _ask("SMTP username (SES access key)", "")
        mail_settings["SMTP_PASSWORD"] = _ask("SMTP password (SES secret key)", "")
        mail_settings["SMTP_USE_TLS"] = "true"
        mail_settings["MAIL_FROM_EMAIL"] = _ask("From email (must be verified in SES)", f"no-reply@{domain}")

    _section("Session & Security")
    session_ttl = _ask("Session TTL (hours)", "168")
    cookie_domain = _ask("Cookie domain (blank for auto)", "")

    cors_origins = f"{scheme}://{domain}"
    extra_cors = _ask("Additional CORS origins (comma-separated, or blank)", "")
    if extra_cors:
        cors_origins += f",{extra_cors}"

    env = {
        "APP_ENV": "production",
        "DATABASE_URL": database_url,
        "CORS_ORIGINS": cors_origins,
        "COOKIE_SECURE": str(use_https).lower(),
        "COOKIE_SAMESITE": "lax",
        "COOKIE_DOMAIN": cookie_domain,
        "SESSION_TTL_HOURS": session_ttl,
        "ALLOW_SELF_REGISTER": "false",
        "REDIS_URL": redis_url,
        "EMAIL_VERIFICATION_BASE_URL": f"{scheme}://{domain}/verify-email",
        "PASSWORD_RESET_BASE_URL": f"{scheme}://{domain}/reset-password",
        **mail_settings,
    }
    env["_domain"] = domain
    env["_scheme"] = scheme
    return env


# ── Write & Run ─────────────────────────────────────────────────────────────

def _write_env(env: dict[str, str]) -> None:
    """Write the .env file, filtering out internal keys (prefixed with _)."""
    lines = [
        "# Perceptrome Web - generated by setup_web.py",
        f"# Environment: {env.get('APP_ENV', 'dev')}",
        "",
    ]

    sections = {
        "Core": ["APP_ENV", "DATABASE_URL", "CORS_ORIGINS"],
        "Session / Cookies": [
            "SESSION_TTL_HOURS", "COOKIE_SECURE", "COOKIE_SAMESITE", "COOKIE_DOMAIN",
        ],
        "Auth": [
            "ALLOW_SELF_REGISTER", "BOOTSTRAP_ADMIN_EMAIL", "BOOTSTRAP_ADMIN_PASSWORD",
        ],
        "Rate Limiting": ["REDIS_URL", "LOGIN_ATTEMPT_STORE"],
        "Email": [
            "MAIL_PROVIDER", "MAIL_FROM_EMAIL",
            "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_USE_TLS",
        ],
        "URLs": ["EMAIL_VERIFICATION_BASE_URL", "PASSWORD_RESET_BASE_URL"],
    }

    for section_name, keys in sections.items():
        section_lines = []
        for key in keys:
            if key in env:
                section_lines.append(f"{key}={env[key]}")
        if section_lines:
            lines.append(f"# {section_name}")
            lines.extend(section_lines)
            lines.append("")

    # Catch any remaining keys not in the defined sections
    all_section_keys = {k for keys in sections.values() for k in keys}
    for key, value in env.items():
        if not key.startswith("_") and key not in all_section_keys:
            lines.append(f"{key}={value}")

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if ENV_PATH.exists():
        if not _ask_yes_no(f"{ENV_PATH} already exists. Overwrite?", default=False):
            print(_yellow("  Skipping .env write. Existing file preserved."))
            return
        # Back up the existing file
        backup = ENV_PATH.with_suffix(".env.bak")
        ENV_PATH.rename(backup)
        print(f"  Backed up existing .env to {_cyan(str(backup))}")

    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"  {_green('Wrote')} {ENV_PATH}")


def _run_cmd(desc: str, cmd: list[str], cwd: Path | None = None) -> bool:
    print(f"  {_yellow('Running:')} {desc}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        print(f"  {_green('Done.')}")
        return True
    except subprocess.CalledProcessError as exc:
        print(_red(f"  Command failed (exit {exc.returncode}): {' '.join(cmd)}"))
        return False
    except FileNotFoundError:
        print(_red(f"  Command not found: {cmd[0]}"))
        return False


def _post_setup(env: dict[str, str]) -> None:
    """Run optional post-setup steps."""
    _section("Post-Setup")

    is_prod = env.get("APP_ENV") == "production"

    # npm install
    if _ask_yes_no("Install client npm dependencies?", default=True):
        _run_cmd("npm install", ["npm", "install"], cwd=CLIENT_DIR)

    # Build client (production only by default)
    if is_prod or _ask_yes_no("Build client for production?", default=False):
        if is_prod:
            print("  Building client for production deployment...")
        _run_cmd("npm run build", ["npm", "run", "build"], cwd=CLIENT_DIR)

    # Alembic migrations
    if _ask_yes_no("Run Alembic database migrations?", default=not is_prod):
        _run_cmd(
            "alembic upgrade head",
            [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "upgrade", "head"],
            cwd=ROOT,
        )

    _section("Next Steps")

    if is_prod:
        domain = env.get("_domain", "app.perceptrome.com")
        print(f"""
  {_bold('AWS Production Deployment')}

  1. Build & push the Docker image:
     {_cyan('docker compose -f docker-compose.prod.yml build')}
     {_cyan('docker tag perceptrome-web:latest <ECR_URI>:latest')}
     {_cyan('docker push <ECR_URI>:latest')}

  2. Or deploy directly with Compose (e.g. on EC2):
     {_cyan('docker compose -f docker-compose.prod.yml up -d')}

  3. Make sure your RDS instance is accessible from the app.

  4. Point DNS for {_green(domain)} to your load balancer / EC2 IP.

  5. First login: use the bootstrap admin credentials above,
     then change the password when prompted.
""")
    else:
        api_port = env.get("_api_port", "8000")
        client_port = env.get("_client_port", "5173")
        use_docker = env.get("_use_docker", "False") == "True"

        if use_docker:
            print(f"""
  {_bold('Local Development (Docker)')}

  Start PostgreSQL + API + Vite in one command:
     {_cyan('docker compose -f docker-compose.dev.yml up')}

  Or start just the database:
     {_cyan('docker compose -f docker-compose.dev.yml up postgres')}

  Then run the API and client separately:
     {_cyan(f'cd perceptrome_web/server && uvicorn app.main:app --reload --host 0.0.0.0 --port {api_port}')}
     {_cyan(f'npm run dev --prefix perceptrome_web/client')}
""")
        else:
            print(f"""
  {_bold('Local Development')}

  Terminal 1 - API server:
     {_cyan(f'cd perceptrome_web/server && uvicorn app.main:app --reload --host 0.0.0.0 --port {api_port}')}

  Terminal 2 - Vite dev server:
     {_cyan(f'npm run dev --prefix perceptrome_web/client')}

  Open: {_green(f'http://localhost:{client_port}')}
""")

        print(f"  Login with: {_green(env.get('BOOTSTRAP_ADMIN_EMAIL', ''))}")
        print(f"  You will be prompted to change your password on first login.")
        print()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    environment = _ask_choice(
        "Which environment are you setting up?",
        ["dev (laptop)", "production (AWS)"],
        default="dev (laptop)",
    )

    is_prod = "production" in environment

    if is_prod:
        env = _collect_prod()
    else:
        env = _collect_dev()

    common = _collect_common()
    env.update(common)

    _section("Configuration Summary")
    print(f"  Environment:  {_green(env.get('APP_ENV', '?'))}")
    print(f"  Database:     {_cyan(env.get('DATABASE_URL', '?')[:60])}...")
    print(f"  CORS:         {env.get('CORS_ORIGINS', '?')}")
    print(f"  Cookie secure:{env.get('COOKIE_SECURE', '?')}")
    print(f"  Mail:         {env.get('MAIL_PROVIDER', '?')}")
    print(f"  Admin:        {env.get('BOOTSTRAP_ADMIN_EMAIL', '?')}")
    print()

    if not _ask_yes_no("Write this configuration?", default=True):
        print(_yellow("  Aborted."))
        return

    _write_env(env)
    _post_setup(env)

    print(_green("  Setup complete."))
    print()


if __name__ == "__main__":
    main()

# server/app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Perceptrome API"
    app_env: str = "dev"  # dev | prod

    database_url: str = "sqlite:///./server/perceptrome_dev.db"

    session_cookie_name: str = "perceptrome_session"
    session_ttl_hours: int = 24 * 7
    cookie_secure: bool = False  # True in prod behind HTTPS
    cookie_samesite: str = "lax"  # "lax" | "strict" | "none"
    cookie_domain: str | None = None

    allow_self_register: bool = False
    bootstrap_admin_email: str | None = None
    bootstrap_admin_password: str | None = None

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,https://perceptrome.com"

    login_rate_limit_window_seconds: int = 60
    login_rate_limit_max_attempts: int = 10

    email_verification_token_ttl_minutes: int = 60
    email_verification_resend_cooldown_seconds: int = 60

    mail_provider: str = "console"  # console | smtp
    mail_from_email: str = "no-reply@perceptrome.local"
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = False

    email_verification_base_url: str = "http://localhost:5173/verify_email.html"


settings = Settings()

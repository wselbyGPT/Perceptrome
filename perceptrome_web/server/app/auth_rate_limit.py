from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .config import settings
from .models import LoginAttempt


@dataclass
class RateLimitStatus:
    limited: bool
    retry_after_seconds: int = 0
    scope: str = ""


class LoginAttemptStore:
    def __init__(self):
        self._last_cleanup_at: datetime | None = None
        self._redis_client = self._build_redis_client()

    def _build_redis_client(self):
        mode = settings.login_attempt_store.lower()
        if mode == "db":
            return None
        if mode == "redis" and not settings.redis_url:
            return None
        if mode == "auto" and not settings.redis_url:
            return None

        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    def check_and_record(self, db: Session, ip: str, email: str, now: datetime) -> RateLimitStatus:
        if self._redis_client is not None:
            return self._check_and_record_redis(ip=ip, email=email, now=now)
        return self._check_and_record_db(db=db, ip=ip, email=email, now=now)

    def _check_and_record_redis(self, ip: str, email: str, now: datetime) -> RateLimitStatus:
        now_ts = now.timestamp()
        window = settings.login_rate_limit_window_seconds

        result = self._redis_window(ip=ip, email=email, now_ts=now_ts, window=window)
        if result.limited:
            return result

        result = self._redis_window(ip=ip, email=None, now_ts=now_ts, window=window)
        return result

    def _redis_window(self, ip: str, email: str | None, now_ts: float, window: int) -> RateLimitStatus:
        if email is None:
            key = f"auth:login:ip:{ip}"
            max_attempts = settings.login_rate_limit_ip_max_attempts
            scope = "ip"
        else:
            key = f"auth:login:ip_email:{ip}:{email}"
            max_attempts = settings.login_rate_limit_max_attempts
            scope = "ip_email"

        cutoff = now_ts - window
        member = f"{now_ts}:{uuid.uuid4()}"
        pipe = self._redis_client.pipeline(transaction=True)
        pipe.zremrangebyscore(key, "-inf", cutoff)
        pipe.zcard(key)
        _, count = pipe.execute()

        if count >= max_attempts:
            oldest = self._redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry = int(max(1, (oldest[0][1] + window) - now_ts))
            else:
                retry = window
            return RateLimitStatus(limited=True, retry_after_seconds=retry, scope=scope)

        pipe = self._redis_client.pipeline(transaction=True)
        pipe.zadd(key, {member: now_ts})
        pipe.expire(key, window + 60)
        pipe.execute()
        return RateLimitStatus(limited=False)

    def _check_and_record_db(self, db: Session, ip: str, email: str, now: datetime) -> RateLimitStatus:
        self._cleanup_if_due(db, now)
        window_seconds = settings.login_rate_limit_window_seconds
        cutoff = now - timedelta(seconds=window_seconds)

        pair_count = db.execute(
            select(func.count(LoginAttempt.id))
            .where(LoginAttempt.ip_address == ip)
            .where(LoginAttempt.email == email)
            .where(LoginAttempt.created_at >= cutoff)
        ).scalar_one()
        if pair_count >= settings.login_rate_limit_max_attempts:
            retry = self._retry_after(db=db, ip=ip, email=email, cutoff=cutoff, now=now)
            return RateLimitStatus(limited=True, retry_after_seconds=retry, scope="ip_email")

        ip_count = db.execute(
            select(func.count(LoginAttempt.id))
            .where(LoginAttempt.ip_address == ip)
            .where(LoginAttempt.created_at >= cutoff)
        ).scalar_one()
        if ip_count >= settings.login_rate_limit_ip_max_attempts:
            retry = self._retry_after(db=db, ip=ip, email=None, cutoff=cutoff, now=now)
            return RateLimitStatus(limited=True, retry_after_seconds=retry, scope="ip")

        db.add(LoginAttempt(ip_address=ip, email=email, created_at=now))
        db.commit()
        return RateLimitStatus(limited=False)

    def _retry_after(self, db: Session, ip: str, email: str | None, cutoff: datetime, now: datetime) -> int:
        q = (
            select(LoginAttempt.created_at)
            .where(LoginAttempt.ip_address == ip)
            .where(LoginAttempt.created_at >= cutoff)
            .order_by(LoginAttempt.created_at.asc())
            .limit(1)
        )
        if email is not None:
            q = q.where(LoginAttempt.email == email)
        oldest = db.execute(q).scalar_one_or_none()
        if oldest is None:
            return settings.login_rate_limit_window_seconds
        retry = oldest + timedelta(seconds=settings.login_rate_limit_window_seconds) - now
        return max(1, int(retry.total_seconds()))

    def _cleanup_if_due(self, db: Session, now: datetime):
        if self._last_cleanup_at and (now - self._last_cleanup_at).total_seconds() < settings.login_attempt_cleanup_interval_seconds:
            return
        retention = now - timedelta(seconds=settings.login_rate_limit_window_seconds * 2)
        db.execute(delete(LoginAttempt).where(LoginAttempt.created_at < retention))
        db.commit()
        self._last_cleanup_at = now


login_attempt_store = LoginAttemptStore()

from __future__ import annotations

import json
import time
from typing import Any, Optional


class ValidationError(Exception):
    pass


def req_str(value: str, field: str) -> str:
    v = (value or "").strip()
    if not v:
        raise ValidationError(f"{field} is required.")
    return v


def opt_str(value: str) -> Optional[str]:
    v = (value or "").strip()
    return v if v else None


def opt_int(value: str, field: str, min_v: Optional[int] = None, max_v: Optional[int] = None) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError as e:
        raise ValidationError(f"{field} must be an integer (got {value!r}).") from e
    if min_v is not None and n < min_v:
        raise ValidationError(f"{field} must be >= {min_v} (got {n}).")
    if max_v is not None and n > max_v:
        raise ValidationError(f"{field} must be <= {max_v} (got {n}).")
    return n


def opt_float(value: str, field: str, min_v: Optional[float] = None, max_v: Optional[float] = None) -> Optional[float]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        x = float(v)
    except ValueError as e:
        raise ValidationError(f"{field} must be a number (got {value!r}).") from e
    if min_v is not None and x < min_v:
        raise ValidationError(f"{field} must be >= {min_v} (got {x}).")
    if max_v is not None and x > max_v:
        raise ValidationError(f"{field} must be <= {max_v} (got {x}).")
    return x


def ts() -> str:
    return time.strftime("%H:%M:%S")


def date_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def now_epoch() -> int:
    return int(time.time())


def fmt_epoch(epoch: Optional[int]) -> str:
    if not epoch:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(int(epoch)))
    except Exception:
        return ""


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def coerce_to_string(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

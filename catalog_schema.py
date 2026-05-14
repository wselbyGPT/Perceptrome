from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CatalogCategorySpec:
    name: str
    count: int
    source: str


@dataclass(frozen=True)
class CatalogSchema:
    schema_version: int
    seed: int | None
    shuffle_within_category: bool
    categories: List[CatalogCategorySpec]


def _as_int(raw: Any, *, field: str) -> int:
    try:
        return int(raw)
    except Exception as exc:  # pragma: no cover - defensive conversion helper
        raise ValueError(f"{field} must be an integer (got {raw!r})") from exc


def _parse_categories(raw: Any, *, base_dir: str) -> List[CatalogCategorySpec]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("catalog schema requires non-empty 'categories' list")

    parsed: List[CatalogCategorySpec] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"categories[{idx}] must be an object")

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"categories[{idx}] missing required 'name'")

        if "count" not in entry:
            raise ValueError(f"categories[{idx}] missing required 'count'")
        count = _as_int(entry["count"], field=f"categories[{idx}].count")
        if count < 0:
            raise ValueError(f"categories[{idx}].count must be >= 0")

        src = str(entry.get("source") or os.path.join("accessions", f"{name}_accessions.txt")).strip()
        if not src:
            raise ValueError(f"categories[{idx}] source path must not be empty")
        if not os.path.isabs(src):
            src = os.path.join(base_dir, src)

        parsed.append(CatalogCategorySpec(name=name, count=count, source=src))

    return parsed


def _decode_schema_text(path: str, text: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML catalog schemas")
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError("catalog schema root must be an object")
    return raw


def parse_catalog_schema(path: str) -> CatalogSchema:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = _decode_schema_text(path, f.read())

    version = _as_int(raw.get("schema_version", CATALOG_SCHEMA_VERSION), field="schema_version")
    if version != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported catalog schema version: {version} (expected {CATALOG_SCHEMA_VERSION})"
        )

    seed = raw.get("seed")
    if seed is not None:
        seed = _as_int(seed, field="seed")

    categories = _parse_categories(raw.get("categories"), base_dir=os.path.dirname(path) or ".")

    return CatalogSchema(
        schema_version=version,
        seed=seed,
        shuffle_within_category=bool(raw.get("shuffle_within_category", False)),
        categories=categories,
    )

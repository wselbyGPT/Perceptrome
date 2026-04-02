from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from perceptrome.io_utils import read_catalog


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _walk_collect(node: Any, out: Dict[str, List[Any]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).lower()
            out.setdefault(low, []).append(value)
            _walk_collect(value, out)
        return
    if isinstance(node, list):
        for child in node:
            _walk_collect(child, out)


def _first_scalar(values: Sequence[Any]) -> str | None:
    for value in values:
        if isinstance(value, (str, int, float, bool)):
            s = str(value).strip()
            if s:
                return s
        if isinstance(value, list):
            nested = _first_scalar(value)
            if nested:
                return nested
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "reference", "refseq"}:
            return True
        if text in {"0", "false", "no", "n", "non-reference", "genbank"}:
            return False
    return None


def _normalize_year(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value)
    digits = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        else:
            if len(digits) >= 4:
                break
            digits = []
    if len(digits) < 4:
        return None
    year = int("".join(digits[:4]))
    if 1800 <= year <= 2200:
        return year
    return None


def _extract_accession(payload: Dict[str, Any]) -> str | None:
    collected: Dict[str, List[Any]] = {}
    _walk_collect(payload, collected)
    for key in ("accession", "accession_version", "genbank_accession", "assembly_accession"):
        value = _first_scalar(collected.get(key, []))
        if value:
            return value
    return None


def _extract_record_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    collected: Dict[str, List[Any]] = {}
    _walk_collect(payload, collected)

    taxon = _first_scalar(
        [
            *_flatten(collected, "tax_id"),
            *_flatten(collected, "taxonomy_id"),
            *_flatten(collected, "taxon"),
            *_flatten(collected, "taxon_id"),
        ]
    )
    host = _first_scalar(
        [
            *_flatten(collected, "host"),
            *_flatten(collected, "host_name"),
            *_flatten(collected, "isolation_host"),
            *_flatten(collected, "hosts"),
        ]
    )
    family = _first_scalar(
        [
            *_flatten(collected, "family"),
            *_flatten(collected, "virus_family"),
            *_flatten(collected, "family_name"),
        ]
    )
    completeness = _first_scalar(
        [
            *_flatten(collected, "completeness"),
            *_flatten(collected, "assembly_level"),
            *_flatten(collected, "genome_completeness"),
        ]
    )

    reference = None
    for key in ("is_reference", "reference", "reference_genome", "is_refseq"):
        vals = _flatten(collected, key)
        for item in vals:
            b = _coerce_bool(item)
            if b is not None:
                reference = b
                break
        if reference is not None:
            break

    date_text = _first_scalar(
        [
            *_flatten(collected, "collection_date"),
            *_flatten(collected, "release_date"),
            *_flatten(collected, "isolation_date"),
            *_flatten(collected, "date"),
        ]
    )

    return {
        "taxon": taxon,
        "host": host,
        "family": family,
        "completeness": completeness,
        "reference": reference,
        "date": date_text,
        "year": _normalize_year(date_text),
    }


def _flatten(collected: Dict[str, List[Any]], key: str) -> List[Any]:
    vals = collected.get(key.lower(), [])
    out: List[Any] = []
    for item in vals:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _load_metadata_from_fetch_manifest(path: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    run = payload.get("run") or {}
    staged_dir = run.get("staged_dir")
    jsonl_paths: List[str] = []
    for entry in payload.get("files") or []:
        rel = str((entry or {}).get("path") or "")
        if not rel.endswith(".jsonl"):
            continue
        if staged_dir:
            jsonl_paths.append(os.path.join(str(staged_dir), rel))

    metadata: Dict[str, Dict[str, Any]] = {}
    for jsonl in jsonl_paths:
        if not os.path.exists(jsonl):
            continue
        for row in _iter_jsonl(jsonl):
            accession = _extract_accession(row)
            if not accession:
                continue
            md = _extract_record_metadata(row)
            metadata.setdefault(accession, {}).update({k: v for k, v in md.items() if v is not None})

    return metadata, jsonl_paths


def _load_metadata_from_snapshot(path: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    base = path
    if os.path.isfile(path) and path.endswith(".json"):
        base = os.path.dirname(path)
    if not os.path.isdir(base):
        return {}, []

    jsonl_paths: List[str] = []
    for root, _, files in os.walk(base):
        for name in files:
            if name.endswith(".jsonl"):
                jsonl_paths.append(os.path.join(root, name))

    metadata: Dict[str, Dict[str, Any]] = {}
    for jsonl in jsonl_paths:
        for row in _iter_jsonl(jsonl):
            accession = _extract_accession(row)
            if not accession:
                continue
            md = _extract_record_metadata(row)
            metadata.setdefault(accession, {}).update({k: v for k, v in md.items() if v is not None})
    return metadata, sorted(jsonl_paths)


def _merge_metadata(*items: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for meta in items:
        for accession, row in meta.items():
            merged.setdefault(accession, {}).update(row)
    return merged


def _calc_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    return n_train, n_val


def _split_random(items: Sequence[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_train, n_val = _calc_counts(len(shuffled), train_ratio, val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _split_grouped(
    items: Sequence[str],
    group_for_accession: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for accession in items:
        buckets[group_for_accession.get(accession, "unknown")].append(accession)

    groups = sorted(buckets.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_train, n_val = _calc_counts(len(items), train_ratio, val_ratio)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for group in groups:
        rows = buckets[group]
        if len(train) < n_train:
            train.extend(rows)
        elif len(val) < n_val:
            val.extend(rows)
        else:
            test.extend(rows)

    if not test and val:
        test.append(val.pop())
    if not test and train:
        test.append(train.pop())

    return {"train": train, "val": val, "test": test}


def _split_date_based(
    items: Sequence[str],
    year_for_accession: Dict[str, int | None],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[str]]:
    unknown_year = 9999
    ordered = sorted(items, key=lambda acc: (year_for_accession.get(acc, unknown_year) or unknown_year, acc))
    n_train, n_val = _calc_counts(len(ordered), train_ratio, val_ratio)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def create_split_payload(args: argparse.Namespace) -> Dict[str, Any]:
    catalog_path = str(args.catalog)
    accessions = read_catalog(catalog_path)
    if len(accessions) < 3:
        raise ValueError("Need at least 3 accessions to create train/val/test splits")

    strategy = str(args.strategy)
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if train_ratio <= 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.0")

    fetch_manifests = list(getattr(args, "fetch_manifest", None) or [])
    snapshots = list(getattr(args, "snapshot", None) or [])

    merged_meta: Dict[str, Dict[str, Any]] = {}
    metadata_refs: List[str] = []

    for manifest in fetch_manifests:
        meta, refs = _load_metadata_from_fetch_manifest(str(manifest))
        merged_meta = _merge_metadata(merged_meta, meta)
        metadata_refs.extend(refs)

    for snapshot in snapshots:
        meta, refs = _load_metadata_from_snapshot(str(snapshot))
        merged_meta = _merge_metadata(merged_meta, meta)
        metadata_refs.extend(refs)

    grouping_key = "accession"
    if strategy == "random":
        splits = _split_random(accessions, train_ratio, val_ratio, int(args.seed))
    elif strategy == "taxon":
        grouping_key = "taxon"
        grouped = {acc: str(merged_meta.get(acc, {}).get("taxon") or "unknown") for acc in accessions}
        splits = _split_grouped(accessions, grouped, train_ratio, val_ratio, int(args.seed))
    elif strategy == "host":
        grouping_key = "host"
        grouped = {acc: str(merged_meta.get(acc, {}).get("host") or "unknown") for acc in accessions}
        splits = _split_grouped(accessions, grouped, train_ratio, val_ratio, int(args.seed))
    elif strategy == "completeness-reference":
        grouping_key = "completeness_reference"
        grouped = {
            acc: f"{str(merged_meta.get(acc, {}).get('completeness') or 'unknown')}|ref={str(bool(merged_meta.get(acc, {}).get('reference', False))).lower()}"
            for acc in accessions
        }
        splits = _split_grouped(accessions, grouped, train_ratio, val_ratio, int(args.seed))
    elif strategy == "family-heldout":
        grouping_key = "family"
        grouped = {acc: str(merged_meta.get(acc, {}).get("family") or "unknown") for acc in accessions}
        observed = {v for v in grouped.values() if v != "unknown"}
        if not observed:
            raise ValueError("family-heldout requires family metadata from snapshot/fetch reports")
        splits = _split_grouped(accessions, grouped, train_ratio, val_ratio, int(args.seed))
    elif strategy == "date-based":
        grouping_key = "year"
        year_map: Dict[str, int | None] = {acc: merged_meta.get(acc, {}).get("year") for acc in accessions}
        if not any(v is not None for v in year_map.values()):
            raise ValueError("date-based strategy requires date metadata from snapshot/fetch reports")
        splits = _split_date_based(accessions, year_map, train_ratio, val_ratio)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    counts = {
        "total": len(accessions),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }

    return {
        "name": str(args.name),
        "created_at": _utc_now_iso(),
        "strategy": strategy,
        "source": {
            "catalog_path": catalog_path,
            "fetch_manifests": [str(v) for v in fetch_manifests],
            "snapshot_references": [str(v) for v in snapshots],
            "metadata_reports": sorted(set(metadata_refs)),
        },
        "seed": int(args.seed),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "counts": counts,
        "splits": splits,
        "replay": {
            "grouping_key": grouping_key,
            "algorithm": "virus-split-v1",
        },
    }

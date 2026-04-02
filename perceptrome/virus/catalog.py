from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import stat
from typing import Any, Dict, Iterable, List, Sequence

from perceptrome.io_utils import read_catalog, write_catalog
from perceptrome.virus.ncbi_datasets import DatasetsCommandError, resolve_datasets_binary

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_SOURCE = "ncbi_datasets_virus"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for accession in lines:
        digest.update(str(accession).strip().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _clean_accessions(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        val = str(item or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _extract_accessions_from_json(payload: Any) -> List[str]:
    found: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                low = str(key).lower()
                if isinstance(value, str) and low in {"accession", "accession_version", "assembly_accession", "genbank_accession"}:
                    found.append(value)
                else:
                    _walk(value)
            return
        if isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(payload)
    return _clean_accessions(found)


def _run_datasets_json(argv: Sequence[str]) -> Dict[str, Any]:
    import subprocess

    proc = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise DatasetsCommandError(
            f"datasets command failed with return code {proc.returncode}: {' '.join(argv)}\n{stderr.strip()}"
        )
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"datasets output was not valid JSON for command: {' '.join(argv)}") from exc
    return {
        "stdout": stdout,
        "stderr": stderr,
        "parsed": parsed,
        "return_code": int(proc.returncode),
    }


def _build_summary_argv(
    *,
    datasets_bin: str,
    mode: str,
    taxon: str | None,
    accessions: Sequence[str],
    filters: Sequence[str],
    includes: Sequence[str],
) -> List[str]:
    argv: List[str] = [datasets_bin, "summary", "virus", "genome"]
    if mode == "taxon":
        if not taxon:
            raise ValueError("taxon mode requires --taxon")
        argv.extend(["taxon", str(taxon)])
    elif mode == "accession":
        if not accessions:
            raise ValueError("accession mode requires at least one accession")
        if len(accessions) == 1:
            argv.extend(["accession", str(accessions[0])])
        else:
            argv.extend(["accession"])
            argv.extend(str(a) for a in accessions)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for item in includes:
        argv.extend(["--include", str(item)])
    argv.extend(str(f) for f in filters)
    argv.append("--as-json")
    return argv


def _mode_from_args(args: argparse.Namespace) -> str:
    provided = int(bool(getattr(args, "taxon", None))) + int(bool(getattr(args, "inputfile", None))) + int(bool(getattr(args, "accession", None)))
    if provided != 1:
        raise ValueError("Choose exactly one query source: --taxon OR --accession (repeatable) OR --inputfile")
    if getattr(args, "taxon", None):
        return "taxon"
    return "accession"


def _resolve_mode_values(args: argparse.Namespace) -> Dict[str, Any]:
    mode = _mode_from_args(args)
    accessions: List[str] = []
    inputfile = getattr(args, "inputfile", None)
    explicit_accessions = list(getattr(args, "accession", None) or [])
    if inputfile:
        accessions.extend(read_catalog(str(inputfile)))
    accessions.extend(explicit_accessions)
    accessions = _clean_accessions(accessions)
    if mode == "accession" and not accessions:
        raise ValueError("No accessions resolved for accession mode")

    return {
        "mode": mode,
        "taxon": getattr(args, "taxon", None),
        "accessions": accessions,
        "provenance": {
            "taxon": str(getattr(args, "taxon", None)) if getattr(args, "taxon", None) else None,
            "explicit_accessions": _clean_accessions(explicit_accessions),
            "inputfile": str(inputfile) if inputfile else None,
            "inputfile_accession_count": len(read_catalog(str(inputfile))) if inputfile else 0,
        },
    }


def _manifest_path_for_catalog(catalog_path: str) -> str:
    return os.path.join(os.path.dirname(catalog_path) or ".", "catalog.manifest.json")


def _build_manifest(
    *,
    catalog_path: str,
    accessions: Sequence[str],
    datasets_argv: Sequence[str],
    query_mode: str,
    filters: Sequence[str],
    includes: Sequence[str],
    provenance: Dict[str, Any],
    snapshot_references: Sequence[str] | None,
) -> Dict[str, Any]:
    digest = _sha256_text_lines(accessions)
    payload: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "source": MANIFEST_SOURCE,
        "catalog": {
            "path": str(catalog_path),
            "accession_count": int(len(accessions)),
            "accession_sha256": digest,
        },
        "query": {
            "mode": str(query_mode),
            "filters": [str(f) for f in filters],
            "includes": [str(i) for i in includes],
        },
        "datasets": {
            "argv": [str(a) for a in datasets_argv],
        },
        "provenance_refs": provenance,
    }
    if snapshot_references:
        payload["snapshot_references"] = [str(item) for item in snapshot_references]
    return payload


def _write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def build_catalog_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mode_values = _resolve_mode_values(args)
    mode = str(mode_values["mode"])
    taxon = mode_values["taxon"]
    accessions = list(mode_values["accessions"])
    filters = list(getattr(args, "filter", None) or [])
    includes = list(getattr(args, "include", None) or [])
    datasets_bin = resolve_datasets_binary(getattr(args, "datasets_bin", None))

    summary_argv = _build_summary_argv(
        datasets_bin=datasets_bin,
        mode=mode,
        taxon=taxon,
        accessions=accessions,
        filters=filters,
        includes=includes,
    )
    result = _run_datasets_json(summary_argv)
    resolved_accessions = _extract_accessions_from_json(result["parsed"])
    if not resolved_accessions and mode == "accession":
        resolved_accessions = accessions
    if not resolved_accessions:
        raise ValueError("datasets query resolved zero accessions")

    catalog_path = str(getattr(args, "output", None) or "catalog.txt")
    write_catalog(catalog_path, resolved_accessions)
    manifest_path = _manifest_path_for_catalog(catalog_path)

    snapshot_refs: List[str] = []
    manifest = _build_manifest(
        catalog_path=catalog_path,
        accessions=resolved_accessions,
        datasets_argv=summary_argv,
        query_mode=mode,
        filters=filters,
        includes=includes,
        provenance=mode_values["provenance"],
        snapshot_references=snapshot_refs,
    )
    _write_manifest(manifest_path, manifest)

    if bool(getattr(args, "snapshot", False)):
        snap = create_snapshot_bundle(
            catalog_path=catalog_path,
            manifest_path=manifest_path,
            metadata_files=list(getattr(args, "snapshot_metadata", None) or []),
            snapshot_dir=getattr(args, "snapshot_dir", None),
        )
        snapshot_refs.append(snap["bundle_dir"])
        manifest["snapshot_references"] = list(snapshot_refs)
        _write_manifest(manifest_path, manifest)

    return {
        "catalog_path": catalog_path,
        "manifest_path": manifest_path,
        "accession_count": len(resolved_accessions),
        "accession_sha256": manifest["catalog"]["accession_sha256"],
        "datasets_argv": summary_argv,
        "snapshot_references": snapshot_refs,
    }


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return payload


def inspect_manifest_payload(manifest: Dict[str, Any], catalog_path: str | None = None) -> Dict[str, Any]:
    catalog = manifest.get("catalog") or {}
    expected_hash = str(catalog.get("accession_sha256") or "")
    expected_count = int(catalog.get("accession_count") or 0)

    selected_catalog = str(catalog_path or catalog.get("path") or "catalog.txt")
    accessions: List[str] = []
    catalog_exists = os.path.exists(selected_catalog)
    if catalog_exists:
        accessions = read_catalog(selected_catalog)

    actual_count = len(accessions)
    actual_hash = _sha256_text_lines(accessions) if accessions else ""
    query = manifest.get("query") or {}

    return {
        "catalog_path": selected_catalog,
        "catalog_exists": catalog_exists,
        "expected_count": expected_count,
        "expected_hash": expected_hash,
        "actual_count": actual_count,
        "actual_hash": actual_hash,
        "hash_matches": bool(actual_hash) and actual_hash == expected_hash,
        "query_mode": query.get("mode"),
        "filters": list(query.get("filters") or []),
        "includes": list(query.get("includes") or []),
        "snapshot_references": list(manifest.get("snapshot_references") or []),
        "accession_preview": accessions[:10],
    }


def rebuild_from_manifest(
    *,
    manifest_path: str,
    output_catalog: str | None = None,
    datasets_bin: str | None = None,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    datasets = manifest.get("datasets") or {}
    argv = list(datasets.get("argv") or [])
    if not argv:
        raise ValueError("Manifest does not contain datasets argv")

    resolved_bin = resolve_datasets_binary(datasets_bin)
    argv[0] = resolved_bin

    result = _run_datasets_json(argv)
    accessions = _extract_accessions_from_json(result["parsed"])
    if not accessions:
        raise ValueError("Rebuild query resolved zero accessions")

    catalog_meta = manifest.get("catalog") or {}
    catalog_path = str(output_catalog or catalog_meta.get("path") or "catalog.txt")
    if not os.path.exists(catalog_path):
        write_catalog(catalog_path, accessions)

    actual_hash = _sha256_text_lines(accessions)
    actual_count = len(accessions)
    stored_hash = str(catalog_meta.get("accession_sha256") or "")

    match = actual_hash == stored_hash
    return {
        "manifest_path": manifest_path,
        "catalog_path": catalog_path,
        "stored_hash": stored_hash,
        "actual_hash": actual_hash,
        "stored_count": int(catalog_meta.get("accession_count") or 0),
        "actual_count": actual_count,
        "match": bool(match),
        "datasets_argv": argv,
    }


def _immutable_copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def create_snapshot_bundle(
    *,
    catalog_path: str,
    manifest_path: str,
    metadata_files: Sequence[str] | None = None,
    snapshot_dir: str | None = None,
) -> Dict[str, Any]:
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    accessions = read_catalog(catalog_path)
    digest = _sha256_text_lines(accessions)[:12]
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = snapshot_dir or os.path.join(os.path.dirname(catalog_path) or ".", "snapshots")
    bundle_dir = os.path.join(root, f"catalog_snapshot_{ts}_{digest}")
    if os.path.exists(bundle_dir):
        raise FileExistsError(f"Snapshot already exists: {bundle_dir}")
    os.makedirs(bundle_dir, exist_ok=False)

    copied: List[str] = []
    catalog_copy = os.path.join(bundle_dir, "catalog.txt")
    manifest_copy = os.path.join(bundle_dir, "catalog.manifest.json")
    _immutable_copy(catalog_path, catalog_copy)
    _immutable_copy(manifest_path, manifest_copy)
    copied.extend([catalog_copy, manifest_copy])

    metadata_out: List[str] = []
    for item in metadata_files or []:
        if not os.path.exists(item):
            continue
        target = os.path.join(bundle_dir, os.path.basename(item))
        _immutable_copy(item, target)
        copied.append(target)
        metadata_out.append(target)

    bundle_manifest_path = os.path.join(bundle_dir, "snapshot.bundle.json")
    bundle_manifest = {
        "created_at": _utc_now_iso(),
        "source": MANIFEST_SOURCE,
        "bundle_dir": bundle_dir,
        "catalog_path": catalog_copy,
        "manifest_path": manifest_copy,
        "catalog_accession_count": len(accessions),
        "catalog_accession_sha256": _sha256_text_lines(accessions),
        "metadata_files": metadata_out,
        "files": copied,
    }
    _write_manifest(bundle_manifest_path, bundle_manifest)
    os.chmod(bundle_manifest_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    return {
        "bundle_dir": bundle_dir,
        "bundle_manifest_path": bundle_manifest_path,
        "files": copied + [bundle_manifest_path],
    }

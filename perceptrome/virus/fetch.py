from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import zipfile
from typing import Any, Dict, Iterable, List, Sequence

from perceptrome.io_utils import read_catalog
from perceptrome.virus.catalog import load_manifest
from perceptrome.virus.ncbi_datasets import (
    DatasetsCommandResult,
    download_virus_genome_by_accession,
    download_virus_genome_by_taxon,
)

EXPECTED_FILES: tuple[str, ...] = (
    "genomic.fna",
    "cds.fna",
    "protein.faa",
    "data_report.jsonl",
    "annotation_report.jsonl",
    "biosample_report.jsonl",
    "dataset_catalog.json",
    "md5sum.txt",
)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_accessions(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _source_from_args(args: argparse.Namespace) -> str:
    has_catalog = bool(getattr(args, "catalog", None))
    has_manifest = bool(getattr(args, "manifest", None))
    has_query = bool(getattr(args, "taxon", None))
    selected = int(has_catalog) + int(has_manifest) + int(has_query)
    if selected != 1:
        raise ValueError("Choose exactly one source: --catalog OR --manifest OR --taxon")
    if has_catalog:
        return "catalog"
    if has_manifest:
        return "manifest"
    return "query"


def _resolve_accessions_from_manifest(manifest_path: str) -> List[str]:
    payload = load_manifest(manifest_path)
    provenance = payload.get("provenance_refs") or {}
    accessions = provenance.get("explicit_accessions") or []
    inputfile = provenance.get("inputfile")
    out: List[str] = list(accessions)
    if inputfile:
        out.extend(read_catalog(str(inputfile)))
    out = _clean_accessions(out)
    if out:
        return out

    catalog = payload.get("catalog") or {}
    catalog_path = catalog.get("path")
    if catalog_path:
        return _clean_accessions(read_catalog(str(catalog_path)))
    raise ValueError(f"Unable to resolve accessions from manifest: {manifest_path}")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_md5sum(path: str) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            checksum, filename = parts[0], parts[-1].lstrip("*")
            checksums[str(filename)] = str(checksum)
    return checksums


def _count_sequences(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def _index_expected_files(staged_dir: str) -> List[Dict[str, Any]]:
    found: Dict[str, List[str]] = {name: [] for name in EXPECTED_FILES}
    for root, _, files in os.walk(staged_dir):
        for name in files:
            if name in found:
                rel = os.path.relpath(os.path.join(root, name), staged_dir)
                found[name].append(rel)

    inventory: List[Dict[str, Any]] = []
    md5_by_file: Dict[str, str] = {}
    for rel in found.get("md5sum.txt", []):
        md5_by_file.update(_parse_md5sum(os.path.join(staged_dir, rel)))

    for expected in EXPECTED_FILES:
        rel_paths = sorted(found[expected])
        for rel in rel_paths:
            abs_path = os.path.join(staged_dir, rel)
            entry: Dict[str, Any] = {
                "name": expected,
                "path": rel,
                "size_bytes": int(os.path.getsize(abs_path)),
            }
            if expected in {"genomic.fna", "cds.fna", "protein.faa"}:
                entry["record_count"] = _count_sequences(abs_path)
            if expected.endswith(".jsonl"):
                entry["line_count"] = _count_lines(abs_path)
            entry["sha256"] = _sha256_file(abs_path)
            md5 = md5_by_file.get(rel) or md5_by_file.get(os.path.basename(rel))
            if md5:
                entry["md5"] = md5
            inventory.append(entry)

    return inventory


def _deterministic_run_root(io_state_file: str) -> str:
    state_dir = os.path.dirname(io_state_file) or "state"
    return os.path.join(state_dir, "virus", "fetch")


def _stable_source_id(source_kind: str, source_payload: Dict[str, Any], filters: Sequence[str], includes: Sequence[str]) -> str:
    payload = {
        "source": source_kind,
        "source_payload": source_payload,
        "filters": [str(v) for v in filters],
        "includes": [str(v) for v in includes],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _extract_zip(zip_path: str, staged_dir: str) -> None:
    if os.path.exists(staged_dir):
        shutil.rmtree(staged_dir)
    os.makedirs(staged_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(staged_dir)


def fetch_virus_from_args(args: argparse.Namespace, *, io_state_file: str) -> Dict[str, Any]:
    source_kind = _source_from_args(args)
    filters = list(getattr(args, "filter", None) or [])
    includes = list(getattr(args, "include", None) or [])
    datasets_bin = getattr(args, "datasets_bin", None)

    source_payload: Dict[str, Any]
    accessions: List[str] = []
    taxon: str | None = None
    if source_kind == "catalog":
        catalog_path = str(args.catalog)
        accessions = _clean_accessions(read_catalog(catalog_path))
        source_payload = {"catalog": catalog_path, "accession_count": len(accessions)}
    elif source_kind == "manifest":
        manifest_path = str(args.manifest)
        accessions = _resolve_accessions_from_manifest(manifest_path)
        source_payload = {"manifest": manifest_path, "accession_count": len(accessions)}
    else:
        taxon = str(args.taxon)
        source_payload = {"taxon": taxon}

    if source_kind in {"catalog", "manifest"} and not accessions:
        raise ValueError(f"{source_kind} source resolved zero accessions")

    run_root = _deterministic_run_root(io_state_file)
    source_id = _stable_source_id(source_kind, source_payload, filters, includes)
    run_dir = os.path.join(run_root, source_id)
    staged_dir = os.path.join(run_dir, "staged")
    os.makedirs(run_dir, exist_ok=True)

    command_results: List[DatasetsCommandResult] = []
    archive_paths: List[str] = []

    if source_kind == "query":
        zip_path = os.path.join(run_dir, "package.query.zip")
        result = download_virus_genome_by_taxon(
            taxon=str(taxon),
            output_path=zip_path,
            datasets_bin=datasets_bin,
            extra_args=[*sum((["--include", v] for v in includes), []), *filters],
        )
        command_results.append(result)
        archive_paths.append(zip_path)
    else:
        for idx, accession in enumerate(accessions):
            zip_path = os.path.join(run_dir, f"package.{idx:04d}.{accession}.zip")
            result = download_virus_genome_by_accession(
                accession=accession,
                output_path=zip_path,
                datasets_bin=datasets_bin,
                extra_args=[*sum((["--include", v] for v in includes), []), *filters],
            )
            command_results.append(result)
            archive_paths.append(zip_path)

    if os.path.exists(staged_dir):
        shutil.rmtree(staged_dir)
    os.makedirs(staged_dir, exist_ok=True)

    for idx, zip_path in enumerate(archive_paths):
        subdir = os.path.join(staged_dir, f"bundle_{idx:04d}")
        _extract_zip(zip_path, subdir)

    inventory = _index_expected_files(staged_dir)
    now = _utc_now_iso()
    manifest = {
        "schema_version": "1.0",
        "created_at": now,
        "updated_at": now,
        "source": {
            "type": source_kind,
            **source_payload,
            "filters": [str(v) for v in filters],
            "includes": [str(v) for v in includes],
        },
        "run": {
            "id": source_id,
            "run_dir": run_dir,
            "staged_dir": staged_dir,
            "archives": archive_paths,
        },
        "files": inventory,
        "counts": {
            "archive_count": len(archive_paths),
            "indexed_file_count": len(inventory),
        },
        "provenance": {
            "command": "perceptrome virus-fetch",
            "argv": [str(v) for v in getattr(args, "_argv", [])],
            "datasets_invocations": [list(item.argv) for item in command_results],
            "datasets_return_codes": [int(item.return_code) for item in command_results],
            "datasets_stderr": [str(item.stderr) for item in command_results],
        },
    }

    manifest_path = os.path.join(run_dir, "fetch.manifest.json")
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp, manifest_path)

    return {
        "run_dir": run_dir,
        "staged_dir": staged_dir,
        "manifest_path": manifest_path,
        "source": source_kind,
        "archive_count": len(archive_paths),
        "indexed_file_count": len(inventory),
    }

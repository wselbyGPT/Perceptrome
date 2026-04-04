from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

from perceptrome.io_utils import write_catalog
from perceptrome.uniprot_api import fetch_uniprot_count, parse_accession_from_fasta_header, stream_uniprot_fasta

_FASTA_HEADER_RE = re.compile(r"^>\S+")
_FASTA_SEQ_RE = re.compile(r"^[A-Za-z*.-]+$")


@dataclass
class FastaRecord:
    header: str
    sequence: str

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.sequence}\n"


@dataclass
class ShardState:
    path: str
    record_count: int
    sha256: str


class FastaValidationError(ValueError):
    pass


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iter_fasta_records(lines: Iterable[str]) -> Iterator[FastaRecord]:
    """Incrementally parse FASTA text into validated records.

    The iterator is safe for streamed chunks/lines and keeps only one record in memory.
    """
    current_header: Optional[str] = None
    seq_parts: List[str] = []

    for raw_line in lines:
        line = (raw_line or "").strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                sequence = "".join(seq_parts)
                if not sequence:
                    raise FastaValidationError(f"FASTA record {current_header!r} has empty sequence")
                if not _FASTA_SEQ_RE.fullmatch(sequence):
                    raise FastaValidationError(f"FASTA record {current_header!r} has invalid sequence characters")
                yield FastaRecord(header=current_header, sequence=sequence)
            if not _FASTA_HEADER_RE.match(line):
                raise FastaValidationError(f"Invalid FASTA header line: {line!r}")
            current_header = line
            seq_parts = []
            continue

        if current_header is None:
            raise FastaValidationError("FASTA stream started with sequence line before any header")
        if not _FASTA_SEQ_RE.fullmatch(line):
            raise FastaValidationError(f"Invalid FASTA sequence line: {line!r}")
        seq_parts.append(line)

    if current_header is None:
        return
    sequence = "".join(seq_parts)
    if not sequence:
        raise FastaValidationError(f"FASTA record {current_header!r} has empty sequence")
    if not _FASTA_SEQ_RE.fullmatch(sequence):
        raise FastaValidationError(f"FASTA record {current_header!r} has invalid sequence characters")
    yield FastaRecord(header=current_header, sequence=sequence)


def _shard_path(prefix_path: str, shard_index: int, use_gzip: bool) -> str:
    ext = ".fasta.gz" if use_gzip else ".fasta"
    return f"{prefix_path}.part-{shard_index:05d}{ext}"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_complete_resume(*, manifest_path: str, prefix_path: str, use_gzip: bool, records_per_shard: int) -> Optional[Dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    if not manifest:
        return None
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        return None

    for index, shard in enumerate(shards, start=1):
        shard_path = str(shard.get("path") or "")
        expected_path = _shard_path(prefix_path, index, use_gzip)
        if shard_path != expected_path:
            return None
        if not os.path.exists(shard_path):
            return None
        if int(shard.get("record_count") or 0) <= 0:
            return None
        if int(shard.get("record_count") or 0) > int(records_per_shard):
            return None
        if str(shard.get("sha256") or "") != _sha256_file(shard_path):
            return None

    return manifest


class ShardWriter:
    def __init__(self, *, prefix_path: str, records_per_shard: int, use_gzip: bool):
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be > 0")
        self.prefix_path = prefix_path
        self.records_per_shard = records_per_shard
        self.use_gzip = use_gzip
        self._shard_index = 0
        self._current_count = 0
        self._total_records = 0
        self._total_residues = 0
        self._current_path: Optional[str] = None
        self._current_handle: Any = None
        self._current_hasher: Optional[hashlib._Hash] = None
        self.shards: List[ShardState] = []

    def _open_next(self) -> None:
        self._shard_index += 1
        self._current_count = 0
        self._current_path = _shard_path(self.prefix_path, self._shard_index, self.use_gzip)
        os.makedirs(os.path.dirname(self._current_path) or ".", exist_ok=True)
        self._current_hasher = hashlib.sha256()
        if self.use_gzip:
            self._current_handle = gzip.open(self._current_path, "wt", encoding="utf-8")
        else:
            self._current_handle = open(self._current_path, "w", encoding="utf-8")

    def _write_raw(self, text: str) -> None:
        assert self._current_handle is not None
        assert self._current_hasher is not None
        self._current_handle.write(text)
        self._current_hasher.update(text.encode("utf-8"))

    def _close_current(self) -> None:
        if self._current_handle is None:
            return
        self._current_handle.close()
        assert self._current_path is not None
        assert self._current_hasher is not None
        self.shards.append(
            ShardState(
                path=self._current_path,
                record_count=self._current_count,
                sha256=self._current_hasher.hexdigest(),
            )
        )
        self._current_handle = None
        self._current_path = None
        self._current_hasher = None

    def add_record(self, record: FastaRecord) -> None:
        if self._current_handle is None:
            self._open_next()
        if self._current_count >= self.records_per_shard:
            self._close_current()
            self._open_next()
        self._write_raw(record.text)
        self._current_count += 1
        self._total_records += 1
        self._total_residues += len(record.sequence)

    def close(self) -> None:
        self._close_current()

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def total_residues(self) -> int:
        return self._total_residues


def build_manifest(*, query: str, include_isoforms: bool, total_records: int, total_residues: int, shards: List[ShardState], accession_preview: List[str], live_count: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    avg_len = (float(total_residues) / float(total_records)) if total_records else 0.0
    payload: Dict[str, Any] = {
        "source": "uniprot",
        "query": query,
        "include_isoforms": bool(include_isoforms),
        "timestamp": _iso_utc_now(),
        "total_records": int(total_records),
        "total_residues": int(total_residues),
        "average_length": avg_len,
        "shards": [
            {
                "path": shard.path,
                "record_count": int(shard.record_count),
                "sha256": shard.sha256,
            }
            for shard in shards
        ],
        "accession_preview": accession_preview,
    }
    if live_count:
        payload["live_count"] = live_count
    return payload


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def write_accession_catalog(path: str, *, accessions: List[str], query: str) -> None:
    header = [
        "Perceptrome UniProt accession catalog",
        f"source: uniprot",
        f"query: {query}",
        f"generated_at: {_iso_utc_now()}",
    ]
    write_catalog(path, accessions, header=header)


def fetch_uniprot_dataset(*, query: str, include_isoforms: bool, prefix_path: str, records_per_shard: int, use_gzip: bool, resume: bool, timeout: float, max_retries: int, backoff_seconds: float, count_only: bool = False, base_url: Optional[str] = None) -> Dict[str, Any]:
    live_count = fetch_uniprot_count(
        query,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        base_url=base_url,
    )
    if count_only:
        return {
            "count_only": True,
            "query": query,
            "include_isoforms": include_isoforms,
            "live_count": live_count,
        }

    manifest_path = f"{prefix_path}.manifest.json"
    catalog_path = f"{prefix_path}.catalog.txt"

    if resume:
        resumed_manifest = detect_complete_resume(
            manifest_path=manifest_path,
            prefix_path=prefix_path,
            use_gzip=use_gzip,
            records_per_shard=records_per_shard,
        )
        if resumed_manifest is not None:
            return {
                "resumed": True,
                "manifest_path": manifest_path,
                "catalog_path": catalog_path,
                "manifest": resumed_manifest,
                "live_count": live_count,
            }

    writer = ShardWriter(prefix_path=prefix_path, records_per_shard=records_per_shard, use_gzip=use_gzip)
    preview: List[str] = []
    all_accessions: List[str] = []

    fasta_lines = stream_uniprot_fasta(
        query,
        include_isoform=include_isoforms,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        base_url=base_url,
    )

    for record in iter_fasta_records(fasta_lines):
        writer.add_record(record)
        accession = parse_accession_from_fasta_header(record.header)
        if accession:
            all_accessions.append(accession)
            if len(preview) < 16:
                preview.append(accession)
    writer.close()

    manifest = build_manifest(
        query=query,
        include_isoforms=include_isoforms,
        total_records=writer.total_records,
        total_residues=writer.total_residues,
        shards=writer.shards,
        accession_preview=preview,
        live_count=live_count,
    )
    write_manifest(manifest_path, manifest)
    write_accession_catalog(catalog_path, accessions=all_accessions, query=query)

    return {
        "resumed": False,
        "manifest_path": manifest_path,
        "catalog_path": catalog_path,
        "manifest": manifest,
        "live_count": live_count,
    }

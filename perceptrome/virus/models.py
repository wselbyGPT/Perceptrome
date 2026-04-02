from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VirusCatalogQuery:
    """Query inputs used to build an NCBI virus catalog manifest."""

    mode: str
    taxon: str | None = None
    accessions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VirusRecordSummary:
    """Manifest-safe summary information for a single virus genome record."""

    accession: str
    organism_name: str | None = None
    tax_id: int | None = None
    sequence_length: int | None = None


@dataclass(frozen=True)
class VirusPackagePaths:
    """Disk locations for downloaded NCBI Datasets artifacts."""

    package_zip_path: str
    metadata_jsonl_path: str | None = None
    readme_path: str | None = None


@dataclass(frozen=True)
class VirusCatalogManifest:
    """Deterministic, JSON-serializable metadata for a virus catalog run."""

    source: str
    generated_at: str
    query: VirusCatalogQuery
    records: tuple[VirusRecordSummary, ...] = field(default_factory=tuple)
    package_paths: VirusPackagePaths | None = None
    extras: dict[str, Any] = field(default_factory=dict)

import logging
import os
from typing import Any, Dict, Optional

import requests

from .config import NCBIConfig, IOConfig
from .io_utils import software_snapshot, utc_now_iso, write_manifest


def fetch_fasta(
    accession: str,
    io_cfg: IOConfig,
    ncbi_cfg: NCBIConfig,
    force: bool = False,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """Download FASTA for accession from NCBI nuccore, returning local path."""
    out_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    fetch_logger = logging.getLogger("fetch")

    if os.path.exists(out_path) and not force:
        fetch_logger.info(f"{accession}: using cached FASTA at {out_path}")
        if not os.path.exists(f"{out_path}.manifest.json"):
            write_manifest(
                out_path,
                {
                    "accession": accession,
                    "artifact": {"kind": "fasta", "path": out_path},
                    "fetch": {
                        "timestamp_utc": utc_now_iso(),
                        "source_type": "fasta",
                        "provider": "ncbi.nuccore",
                        "cache_hit": True,
                    },
                    "effective_settings": {},
                    "filtering": {},
                    "software": software_snapshot(config_snapshot=config_snapshot),
                },
            )
        return out_path

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "email": ncbi_cfg.email,
    }
    if ncbi_cfg.api_key:
        params["api_key"] = ncbi_cfg.api_key  # type: ignore[assignment]

    fetch_logger.info(f"{accession}: fetching from NCBI {url}")
    last_error: Optional[str] = None
    for attempt in range(ncbi_cfg.max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200 and resp.text.strip().startswith(">"):
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                write_manifest(
                    out_path,
                    {
                        "accession": accession,
                        "artifact": {"kind": "fasta", "path": out_path},
                        "fetch": {
                            "timestamp_utc": utc_now_iso(),
                            "source_type": "fasta",
                            "provider": "ncbi.nuccore",
                        },
                        "effective_settings": {},
                        "filtering": {},
                        "software": software_snapshot(config_snapshot=config_snapshot),
                    },
                )
                fetch_logger.info(
                    f"{accession}: fetched and saved FASTA ({len(resp.text)} bytes)"
                )
                return out_path
            else:
                msg = (
                    f"HTTP {resp.status_code}, "
                    f"first 80 chars: {resp.text[:80]!r}"
                )
                last_error = msg
        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        if attempt < ncbi_cfg.max_retries:
            delay = ncbi_cfg.backoff_seconds * (2 ** attempt)
            fetch_logger.warning(
                f"{accession}: fetch attempt {attempt+1} failed ({last_error}); "
                f"retrying in {delay:.1f}s"
            )
            import time
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {accession} from NCBI: {last_error}")


def fetch_genbank(
    accession: str,
    io_cfg: IOConfig,
    ncbi_cfg: NCBIConfig,
    force: bool = False,
    rettype: str = "gbwithparts",
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """Download GenBank flatfile for accession from NCBI nuccore, returning local path."""
    out_path = os.path.join(getattr(io_cfg, "cache_genbank_dir", "cache/genbank"), f"{accession}.gb")
    fetch_logger = logging.getLogger("fetch")

    if os.path.exists(out_path) and not force:
        fetch_logger.info(f"{accession}: using cached GenBank at {out_path}")
        if not os.path.exists(f"{out_path}.manifest.json"):
            write_manifest(
                out_path,
                {
                    "accession": accession,
                    "artifact": {"kind": "genbank", "path": out_path},
                    "fetch": {
                        "timestamp_utc": utc_now_iso(),
                        "source_type": "genbank",
                        "provider": "ncbi.nuccore",
                        "cache_hit": True,
                    },
                    "effective_settings": {"rettype": rettype},
                    "filtering": {},
                    "software": software_snapshot(config_snapshot=config_snapshot),
                },
            )
        return out_path

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": rettype,
        "retmode": "text",
        "email": ncbi_cfg.email,
    }
    if ncbi_cfg.api_key:
        params["api_key"] = ncbi_cfg.api_key  # type: ignore[assignment]

    fetch_logger.info(f"{accession}: fetching GenBank from NCBI {url}")
    last_error: Optional[str] = None
    for attempt in range(ncbi_cfg.max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            # GenBank flatfile usually begins with 'LOCUS'
            if resp.status_code == 200 and resp.text.lstrip().startswith("LOCUS"):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                write_manifest(
                    out_path,
                    {
                        "accession": accession,
                        "artifact": {"kind": "genbank", "path": out_path},
                        "fetch": {
                            "timestamp_utc": utc_now_iso(),
                            "source_type": "genbank",
                            "provider": "ncbi.nuccore",
                        },
                        "effective_settings": {"rettype": rettype},
                        "filtering": {},
                        "software": software_snapshot(config_snapshot=config_snapshot),
                    },
                )
                fetch_logger.info(
                    f"{accession}: fetched and saved GenBank ({len(resp.text)} bytes)"
                )
                return out_path
            else:
                msg = (
                    f"HTTP {resp.status_code}, "
                    f"first 80 chars: {resp.text[:80]!r}"
                )
                last_error = msg
        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        if attempt < ncbi_cfg.max_retries:
            delay = ncbi_cfg.backoff_seconds * (2 ** attempt)
            fetch_logger.warning(
                f"{accession}: GenBank fetch attempt {attempt+1} failed ({last_error}); "
                f"retrying in {delay:.1f}s"
            )
            import time
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch GenBank for {accession} from NCBI: {last_error}")

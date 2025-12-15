# server/main.py
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------------------------------------------------------
# Path / import setup (ensure repo root on sys.path)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("perceptrome")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------
# Optional imports (perceptrome core + biopython)
# -----------------------------------------------------------------------------
try:
    import perceptrome  # noqa: F401
except Exception:
    perceptrome = None  # type: ignore

try:
    from perceptrome import generate as perceptrome_generate  # type: ignore
except Exception:
    perceptrome_generate = None  # type: ignore

try:
    from Bio import SeqIO  # type: ignore
except Exception:
    SeqIO = None  # type: ignore

# -----------------------------------------------------------------------------
# App
# NOTE: nginx is proxying /api/* -> uvicorn WITHOUT the /api prefix (proxy_pass .../).
# So these routes are defined WITHOUT /api here:
#   /health, /catalogs, /cache, /genome/{acc}/summary, etc.
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Perceptrome API",
    version=getattr(perceptrome, "__version__", "dev") if perceptrome else "dev",
)

# -----------------------------------------------------------------------------
# Constants / dirs
# -----------------------------------------------------------------------------
SAFE_ACC_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")

CACHE_DIR = ROOT / "cache"
CACHE_GENBANK = CACHE_DIR / "genbank"
CACHE_FASTA = CACHE_DIR / "fasta"
GENERATED_DIR = ROOT / "generated"
CATALOGS_DIR = ROOT / "catalogs"

GENBANK_EXTS = (".gb", ".gbk", ".gbff", ".genbank")
FASTA_EXTS = (".fasta", ".fa", ".fna", ".ffn")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_acc(acc: str) -> str:
    acc = (acc or "").strip()
    if not acc or not SAFE_ACC_RE.match(acc):
        raise HTTPException(status_code=400, detail="Invalid accession")
    return acc


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    if not name or not SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    return name


def _stat(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {
        "name": p.name,
        "path": str(p.resolve()),
        "bytes": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def _find_cached_record(acc: str) -> Tuple[str, Path]:
    """
    Return ("genbank", path) or ("fasta", path). Prefer GenBank if present.
    """
    for ext in GENBANK_EXTS:
        p = CACHE_GENBANK / f"{acc}{ext}"
        if p.exists() and p.is_file():
            return "genbank", p
    for ext in FASTA_EXTS:
        p = CACHE_FASTA / f"{acc}{ext}"
        if p.exists() and p.is_file():
            return "fasta", p
    raise HTTPException(
        status_code=404,
        detail=f"No cached file for {acc} in cache/genbank or cache/fasta.",
    )


def _read_fasta_seq(fp: Path) -> str:
    seq_parts: List[str] = []
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip())
    return "".join(seq_parts).upper()


def _count_bases(seq: str) -> Dict[str, int]:
    # Fast, explicit (avoid surprises with non-ACGTN).
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0, "OTHER": 0}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
        elif ch in ("U",):  # tolerate RNA
            counts["OTHER"] += 1
        else:
            counts["OTHER"] += 1
    return counts


def _pct(num: float, den: float) -> float:
    return (num / den) * 100.0 if den else 0.0


def _features_summary_from_genbank(fp: Path) -> Dict[str, Any]:
    if SeqIO is None:
        # Keep API up even if biopython is missing (features become unknown).
        return {"total": 0, "by_type": {}, "top": []}

    with fp.open("r", encoding="utf-8", errors="replace") as f:
        rec = next(SeqIO.parse(f, "genbank"), None)
    if rec is None:
        return {"total": 0, "by_type": {}, "top": []}

    by_type: Dict[str, int] = {}
    feats = getattr(rec, "features", []) or []
    for feat in feats:
        t = getattr(feat, "type", "") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1

    top = sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {
        "total": int(len(feats)),
        "by_type": by_type,
        "top": [{"type": t, "count": c} for t, c in top],
    }


def _seq_len_from_genbank(fp: Path) -> Optional[int]:
    if SeqIO is None:
        return None
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        rec = next(SeqIO.parse(f, "genbank"), None)
    if rec is None:
        return None
    return int(len(rec.seq))


def _seq_counts_from_genbank(fp: Path) -> Tuple[int, Dict[str, int]]:
    if SeqIO is None:
        raise HTTPException(status_code=500, detail="Biopython is required to parse GenBank.")
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        rec = next(SeqIO.parse(f, "genbank"), None)
    if rec is None:
        raise HTTPException(status_code=500, detail="Failed to parse GenBank record.")
    seq = str(rec.seq).upper()
    return len(seq), _count_bases(seq)


def _walk_dir_stats(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"path": str(p), "exists": False, "files": 0, "bytes": 0}
    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(p):
        for fn in files:
            fp = Path(root) / fn
            try:
                st = fp.stat()
            except Exception:
                continue
            total_files += 1
            total_bytes += int(st.st_size)
    return {"path": str(p.resolve()), "exists": True, "files": total_files, "bytes": total_bytes}


# -----------------------------------------------------------------------------
# Basic endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "perceptrome",
        "version": app.version,
        "time": now_utc_iso(),
        "hint": "Try /health, /catalogs, /cache, /genome/{acc}/summary",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "time": now_utc_iso(),
        "version": app.version,
        "perceptrome_imported": bool(perceptrome is not None),
        "generate_available": bool(callable(perceptrome_generate)),
    }


@app.get("/cache")
def cache_info():
    return {
        "time": now_utc_iso(),
        "cache": {
            "root": _walk_dir_stats(CACHE_DIR),
            "genbank": _walk_dir_stats(CACHE_GENBANK),
            "fasta": _walk_dir_stats(CACHE_FASTA),
        },
        "generated": _walk_dir_stats(GENERATED_DIR),
    }


@app.get("/catalogs")
def catalogs():
    """
    Lists catalog files (best-effort). Looks in ./catalogs if present,
    otherwise returns an empty list.
    """
    out: List[Dict[str, Any]] = []
    if not CATALOGS_DIR.exists():
        return {"time": now_utc_iso(), "dir": str(CATALOGS_DIR), "catalogs": out}

    for p in sorted(CATALOGS_DIR.glob("*")):
        if not p.is_file():
            continue
        # Only common text/json extensions; adjust if you want.
        if p.suffix.lower() not in (".txt", ".tsv", ".csv", ".json"):
            continue
        row = _stat(p)
        # line count (best-effort)
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                row["lines"] = sum(1 for _ in f)
        except Exception:
            row["lines"] = None
        out.append(row)

    return {"time": now_utc_iso(), "dir": str(CATALOGS_DIR.resolve()), "catalogs": out}


# -----------------------------------------------------------------------------
# Genome inspector: summary (this is what is working for you now)
# -----------------------------------------------------------------------------
@app.get("/genome/{acc}/summary")
def genome_summary(acc: str):
    acc = _check_acc(acc)
    source, fp = _find_cached_record(acc)

    if source == "genbank":
        length, counts = _seq_counts_from_genbank(fp)
        features = _features_summary_from_genbank(fp)
    else:
        seq = _read_fasta_seq(fp)
        length = len(seq)
        counts = _count_bases(seq)
        features = {"total": 0, "by_type": {}, "top": []}

    gc = counts["G"] + counts["C"]
    n = counts["N"]
    gc_pct = _pct(gc, length)
    n_pct = _pct(n, length)

    return {
        "accession": acc,
        "source": source,
        "file": _stat(fp),
        "length": int(length),
        "gc_pct": float(gc_pct),
        "n_pct": float(n_pct),
        "counts": counts,
        "features": features,
    }


# -----------------------------------------------------------------------------
# Optional: generation endpoint (kept flexible so it won’t break your core)
# -----------------------------------------------------------------------------
def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    # last resort
    return str(x)


@app.post("/generate")
async def generate(payload: Dict[str, Any] = Body(default_factory=dict)):
    if not callable(perceptrome_generate):
        raise HTTPException(status_code=503, detail="perceptrome.generate not available on this server.")

    # Best-effort: map payload keys to the function signature.
    try:
        sig = inspect.signature(perceptrome_generate)  # type: ignore[arg-type]
        params = sig.parameters
    except Exception:
        params = {}

    try:
        if "payload" in params:
            result = perceptrome_generate(payload=payload)  # type: ignore[misc]
        else:
            kwargs = {k: v for k, v in payload.items() if k in params} if params else {}
            if kwargs:
                result = perceptrome_generate(**kwargs)  # type: ignore[misc]
            else:
                # Fall back to a single positional payload if possible
                result = perceptrome_generate(payload)  # type: ignore[misc]
    except TypeError:
        # Last-ditch
        result = perceptrome_generate(payload=payload)  # type: ignore[misc]
    except Exception as e:
        log.exception("generate failed")
        raise HTTPException(status_code=500, detail=f"Generate failed: {e}")

    return JSONResponse(_jsonable(result))


# -----------------------------------------------------------------------------
# Generated files (minimal utility for the dashboard)
# -----------------------------------------------------------------------------
@app.get("/generated")
def generated_list(
    limit: int = Query(default=200, ge=1, le=5000),
):
    out: List[Dict[str, Any]] = []
    if not GENERATED_DIR.exists():
        return {"time": now_utc_iso(), "dir": str(GENERATED_DIR), "files": out}

    for p in sorted(GENERATED_DIR.glob("*")):
        if not p.is_file():
            continue
        out.append(_stat(p))
        if len(out) >= limit:
            break

    return {"time": now_utc_iso(), "dir": str(GENERATED_DIR.resolve()), "files": out}


@app.get("/generated/{name}")
def generated_get(name: str):
    name = _safe_name(name)
    fp = GENERATED_DIR / name
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="Generated file not found")
    # Let the browser decide; common for .fasta/.gb/.json etc.
    return FileResponse(str(fp), filename=fp.name)


# -----------------------------------------------------------------------------
# Mount Genome Extras (features + map.svg + map.pdf)
# IMPORTANT: these routes are required by web/js/inspector.js.
# -----------------------------------------------------------------------------
try:
    import server.api_genome_extras as gx  # <-- your file: server/api_genome_extras.py

    app.include_router(gx.router)
    log.info("Mounted genome extras router: %s", getattr(gx.router, "prefix", "?"))
except Exception as e:
    # Keep server up even if extras module has issues; log it loudly.
    log.warning("Genome extras router NOT mounted: %s", e, exc_info=True)

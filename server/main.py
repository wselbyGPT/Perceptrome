from __future__ import annotations

import inspect
import math
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import perceptrome  # noqa: F401
    from perceptrome import generate as perceptrome_generate  # type: ignore
except Exception:
    perceptrome = None  # type: ignore
    perceptrome_generate = None  # type: ignore

app = FastAPI(
    title="Perceptrome API",
    version=getattr(perceptrome, "__version__", "dev") if perceptrome else "dev",
)

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_ACC_RE = re.compile(r"^[A-Za-z0-9._-]+$")

GENBANK_EXTS = [".gb", ".gbk", ".gbff", ".gbf"]
FASTA_EXTS = [".fasta", ".fa", ".fna", ".ffn", ".faa", ".txt", ".fsa"]
SVG_EXTS = [".svg"]

_SEQ_CLEAN_RE = re.compile(r"[^A-Za-z]")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_info(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {
        "name": p.name,
        "path": str(p),
        "bytes": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


# --------------------------
# Catalogs + Cache
# --------------------------

def resolve_accession_file(name: str) -> Path:
    if not SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid name.")

    accessions_dir = ROOT / "accessions"
    if not accessions_dir.exists():
        raise HTTPException(status_code=500, detail="accessions/ directory not found.")

    candidates: List[Path] = []
    n = name
    if n.endswith(".txt"):
        candidates.append(accessions_dir / n)
    else:
        candidates.append(accessions_dir / f"{n}.txt")
        candidates.append(accessions_dir / f"{n}_accessions.txt")

    for p in accessions_dir.glob("*.txt"):
        if p.stem == n:
            candidates.insert(0, p)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    raise HTTPException(status_code=404, detail=f"Catalog '{name}' not found in accessions/.")


def read_lines_slice(path: Path, offset: int, limit: int) -> List[str]:
    if offset < 0 or limit <= 0 or limit > 20000:
        raise HTTPException(status_code=400, detail="Bad offset/limit.")
    out: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(offset):
            if f.readline() == "":
                return []
        for _ in range(limit):
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            if line:
                out.append(line)
    return out


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time_utc": now_iso(), "repo_root": str(ROOT), "version": app.version}


@app.get("/catalogs")
def catalogs(with_counts: bool = Query(False)) -> Dict[str, Any]:
    d = ROOT / "accessions"
    if not d.exists():
        raise HTTPException(status_code=500, detail="accessions/ directory not found.")

    files = sorted(d.glob("*.txt"), key=lambda p: p.name)
    items: List[Dict[str, Any]] = []
    for p in files:
        info = file_info(p)
        info["catalog"] = p.stem
        if with_counts:
            info["lines"] = count_lines(p)
        items.append(info)

    return {"count": len(items), "items": items}


@app.get("/catalogs/{name}")
def catalog_preview(
    name: str,
    limit: int = Query(200, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    with_total: bool = Query(False),
) -> Dict[str, Any]:
    path = resolve_accession_file(name)
    lines = read_lines_slice(path, offset=offset, limit=limit)
    resp: Dict[str, Any] = {
        "name": name,
        "resolved": path.name,
        "offset": offset,
        "limit": limit,
        "returned": len(lines),
        "items": lines,
    }
    if with_total:
        resp["total_lines"] = count_lines(path)
    return resp


def list_dir(dirpath: Path, limit: int, exts: Optional[List[str]] = None) -> Dict[str, Any]:
    if not dirpath.exists():
        return {"exists": False, "path": str(dirpath), "count": 0, "items": []}

    files: List[Path] = []
    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        files.append(p)

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items = [file_info(p) for p in files[:limit]]
    return {"exists": True, "path": str(dirpath), "count": len(files), "items": items}


@app.get("/cache")
def cache(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    c = ROOT / "cache"
    return {
        "root": str(c),
        "genbank": list_dir(c / "genbank", limit=limit, exts=GENBANK_EXTS),
        "fasta": list_dir(c / "fasta", limit=limit, exts=FASTA_EXTS),
        "encoded": list_dir(c / "encoded", limit=limit, exts=[".npy"]),
    }


# --------------------------
# Parsing + summary helpers
# --------------------------

def _resolve_cached_genome_file(acc: str) -> Tuple[str, Path]:
    if not SAFE_ACC_RE.match(acc):
        raise HTTPException(status_code=400, detail="Invalid accession.")

    cache_root = ROOT / "cache"
    gb_dir = cache_root / "genbank"
    fa_dir = cache_root / "fasta"

    for ext in GENBANK_EXTS:
        p = gb_dir / f"{acc}{ext}"
        if p.exists():
            return ("genbank", p)
    for ext in FASTA_EXTS:
        p = fa_dir / f"{acc}{ext}"
        if p.exists():
            return ("fasta", p)

    if gb_dir.exists():
        for p in gb_dir.iterdir():
            if p.is_file() and p.suffix.lower() in GENBANK_EXTS and p.stem == acc:
                return ("genbank", p)
    if fa_dir.exists():
        for p in fa_dir.iterdir():
            if p.is_file() and p.suffix.lower() in FASTA_EXTS and p.stem == acc:
                return ("fasta", p)

    raise HTTPException(status_code=404, detail=f"No cached file for {acc} in cache/genbank or cache/fasta.")


def _seq_stats_dna(seq: str) -> Dict[str, Any]:
    s = seq.upper()
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0, "OTHER": 0}
    for ch in s:
        if ch in counts:
            counts[ch] += 1
        else:
            counts["OTHER"] += 1

    length = len(s)
    acgt = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    gc = counts["G"] + counts["C"]
    gc_pct = (gc / acgt * 100.0) if acgt > 0 else None
    n_pct = (counts["N"] / length * 100.0) if length > 0 else None
    return {"length": length, "counts": counts, "gc_pct": gc_pct, "n_pct": n_pct, "acgt": acgt}


def _parse_fasta_sequence(path: Path) -> str:
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                continue
            line = line.strip()
            if not line:
                continue
            seq_parts.append(_SEQ_CLEAN_RE.sub("", line))
    return "".join(seq_parts)


def _parse_genbank_sequence_and_features(path: Path) -> Tuple[str, Dict[str, Any]]:
    seq_parts: List[str] = []
    in_origin = False
    in_features = False
    feat_counts: Dict[str, int] = {}
    total_feats = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if line.startswith("FEATURES"):
                in_features = True
                in_origin = False
                continue

            if line.startswith("ORIGIN"):
                in_origin = True
                in_features = False
                continue

            if line.startswith("//"):
                break

            if in_features:
                if len(line) >= 6 and line[:5] == "     " and line[5] != " ":
                    key = line[5:].split()[0]
                    feat_counts[key] = feat_counts.get(key, 0) + 1
                    total_feats += 1
                continue

            if in_origin:
                seq_parts.append(_SEQ_CLEAN_RE.sub("", line))

    seq = "".join(seq_parts)
    top = sorted(feat_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    feats = {"total": total_feats, "by_type": feat_counts, "top": [{"type": k, "count": v} for k, v in top]}
    return seq, feats


@app.get("/genome/{acc}/summary")
def genome_summary(acc: str) -> Dict[str, Any]:
    source, path = _resolve_cached_genome_file(acc)
    if source == "fasta":
        seq = _parse_fasta_sequence(path)
        feats = None
    else:
        seq, feats = _parse_genbank_sequence_and_features(path)

    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGTN]", "N", seq)
    stats = _seq_stats_dna(seq)

    return {
        "accession": acc,
        "source": source,
        "file": file_info(path),
        "length": stats["length"],
        "gc_pct": stats["gc_pct"],
        "n_pct": stats["n_pct"],
        "counts": stats["counts"],
        "features": feats,
    }


# --------------------------
# Generation jobs
# --------------------------

class GenerateRequest(BaseModel):
    kind: str = Field("genome", description="genome or protein")
    length: int = Field(2000, ge=10, le=20_000_000)
    n: int = Field(1, ge=1, le=50)
    seed: Optional[int] = Field(None, ge=0, le=2_147_483_647)
    temperature: Optional[float] = Field(None, ge=0.05, le=5.0)
    top_p: Optional[float] = Field(None, ge=0.01, le=1.0)
    notes: Optional[str] = Field(None, max_length=2000)


@dataclass
class Job:
    id: str
    kind: str
    created_utc: str
    status: str = "queued"  # queued|running|done|error
    message: str = ""
    progress: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    result_files: List[Dict[str, Any]] = field(default_factory=list)
    result_stats: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    error: Optional[str] = None
    ended_utc: Optional[str] = None


_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Job] = {}


def _job_get(job_id: str) -> Job:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job not found.")
        return j


def _job_put(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job


def _jobs_list() -> List[Job]:
    with _JOBS_LOCK:
        return list(_JOBS.values())


def _safe_kind(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k not in ("genome", "protein"):
        raise HTTPException(status_code=400, detail="kind must be 'genome' or 'protein'.")
    return k


def _next_generated_name(prefix: str, ext: str) -> str:
    outdir = ROOT / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in outdir.glob(f"{prefix}_*.{ext}"):
        m = re.search(r"_(\d+)\." + re.escape(ext) + r"$", p.name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    n = (max(nums) + 1) if nums else 1
    return f"{prefix}_{n:03d}.{ext}"


def _wrap_fasta(header: str, seq: str, width: int = 70) -> str:
    lines = [f">{header}"]
    for i in range(0, len(seq), width):
        lines.append(seq[i:i + width])
    return "\n".join(lines) + "\n"


def _compute_base_freq_from_cache(max_files: int = 30, max_total_chars: int = 3_000_000) -> Dict[str, float]:
    gb_dir = ROOT / "cache" / "genbank"
    fa_dir = ROOT / "cache" / "fasta"
    files: List[Path] = []

    if gb_dir.exists():
        files += [p for p in gb_dir.iterdir() if p.is_file() and p.suffix.lower() in GENBANK_EXTS]
    if fa_dir.exists():
        files += [p for p in fa_dir.iterdir() if p.is_file() and p.suffix.lower() in FASTA_EXTS]

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[:max_files]

    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    total = 0

    for p in files:
        if total >= max_total_chars:
            break
        try:
            if p.suffix.lower() in GENBANK_EXTS:
                seq, _ = _parse_genbank_sequence_and_features(p)
            else:
                seq = _parse_fasta_sequence(p)
            s = seq.upper().replace("U", "T")
            s = re.sub(r"[^ACGT]", "", s)
            for ch in s:
                counts[ch] += 1
                total += 1
                if total >= max_total_chars:
                    break
        except Exception:
            continue

    if total <= 0:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    return {k: counts[k] / total for k in counts}


def _sample_genome(length: int, seed: Optional[int], freq: Dict[str, float]) -> str:
    import random
    rng = random.Random(seed if seed is not None else None)
    keys = ["A", "C", "G", "T"]
    w = [max(0.0, float(freq.get(k, 0.0))) for k in keys]
    s = sum(w) or 1.0
    w = [x / s for x in w]
    c = []
    acc = 0.0
    for x in w:
        acc += x
        c.append(acc)
    out = []
    for _ in range(length):
        r = rng.random()
        if r <= c[0]:
            out.append("A")
        elif r <= c[1]:
            out.append("C")
        elif r <= c[2]:
            out.append("G")
        else:
            out.append("T")
    return "".join(out)


_AA = "ACDEFGHIKLMNPQRSTVWY"
def _sample_protein(length: int, seed: Optional[int]) -> str:
    import random
    rng = random.Random(seed if seed is not None else None)
    return "".join(rng.choice(_AA) for _ in range(length))


def _try_call_perceptrome_generator(kind: str, length: int, seed: Optional[int], temperature: Optional[float], top_p: Optional[float]) -> Optional[Tuple[str, str]]:
    mod = perceptrome_generate
    if mod is None:
        return None

    candidates = []
    for name in dir(mod):
        if "gen" in name.lower() and callable(getattr(mod, name)):
            candidates.append(name)

    prefer = ["generate"]
    if kind == "genome":
        prefer = ["generate_plasmid", "generate_genome", "generate_sequence", "generate"]
    else:
        prefer = ["generate_protein", "generate_aa", "generate_sequence", "generate"]

    ordered = prefer + [n for n in candidates if n not in prefer]

    for fn_name in ordered:
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            for key, val in [
                ("kind", kind),
                ("mode", kind),
                ("length", length),
                ("seq_len", length),
                ("n_tokens", length),
                ("seed", seed),
                ("temperature", temperature),
                ("temp", temperature),
                ("top_p", top_p),
                ("p", top_p),
            ]:
                if key in sig.parameters and val is not None:
                    kwargs[key] = val

            result = fn(**kwargs)  # type: ignore

            if isinstance(result, str):
                return (fn_name, result)
            if isinstance(result, dict):
                seq = result.get("sequence") or result.get("seq")
                if isinstance(seq, str) and seq:
                    return (fn_name, seq)
            if isinstance(result, (list, tuple)) and result and all(isinstance(x, str) for x in result):
                return (fn_name, "".join(result))
        except Exception:
            continue

    return None


def _run_generation_job(job_id: str) -> None:
    job = _job_get(job_id)
    job.status = "running"
    job.message = "Generating…"
    job.progress = 0.05
    _job_put(job)

    req = job.params
    kind = req["kind"]
    length = int(req["length"])
    n = int(req["n"])
    seed = req.get("seed")
    temperature = req.get("temperature")
    top_p = req.get("top_p")
    notes = req.get("notes")

    outdir = ROOT / "generated"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        method = ""
        sequences: List[str] = []
        adapter = _try_call_perceptrome_generator(kind, length, seed, temperature, top_p)
        if adapter is not None:
            method, seq0 = adapter
            sequences = [seq0]
            for i in range(1, n):
                adapter_i = _try_call_perceptrome_generator(kind, length, (seed + i) if seed is not None else None, temperature, top_p)
                if adapter_i is None:
                    break
                sequences.append(adapter_i[1])
        else:
            if kind == "genome":
                job.message = "Estimating base frequencies from cache…"
                job.progress = 0.10
                _job_put(job)
                freq = _compute_base_freq_from_cache()
                method = "fallback_cache_freq"
                for i in range(n):
                    sequences.append(_sample_genome(length, (seed + i) if seed is not None else None, freq))
            else:
                method = "fallback_uniform_aa"
                for i in range(n):
                    sequences.append(_sample_protein(length, (seed + i) if seed is not None else None))

        job.method = method
        job.message = "Writing outputs…"
        job.progress = 0.75
        _job_put(job)

        files_out: List[Dict[str, Any]] = []
        stats_out: List[Dict[str, Any]] = []

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        for i, seq in enumerate(sequences):
            seq = _SEQ_CLEAN_RE.sub("", seq)

            if kind == "genome":
                seq = seq.upper().replace("U", "T")
                seq = re.sub(r"[^ACGTN]", "N", seq)
                ext = "fasta"
                prefix = "perceptrome_novel_genome"
                name = _next_generated_name(prefix, ext)
                header = f"{name} kind=genome len={len(seq)} seed={seed} i={i} method={method} utc={ts}"
            else:
                seq = seq.upper()
                seq = re.sub(rf"[^{_AA}X]", "X", seq)
                ext = "faa"
                prefix = "perceptrome_novel_protein"
                name = _next_generated_name(prefix, ext)
                header = f"{name} kind=protein len={len(seq)} seed={seed} i={i} method={method} utc={ts}"

            if notes:
                header += f" notes={notes[:120].replace(' ', '_')}"

            text = _wrap_fasta(header, seq)
            path = outdir / name
            path.write_text(text, encoding="utf-8")

            if kind == "genome":
                st = _seq_stats_dna(seq)
            else:
                counts = {aa: 0 for aa in _AA}
                counts["X"] = 0
                for ch in seq:
                    counts[ch] = counts.get(ch, 0) + 1
                lengthp = len(seq)
                x_pct = (counts["X"] / lengthp * 100.0) if lengthp else None
                st = {"length": lengthp, "counts": counts, "gc_pct": None, "n_pct": x_pct, "acgt": None}

            files_out.append(file_info(path))
            stats_out.append({"file": path.name, **st})

            job.progress = 0.75 + 0.20 * ((i + 1) / max(1, len(sequences)))
            _job_put(job)

        job.result_files = files_out
        job.result_stats = stats_out
        job.status = "done"
        job.message = "Done"
        job.progress = 1.0
        job.ended_utc = now_iso()
        _job_put(job)

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.message = "Error"
        job.ended_utc = now_iso()
        _job_put(job)


@app.post("/generate")
def generate(req: GenerateRequest, bg: BackgroundTasks) -> Dict[str, Any]:
    kind = _safe_kind(req.kind)
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        kind=kind,
        created_utc=now_iso(),
        status="queued",
        message="Queued",
        params=req.model_dump(),
    )
    job.params["kind"] = kind
    _job_put(job)
    bg.add_task(_run_generation_job, job_id)
    return {"job_id": job_id, "status": job.status}


@app.get("/jobs")
def jobs(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    items = sorted(_jobs_list(), key=lambda j: j.created_utc, reverse=True)[:limit]
    return {"count": len(items), "items": [j.__dict__ for j in items]}


@app.get("/jobs/{job_id}")
def job(job_id: str) -> Dict[str, Any]:
    return _job_get(job_id).__dict__


@app.get("/generated")
def generated(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    d = ROOT / "generated"
    d.mkdir(parents=True, exist_ok=True)
    files = [p for p in d.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items = [file_info(p) for p in files[:limit]]
    return {"count": len(files), "items": items}


@app.get("/generated/{filename}")
def generated_download(filename: str) -> FileResponse:
    if not SAFE_NAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    p = (ROOT / "generated" / filename).resolve()
    gen = (ROOT / "generated").resolve()
    if not str(p).startswith(str(gen) + "/"):
        raise HTTPException(status_code=400, detail="Bad path.")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(str(p), filename=p.name)


# --------------------------
# Validate (NEW)
# --------------------------

_STOP = {"TAA", "TAG", "TGA"}

def _max_run(s: str) -> Dict[str, Any]:
    if not s:
        return {"base": None, "run": 0}
    best_base = s[0]
    best_run = 1
    cur_base = s[0]
    cur_run = 1
    for ch in s[1:]:
        if ch == cur_base:
            cur_run += 1
        else:
            if cur_run > best_run:
                best_run = cur_run
                best_base = cur_base
            cur_base = ch
            cur_run = 1
    if cur_run > best_run:
        best_run = cur_run
        best_base = cur_base
    return {"base": best_base, "run": best_run}


def _find_orfs_forward(seq: str, min_aa: int = 30) -> Dict[str, Any]:
    """
    Simple forward-strand ORFs in 3 frames.
    """
    s = seq.upper()
    orfs = []
    L = len(s)
    for frame in (0, 1, 2):
        i = frame
        while i + 3 <= L:
            codon = s[i:i+3]
            if codon == "ATG":
                j = i + 3
                while j + 3 <= L:
                    stop = s[j:j+3]
                    if stop in _STOP:
                        aa_len = (j - i) // 3
                        if aa_len >= min_aa:
                            orfs.append({"frame": frame, "start": i, "end": j + 3, "aa_len": aa_len})
                        i = j + 3
                        break
                    j += 3
                else:
                    # no stop; end
                    i += 3
            else:
                i += 3

    orfs.sort(key=lambda x: (-x["aa_len"], x["start"]))
    longest = orfs[0]["aa_len"] if orfs else 0
    return {
        "min_aa": min_aa,
        "count": len(orfs),
        "longest_aa": longest,
        "top": orfs[:10],
    }


def _load_generated_sequence(filename: str) -> Tuple[str, str, Path]:
    """
    Returns (kind, sequence, path)
    kind inferred from extension, but validated by content.
    """
    if not SAFE_NAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    p = (ROOT / "generated" / filename).resolve()
    gen = (ROOT / "generated").resolve()
    if not str(p).startswith(str(gen) + "/"):
        raise HTTPException(status_code=400, detail="Bad path.")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Not found.")

    seq = _parse_fasta_sequence(p)
    seq = _SEQ_CLEAN_RE.sub("", seq).upper()

    # infer kind
    ext = p.suffix.lower()
    if ext in (".faa", ".aa"):
        kind = "protein"
    elif ext in (".fasta", ".fa", ".fna", ".ffn", ".txt"):
        kind = "genome"
    else:
        kind = "genome"

    # content sanity
    dna_like = sum(1 for ch in seq if ch in "ACGTN") / max(1, len(seq))
    if kind == "protein" and dna_like > 0.95:
        kind = "genome"
    if kind == "genome" and dna_like < 0.70:
        kind = "protein"

    return kind, seq, p


@app.post("/generated/{filename}/validate")
def validate_generated(filename: str) -> Dict[str, Any]:
    kind, seq, path = _load_generated_sequence(filename)

    if kind == "genome":
        seq = seq.replace("U", "T")
        seq = re.sub(r"[^ACGTN]", "N", seq)
        st = _seq_stats_dna(seq)
        maxrun = _max_run(seq)
        orfs = _find_orfs_forward(seq, min_aa=30)

        invalid = st["counts"].get("OTHER", 0)
        invalid_pct = (invalid / max(1, st["length"]) * 100.0)

        return {
            "file": file_info(path),
            "kind": kind,
            "length": st["length"],
            "gc_pct": st["gc_pct"],
            "n_pct": st["n_pct"],
            "invalid_pct": invalid_pct,
            "max_run": maxrun,
            "orfs_forward": orfs,
        }

    # protein
    counts = {aa: 0 for aa in _AA}
    counts["X"] = 0
    other = 0
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
        else:
            other += 1

    L = len(seq)
    x_pct = ((counts["X"] + other) / max(1, L)) * 100.0
    maxrun = _max_run(seq)

    return {
        "file": file_info(path),
        "kind": kind,
        "length": L,
        "x_pct": x_pct,
        "max_run": maxrun,
        "aa_counts_top": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }


# --------------------------
# Compare-to-training-distribution (NEW)
# --------------------------

def _all_kmers(k: int) -> List[str]:
    if k <= 0:
        return [""]
    prev = _all_kmers(k - 1)
    out = []
    for p in prev:
        for b in "ACGT":
            out.append(p + b)
    return out


_K3 = _all_kmers(3)
_K3_INDEX = {k: i for i, k in enumerate(_K3)}


def _kmer_dist_dna(seq: str, k: int = 3) -> List[float]:
    s = re.sub(r"[^ACGT]", "", seq.upper())
    if len(s) < k:
        return [0.0] * (4 ** k)
    counts = [0] * (4 ** k)
    total = 0
    for i in range(0, len(s) - k + 1):
        kmer = s[i:i+k]
        idx = _K3_INDEX.get(kmer)
        if idx is None:
            continue
        counts[idx] += 1
        total += 1
    if total <= 0:
        return [0.0] * (4 ** k)
    return [c / total for c in counts]


def _js_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    if len(p) != len(q):
        raise ValueError("dist size mismatch")
    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]

    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0.0:
                continue
            s += ai * math.log((ai + eps) / (bi + eps), 2)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _percentile(sorted_vals: List[float], x: float) -> float:
    if not sorted_vals:
        return float("nan")
    # rank-based percentile
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / len(sorted_vals)


_BASELINE_LOCK = threading.Lock()
_BASELINE: Optional[Dict[str, Any]] = None


def _compute_baseline(max_files: int = 80, max_total_bases: int = 6_000_000) -> Dict[str, Any]:
    gb_dir = ROOT / "cache" / "genbank"
    fa_dir = ROOT / "cache" / "fasta"
    files: List[Path] = []

    if gb_dir.exists():
        files += [p for p in gb_dir.iterdir() if p.is_file() and p.suffix.lower() in GENBANK_EXTS]
    if fa_dir.exists():
        files += [p for p in fa_dir.iterdir() if p.is_file() and p.suffix.lower() in FASTA_EXTS]

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[:max_files]

    lengths: List[float] = []
    gcs: List[float] = []
    ns: List[float] = []
    k3_acc = [0.0] * (4 ** 3)
    k3_total = 0.0
    used = 0
    total_bases = 0

    for p in files:
        if total_bases >= max_total_bases:
            break
        try:
            if p.suffix.lower() in GENBANK_EXTS:
                seq, _ = _parse_genbank_sequence_and_features(p)
            else:
                seq = _parse_fasta_sequence(p)
            seq = seq.upper().replace("U", "T")
            seq = re.sub(r"[^ACGTN]", "N", seq)
            st = _seq_stats_dna(seq)
            if st["length"] <= 0:
                continue

            lengths.append(float(st["length"]))
            if st["gc_pct"] is not None:
                gcs.append(float(st["gc_pct"]))
            if st["n_pct"] is not None:
                ns.append(float(st["n_pct"]))

            k3 = _kmer_dist_dna(seq, k=3)
            # accumulate as mean-of-distributions approximation
            for i, v in enumerate(k3):
                k3_acc[i] += v
            k3_total += 1.0

            used += 1
            total_bases += min(st["length"], max_total_bases - total_bases)
        except Exception:
            continue

    lengths.sort()
    gcs.sort()
    ns.sort()

    k3_mean = [v / max(1.0, k3_total) for v in k3_acc]

    def quantiles(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0}
        def q(pct: float) -> float:
            if not vals:
                return float("nan")
            idx = int(round((pct / 100.0) * (len(vals) - 1)))
            idx = max(0, min(len(vals) - 1, idx))
            return float(vals[idx])
        return {"n": len(vals), "p05": q(5), "p25": q(25), "p50": q(50), "p75": q(75), "p95": q(95)}

    return {
        "built_utc": now_iso(),
        "n_files": used,
        "total_bases_sampled": int(total_bases),
        "length": quantiles(lengths),
        "gc_pct": quantiles(gcs),
        "n_pct": quantiles(ns),
        "k3_mean": k3_mean,
        "_sorted": {"lengths": lengths, "gcs": gcs, "ns": ns},
    }


def _get_baseline() -> Dict[str, Any]:
    global _BASELINE
    with _BASELINE_LOCK:
        if _BASELINE is None:
            _BASELINE = _compute_baseline()
        return _BASELINE


@app.get("/distribution/summary")
def distribution_summary() -> Dict[str, Any]:
    b = _get_baseline()
    return {
        "built_utc": b["built_utc"],
        "n_files": b["n_files"],
        "total_bases_sampled": b["total_bases_sampled"],
        "length": b["length"],
        "gc_pct": b["gc_pct"],
        "n_pct": b["n_pct"],
    }


@app.post("/generated/{filename}/compare")
def compare_generated(filename: str) -> Dict[str, Any]:
    kind, seq, path = _load_generated_sequence(filename)
    if kind != "genome":
        raise HTTPException(status_code=400, detail="Compare-to-training-distribution is implemented for genome/DNA outputs only (for now).")

    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGTN]", "N", seq)
    st = _seq_stats_dna(seq)

    b = _get_baseline()
    lengths = b["_sorted"]["lengths"]
    gcs = b["_sorted"]["gcs"]
    ns = b["_sorted"]["ns"]

    length_pct = _percentile(lengths, float(st["length"])) if lengths else float("nan")
    gc_pctile = _percentile(gcs, float(st["gc_pct"])) if (gcs and st["gc_pct"] is not None) else float("nan")
    n_pctile = _percentile(ns, float(st["n_pct"])) if (ns and st["n_pct"] is not None) else float("nan")

    k3_seq = _kmer_dist_dna(seq, k=3)
    js = _js_divergence(k3_seq, b["k3_mean"])

    # simple “overall” score: lower JS + central percentiles are better
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    js_score = 1.0 - clamp01(js / 1.0)  # JS is [0,1] in bits-ish for bounded dists
    center_score = 0.0
    for pct in [length_pct, gc_pctile, n_pctile]:
        if math.isnan(pct):
            continue
        center_score += (1.0 - abs(pct - 50.0) / 50.0)
    center_score = center_score / 3.0 if center_score > 0 else 0.0

    overall = 100.0 * (0.65 * js_score + 0.35 * center_score)

    return {
        "file": file_info(path),
        "kind": kind,
        "sequence": {
            "length": st["length"],
            "gc_pct": st["gc_pct"],
            "n_pct": st["n_pct"],
        },
        "baseline": {
            "built_utc": b["built_utc"],
            "n_files": b["n_files"],
            "total_bases_sampled": b["total_bases_sampled"],
            "length": b["length"],
            "gc_pct": b["gc_pct"],
            "n_pct": b["n_pct"],
        },
        "percentiles": {
            "length_pct": length_pct,
            "gc_pct": gc_pctile,
            "n_pct": n_pctile,
        },
        "divergence": {
            "k3_js": js,
        },
        "score": {
            "overall_0_100": overall,
            "js_component_0_1": js_score,
            "center_component_0_1": center_score,
        },
    }


# --------------------------
# Generate → Map (NEW): SVG plasmid-style map
# --------------------------

def _orfs_for_map(seq: str, min_aa: int = 60) -> List[Dict[str, Any]]:
    orfs = _find_orfs_forward(seq, min_aa=min_aa)
    return orfs["top"]  # already sorted


def _svg_map(seq: str, title: str, subtitle: str, orfs: List[Dict[str, Any]]) -> str:
    """
    Clean, modern plasmid-like circular map as SVG.
    """
    L = max(1, len(seq))
    st = _seq_stats_dna(seq)
    gc = st["gc_pct"]
    npct = st["n_pct"]

    W, H = 980, 640
    cx, cy = 320, 320
    R = 210
    ring = 26

    def ang(pos: int) -> float:
        return (pos / L) * 2.0 * math.pi - math.pi / 2.0

    def polar(a: float, r: float) -> Tuple[float, float]:
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    def arc_path(a0: float, a1: float, r0: float, r1: float) -> str:
        # Ensure a1 >= a0
        while a1 < a0:
            a1 += 2 * math.pi
        large = 1 if (a1 - a0) > math.pi else 0

        x0, y0 = polar(a0, r1)
        x1, y1 = polar(a1, r1)
        x2, y2 = polar(a1, r0)
        x3, y3 = polar(a0, r0)

        return (
            f"M {x0:.2f} {y0:.2f} "
            f"A {r1:.2f} {r1:.2f} 0 {large} 1 {x1:.2f} {y1:.2f} "
            f"L {x2:.2f} {y2:.2f} "
            f"A {r0:.2f} {r0:.2f} 0 {large} 0 {x3:.2f} {y3:.2f} "
            "Z"
        )

    # Build tick marks (every 10% major)
    ticks = []
    for i in range(0, 11):
        a = -math.pi/2 + (i/10.0) * 2*math.pi
        x1, y1 = polar(a, R + ring/2 + 4)
        x2, y2 = polar(a, R + ring/2 + 18)
        ticks.append(f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" class="tick"/>')

    # ORF arcs
    arcs = []
    for idx, o in enumerate(orfs[:12]):
        s = o["start"]
        e = o["end"]
        a0 = ang(s)
        a1 = ang(e)
        p = arc_path(a0, a1, R - ring/2, R + ring/2)
        arcs.append(f'<path d="{p}" class="orf orf{idx%4}"/>')

    legend = []
    for idx in range(min(8, len(orfs))):
        o = orfs[idx]
        legend.append(
            f'<div class="li"><span class="dot c{idx%4}"></span>'
            f'<span class="t">ORF {idx+1}</span>'
            f'<span class="m">frame {o["frame"]} • aa {o["aa_len"]} • {o["start"]}-{o["end"]}</span>'
            f'</div>'
        )
    legend_html = "\n".join(legend) if legend else '<div class="li"><span class="t">No ORFs (min_aa threshold) found</span></div>'

    gc_str = f"{gc:.2f}%" if gc is not None else "—"
    n_str = f"{npct:.2f}%" if npct is not None else "—"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#0b1020"/>
      <stop offset="1" stop-color="#050814"/>
    </linearGradient>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="8"/>
      <feOffset dx="0" dy="6" result="off"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.18"/>
      </feComponentTransfer>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#bg)"/>
  <g filter="url(#soft)">
    <rect x="540" y="60" width="400" height="520" rx="22" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.10)"/>
  </g>

  <g>
    <text x="560" y="108" class="h1">{title}</text>
    <text x="560" y="132" class="sub">{subtitle}</text>

    <g transform="translate(560,160)">
      <foreignObject x="0" y="0" width="360" height="380">
        <div xmlns="http://www.w3.org/1999/xhtml" class="legend">
          <div class="kv">
            <div><span class="k">length</span><span class="v">{L:,}</span></div>
            <div><span class="k">GC%</span><span class="v">{gc_str}</span></div>
            <div><span class="k">N%</span><span class="v">{n_str}</span></div>
          </div>
          <div class="hr"></div>
          <div class="sec">Top ORFs (forward)</div>
          {legend_html}
        </div>
      </foreignObject>
    </g>
  </g>

  <g>
    <circle cx="{cx}" cy="{cy}" r="{R+ring/2+26}" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)"/>
    <circle cx="{cx}" cy="{cy}" r="{R+ring/2+2}" fill="none" stroke="rgba(255,255,255,0.08)"/>
    <circle cx="{cx}" cy="{cy}" r="{R-ring/2-2}" fill="none" stroke="rgba(255,255,255,0.08)"/>

    <path d="{arc_path(-math.pi/2, -math.pi/2 + 2*math.pi, R-ring/2, R+ring/2)}" class="ring"/>
    {''.join(ticks)}
    {''.join(arcs)}

    <circle cx="{cx}" cy="{cy}" r="{R-78}" fill="rgba(0,0,0,0.20)" stroke="rgba(255,255,255,0.08)"/>
    <text x="{cx}" y="{cy-4}" text-anchor="middle" class="center">{L:,} bp</text>
    <text x="{cx}" y="{cy+20}" text-anchor="middle" class="center2">Perceptrome Map</text>
  </g>

  <style>
    .h1 {{ font: 700 18px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; fill: rgba(255,255,255,0.92); }}
    .sub {{ font: 400 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; fill: rgba(255,255,255,0.64); }}
    .center {{ font: 800 22px ui-sans-serif, system-ui; fill: rgba(255,255,255,0.92); letter-spacing: 0.2px; }}
    .center2 {{ font: 500 12px ui-sans-serif, system-ui; fill: rgba(255,255,255,0.60); }}
    .ring {{ fill: rgba(255,255,255,0.04); stroke: rgba(255,255,255,0.12); }}
    .tick {{ stroke: rgba(255,255,255,0.25); stroke-width: 2; stroke-linecap: round; }}
    .orf {{ stroke: rgba(255,255,255,0.22); stroke-width: 1; }}
    .orf0 {{ fill: rgba(120,190,255,0.30); }}
    .orf1 {{ fill: rgba(170,255,170,0.26); }}
    .orf2 {{ fill: rgba(255,200,120,0.26); }}
    .orf3 {{ fill: rgba(255,120,190,0.26); }}

    .legend {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: rgba(255,255,255,0.88);
      padding: 14px;
    }}
    .kv div {{
      display:flex; justify-content:space-between; gap:12px;
      margin-bottom: 6px;
      font-size: 12px;
    }}
    .k {{ color: rgba(255,255,255,0.62); }}
    .v {{ font-weight: 700; color: rgba(255,255,255,0.92); }}
    .hr {{ height: 1px; background: rgba(255,255,255,0.10); margin: 10px 0; }}
    .sec {{ font-weight: 700; font-size: 12px; color: rgba(255,255,255,0.78); margin-bottom: 8px; }}
    .li {{ display:flex; align-items:center; gap:10px; margin-bottom: 8px; font-size: 11px; }}
    .dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; }}
    .c0 {{ background: rgba(120,190,255,0.85); }}
    .c1 {{ background: rgba(170,255,170,0.80); }}
    .c2 {{ background: rgba(255,200,120,0.80); }}
    .c3 {{ background: rgba(255,120,190,0.80); }}
    .t {{ font-weight: 700; color: rgba(255,255,255,0.90); }}
    .m {{ color: rgba(255,255,255,0.62); }}
  </style>
</svg>
"""


@app.post("/generated/{filename}/map")
def map_generated(filename: str) -> Dict[str, Any]:
    kind, seq, path = _load_generated_sequence(filename)
    if kind != "genome":
        raise HTTPException(status_code=400, detail="Map is implemented for genome/DNA outputs only (for now).")

    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGTN]", "N", seq)

    outdir = ROOT / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    outname = f"{stem}_map.svg"
    outpath = (outdir / outname).resolve()

    orfs = _orfs_for_map(seq, min_aa=60)
    st = _seq_stats_dna(seq)
    title = f"{filename}"
    subtitle = f"len={st['length']:,} • GC={st['gc_pct']:.2f}% • N={st['n_pct']:.2f}%  (if available)"
    svg = _svg_map(seq, title=title, subtitle=subtitle, orfs=orfs)

    outpath.write_text(svg, encoding="utf-8")
    return {
        "ok": True,
        "map_file": file_info(outpath),
        "download_url": f"/api/generated/{outname}",
    }

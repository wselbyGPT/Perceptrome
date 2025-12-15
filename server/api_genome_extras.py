# server/api_genome_extras.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/genome", tags=["genome-extras"])

SAFE_ACC_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

ROOT = Path(__file__).resolve().parents[1]
GB_DIR = ROOT / "cache" / "genbank"
MAP_DIR = ROOT / "cache" / "maps"

try:
    from Bio import SeqIO
except Exception:
    SeqIO = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
except Exception:
    plt = None  # type: ignore
    Wedge = None  # type: ignore


def _acc_ok(acc: str) -> str:
    acc = acc.strip()
    if not acc or not SAFE_ACC_RE.match(acc):
        raise HTTPException(400, "Invalid accession")
    return acc


def _gb_path(acc: str) -> Path:
    for ext in (".gb", ".gbk", ".gbff"):
        p = GB_DIR / f"{acc}{ext}"
        if p.exists():
            return p
    raise HTTPException(404, f"No cached GenBank for {acc} in cache/genbank")


def _load_record(p: Path):
    if SeqIO is None:
        raise HTTPException(500, "Biopython missing: pip install biopython")
    with p.open("r", encoding="utf-8", errors="replace") as f:
        rec = next(SeqIO.parse(f, "genbank"), None)
    if rec is None:
        raise HTTPException(500, "Failed to parse GenBank")
    return rec


def _segments(feature) -> List[Tuple[int, int]]:
    loc = feature.location
    parts = getattr(loc, "parts", None)
    if parts:
        out = []
        for part in parts:
            s, e = int(part.start), int(part.end)
            if e > s:
                out.append((s, e))
        return out
    s, e = int(loc.start), int(loc.end)
    return [(s, e)] if e > s else []


@router.get("/{acc}/features")
def genome_features(
    acc: str,
    types: Optional[str] = Query(default=None, description="Comma-separated types, e.g. CDS,gene,tRNA"),
    q: Optional[str] = Query(default=None, description="Search gene/product/locus_tag substring"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
):
    acc = _acc_ok(acc)
    gb = _gb_path(acc)
    rec = _load_record(gb)

    want = None
    if types:
        want = {t.strip() for t in types.split(",") if t.strip()}

    needle = (q or "").strip().lower() or None

    feats: List[Dict[str, Any]] = []
    for f in getattr(rec, "features", []) or []:
        ftype = getattr(f, "type", "") or ""
        if want and ftype not in want:
            continue

        quals = getattr(f, "qualifiers", {}) or {}
        gene = (quals.get("gene", [""]) or [""])[0]
        product = (quals.get("product", [""]) or [""])[0]
        locus_tag = (quals.get("locus_tag", [""]) or [""])[0]

        if needle:
            hay = f"{gene} {product} {locus_tag}".lower()
            if needle not in hay:
                continue

        segs = _segments(f)
        if not segs:
            continue

        start = min(s for s, _ in segs)
        end = max(e for _, e in segs)
        strand = getattr(getattr(f, "location", None), "strand", None)

        feats.append({
            "type": ftype,
            "start": start,
            "end": end,
            "length": max(0, end - start),
            "strand": strand,
            "gene": gene,
            "product": product,
            "locus_tag": locus_tag,
            "segments": segs if len(segs) > 1 else None,
        })

    feats.sort(key=lambda d: (d["start"], d["end"]))
    total = len(feats)
    chunk = feats[offset: offset + limit]
    return {"accession": acc, "total": total, "offset": offset, "limit": limit, "features": chunk}


def _ensure_maps():
    MAP_DIR.mkdir(parents=True, exist_ok=True)


def _make_map(acc: str, gb: Path, svg: Path, pdf: Path):
    if plt is None or Wedge is None:
        raise HTTPException(500, "matplotlib missing: pip install matplotlib")

    rec = _load_record(gb)
    L = len(rec.seq)
    if L <= 0:
        raise HTTPException(500, "Record length is 0")

    # simple ring map
    ring = {"CDS": (1.00, 0.10), "tRNA": (0.86, 0.07), "rRNA": (0.78, 0.07), "gene": (0.66, 0.06)}
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")

    def ang(pos: int) -> float:
        return 90.0 - (360.0 * (pos / L))

    for f in getattr(rec, "features", []) or []:
        t = getattr(f, "type", "")
        if t not in ring:
            continue
        r, w = ring[t]
        strand = getattr(getattr(f, "location", None), "strand", None)
        alpha = 0.9 if strand != -1 else 0.6

        for s, e in _segments(f):
            theta1 = ang(e)
            theta2 = ang(s)
            col = cmap(s / max(1, L))
            ax.add_patch(Wedge((0, 0), r, theta1, theta2, width=w, facecolor=col, edgecolor="none", alpha=alpha))

    ax.text(0, 1.18, acc, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, 1.10, f"length: {L:,} bp", ha="center", va="center", fontsize=10, alpha=0.8)

    fig.savefig(svg, format="svg", transparent=True)
    fig.savefig(pdf, format="pdf", transparent=True)
    plt.close(fig)


@router.get("/{acc}/map.svg")
def genome_map_svg(acc: str):
    acc = _acc_ok(acc)
    gb = _gb_path(acc)
    _ensure_maps()
    svg = MAP_DIR / f"{acc}.svg"
    pdf = MAP_DIR / f"{acc}.pdf"
    if (not svg.exists()) or (svg.stat().st_mtime < gb.stat().st_mtime):
        _make_map(acc, gb, svg, pdf)
    return FileResponse(str(svg), media_type="image/svg+xml", filename=svg.name)


@router.get("/{acc}/map.pdf")
def genome_map_pdf(acc: str):
    acc = _acc_ok(acc)
    gb = _gb_path(acc)
    _ensure_maps()
    svg = MAP_DIR / f"{acc}.svg"
    pdf = MAP_DIR / f"{acc}.pdf"
    if (not pdf.exists()) or (pdf.stat().st_mtime < gb.stat().st_mtime):
        _make_map(acc, gb, svg, pdf)
    return FileResponse(str(pdf), media_type="application/pdf", filename=pdf.name)

#!/usr/bin/env python3
"""
plasmid_circle.py
Draw a circular plasmid map (PDF) from a FASTA sequence using Biopython GenomeDiagram.

- Predicts ORFs (simple ATG -> stop scan) on both strands and draws them as arrows.
- Adds a GC% ring (sliding window).
This is a visualization / smoke-test map, not a real annotation pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from Bio.Graphics import GenomeDiagram
except ImportError as e:
    raise SystemExit("Missing Biopython. Install: pip install biopython reportlab") from e

from reportlab.lib import colors
from reportlab.lib.units import cm


@dataclass
class ORFHit:
    start: int
    end: int
    strand: int
    aa_len: int


STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODON = "ATG"


def _clean_seq(record) -> str:
    s = str(record.seq).upper().replace("U", "T")
    # Allow N; drop anything else to N
    s = "".join(ch if ch in "ACGTN" else "N" for ch in s)
    if not s:
        raise ValueError("Empty sequence")
    return s


def find_orfs_simple(seq: str, min_aa: int = 90) -> List[ORFHit]:
    """
    Very simple ORF finder:
      - both strands
      - frames 0,1,2
      - start ATG, stop TAA/TAG/TGA
    Returns ORFHit in ORIGINAL forward coordinates (0-based, end exclusive).
    """
    seq = seq.upper()
    L = len(seq)
    hits: List[ORFHit] = []

    def scan(s: str, strand: int):
        nonlocal hits
        n = len(s)
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < n:
                if s[i:i+3] == START_CODON:
                    j = i + 3
                    while j + 2 < n:
                        cod = s[j:j+3]
                        if cod in STOP_CODONS:
                            aa_len = (j - i) // 3
                            if aa_len >= min_aa:
                                if strand == 1:
                                    start_f = i
                                    end_f = j + 3
                                else:
                                    # Map rc-coordinates back to forward:
                                    # rc segment [i, j+3) corresponds to forward [L-(j+3), L-i)
                                    start_f = L - (j + 3)
                                    end_f = L - i
                                hits.append(ORFHit(start=start_f, end=end_f, strand=strand, aa_len=aa_len))
                            i = j + 3
                            break
                        j += 3
                    else:
                        i += 3
                else:
                    i += 3

    scan(seq, strand=1)
    rc = str(Seq(seq).reverse_complement())
    scan(rc, strand=-1)

    # Deduplicate identical ORFs (can happen with Ns)
    uniq = {}
    for h in hits:
        key = (h.start, h.end, h.strand)
        if key not in uniq or h.aa_len > uniq[key].aa_len:
            uniq[key] = h
    return list(uniq.values())


def gc_ring(seq: str, window: int = 200, step: int = 10) -> List[Tuple[int, float]]:
    """
    Circular GC% ring data: returns [(pos, gc_frac), ...] where pos is bp index.
    Uses wrap-around by indexing modulo length.
    """
    seq = seq.upper()
    L = len(seq)
    if window <= 0 or window > L:
        window = min(max(50, L // 20), L)

    def gc_at(center: int) -> float:
        half = window // 2
        gc = 0
        for k in range(center - half, center + half):
            b = seq[k % L]
            if b in ("G", "C"):
                gc += 1
        return gc / float(window)

    data: List[Tuple[int, float]] = []
    for pos in range(0, L, max(1, step)):
        data.append((pos, gc_at(pos)))
    return data


def build_diagram(
    fasta_path: Path,
    out_path: Path,
    min_aa: int,
    max_orfs: int,
    gc_window: int,
    gc_step: int,
    title: str | None,
):
    record = next(SeqIO.parse(str(fasta_path), "fasta"))
    seq = _clean_seq(record)
    L = len(seq)
    name = title or (record.id if record.id else fasta_path.stem)

    # ORFs
    orfs = find_orfs_simple(seq, min_aa=min_aa)
    orfs.sort(key=lambda h: h.aa_len, reverse=True)
    if max_orfs > 0:
        orfs = orfs[:max_orfs]

    # Create SeqFeatures for GenomeDiagram
    features: List[SeqFeature] = []
    for i, h in enumerate(orfs, start=1):
        loc = FeatureLocation(int(h.start), int(h.end), strand=int(h.strand))
        feat = SeqFeature(
            location=loc,
            type="CDS",
            qualifiers={"label": f"ORF{i} ({h.aa_len}aa)"},
        )
        features.append(feat)

    diagram = GenomeDiagram.Diagram(name)

    # Track 1: ORFs
    track_orf = diagram.new_track(
        1, name="Predicted ORFs", greytrack=True, greytrack_labels=1, start=0, end=L
    )
    set_orf = track_orf.new_set()

    palette = [colors.darkblue, colors.darkgreen, colors.purple, colors.brown, colors.darkred]
    for idx, feat in enumerate(features):
        set_orf.add_feature(
            feat,
            sigil="ARROW",
            arrowshaft_height=0.6,
            arrowhead_length=0.4,
            color=palette[idx % len(palette)],
            label=True,
            label_size=6,
            label_angle=0,
            label_position="middle",
        )

    # Track 2: GC ring
    track_gc = diagram.new_track(
        2, name="GC%", greytrack=True, greytrack_labels=1, start=0, end=L, height=0.25
    )
    set_gc = track_gc.new_set("graph")
    gc_data = gc_ring(seq, window=gc_window, step=gc_step)
    set_gc.new_graph(
        gc_data,
        name="GC fraction",
        style="line",
        color=colors.darkgray,
        altcolour=colors.lightgrey,
    )

    # Draw
    diagram.draw(
        format="circular",
        circular=True,
        start=0,
        end=L,
        pagesize=(20 * cm, 20 * cm),
        circle_core=0.35,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lower().lstrip(".") or "pdf"
    # GenomeDiagram expects uppercase format strings like "PDF"
    diagram.write(str(out_path), fmt.upper())
    print(f"[ok] wrote {out_path}  (len={L}bp, ORFs_shown={len(features)}, min_aa={min_aa}, GCwin={gc_window})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", type=Path, help="Input FASTA (single record recommended)")
    ap.add_argument("--out", type=Path, default=None, help="Output file (default: <fasta>.plasmid_map.pdf)")
    ap.add_argument("--min-aa", type=int, default=90, help="Minimum ORF length in amino acids")
    ap.add_argument("--max-orfs", type=int, default=60, help="Show only top-N ORFs by length (0 = no limit)")
    ap.add_argument("--gc-window", type=int, default=200, help="GC window size (bp)")
    ap.add_argument("--gc-step", type=int, default=10, help="GC sampling step (bp)")
    ap.add_argument("--title", type=str, default=None, help="Diagram title override")
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = args.fasta.with_suffix("")  # drop .fasta
        out = out.parent / (out.name + ".plasmid_map.pdf")

    build_diagram(
        fasta_path=args.fasta,
        out_path=out,
        min_aa=args.min_aa,
        max_orfs=args.max_orfs,
        gc_window=args.gc_window,
        gc_step=args.gc_step,
        title=args.title,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plasmid_map.py — fully-labeled circular plasmid map PDF from a FASTA sequence.

Includes:
- Outer bp scale w/ labeled major ticks
- Predicted ORFs (ATG->stop; both strands; 3 frames) drawn as arrows + labeled ORF#
- GC% ring graph (sliding window; circular wrap)
- Legend + ORF table embedded into the PDF

Example:
  python3 tools/plasmid_map.py generated/perceptrome_novel_001.fasta \
    --out generated/perceptrome_novel_001_map.pdf \
    --title "Perceptrome Novel Plasmid 001" \
    --min-orf-aa 90 --gc-window 200 --gc-step 50
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Graphics import GenomeDiagram

from reportlab.lib import colors
from reportlab.graphics.shapes import String, Rect


STOP = {"TAA", "TAG", "TGA"}
START = "ATG"


def revcomp(dna: str) -> str:
    return str(Seq(dna).reverse_complement())


@dataclass(frozen=True)
class ORF:
    start: int          # 0-based inclusive (forward coordinate system)
    end: int            # 0-based exclusive
    strand: int         # +1 or -1
    frame: int          # 0,1,2
    aa_len: int


def find_orfs(dna: str, min_aa: int) -> List[ORF]:
    """Naive ORF finder: ATG -> first in-frame stop; scans both strands and 3 frames."""
    dna = dna.upper().replace("U", "T")
    L = len(dna)
    out: List[ORF] = []

    def scan(seq: str, strand: int):
        nonlocal out
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < L:
                cod = seq[i:i+3]
                if cod == START:
                    j = i + 3
                    while j + 2 < L:
                        cod2 = seq[j:j+3]
                        if cod2 in STOP:
                            aa_len = (j - i) // 3
                            if aa_len >= min_aa:
                                if strand == 1:
                                    start, end = i, j
                                else:
                                    # map reverse-complement coords back to forward coords
                                    start = L - j
                                    end = L - i
                                out.append(ORF(start=start, end=end, strand=strand, frame=frame, aa_len=aa_len))
                            i = j + 3  # skip past stop
                            break
                        j += 3
                    else:
                        i += 3
                else:
                    i += 3

    scan(dna, strand=1)
    scan(revcomp(dna), strand=-1)

    out.sort(key=lambda o: (o.start, -(o.end - o.start)))
    return out


def gc_series_percent(dna: str, window: int, step: int) -> List[Tuple[int, float]]:
    """Return list of (pos, GC%) in [0..100], treating DNA as circular."""
    dna = dna.upper()
    L = len(dna)
    if L == 0:
        return [(0, 0.0)]

    if window <= 0:
        window = min(200, L)
    if step <= 0:
        step = max(1, window // 4)

    doubled = dna + dna  # circular wrap
    data: List[Tuple[int, float]] = []
    for pos in range(0, L, step):
        w = doubled[pos:pos+window]
        if not w:
            continue
        gc = (w.count("G") + w.count("C")) / float(len(w))
        data.append((pos, 100.0 * gc))
    return data or [(0, 0.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", type=Path, help="Input FASTA (single sequence preferred)")
    ap.add_argument("--out", type=Path, default=None, help="Output PDF path")
    ap.add_argument("--title", type=str, default=None, help="Map title")
    ap.add_argument("--min-orf-aa", type=int, default=90, help="Minimum ORF length (aa) to display")
    ap.add_argument("--max-orfs-on-map", type=int, default=40, help="Cap ORFs drawn/labeled (reduces clutter)")
    ap.add_argument("--top-orfs-in-legend", type=int, default=25, help="How many ORFs to list in legend table")
    ap.add_argument("--gc-window", type=int, default=200, help="GC%% window (bp)")
    ap.add_argument("--gc-step", type=int, default=50, help="GC%% step (bp)")
    ap.add_argument("--tick-major", type=int, default=1000, help="Major tick interval (bp)")
    ap.add_argument("--tick-minor", type=int, default=200, help="Minor tick interval (bp)")
    args = ap.parse_args()

    if not args.fasta.exists():
        raise SystemExit(f"FASTA not found: {args.fasta}")

    record = SeqIO.read(str(args.fasta), "fasta")
    dna = str(record.seq).upper().replace("U", "T")
    L = len(dna)
    if L == 0:
        raise SystemExit("Sequence is empty")

    out_pdf = args.out or (args.fasta.with_suffix("").with_suffix(".map.pdf"))
    title = args.title or (record.id or args.fasta.stem)
    global_gc = (dna.count("G") + dna.count("C")) / float(L)

    # ORFs
    orfs = find_orfs(dna, min_aa=int(args.min_orf_aa))
    orfs_by_len = sorted(orfs, key=lambda o: (-(o.end - o.start), o.start))
    orfs_on_map = sorted(orfs_by_len[: max(0, int(args.max_orfs_on_map))], key=lambda o: o.start)
    numbered = [(i, orf) for i, orf in enumerate(orfs_on_map, start=1)]

    # GC%
    gc_data = gc_series_percent(dna, window=int(args.gc_window), step=int(args.gc_step))

    diagram = GenomeDiagram.Diagram(f"{title}")

    # Track 1: scale (outer)
    tr_scale = diagram.new_track(1, name="bp scale", greytrack=False, start=0, end=L)
    tr_scale.scale = True
    tr_scale.scale_ticks = True
    tr_scale.scale_labels = True
    tr_scale.scale_fontsize = 7
    tr_scale.scale_largeticks = int(args.tick_major)
    tr_scale.scale_smallticks = int(args.tick_minor)
    tr_scale.scale_largetick_labels = True
    tr_scale.scale_smalltick_labels = False

    # Track 2: ORFs
    tr_orf = diagram.new_track(2, name=f"Predicted ORFs (min {args.min_orf_aa} aa)", greytrack=True, start=0, end=L)
    fs_orf = tr_orf.new_set("feature")

    fwd_color = colors.HexColor("#2B6CB0")  # blue
    rev_color = colors.HexColor("#C53030")  # red

    for n, orf in numbered:
        loc = FeatureLocation(int(orf.start), int(orf.end), strand=int(orf.strand))
        feat = SeqFeature(loc, type="CDS", qualifiers={"label": [f"ORF{n}"]})
        fs_orf.add_feature(
            feat,
            sigil="ARROW",
            arrowshaft_height=0.6,
            arrowhead_length=0.5,
            color=(fwd_color if orf.strand == 1 else rev_color),
            border=colors.black,
            label=True,
            label_position="middle",
            label_size=6,
            label_angle=0,
        )

    # Track 3: GC% (inner)
    tr_gc = diagram.new_track(3, name=f"GC% (win {args.gc_window}bp / step {args.gc_step}bp)", greytrack=True, start=0, end=L)
    gs = tr_gc.new_set("graph")
    gs.new_graph(
        gc_data,
        name="GC%",
        style="line",
        color=colors.HexColor("#2F855A"),
        center=50.0,
    )

    diagram.draw(format="circular", start=0, end=L, circle_core=0.35)

    # Overlay legend + ORF table into the PDF drawing
    d = diagram.drawing
    x0, y0 = 30, 20
    line = 11

    d.add(String(x0, y0 + line * 9, f"{title}", fontSize=14))
    d.add(String(
        x0, y0 + line * 8,
        f"Length: {L:,} bp    Global GC: {global_gc*100:.2f}%    ORFs shown: {len(numbered)} / found: {len(orfs)}",
        fontSize=9
    ))

    d.add(String(x0, y0 + line * 7, "Legend:", fontSize=10))
    d.add(Rect(x0, y0 + line * 6.2, 10, 8, fillColor=fwd_color, strokeColor=colors.black))
    d.add(String(x0 + 14, y0 + line * 6.4, "ORF (+ strand)", fontSize=9))
    d.add(Rect(x0 + 120, y0 + line * 6.2, 10, 8, fillColor=rev_color, strokeColor=colors.black))
    d.add(String(x0 + 134, y0 + line * 6.4, "ORF (- strand)", fontSize=9))
    d.add(Rect(x0 + 240, y0 + line * 6.2, 10, 8, fillColor=colors.HexColor("#2F855A"), strokeColor=colors.black))
    d.add(String(x0 + 254, y0 + line * 6.4, "GC% (line)", fontSize=9))

    d.add(String(x0, y0 + line * 5.2, "ORF table (from map labels):", fontSize=10))
    numbered_by_len = sorted(numbered, key=lambda t: (-(t[1].end - t[1].start), t[1].start))
    max_rows = max(0, int(args.top_orfs_in_legend))
    for r, (n, orf) in enumerate(numbered_by_len[:max_rows], start=1):
        start1 = orf.start + 1
        end1 = orf.end
        strand = "+" if orf.strand == 1 else "-"
        txt = f"ORF{n:02d}  {start1:>6}-{end1:<6}  ({strand})  {orf.aa_len:>4} aa  frame {orf.frame}"
        d.add(String(x0, y0 + line * (5.2 - r), txt, fontSize=8))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    diagram.write(str(out_pdf), "PDF")
    print(f"[plasmid_map] wrote: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

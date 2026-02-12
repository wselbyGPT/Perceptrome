from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from perceptrome.encoding.parse import reverse_complement

STOP_CODONS = {"TAA", "TAG", "TGA"}
CODON_TO_AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_orf(dna: str) -> str:
    out: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i+3]
        if cod in STOP_CODONS:
            break
        out.append(CODON_TO_AA.get(cod, "X"))
    return "".join(out)


@dataclass
class GenBankBuilderConfig:
    min_orf_aa: int = 90
    start_codons: Tuple[str, ...] = ("ATG",)
    include_partial_cds: bool = False
    allow_no_start: bool = False


@dataclass
class OrfFeature:
    start: int
    end: int
    strand: int
    dna: str
    protein: str
    partial_left: bool
    partial_right: bool
    frame: int
    source: str


def _sanitize_seq(seq: str) -> str:
    return "".join(ch for ch in seq.upper().replace("U", "T") if ch in "ACGTN")


def _parse_fasta_header(header: str) -> Dict[str, str]:
    text = (header or "").strip()
    if text.startswith(">"):
        text = text[1:].strip()
    out: Dict[str, str] = {"raw": text}
    if not text:
        out["name"] = "sequence"
        return out

    parts = text.split()
    out["name"] = parts[0]
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def _read_fasta(path: str) -> Tuple[str, str, Dict[str, str]]:
    header = ""
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if not header:
                    header = s
                continue
            seq_parts.append(s)
    seq = _sanitize_seq("".join(seq_parts))
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return header, seq, _parse_fasta_header(header)


def _scan_orfs_on_strand(seq: str, strand: int, cfg: GenBankBuilderConfig) -> Iterable[OrfFeature]:
    starts = {s.upper() for s in cfg.start_codons}
    oriented = seq if strand == 1 else reverse_complement(seq)
    n = len(oriented)

    for frame in (0, 1, 2):
        i = frame
        while i + 2 < n:
            start_here = oriented[i:i + 3] in starts
            if not start_here and cfg.allow_no_start:
                start_here = oriented[i:i + 3] not in STOP_CODONS
            if not start_here:
                i += 3
                continue

            j = i
            stop_at: Optional[int] = None
            while j + 2 < n:
                cod = oriented[j:j + 3]
                if cod in STOP_CODONS:
                    stop_at = j
                    break
                j += 3

            partial_right = False
            if stop_at is None:
                if not cfg.include_partial_cds:
                    i += 3
                    continue
                tail = n - i
                tail -= tail % 3
                if tail < 3:
                    i += 3
                    continue
                end_idx = i + tail
                partial_right = True
            else:
                end_idx = stop_at + 3

            dna = oriented[i:end_idx]
            coding = dna[:-3] if not partial_right else dna
            if len(coding) < cfg.min_orf_aa * 3:
                i += 3
                continue

            protein = translate_orf(coding)
            if len(protein) < cfg.min_orf_aa:
                i += 3
                continue

            if strand == 1:
                start = i + 1
                end = end_idx
            else:
                start = n - end_idx + 1
                end = n - i

            partial_left = oriented[i:i + 3] not in starts
            yield OrfFeature(
                start=start,
                end=end,
                strand=strand,
                dna=dna,
                protein=protein,
                partial_left=partial_left,
                partial_right=partial_right,
                frame=frame,
                source="orf_prediction",
            )
            i += 3


def _location_for_feature(f: OrfFeature) -> str:
    left = f"<{f.start}" if f.partial_left else str(f.start)
    right = f">{f.end}" if f.partial_right else str(f.end)
    loc = f"{left}..{right}"
    return f"complement({loc})" if f.strand < 0 else loc


def _genbank_wrap_origin(seq: str) -> List[str]:
    lines: List[str] = []
    for i in range(0, len(seq), 60):
        chunk = seq[i:i + 60].lower()
        groups = [chunk[j:j + 10] for j in range(0, len(chunk), 10)]
        lines.append(f"{i + 1:>9} {' '.join(groups)}")
    return lines


def _safe_locus_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name or "sequence")
    return name[:16] or "sequence"


def build_genbank_from_fasta(
    fasta_header: str,
    seq: str,
    config: Optional[GenBankBuilderConfig] = None,
) -> Tuple[str, str]:
    cfg = config or GenBankBuilderConfig()
    meta = _parse_fasta_header(fasta_header)
    seq = _sanitize_seq(seq)
    if not seq:
        raise ValueError("Empty sequence")

    locus = _safe_locus_name(meta.get("name", "sequence"))
    date_str = datetime.utcnow().strftime("%d-%b-%Y").upper()
    defn = meta.get("organism") or meta.get("raw") or locus

    features: List[OrfFeature] = []
    features.extend(_scan_orfs_on_strand(seq, strand=1, cfg=cfg))
    features.extend(_scan_orfs_on_strand(seq, strand=-1, cfg=cfg))
    features.sort(key=lambda x: (x.start, x.end, -x.strand))

    lines: List[str] = []
    lines.append(f"LOCUS       {locus:<16}{len(seq):>11} bp    DNA     PLN       {date_str}")
    lines.append(f"DEFINITION  {defn}.")
    lines.append("ACCESSION   .")
    lines.append("VERSION     .")
    lines.append("KEYWORDS    .")
    lines.append("SOURCE      synthetic construct")
    lines.append("  ORGANISM  synthetic construct")
    lines.append("FEATURES             Location/Qualifiers")
    lines.append(f"     source          1..{len(seq)}")
    lines.append('                     /organism="synthetic construct"')
    lines.append('                     /mol_type="genomic DNA"')

    for idx, feat in enumerate(features, start=1):
        lines.append(f"     CDS             {_location_for_feature(feat)}")
        codon_start = (feat.frame % 3) + 1
        product = f"predicted protein {idx}"
        conf = "high" if not (feat.partial_left or feat.partial_right) else "medium"
        note = (
            f"source={feat.source}; confidence={conf}; strand={'+' if feat.strand > 0 else '-'}; "
            f"start_codons={','.join(cfg.start_codons)}"
        )
        lines.append(f'                     /codon_start={codon_start}')
        lines.append(f'                     /product="{product}"')
        lines.append(f'                     /note="{note}"')
        lines.append(f'                     /translation="{feat.protein}"')

    lines.append("ORIGIN")
    lines.extend(_genbank_wrap_origin(seq))
    lines.append("//")
    return locus, "\n".join(lines) + "\n"


def build_genbank_from_fasta_file(
    fasta_path: str,
    output_path: Optional[str] = None,
    config: Optional[GenBankBuilderConfig] = None,
) -> str:
    header, seq, meta = _read_fasta(fasta_path)
    locus, text = build_genbank_from_fasta(header, seq, config=config)
    if output_path is None:
        base = _safe_locus_name(meta.get("name") or os.path.splitext(os.path.basename(fasta_path))[0])
        output_path = os.path.join("generated", f"{base}.gb")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return output_path


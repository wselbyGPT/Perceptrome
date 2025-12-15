import os
from typing import List, Tuple

def parse_fasta_sequence(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA not found: {path}")
    seq_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence found in {path}")
    return seq


def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def parse_genbank_dna(path: str) -> str:
    """Extract the DNA sequence from a GenBank flatfile (ORIGIN section)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GenBank not found: {path}")
    seq_parts: List[str] = []
    in_origin = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ORIGIN"):
                in_origin = True
                continue
            if in_origin:
                if line.startswith("//"):
                    break
                # lines look like: '     1 atgc...'
                s = "".join(ch for ch in line.strip() if ch.isalpha())
                if s:
                    seq_parts.append(s)
    seq = "".join(seq_parts).upper().replace("U", "T")
    if not seq:
        raise ValueError(f"No ORIGIN sequence found in {path}")
    return seq


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_loc_range(token: str) -> Tuple[int, int]:
    # Handles: 123..456, <123..>456, 123, 123^124 (we treat as 123..124)
    t = token.strip()
    t = t.replace("<", "").replace(">", "")
    if "^" in t:
        a, b = t.split("^", 1)
        a = a.strip()
        b = b.strip()
        return int(a), int(b)
    if ".." in t:
        a, b = t.split("..", 1)
        return int(a.strip()), int(b.strip())
    return int(t), int(t)


def extract_dna_from_location(dna: str, loc: str) -> str:
    """Extract CDS DNA from a GenBank location string using the ORIGIN DNA."""
    s = loc.strip().replace(" ", "")

    def parse_expr(expr: str) -> Tuple[str, bool]:
        # returns (sequence, is_complement)
        if expr.startswith("complement(") and expr.endswith(")"):
            inner = expr[len("complement("):-1]
            seq0, _ = parse_expr(inner)
            return seq0, True
        if (expr.startswith("join(") or expr.startswith("order(")) and expr.endswith(")"):
            inner = expr[expr.find("(") + 1:-1]
            segs = _split_top_level_commas(inner)
            out = []
            for seg in segs:
                seq1, comp1 = parse_expr(seg)
                if comp1:
                    seq1 = reverse_complement(seq1)
                out.append(seq1)
            return "".join(out), False
        # simple range
        start, end = _parse_loc_range(expr)
        if start < 1 or end < 1 or end < start:
            return "", False
        # GenBank is 1-based inclusive
        return dna[start - 1:end], False

    seq, is_comp = parse_expr(s)
    if is_comp:
        seq = reverse_complement(seq)
    return seq



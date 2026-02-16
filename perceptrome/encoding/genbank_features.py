from dataclasses import dataclass
import os
from typing import List, Optional

from .parse import extract_dna_from_location, parse_genbank_dna

CODON_TO_AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
    "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R",
    "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
STOP_CODONS = {"TAA", "TAG", "TGA"}



def _translate_cds(dna: str, codon_start: int = 1) -> str:
    if codon_start not in (1, 2, 3):
        codon_start = 1
    dna = dna.upper().replace("U", "T")
    if codon_start > 1 and len(dna) >= codon_start - 1:
        dna = dna[codon_start - 1:]
    aa = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i + 3]
        if cod in STOP_CODONS:
            break
        aa.append(CODON_TO_AA.get(cod, "X"))
    return "".join(aa)


@dataclass
class CDSFeature:
    start: int
    end: int
    strand: int
    gene_or_locus_tag: str
    product: str
    protein_length: int
    translation_source: str


def _location_bounds(loc: str) -> tuple[int, int, int]:
    """Return (start, end, strand) from a GenBank location string."""
    cleaned = loc.strip().replace(" ", "")
    strand = -1 if "complement(" in cleaned else 1

    nums: List[int] = []
    cur = ""
    for ch in cleaned:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(int(cur))
                cur = ""
    if cur:
        nums.append(int(cur))

    if not nums:
        return 1, 1, strand
    return min(nums), max(nums), strand


def parse_cds_features_from_genbank(path: str) -> List[CDSFeature]:
    """Parse CDS features from GenBank and return structured feature objects."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GenBank not found: {path}")

    try:
        origin_dna = parse_genbank_dna(path)
    except Exception:
        origin_dna = ""

    features: List[CDSFeature] = []
    in_features = False
    in_origin = False

    cur_key: Optional[str] = None
    cur_loc: Optional[str] = None
    cur_gene = ""
    cur_locus_tag = ""
    cur_product = ""
    cur_translation: Optional[str] = None
    cur_codon_start = 1
    cur_is_pseudo = False

    reading_qualifier_key: Optional[str] = None
    qualifier_buf: List[str] = []

    def _reset_cds() -> None:
        nonlocal cur_key, cur_loc, cur_gene, cur_locus_tag, cur_product
        nonlocal cur_translation, cur_codon_start, cur_is_pseudo
        nonlocal reading_qualifier_key, qualifier_buf
        cur_key = None
        cur_loc = None
        cur_gene = ""
        cur_locus_tag = ""
        cur_product = ""
        cur_translation = None
        cur_codon_start = 1
        cur_is_pseudo = False
        reading_qualifier_key = None
        qualifier_buf = []

    def _apply_qualifier(key: str, value: str) -> None:
        nonlocal cur_gene, cur_locus_tag, cur_product, cur_translation, cur_codon_start, cur_is_pseudo
        v = value.strip()
        if key == "gene":
            cur_gene = v
        elif key == "locus_tag":
            cur_locus_tag = v
        elif key == "product":
            cur_product = v
        elif key == "translation":
            cur_translation = v.replace(" ", "").replace("\n", "")
        elif key == "codon_start":
            try:
                cur_codon_start = int(v)
            except Exception:
                cur_codon_start = 1
        elif key in ("pseudo", "pseudogene"):
            cur_is_pseudo = True

    def _flush_pending_qualifier() -> None:
        nonlocal reading_qualifier_key, qualifier_buf
        if reading_qualifier_key is None:
            return
        _apply_qualifier(reading_qualifier_key, "".join(qualifier_buf))
        reading_qualifier_key = None
        qualifier_buf = []

    def _flush_feature() -> None:
        nonlocal cur_loc
        if cur_key != "CDS" or cur_loc is None or cur_is_pseudo:
            _reset_cds()
            return

        _flush_pending_qualifier()

        start, end, strand = _location_bounds(cur_loc)
        gene_or_locus_tag = cur_gene or cur_locus_tag or "CDS"
        product = cur_product or "hypothetical protein"

        protein = ""
        source = "inferred"
        if cur_translation:
            protein = cur_translation.strip().upper().rstrip("*")
            source = "provided"
        elif origin_dna:
            cds_dna = extract_dna_from_location(origin_dna, cur_loc)
            protein = _translate_cds(cds_dna, codon_start=cur_codon_start).strip().upper().rstrip("*")

        features.append(
            CDSFeature(
                start=start,
                end=end,
                strand=strand,
                gene_or_locus_tag=gene_or_locus_tag,
                product=product,
                protein_length=len(protein),
                translation_source=source,
            )
        )
        _reset_cds()

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                in_origin = True
                in_features = False
                _flush_feature()
                continue
            if in_origin:
                continue
            if not in_features:
                continue

            if line.startswith("     ") and len(line) > 5 and line[5] != " ":
                _flush_feature()
                cur_key = line[5:21].strip()
                cur_loc = line[21:].strip() if cur_key == "CDS" else None
                continue

            if cur_key == "CDS" and line.startswith("                     "):
                q = line.strip()

                if reading_qualifier_key is not None:
                    if q.endswith('"'):
                        qualifier_buf.append(q[:-1])
                        _flush_pending_qualifier()
                    else:
                        qualifier_buf.append(q)
                    continue

                if not q.startswith("/"):
                    continue

                body = q[1:]
                if "=" not in body:
                    _apply_qualifier(body, "")
                    continue

                key, val = body.split("=", 1)
                key = key.strip()
                val = val.strip()
                if val.startswith('"'):
                    val = val[1:]
                    if val.endswith('"'):
                        _apply_qualifier(key, val[:-1])
                    else:
                        reading_qualifier_key = key
                        qualifier_buf = [val]
                else:
                    _apply_qualifier(key, val)

    _flush_feature()
    return features

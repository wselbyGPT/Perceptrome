from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np


from .constants import CODON_TO_AA, STOP_CODONS, START_CODON, AA_TO_IDX, AA_UNK, AA_VOCAB_SIZE

from .parse import parse_genbank_dna, extract_dna_from_location, reverse_complement


def translate_cds(dna: str, codon_start: int = 1) -> str:
    """Translate CDS DNA to AA; respects codon_start (1..3)."""
    if codon_start not in (1, 2, 3):
        codon_start = 1
    dna = dna.upper().replace("U", "T")
    if codon_start > 1 and len(dna) >= codon_start - 1:
        dna = dna[codon_start - 1:]
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i + 3]
        if cod in STOP_CODONS:
            break
        aas.append(CODON_TO_AA.get(cod, "X"))
    return "".join(aas)


def extract_cds_proteins_from_genbank(
    path: str,
    min_aa: int = 90,
    return_sources: bool = False,
    require_translation: bool = False,
    reject_partial_cds: bool = False,
    require_start_m: bool = False,
    x_free: bool = False,
    max_protein_aa: int = 0,
):
    """
    Extract protein sequences from GenBank FEATURES/CDS.

    Preference order per CDS:
      1) /translation="..." qualifier (most reliable)
      2) location + ORIGIN DNA + /codon_start (fallback, unless require_translation=True)

    Notes:
      - Skips CDS with /pseudo or /pseudogene.
      - Handles common location forms: 123..456, complement(123..456), join(...), complement(join(...)).
      - If reject_partial_cds=True, skip locations containing < or >.
      - If x_free=True, skip proteins containing X or internal *.
      - If require_start_m=True, require leading M.
      - If max_protein_aa>0, skip proteins longer than that.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GenBank not found: {path}")

    # Parse ORIGIN DNA once for fallback translation
    try:
        origin_dna = parse_genbank_dna(path)
    except Exception:
        origin_dna = ""

    proteins: List[str] = []
    sources: List[bool] = []
    in_features = False
    in_origin = False

    cur_loc: Optional[str] = None
    cur_translation: Optional[str] = None
    cur_codon_start: int = 1
    cur_is_pseudo: bool = False
    reading_translation = False
    trans_buf: List[str] = []

    def _reset():
        nonlocal cur_loc, cur_translation, cur_codon_start, cur_is_pseudo, reading_translation, trans_buf
        cur_loc = None
        cur_translation = None
        cur_codon_start = 1
        cur_is_pseudo = False
        reading_translation = False
        trans_buf = []

    def flush_cds():
        """Finalize current CDS feature and append protein if it passes filters."""
        nonlocal cur_loc, cur_translation, cur_codon_start, cur_is_pseudo, reading_translation, trans_buf
        if cur_loc is None:
            return

        # pseudo/pseudogene -> skip
        if cur_is_pseudo:
            _reset()
            return

        # partial location -> skip
        if reject_partial_cds and ("<" in cur_loc or ">" in cur_loc):
            _reset()
            return

        from_translation = bool(cur_translation)

        # build protein
        prot = ""
        if cur_translation:
            prot = cur_translation
        else:
            if require_translation:
                _reset()
                return
            if origin_dna:
                cds_dna = extract_dna_from_location(origin_dna, cur_loc)
                if cds_dna:
                    prot = translate_cds(cds_dna, codon_start=cur_codon_start)

        prot = prot.replace(" ", "").replace("\n", "").strip().upper()

        # normalize trailing '*'
        if prot.endswith("*"):
            prot = prot[:-1]

        # filters
        if not prot:
            _reset()
            return
        if len(prot) < int(min_aa):
            _reset()
            return
        if max_protein_aa and int(max_protein_aa) > 0 and len(prot) > int(max_protein_aa):
            _reset()
            return
        if require_start_m and not prot.startswith("M"):
            _reset()
            return
        if x_free and ("X" in prot or "*" in prot):
            _reset()
            return

        proteins.append(prot)
        sources.append(from_translation)
        _reset()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                in_origin = True
                in_features = False
                if reading_translation and trans_buf:
                    cur_translation = "".join(trans_buf)
                flush_cds()
                continue
            if in_origin:
                continue
            if not in_features:
                continue

            # New feature begins at column 6 (5 spaces) with a key
            if line.startswith("     ") and len(line) > 5 and line[5] != " ":
                # starting a new feature; flush previous CDS if any
                if reading_translation and trans_buf:
                    cur_translation = "".join(trans_buf)
                flush_cds()

                key = line[5:21].strip()
                loc = line[21:].strip()
                if key == "CDS":
                    cur_loc = loc
                    cur_translation = None
                    cur_codon_start = 1
                    cur_is_pseudo = False
                    reading_translation = False
                    trans_buf = []
                else:
                    cur_loc = None
                continue

            # Qualifiers: start around column 22 (21 spaces)
            if cur_loc is not None and line.startswith("                     "):
                q = line.strip()

                if reading_translation:
                    if q.endswith('"'):
                        trans_buf.append(q.rstrip('"'))
                        cur_translation = "".join(trans_buf)
                        reading_translation = False
                        trans_buf = []
                    else:
                        trans_buf.append(q)
                    continue

                if q.startswith("/pseudo") or q.startswith("/pseudogene"):
                    cur_is_pseudo = True
                    continue
                if q.startswith("/codon_start="):
                    try:
                        cur_codon_start = int(q.split("=", 1)[1].strip().strip('"'))
                    except Exception:
                        cur_codon_start = 1
                    continue
                if q.startswith("/translation="):
                    val = q.split("=", 1)[1].lstrip()
                    if val.startswith('"'):
                        val = val[1:]
                    if val.endswith('"'):
                        cur_translation = val.rstrip('"')
                        reading_translation = False
                        trans_buf = []
                    else:
                        reading_translation = True
                        trans_buf = [val]
                    continue

    # flush at EOF
    if reading_translation and trans_buf:
        cur_translation = "".join(trans_buf)
    flush_cds()

    if return_sources:
        return list(zip(proteins, sources))
    return proteins


# -----------------------------
# Base encoding
# -----------------------------

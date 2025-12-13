import logging, os
from typing import List, Optional, Tuple

import numpy as np
from .config import IOConfig

# -----------------------------
# CODON tokenizer constants
# -----------------------------
_CODON_BASES = "ACGT"
CODONS: List[str] = [a + b + c for a in _CODON_BASES for b in _CODON_BASES for c in _CODON_BASES]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
UNK_IDX = len(CODONS)  # 64
IDX_TO_CODON: List[str] = CODONS + ["NNN"]
CODON_VOCAB_SIZE = len(IDX_TO_CODON)  # 65

_gc_frac = []
_gc_count = []
for c in CODONS:
    gc = sum(1 for ch in c if ch in ("G", "C"))
    _gc_count.append(float(gc))
    _gc_frac.append(float(gc) / 3.0)
_gc_count.append(0.0)
_gc_frac.append(0.0)
GC_COUNT_PER_TOKEN = np.array(_gc_count, dtype=np.float32)     # (65,)
GC_FRAC_PER_TOKEN = np.array(_gc_frac, dtype=np.float32)       # (65,)

# -----------------------------
# AA tokenizer constants
# -----------------------------
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20
AA_UNK = "X"
AA_VOCAB = AA_ALPHABET + AA_UNK       # 21
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
IDX_TO_AA = list(AA_VOCAB)
AA_VOCAB_SIZE = len(AA_VOCAB)

HYDROPHOBIC = set(list("AILMFWVYC"))  # simple set
HYDRO_IDX = np.array([1.0 if aa in HYDROPHOBIC else 0.0 for aa in IDX_TO_AA], dtype=np.float32)

# Standard genetic code (DNA codons)
CODON_TO_AA = {
    # Phenylalanine
    "TTT":"F","TTC":"F",
    # Leucine
    "TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    # Isoleucine
    "ATT":"I","ATC":"I","ATA":"I",
    # Methionine (start)
    "ATG":"M",
    # Valine
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    # Serine
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    # Proline
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    # Threonine
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    # Alanine
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    # Tyrosine
    "TAT":"Y","TAC":"Y",
    # Histidine
    "CAT":"H","CAC":"H",
    # Glutamine
    "CAA":"Q","CAG":"Q",
    # Asparagine
    "AAT":"N","AAC":"N",
    # Lysine
    "AAA":"K","AAG":"K",
    # Aspartic acid
    "GAT":"D","GAC":"D",
    # Glutamic acid
    "GAA":"E","GAG":"E",
    # Cysteine
    "TGT":"C","TGC":"C",
    # Tryptophan
    "TGG":"W",
    # Arginine
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    # Glycine
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
STOP_CODONS = {"TAA","TAG","TGA"}
START_CODON = "ATG"

def tokenizer_meta(tokenizer: str, window_size: int) -> Tuple[int, int]:
    """
    Returns (seq_len_units, vocab_size)

    base: units = window_size (bp), vocab = 4
    codon: units = window_size//3 (codons), vocab = 65
    aa: units = window_size (amino acids), vocab = 21
    """
    tok = tokenizer.lower()
    if tok == "base":
        return window_size, 4
    if tok == "codon":
        if window_size % 3 != 0:
            raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size})")
        return window_size // 3, CODON_VOCAB_SIZE
    if tok == "aa":
        if window_size <= 0:
            raise ValueError("aa tokenizer requires window_size > 0 (amino acids)")
        return window_size, AA_VOCAB_SIZE
    raise ValueError(f"Unknown tokenizer: {tokenizer}")

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
    comp = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
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
        a = a.strip(); b = b.strip()
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
            inner = expr[expr.find("(")+1:-1]
            segs = _split_top_level_commas(inner)
            out = []
            for seg in segs:
                seq1, comp1 = parse_expr(seg)
                if comp1:
                    # If any segment returns complement, apply at segment level
                    seq1 = reverse_complement(seq1)
                out.append(seq1)
            return "".join(out), False
        # simple range
        start, end = _parse_loc_range(expr)
        if start < 1 or end < 1 or end < start:
            return "", False
        # GenBank is 1-based inclusive
        return dna[start-1:end], False

    seq, is_comp = parse_expr(s)
    if is_comp:
        seq = reverse_complement(seq)
    return seq

def translate_cds(dna: str, codon_start: int = 1) -> str:
    """Translate CDS DNA to AA; respects codon_start (1..3)."""
    if codon_start not in (1, 2, 3):
        codon_start = 1
    dna = dna.upper().replace("U", "T")
    if codon_start > 1 and len(dna) >= codon_start - 1:
        dna = dna[codon_start - 1 :]
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i+3]
        if cod in STOP_CODONS:
            break
        aas.append(CODON_TO_AA.get(cod, "X"))
    return "".join(aas)

def extract_cds_proteins_from_genbank(path: str, min_aa: int = 90, return_sources: bool = False):
    """
    Extract protein sequences from GenBank FEATURES/CDS.

    Preference order per CDS:
      1) /translation="..." qualifier (most reliable)
      2) location + ORIGIN DNA + /codon_start (fallback)

    Notes:
      - Skips CDS with /pseudo or /pseudogene.
      - Handles common location forms: 123..456, complement(123..456), join(...), complement(join(...)).
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

    def flush_cds():
        nonlocal cur_loc, cur_translation, cur_codon_start, cur_is_pseudo, reading_translation, trans_buf
        if cur_loc is None:
            return
        if cur_is_pseudo:
            # reset
            cur_loc = None; cur_translation = None; cur_codon_start = 1; cur_is_pseudo = False
            reading_translation = False; trans_buf = []
            return

        from_translation = bool(cur_translation)

        prot = ""
        if cur_translation:
            prot = cur_translation
        elif origin_dna:
            cds_dna = extract_dna_from_location(origin_dna, cur_loc)
            if cds_dna:
                prot = translate_cds(cds_dna, codon_start=cur_codon_start)

        prot = prot.replace(" ", "").replace("\n", "").strip().upper()
        # Remove trailing '*' if present
        if prot.endswith("*"):
            prot = prot[:-1]
        if prot and len(prot) >= min_aa:
            proteins.append(prot)
            sources.append(from_translation)

        # reset
        cur_loc = None; cur_translation = None; cur_codon_start = 1; cur_is_pseudo = False
        reading_translation = False; trans_buf = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                in_origin = True
                # end of FEATURES parsing
                in_features = False
                # flush any CDS we were building
                if reading_translation and trans_buf:
                    cur_translation = "".join(trans_buf)
                flush_cds()
                continue
            if in_origin:
                # no need to parse further here
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
                    # keep accumulating until closing quote
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
                    # translation is quoted and can span lines
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


    if return_sources:
        return list(zip(proteins, sources))
    return proteins
# -----------------------------
# Base encoding
# -----------------------------
def encode_sequence_one_hot(seq: str, window_size: int, stride: int) -> np.ndarray:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    length = len(seq)

    if length < window_size:
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        return arr[None, ...]

    windows: List[np.ndarray] = []
    for start in range(0, length - window_size + 1, stride):
        window_seq = seq[start : start + window_size]
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(window_seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    if not windows:
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq[:window_size]):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    return np.stack(windows, axis=0)

# -----------------------------
# Codon encoding
# -----------------------------
def encode_sequence_codons(seq: str, window_size_bp: int, stride_bp: int, frame_offset: int = 0) -> np.ndarray:
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")
    if window_size_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires window_size divisible by 3 (got {window_size_bp})")
    if stride_bp % 3 != 0:
        raise ValueError(f"codon tokenizer requires stride divisible by 3 (got {stride_bp})")

    seq = seq.upper()
    if len(seq) <= frame_offset + 2:
        window_codons = window_size_bp // 3
        out = np.zeros((1, window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        return out

    usable = len(seq) - ((len(seq) - frame_offset) % 3)
    codon_count = max(0, (usable - frame_offset) // 3)
    window_codons = window_size_bp // 3
    stride_codons = stride_bp // 3

    def codon_at(i: int) -> int:
        start = frame_offset + 3 * i
        c = seq[start : start + 3]
        return CODON_TO_IDX.get(c, UNK_IDX)

    if codon_count <= 0:
        out = np.zeros((1, window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        return out

    windows: List[np.ndarray] = []

    if codon_count < window_codons:
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(codon_count):
            idx = codon_at(j)
            arr[j, idx] = 1.0
        windows.append(arr)
        return np.stack(windows, axis=0)

    for start_c in range(0, codon_count - window_codons + 1, stride_codons):
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(window_codons):
            idx = codon_at(start_c + j)
            arr[j, idx] = 1.0
        windows.append(arr)

    if not windows:
        arr = np.zeros((window_codons, CODON_VOCAB_SIZE), dtype=np.float32)
        for j in range(window_codons):
            idx = codon_at(j)
            arr[j, idx] = 1.0
        windows.append(arr)

    return np.stack(windows, axis=0)

# -----------------------------
# Proteome (ORF -> AA) extraction + encoding
# -----------------------------
def translate_orf(dna: str) -> str:
    """
    dna length must be multiple of 3; stops are not included here.
    """
    dna = dna.upper()
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i+3]
        if cod in STOP_CODONS:
            break
        aa = CODON_TO_AA.get(cod, "X")
        aas.append(aa)
    return "".join(aas)

def find_orfs_proteins(seq: str, min_orf_aa: int = 90) -> List[str]:
    """
    Very simple ORF finder:
      - scans 3 frames on forward and reverse-complement
      - start codon ATG
      - stops at TAA/TAG/TGA
      - returns translated proteins length >= min_orf_aa
    """
    seq = seq.upper()
    proteins: List[str] = []

    def scan_strand(s: str):
        L = len(s)
        for frame in (0, 1, 2):
            i = frame
            while i + 2 < L:
                cod = s[i:i+3]
                if cod == START_CODON:
                    # find stop
                    j = i
                    while j + 2 < L:
                        cod2 = s[j:j+3]
                        if cod2 in STOP_CODONS:
                            orf_dna = s[i:j]  # exclude stop codon
                            if len(orf_dna) >= min_orf_aa * 3:
                                prot = translate_orf(orf_dna)
                                if len(prot) >= min_orf_aa:
                                    proteins.append(prot)
                            i = j + 3
                            break
                        j += 3
                    else:
                        # no stop found
                        i += 3
                else:
                    i += 3

    scan_strand(seq)
    scan_strand(reverse_complement(seq))
    return proteins

def encode_proteins_aa_windows(
    proteins: List[str],
    window_aa: int,
    stride_aa: int,
    max_windows_per_protein: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns: (num_windows, window_aa, 21)
    Pads short proteins to one window.
    """
    if window_aa <= 0 or stride_aa <= 0:
        raise ValueError("window_aa and stride_aa must be > 0")
    if rng is None:
        rng = np.random.default_rng(0)
    windows: List[np.ndarray] = []

    for prot in proteins:
        prot = prot.strip().upper()
        if not prot:
            continue

        if len(prot) < window_aa:
            arr = np.zeros((window_aa, AA_VOCAB_SIZE), dtype=np.float32)
            for i, aa in enumerate(prot):
                idx = AA_TO_IDX.get(aa, AA_TO_IDX[AA_UNK])
                arr[i, idx] = 1.0
            windows.append(arr)
            continue

        starts = list(range(0, len(prot) - window_aa + 1, stride_aa))
        if not starts:
            starts = [0]
        if max_windows_per_protein is not None and max_windows_per_protein > 0 and len(starts) > max_windows_per_protein:
            starts = list(rng.choice(starts, size=int(max_windows_per_protein), replace=False))
            starts.sort()

        for start in starts:
            chunk = prot[start:start+window_aa]
            arr = np.zeros((window_aa, AA_VOCAB_SIZE), dtype=np.float32)
            for i, aa in enumerate(chunk):
                idx = AA_TO_IDX.get(aa, AA_TO_IDX[AA_UNK])
                arr[i, idx] = 1.0
            windows.append(arr)

    if not windows:
        # return a single all-zero window to keep pipeline alive
        return np.zeros((1, window_aa, AA_VOCAB_SIZE), dtype=np.float32)

    return np.stack(windows, axis=0)

def encode_accession(
    accession: str,
    io_cfg: IOConfig,
    window_size: int,
    stride: int,
    tokenizer: str = "base",
    frame_offset: int = 0,
    min_orf_aa: int = 90,
    max_windows_per_protein: int | None = None,
    protein_len_min: int | None = None,
    protein_len_max: int | None = None,
    translation_only: bool = False,
    rng_seed: int | None = None,
    source: str = "fasta",
    save_to_disk: bool = True,
    out_path: Optional[str] = None,
) -> np.ndarray:
    tok = tokenizer.lower()
    src = source.lower()
    if src == "genbank":
        gb_path = os.path.join(getattr(io_cfg, "cache_genbank_dir", "cache/genbank"), f"{accession}.gb")
        if not os.path.exists(gb_path):
            raise FileNotFoundError(f"GenBank for {accession} not found at {gb_path}")
        seq = parse_genbank_dna(gb_path)
        gb_for_proteins = gb_path
    else:
        fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA for {accession} not found at {fasta_path}")
        seq = parse_fasta_sequence(fasta_path)
        gb_for_proteins = None

    if tok == "base":
        encoded = encode_sequence_one_hot(seq, window_size, stride)
        logging.info(f"{accession}: encoded(BASE) len={len(seq)} -> windows={encoded.shape[0]} shape={encoded.shape}")

    elif tok == "codon":
        encoded = encode_sequence_codons(seq, window_size_bp=window_size, stride_bp=stride, frame_offset=frame_offset)
        logging.info(f"{accession}: encoded(CODON) len={len(seq)} -> windows={encoded.shape[0]} shape={encoded.shape}")

    elif tok == "aa":
        proteins: List[str] = []
        if src == "genbank" and gb_for_proteins is not None:
            try:
                prot_items = extract_cds_proteins_from_genbank(gb_for_proteins, min_aa=min_orf_aa, return_sources=True)
                # prot_items may be list[str] or list[(str, bool)] depending on return_sources
                if prot_items and isinstance(prot_items[0], tuple):
                    if translation_only:
                        proteins = [p for (p, from_tr) in prot_items if from_tr]
                    else:
                        proteins = [p for (p, _) in prot_items]
                else:
                    proteins = list(prot_items)
                if proteins:
                    logging.info(f"{accession}: extracted CDS proteins from GenBank: {len(proteins)} (min_aa={min_orf_aa})")
                else:
                    logging.warning(f"{accession}: GenBank contained no CDS translations >= {min_orf_aa}aa; falling back to naive ORFs.")
            except Exception as e:
                logging.warning(f"{accession}: failed to parse GenBank CDS ({e}); falling back to naive ORFs.")
                proteins = []
        if not proteins:
            proteins = find_orfs_proteins(seq, min_orf_aa=min_orf_aa)
        # Optional protein length filters (in addition to min_orf_aa)
        if proteins and protein_len_min is not None:
            proteins = [p for p in proteins if len(p) >= int(protein_len_min)]
        if proteins and protein_len_max is not None:
            proteins = [p for p in proteins if len(p) <= int(protein_len_max)]

        # Deterministic sampling seed for balancing across proteins
        if rng_seed is None:
            import zlib
            rng_seed = zlib.crc32(accession.encode('utf-8')) & 0xFFFFFFFF
        rng = np.random.default_rng(int(rng_seed))

        encoded = encode_proteins_aa_windows(
            proteins, window_aa=window_size, stride_aa=stride,
            max_windows_per_protein=max_windows_per_protein,
            rng=rng,
        )
        logging.info(
            f"{accession}: encoded(AA) src={src} genome_len={len(seq)} proteins={len(proteins)} "
            f"-> windows={encoded.shape[0]} window_aa={window_size} stride_aa={stride} shape={encoded.shape}"
        )

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    if save_to_disk:
        os.makedirs(io_cfg.cache_encoded_dir, exist_ok=True)
        if out_path is None:
            out_path = os.path.join(io_cfg.cache_encoded_dir, f"{accession}.npy")
        np.save(out_path, encoded)
        logging.info(f"{accession}: saved encoded tensor to {out_path}")

    return encoded

def compute_gc_from_encoded(encoded: np.ndarray, tokenizer: str = "base") -> np.ndarray:
    """
    Returns per-window metric:
      - base/codon: GC fraction
      - aa: hydrophobic fraction (simple proxy), returned in same slot
    """
    tok = tokenizer.lower()

    if tok == "base":
        if encoded.ndim != 3 or encoded.shape[2] != 4:
            raise ValueError("base encoded must have shape (num_windows, window_size, 4)")
        gc_counts = encoded[:, :, 1] + encoded[:, :, 2]
        window_size = encoded.shape[1]
        return (gc_counts.sum(axis=1) / float(window_size)).astype(np.float32)

    if tok == "codon":
        if encoded.ndim != 3 or encoded.shape[2] != CODON_VOCAB_SIZE:
            raise ValueError("codon encoded must have shape (num_windows, window_codons, 65)")
        gc_pos = (encoded * GC_FRAC_PER_TOKEN[None, None, :]).sum(axis=2)
        return gc_pos.mean(axis=1).astype(np.float32)

    if tok == "aa":
        if encoded.ndim != 3 or encoded.shape[2] != AA_VOCAB_SIZE:
            raise ValueError("aa encoded must have shape (num_windows, window_aa, 21)")
        # expected hydrophobic indicator per position, then mean
        hydro_pos = (encoded * HYDRO_IDX[None, None, :]).sum(axis=2)  # (N, W)
        return hydro_pos.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unknown tokenizer: {tokenizer}")

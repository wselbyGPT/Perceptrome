from typing import List

import numpy as np


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
GC_COUNT_PER_TOKEN = np.array(_gc_count, dtype=np.float32)  # (65,)
GC_FRAC_PER_TOKEN = np.array(_gc_frac, dtype=np.float32)    # (65,)

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
    "TTT": "F", "TTC": "F",
    # Leucine
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    # Isoleucine
    "ATT": "I", "ATC": "I", "ATA": "I",
    # Methionine (start)
    "ATG": "M",
    # Valine
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    # Serine
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    # Proline
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    # Threonine
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    # Alanine
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    # Tyrosine
    "TAT": "Y", "TAC": "Y",
    # Histidine
    "CAT": "H", "CAC": "H",
    # Glutamine
    "CAA": "Q", "CAG": "Q",
    # Asparagine
    "AAT": "N", "AAC": "N",
    # Lysine
    "AAA": "K", "AAG": "K",
    # Aspartic acid
    "GAT": "D", "GAC": "D",
    # Glutamic acid
    "GAA": "E", "GAG": "E",
    # Cysteine
    "TGT": "C", "TGC": "C",
    # Tryptophan
    "TGG": "W",
    # Arginine
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    # Glycine
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODON = "ATG"

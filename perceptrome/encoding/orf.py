from typing import List, Optional

import numpy as np


from .constants import CODON_TO_AA, STOP_CODONS, START_CODON, AA_TO_IDX, AA_UNK, AA_VOCAB_SIZE

from .parse import reverse_complement


def translate_orf(dna: str) -> str:
    """dna length must be multiple of 3; stops are not included here."""
    dna = dna.upper()
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        cod = dna[i:i + 3]
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
                cod = s[i:i + 3]
                if cod == START_CODON:
                    j = i
                    while j + 2 < L:
                        cod2 = s[j:j + 3]
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
    max_windows_per_protein: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
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
            chunk = prot[start:start + window_aa]
            arr = np.zeros((window_aa, AA_VOCAB_SIZE), dtype=np.float32)
            for i, aa in enumerate(chunk):
                idx = AA_TO_IDX.get(aa, AA_TO_IDX[AA_UNK])
                arr[i, idx] = 1.0
            windows.append(arr)

    if not windows:
        return np.zeros((1, window_aa, AA_VOCAB_SIZE), dtype=np.float32)

    return np.stack(windows, axis=0)



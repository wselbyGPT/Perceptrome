from typing import List

import numpy as np


from .constants import (CODON_TO_IDX, UNK_IDX, CODON_VOCAB_SIZE, GC_FRAC_PER_TOKEN, AA_VOCAB_SIZE, HYDRO_IDX)


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
        window_seq = seq[start:start + window_size]
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
        c = seq[start:start + 3]
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
        hydro_pos = (encoded * HYDRO_IDX[None, None, :]).sum(axis=2)  # (N, W)
        return hydro_pos.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unknown tokenizer: {tokenizer}")

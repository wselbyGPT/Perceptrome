import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from .config import IOConfig
from .encoding.constants import (
    CODONS, CODON_TO_IDX, UNK_IDX, IDX_TO_CODON, CODON_VOCAB_SIZE,
    GC_COUNT_PER_TOKEN, GC_FRAC_PER_TOKEN,
    AA_ALPHABET, AA_UNK, AA_VOCAB, AA_TO_IDX, IDX_TO_AA, AA_VOCAB_SIZE,
    HYDROPHOBIC, HYDRO_IDX,
    CODON_TO_AA, STOP_CODONS, START_CODON,

)
from .encoding.parse import (
    parse_fasta_sequence, reverse_complement, parse_genbank_dna,
    extract_dna_from_location,
)
from .encoding.protein import (translate_cds, extract_cds_proteins_from_genbank)
from .encoding.orf import (translate_orf, find_orfs_proteins, encode_proteins_aa_windows)
from .encoding.protein_opts import _normalize_protein_opts
from .encoding.encode import (encode_sequence_one_hot, encode_sequence_codons, compute_gc_from_encoded)








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


def encode_accession(
    accession: str,
    io_cfg: IOConfig,
    window_size: int,
    stride: int,
    tokenizer: str = "base",
    frame_offset: int = 0,
    min_orf_aa: int = 90,
    max_windows_per_protein: Optional[int] = None,
    protein_len_min: Optional[int] = None,
    protein_len_max: Optional[int] = None,
    translation_only: bool = False,
    rng_seed: Optional[int] = None,
    source: str = "fasta",
    save_to_disk: bool = True,
    out_path: Optional[str] = None,
    # optional grounded protein options
    protein_opts: Optional[Dict[str, Any]] = None,
    strict_cds: Optional[bool] = None,
    require_translation: Optional[bool] = None,
    x_free: Optional[bool] = None,
    require_start_m: Optional[bool] = None,
    reject_partial_cds: Optional[bool] = None,
    max_protein_aa: Optional[int] = None,
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
        # ---- define grounded options safely (prevents NameError) ----
        p = _normalize_protein_opts(protein_opts)
        strict_cds_v = bool(strict_cds) if strict_cds is not None else bool(p["strict_cds"])
        require_translation_v = bool(require_translation) if require_translation is not None else bool(p["require_translation"])
        x_free_v = bool(x_free) if x_free is not None else bool(p["x_free"])
        require_start_m_v = bool(require_start_m) if require_start_m is not None else bool(p["require_start_m"])
        reject_partial_cds_v = bool(reject_partial_cds) if reject_partial_cds is not None else bool(p["reject_partial_cds"])
        if max_protein_aa is None:
            max_protein_aa_v = int(p["max_protein_aa"]) if p.get("max_protein_aa", 0) else 0
        else:
            max_protein_aa_v = int(max_protein_aa)

        proteins: List[str] = []

        if src == "genbank" and gb_for_proteins is not None:
            try:
                prot_items = extract_cds_proteins_from_genbank(
                    gb_for_proteins,
                    min_aa=min_orf_aa,
                    return_sources=True,
                    require_translation=require_translation_v,
                    reject_partial_cds=reject_partial_cds_v,
                    require_start_m=require_start_m_v,
                    x_free=x_free_v,
                    max_protein_aa=max_protein_aa_v,
                )
                # prot_items is list[(str, bool)]
                if translation_only:
                    proteins = [pseq for (pseq, from_tr) in prot_items if from_tr]
                else:
                    proteins = [pseq for (pseq, _from_tr) in prot_items]

                if proteins:
                    logging.info(f"{accession}: extracted CDS proteins from GenBank: {len(proteins)} (min_aa={min_orf_aa})")
                else:
                    logging.warning(
                        f"{accession}: GenBank contained no CDS proteins >= {min_orf_aa}aa (filters applied); "
                        f"{'strict-cds enabled' if strict_cds_v else 'falling back to naive ORFs'}."
                    )
            except Exception as e:
                if strict_cds_v:
                    raise
                logging.warning(f"{accession}: failed to parse GenBank CDS ({e}); falling back to naive ORFs.")
                proteins = []

        if not proteins:
            if strict_cds_v:
                raise ValueError(
                    f"{accession}: strict-cds enabled but no CDS proteins passed filters "
                    f"(min_aa={min_orf_aa}, require_translation={require_translation_v}, x_free={x_free_v}, "
                    f"require_start_m={require_start_m_v}, reject_partial_cds={reject_partial_cds_v}, "
                    f"max_protein_aa={max_protein_aa_v})"
                )
            proteins = find_orfs_proteins(seq, min_orf_aa=min_orf_aa)

        # Optional protein length filters (in addition to min_orf_aa)
        if proteins and protein_len_min is not None:
            proteins = [pp for pp in proteins if len(pp) >= int(protein_len_min)]
        if proteins and protein_len_max is not None:
            proteins = [pp for pp in proteins if len(pp) <= int(protein_len_max)]

        # Deterministic sampling seed for balancing across proteins
        if rng_seed is None:
            import zlib
            rng_seed = zlib.crc32(accession.encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.default_rng(int(rng_seed))

        encoded = encode_proteins_aa_windows(
            proteins,
            window_aa=window_size,
            stride_aa=stride,
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



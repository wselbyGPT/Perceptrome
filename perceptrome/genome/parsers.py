from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from perceptrome.encoding.genbank_features import CDSFeature, parse_cds_features_from_genbank
from perceptrome.encoding.parse import parse_fasta_sequence, parse_genbank_dna


GENBANK_EXTENSIONS = (".gb", ".gbk", ".genbank")
FASTA_EXTENSIONS = (".fa", ".fasta", ".fna")
SUPPORTED_EXTENSIONS = GENBANK_EXTENSIONS + FASTA_EXTENSIONS


@dataclass(frozen=True)
class GenomeInputContents:
    sequence: str
    cds_features: Optional[List[CDSFeature]]
    source_format: str


def detect_input_format(path: str) -> str:
    low = path.lower()
    if low.endswith(GENBANK_EXTENSIONS):
        return "genbank"
    if low.endswith(FASTA_EXTENSIONS):
        return "fasta"
    raise ValueError(f"Unsupported genome input extension: {path}")


def genome_input_files_in_dir(input_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if name.lower().endswith(SUPPORTED_EXTENSIONS):
            out.append(path)
    return out


def load_genome_input(path: str) -> GenomeInputContents:
    fmt = detect_input_format(path)
    if fmt == "genbank":
        sequence = parse_genbank_dna(path)
        try:
            features = parse_cds_features_from_genbank(path)
        except Exception:
            features = []
        return GenomeInputContents(
            sequence=sequence,
            cds_features=features,
            source_format="genbank",
        )
    sequence = parse_fasta_sequence(path)
    return GenomeInputContents(
        sequence=sequence,
        cds_features=None,
        source_format="fasta",
    )

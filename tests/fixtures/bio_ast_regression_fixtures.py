"""Synthetic regression fixtures for Bio-AST related tests."""

from __future__ import annotations

SYNTHETIC_FASTA_SEQUENCE = (
    "ATG"
    + ("GCT" * 18)
    + ("GAA" * 12)
    + ("TTC" * 8)
    + "TAA"
)

SECONDARY_TAG_SET = ["h", "E", "c", "T", "g", "i"]

SME_MOTIF_FIXTURES = [
    {
        "motif_name": "helix-entry",
        "secondary_tag": "h",
        "motif_family": "structural",
        "motif_subtype": "coiled_coil",
        "energetic_evolutionary": {
            "folding_energy_estimate": -8.5,
            "phi_bin": -65,
            "psi_bin": -45,
            "conservation_score": 0.82,
            "prion_likelihood": 0.03,
            "variant_sensitivity": 0.21,
        },
    },
    {
        "motif_name": "HTH",
        "secondary_tag": "E",
        "motif_family": "regulatory",
        "motif_subtype": "binding_loop",
        "energetic_evolutionary": {
            "folding_energy_estimate": -5.2,
            "phi_bin": -120,
            "psi_bin": 130,
            "conservation_score": 0.91,
            "prion_likelihood": 0.01,
            "variant_sensitivity": 0.34,
        },
    },
]


def render_fasta(sequence: str = SYNTHETIC_FASTA_SEQUENCE, header: str = "synthetic") -> str:
    return f">{header}\n{sequence}\n"


def render_genbank(sequence: str = SYNTHETIC_FASTA_SEQUENCE) -> str:
    cds_start = 1
    cds_end = len(sequence)
    origin_lines = []
    for i in range(0, len(sequence), 60):
        chunk = sequence[i : i + 60].lower()
        grouped = " ".join(chunk[j : j + 10] for j in range(0, len(chunk), 10))
        origin_lines.append(f"{i + 1:>9} {grouped}")

    origin_block = "\n".join(origin_lines)
    return (
        "LOCUS       SYNTH001         {length} bp    DNA     PLN       01-JAN-2000\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             1..{length}\n"
        "                     /gene=\"hthA\"\n"
        "                     /product=\"helix-turn-helix regulator\"\n"
        "ORIGIN\n"
        "{origin}\n"
        "//\n"
    ).format(length=cds_end, origin=origin_block)

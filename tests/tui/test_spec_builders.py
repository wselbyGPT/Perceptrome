from __future__ import annotations

import pytest

from perceptrome.tui.spec_builders import (
    SpecValidationError,
    build_generate_plasmid_spec,
    build_train_one_spec,
)


def test_build_train_one_requires_accession() -> None:
    with pytest.raises(SpecValidationError):
        build_train_one_spec(accession="")


def test_build_generate_plasmid_happy_path() -> None:
    spec = build_generate_plasmid_spec(output="out.fasta", length=128, top_k=5)
    assert spec.kind == "generate_plasmid"
    assert spec.params["output"] == "out.fasta"
    assert spec.params["length"] == 128

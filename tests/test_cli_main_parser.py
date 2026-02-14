from __future__ import annotations

import pytest


def test_cli_parser_init_command(cli_main_module) -> None:
    parser = cli_main_module.build_parser()

    args = parser.parse_args(["--config", "custom.yaml", "init"])

    assert args.command == "init"
    assert args.config == "custom.yaml"


def test_cli_parser_train_one_arguments(cli_main_module) -> None:
    parser = cli_main_module.build_parser()

    args = parser.parse_args(
        [
            "train-one",
            "ABC123",
            "--tokenizer",
            "codon",
            "--frame-offset",
            "2",
            "--window-size",
            "600",
            "--stride",
            "300",
            "--source",
            "fasta",
        ]
    )

    assert args.accession == "ABC123"
    assert args.tokenizer == "codon"
    assert args.frame_offset == 2
    assert args.window_size == 600
    assert args.stride == 300


def test_cli_parser_rejects_invalid_frame_choice(cli_main_module) -> None:
    parser = cli_main_module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["encode-one", "ABC123", "--tokenizer", "codon", "--frame-offset", "4"])

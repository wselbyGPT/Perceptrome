from __future__ import annotations

import argparse
import traceback
from typing import Callable, Dict, Optional

CommandFunc = Callable[[argparse.Namespace], int]

_COMMANDS: Optional[Dict[str, CommandFunc]] = None
_COMMAND_IMPORT_ERROR: Optional[BaseException] = None


def load_commands() -> Dict[str, CommandFunc]:
    global _COMMANDS, _COMMAND_IMPORT_ERROR
    if _COMMANDS is not None:
        return _COMMANDS

    try:
        from perceptrome.cli.commands import (  # type: ignore
            cmd_encode_one,
            cmd_fetch_one,
            cmd_generate_plasmid,
            cmd_generate_protein,
            cmd_train_one,
        )

        _COMMANDS = {
            "fetch_one": cmd_fetch_one,
            "encode_one": cmd_encode_one,
            "train_one": cmd_train_one,
            "gen_plasmid": cmd_generate_plasmid,
            "gen_protein": cmd_generate_protein,
        }
        _COMMAND_IMPORT_ERROR = None
    except BaseException as e:  # noqa: BLE001
        _COMMANDS = {}
        _COMMAND_IMPORT_ERROR = e

    return _COMMANDS


def get_import_error() -> Optional[BaseException]:
    return _COMMAND_IMPORT_ERROR


def import_help_message(err: BaseException) -> str:
    msg = [
        "Perceptrome command import failed.",
        "",
        f"Error: {type(err).__name__}: {err}",
        "",
        "Common fixes:",
        "  • Ensure you're in the correct venv for Perceptrome",
        "  • If PyTorch is missing/broken, install a CPU build:",
        "      python -m pip install --upgrade pip",
        "      python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "",
        "Full traceback is shown below.",
    ]
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    return "\n".join(msg) + "\n\n" + tb


from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def fixture_config_path() -> Path:
    return Path(__file__).parent / "fixtures" / "stream_config_test.yaml"


def _make_fake_common() -> types.ModuleType:
    from perceptrome.config import extract_configs, load_full_config
    from perceptrome.io_utils import (
        encoded_cache_path,
        ensure_dirs,
        load_state,
        read_catalog,
        save_state,
        setup_logging,
    )

    mod = types.ModuleType("perceptrome.cli.common")
    mod.extract_configs = extract_configs
    mod.load_full_config = load_full_config
    mod.encoded_cache_path = encoded_cache_path
    mod.ensure_dirs = ensure_dirs
    mod.load_state = load_state
    mod.read_catalog = read_catalog
    mod.save_state = save_state
    mod.setup_logging = setup_logging

    # placeholders for names imported by perceptrome.cli.commands
    mod.compute_gc_from_encoded = lambda *a, **k: None
    mod.encode_accession = lambda *a, **k: None
    mod.generate_plasmid_sequence = lambda *a, **k: None
    mod.generate_protein_sequence = lambda *a, **k: None
    mod.fetch_fasta = lambda *a, **k: None
    mod.fetch_genbank = lambda *a, **k: None
    mod.cleanup_accession_files = lambda *a, **k: None
    mod.compute_window_errors = lambda *a, **k: None
    mod.train_on_encoded = lambda *a, **k: 0.0
    mod.curses = None
    mod.run_scope_ui = lambda *a, **k: None
    mod.run_scope_stream_ui = lambda *a, **k: None
    mod.ScopeStreamContext = object
    mod._get_tok = lambda *a, **k: "base"
    mod._get_frame = lambda *a, **k: 0
    mod._get_min_orf = lambda *a, **k: 90
    mod._get_grounded = lambda *a, **k: {}
    mod._get_protein_opts = lambda *a, **k: {}
    mod._get_source = lambda *a, **k: "fasta"
    mod._ensure_record = lambda *a, **k: ""
    return mod


@pytest.fixture
def commands_module(monkeypatch):
    """Load perceptrome.cli.commands with lightweight stubs for heavy deps."""
    import perceptrome

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.load = lambda *a, **k: []

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "perceptrome.cli.common", _make_fake_common())

    module_name = "perceptrome.cli.commands"
    commands_path = Path(perceptrome.__file__).resolve().parent / "cli" / "commands.py"
    spec = importlib.util.spec_from_file_location(module_name, commands_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_main_module(monkeypatch):
    fake_commands = types.ModuleType("perceptrome.cli.commands")
    for name in [
        "cmd_init",
        "cmd_catalog_show",
        "cmd_fetch_one",
        "cmd_encode_one",
        "cmd_train_one",
        "cmd_scope_one",
        "cmd_scope_stream",
        "cmd_stream",
        "cmd_generate_plasmid",
        "cmd_generate_protein",
    ]:
        setattr(fake_commands, name, lambda args, _n=name: _n)

    monkeypatch.setitem(sys.modules, "perceptrome.cli.commands", fake_commands)
    import perceptrome.cli_main as cli_main

    return importlib.reload(cli_main)

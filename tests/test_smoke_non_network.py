from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from perceptrome.io_utils import encoded_cache_path


def test_cmd_init_creates_state_and_dirs(commands_module, tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "io": {
                    "cache_fasta_dir": "test_cache/fasta",
                    "cache_genbank_dir": "test_cache/genbank",
                    "cache_encoded_dir": "test_cache/encoded",
                    "model_dir": "test_model",
                    "checkpoints_dir": "test_model/checkpoints",
                    "logs_dir": "test_logs",
                    "state_file": "test_state/progress.json",
                }
            }
        ),
        encoding="utf-8",
    )

    import perceptrome.config as config_mod

    monkeypatch.setattr(config_mod, "yaml", types.SimpleNamespace(safe_load=json.load))

    args = argparse.Namespace(config=str(cfg_path))

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        rc = commands_module.cmd_init(args)
    finally:
        os.chdir(old_cwd)

    assert rc == 0
    state_file = tmp_path / "test_state" / "progress.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["total_steps"] == 0
    assert (tmp_path / "test_cache" / "encoded").exists()


def test_encoded_cache_path_is_deterministic() -> None:
    io_cfg = argparse.Namespace(cache_encoded_dir="cache/encoded")

    path = encoded_cache_path(
        io_cfg=io_cfg,
        accession="ACC001",
        tokenizer="aa",
        window_size=256,
        stride=128,
        frame_offset=0,
        source="genbank",
        min_orf_aa=90,
        max_windows_per_protein=4,
        protein_len_min=80,
        protein_len_max=600,
        translation_only=True,
        curriculum_tag="cur1",
    )

    assert path == "cache/encoded/ACC001.aa.w256.s128.srcgb.min90.wpp4.pmin80.pmax600.tronly.cur1.npy"

from __future__ import annotations

import copy
import json
import types

from perceptrome.config import DEFAULT_CONFIG, deep_update, extract_configs, load_full_config


def test_deep_update_nested_dict_and_list_replacement() -> None:
    base = {"a": {"x": 1, "lst": [1, 2]}, "b": 3}
    updates = {"a": {"x": 9, "lst": [7]}, "b": {"nested": True}}

    merged = deep_update(base, updates)

    assert merged["a"]["x"] == 9
    assert merged["a"]["lst"] == [7]
    assert merged["b"] == {"nested": True}


def test_load_full_config_merges_fixture_values(fixture_config_path, monkeypatch) -> None:
    import perceptrome.config as config_mod

    monkeypatch.setattr(config_mod, "yaml", types.SimpleNamespace(safe_load=json.load))
    cfg = load_full_config(str(fixture_config_path))

    assert cfg["ncbi"]["email"] == "fixture@example.com"
    assert cfg["training"]["tokenizer"] == "codon"
    assert cfg["training"]["batch_size"] == DEFAULT_CONFIG["training"]["batch_size"]
    assert cfg["training"]["curriculum_steps"] == [0, 1]


def test_extract_configs_converts_types_from_merged_config(fixture_config_path, monkeypatch) -> None:
    import perceptrome.config as config_mod

    monkeypatch.setattr(config_mod, "yaml", types.SimpleNamespace(safe_load=json.load))
    cfg = load_full_config(str(fixture_config_path))
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)

    assert ncbi_cfg.email == "fixture@example.com"
    assert train_cfg.frame_offset == 2
    assert train_cfg.window_size == 600
    assert io_cfg.cache_encoded_dir.endswith("fixture_cache/encoded")


def test_load_full_config_missing_file_keeps_defaults() -> None:
    cfg = load_full_config("/definitely/not/here.yaml")
    default_copy = copy.deepcopy(DEFAULT_CONFIG)
    assert cfg == default_copy

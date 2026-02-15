import tempfile
import textwrap
import unittest

from perceptrome import config


class ConfigBehaviorTests(unittest.TestCase):
    def test_deep_update_merges_dicts_and_replaces_lists(self):
        base = {"training": {"window_size": 512, "curriculum_steps": [0, 100]}, "io": {"logs_dir": "logs"}}
        updates = {"training": {"window_size": 256, "curriculum_steps": [5]}, "io": {"model_dir": "modelx"}}

        merged = config.deep_update(base, updates)

        self.assertEqual(merged["training"]["window_size"], 256)
        self.assertEqual(merged["training"]["curriculum_steps"], [5])
        self.assertEqual(merged["io"]["logs_dir"], "logs")
        self.assertEqual(merged["io"]["model_dir"], "modelx")

    @unittest.skipIf(config.yaml is None, "PyYAML not installed")
    def test_load_full_config_merges_yaml_with_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=True) as f:
            f.write(textwrap.dedent(
                """
                training:
                  tokenizer: codon
                  window_size: 300
                  curriculum_steps: [1, 2]
                io:
                  logs_dir: custom_logs
                """
            ))
            f.flush()

            cfg = config.load_full_config(f.name)

        self.assertEqual(cfg["training"]["tokenizer"], "codon")
        self.assertEqual(cfg["training"]["window_size"], 300)
        self.assertEqual(cfg["training"]["curriculum_steps"], [1, 2])
        self.assertEqual(cfg["io"]["logs_dir"], "custom_logs")
        self.assertIn("batch_size", cfg["training"])  # default retained

    def test_extract_configs_coerces_optional_ints(self):
        cfg = {
            "training": {
                "max_protein_aa": "",
                "protein_len_min": "40",
                "protein_len_max": None,
            }
        }
        _, train_cfg, _ = config.extract_configs(cfg)

        self.assertIsNone(train_cfg.max_protein_aa)
        self.assertEqual(train_cfg.protein_len_min, 40)
        self.assertIsNone(train_cfg.protein_len_max)


if __name__ == "__main__":
    unittest.main()

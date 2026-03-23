import tempfile
import unittest


class HierarchicalModelTests(unittest.TestCase):
    def test_config_extract_supports_hierarchical(self):
        from perceptrome.config import extract_configs

        _, train_cfg, _ = extract_configs(
            {
                "training": {
                    "model_type": "hierarchical",
                    "hierarchical_ablation_mode": "cnn_only",
                    "stage_a_steps": 5,
                    "stage_d_steps": 3,
                }
            }
        )
        self.assertEqual(train_cfg.model_type, "hierarchical")
        self.assertEqual(train_cfg.hierarchical_ablation_mode, "cnn_only")
        self.assertEqual(train_cfg.stage_a_steps, 5)
        self.assertEqual(train_cfg.stage_d_steps, 3)

    def test_hierarchical_forward_sequence_only_and_fused_ablation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        from perceptrome.models.hierarchical_vae import HierarchicalVAE

        b, l, v = 2, 12, 4
        x = torch.zeros((b, l, v), dtype=torch.float32)
        x[:, :, 0] = 1.0
        flat = x.view(b, -1)
        ast_batch = {
            "node_type_ids": torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long),
            "coords": torch.tensor([[[1, 2], [3, 4], [5, 6]], [[1, 2], [2, 3], [0, 0]]], dtype=torch.float32),
            "strand": torch.tensor([[0, 1, -1], [0, 1, 0]], dtype=torch.long),
            "frame": torch.tensor([[0, 1, 2], [0, 1, -1]], dtype=torch.long),
            "node_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            "edge_index": [torch.tensor([[0, 1], [1, 2]], dtype=torch.long), torch.tensor([[0], [1]], dtype=torch.long)],
        }

        seq_only = HierarchicalVAE(seq_len=l, vocab_size=v, hidden_dim=16, latent_dim=8, ast_tree_layers=2)
        seq_aux = seq_only.forward_with_aux(flat)
        self.assertFalse(seq_aux.ast_used)
        self.assertEqual(tuple(seq_aux.fused_token.shape), (b, l, 16))

        for mode in ("pooled_fusion", "cnn_only", "ast_only"):
            model = HierarchicalVAE(seq_len=l, vocab_size=v, hidden_dim=16, latent_dim=8, ast_tree_layers=2, ablation_mode=mode)
            out = model.forward_with_aux(flat, ast_batch=ast_batch)
            self.assertTrue(out.ast_used)
            self.assertEqual(tuple(out.recon_logits.shape), (b, l * v))
            self.assertEqual(tuple(out.mu.shape), (b, 8))
            self.assertEqual(tuple(out.logvar.shape), (b, 8))
            self.assertEqual(tuple(out.fused_global.shape), (b, 16))

    def test_checkpoint_roundtrip_hierarchical(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        from perceptrome.config import IOConfig
        from perceptrome.model import load_or_init_model, save_checkpoint, get_device

        with tempfile.TemporaryDirectory() as td:
            io_cfg = IOConfig(
                cache_fasta_dir=td,
                cache_genbank_dir=td,
                cache_encoded_dir=td,
                model_dir=td,
                checkpoints_dir=td,
                logs_dir=td,
                state_file=f"{td}/state.json",
            )
            model, opt, step, ckpt = load_or_init_model(
                io_cfg=io_cfg,
                seq_len=8,
                vocab_size=4,
                hidden_dim=16,
                learning_rate=1e-3,
                device=get_device(),
                tokenizer="base",
                loss_type="mse",
                model_type="hierarchical",
                transformer_d_model=16,
                transformer_nhead=2,
                transformer_layers=2,
                transformer_dropout=0.1,
                hierarchical_latent_dim=8,
                hierarchical_ablation_mode="pooled_fusion",
            )
            save_checkpoint(
                ckpt_path=ckpt,
                model=model,
                optimizer=opt,
                global_step=step,
                tokenizer="base",
                seq_len=8,
                vocab_size=4,
                hidden_dim=16,
                loss_type="mse",
                model_type="hierarchical",
                transformer_d_model=16,
                transformer_nhead=2,
                transformer_layers=2,
                transformer_dropout=0.1,
                learning_rate=1e-3,
                beta_kl=1e-3,
                hierarchical_latent_dim=8,
                hierarchical_ablation_mode="pooled_fusion",
            )
            model2, _, _, _ = load_or_init_model(
                io_cfg=io_cfg,
                seq_len=8,
                vocab_size=4,
                hidden_dim=16,
                learning_rate=1e-3,
                device=get_device(),
                tokenizer="base",
                loss_type="mse",
                model_type="hierarchical",
                transformer_d_model=16,
                transformer_nhead=2,
                transformer_layers=2,
                transformer_dropout=0.1,
                hierarchical_latent_dim=8,
                hierarchical_ablation_mode="pooled_fusion",
            )
            self.assertIsNotNone(model2)


if __name__ == "__main__":
    unittest.main()

"""Tests for advanced generation strategies: sampling and latent space manipulation.

Tests cover:
  - Token sampling: temperature, top-k, top-p, beam search, anneal
  - Latent strategies: random, SLERP, arithmetic, gradient, walk, inpainting
  - Integration: sample_window dispatcher, generate_latent_vectors dispatcher
"""

import unittest
from typing import List

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Sampling strategy tests
# ---------------------------------------------------------------------------


class TemperatureSamplingTests(unittest.TestCase):
    def test_basic_sampling(self):
        from perceptrome.sampling import sample_temperature
        logits = np.array([10.0, 0.0, 0.0, 0.0])  # strongly favor idx 0
        counts = [0] * 4
        for _ in range(200):
            counts[sample_temperature(logits, 1.0)] += 1
        self.assertGreater(counts[0], 100)  # idx 0 should dominate

    def test_high_temperature_more_uniform(self):
        from perceptrome.sampling import sample_temperature
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        counts = [0] * 4
        for _ in range(400):
            counts[sample_temperature(logits, 10.0)] += 1
        # High temperature should spread probability more evenly
        self.assertGreater(counts[1] + counts[2] + counts[3], 50)

    def test_low_temperature_greedy(self):
        from perceptrome.sampling import sample_temperature
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        counts = [0] * 4
        for _ in range(100):
            counts[sample_temperature(logits, 0.01)] += 1
        self.assertEqual(counts[0], 100)  # should be purely greedy


class TopKSamplingTests(unittest.TestCase):
    def test_top_k_restricts_choices(self):
        from perceptrome.sampling import sample_top_k
        logits = np.array([10.0, 8.0, 0.0, -10.0])
        counts = [0] * 4
        for _ in range(200):
            counts[sample_top_k(logits, k=2, temperature=1.0)] += 1
        # Only top-2 (indices 0,1) should be sampled
        self.assertEqual(counts[2], 0)
        self.assertEqual(counts[3], 0)
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)

    def test_top_k_1_is_greedy(self):
        from perceptrome.sampling import sample_top_k
        logits = np.array([5.0, 10.0, 3.0, 1.0])
        for _ in range(50):
            self.assertEqual(sample_top_k(logits, k=1, temperature=1.0), 1)


class TopPSamplingTests(unittest.TestCase):
    def test_nucleus_basic(self):
        from perceptrome.sampling import sample_top_p
        # One token has 99% probability after softmax
        logits = np.array([100.0, 0.0, 0.0, 0.0])
        counts = [0] * 4
        for _ in range(100):
            counts[sample_top_p(logits, p=0.9, temperature=1.0)] += 1
        self.assertEqual(counts[0], 100)

    def test_nucleus_includes_more_with_high_p(self):
        from perceptrome.sampling import sample_top_p
        logits = np.array([2.0, 1.9, 1.8, 0.0])
        counts = [0] * 4
        for _ in range(400):
            counts[sample_top_p(logits, p=0.99, temperature=1.0)] += 1
        # With p=0.99, all top tokens should be reachable
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)
        self.assertGreater(counts[2], 0)


class AnnealSamplingTests(unittest.TestCase):
    def test_anneal_varies_with_position(self):
        from perceptrome.sampling import sample_anneal
        logits = np.array([5.0, 3.0, 1.0, 0.0])
        # Early position (high temp) should be more diverse
        # Late position (low temp) should be more focused
        early_counts = [0] * 4
        late_counts = [0] * 4
        for _ in range(400):
            early_counts[sample_anneal(logits, 0, 100, 2.0, 0.1)] += 1
            late_counts[sample_anneal(logits, 99, 100, 2.0, 0.1)] += 1
        # Late should be more concentrated on index 0
        self.assertGreater(late_counts[0], early_counts[0])


class BeamSearchTests(unittest.TestCase):
    def test_beam_search_greedy(self):
        from perceptrome.sampling import beam_search_window
        # 3 positions, 4 vocab
        logits = np.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ])
        tokens = beam_search_window(logits, beam_width=1, temperature=0.01)
        self.assertEqual(tokens, [0, 1, 2])

    def test_beam_search_wider_finds_better(self):
        from perceptrome.sampling import beam_search_window
        # logits where greedy fails but beam search succeeds
        logits = np.array([
            [1.0, 5.0, 0.0, 0.0],  # greedy picks 1
            [0.0, -10.0, 5.0, 0.0],  # after 1, best is 2
            [5.0, 0.0, 0.0, 0.0],
        ])
        # beam width > 1 should explore alternatives
        tokens = beam_search_window(logits, beam_width=4, temperature=1.0)
        self.assertEqual(len(tokens), 3)

    def test_beam_returns_correct_length(self):
        from perceptrome.sampling import beam_search_window
        seq_len, vocab = 16, 4
        logits = np.random.randn(seq_len, vocab)
        tokens = beam_search_window(logits, beam_width=3)
        self.assertEqual(len(tokens), seq_len)
        self.assertTrue(all(0 <= t < vocab for t in tokens))


class SampleWindowDispatcherTests(unittest.TestCase):
    def test_temperature_dispatch(self):
        from perceptrome.sampling import SamplingConfig, sample_window
        cfg = SamplingConfig(strategy="temperature", temperature=1.0)
        logits = np.random.randn(32, 4)
        tokens = sample_window(logits, cfg)
        self.assertEqual(len(tokens), 32)

    def test_top_k_dispatch(self):
        from perceptrome.sampling import SamplingConfig, sample_window
        cfg = SamplingConfig(strategy="top_k", top_k=2, temperature=1.0)
        logits = np.random.randn(16, 4)
        tokens = sample_window(logits, cfg)
        self.assertEqual(len(tokens), 16)

    def test_beam_dispatch(self):
        from perceptrome.sampling import SamplingConfig, sample_window
        cfg = SamplingConfig(strategy="beam", beam_width=3, temperature=1.0)
        logits = np.random.randn(8, 4)
        tokens = sample_window(logits, cfg)
        self.assertEqual(len(tokens), 8)

    def test_bias_weights_applied(self):
        from perceptrome.sampling import SamplingConfig, sample_window
        cfg = SamplingConfig(strategy="temperature", temperature=0.01)
        # All logits equal, but bias heavily favors idx 2
        logits = np.zeros((50, 4))
        bias = np.array([0.001, 0.001, 100.0, 0.001])
        tokens = sample_window(logits, cfg, bias_weights=bias)
        # Most tokens should be idx 2
        self.assertGreater(tokens.count(2), 40)


# ---------------------------------------------------------------------------
# Latent strategy tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "PyTorch not installed")
class RandomLatentTests(unittest.TestCase):
    def test_shape(self):
        from perceptrome.latent_strategies import random_latent
        z = random_latent(64, torch.device("cpu"), scale=1.0, batch_size=2)
        self.assertEqual(z.shape, (2, 64))

    def test_scale(self):
        from perceptrome.latent_strategies import random_latent
        torch.manual_seed(42)
        z1 = random_latent(256, torch.device("cpu"), scale=0.1)
        torch.manual_seed(42)
        z2 = random_latent(256, torch.device("cpu"), scale=10.0)
        self.assertGreater(z2.abs().mean().item(), z1.abs().mean().item())


@unittest.skipIf(torch is None, "PyTorch not installed")
class SLERPTests(unittest.TestCase):
    def test_endpoints(self):
        from perceptrome.latent_strategies import interpolate_latents
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)
        points = interpolate_latents(z1, z2, steps=5, method="slerp")
        self.assertEqual(len(points), 5)
        self.assertTrue(torch.allclose(points[0], z1, atol=1e-4))
        self.assertTrue(torch.allclose(points[-1], z2, atol=1e-4))

    def test_lerp_endpoints(self):
        from perceptrome.latent_strategies import interpolate_latents
        z1 = torch.randn(1, 32)
        z2 = torch.randn(1, 32)
        points = interpolate_latents(z1, z2, steps=3, method="lerp")
        self.assertEqual(len(points), 3)
        self.assertTrue(torch.allclose(points[0], z1, atol=1e-5))
        self.assertTrue(torch.allclose(points[-1], z2, atol=1e-5))
        # Midpoint should be average
        mid = (z1 + z2) / 2
        self.assertTrue(torch.allclose(points[1], mid, atol=1e-5))

    def test_slerp_preserves_norm(self):
        from perceptrome.latent_strategies import interpolate_latents
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)
        # Normalize both to same norm
        z1 = z1 / z1.norm() * 5.0
        z2 = z2 / z2.norm() * 5.0
        points = interpolate_latents(z1, z2, steps=11, method="slerp")
        for p in points:
            # SLERP should stay near the norm
            self.assertAlmostEqual(p.norm().item(), 5.0, places=1)


@unittest.skipIf(torch is None, "PyTorch not installed")
class LatentArithmeticTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        from perceptrome.latent_strategies import latent_arithmetic
        z_a = torch.tensor([[3.0, 1.0, 0.0]])
        z_b = torch.tensor([[1.0, 1.0, 0.0]])
        z_c = torch.tensor([[0.0, 0.0, 5.0]])
        result = latent_arithmetic(z_a, z_b, z_c, scale=1.0)
        expected = torch.tensor([[2.0, 0.0, 5.0]])  # (3-1+0, 1-1+0, 0-0+5)
        self.assertTrue(torch.allclose(result, expected))

    def test_scale_parameter(self):
        from perceptrome.latent_strategies import latent_arithmetic
        z_a = torch.tensor([[4.0]])
        z_b = torch.tensor([[2.0]])
        z_c = torch.tensor([[0.0]])
        result = latent_arithmetic(z_a, z_b, z_c, scale=0.5)
        # delta = 4 - 2 = 2, scaled by 0.5 = 1, plus z_c = 0 -> 1.0
        self.assertAlmostEqual(result.item(), 1.0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class GradientOptimizationTests(unittest.TestCase):
    def test_optimizes_toward_target(self):
        from perceptrome.model import PlasmidVAE
        from perceptrome.latent_strategies import gradient_optimize_latent

        seq_len, vocab_size, hidden_dim = 16, 4, 32
        model = PlasmidVAE(input_dim=seq_len * vocab_size, hidden_dim=hidden_dim)
        model.eval()

        z_init = torch.randn(1, hidden_dim)
        z_opt = gradient_optimize_latent(
            model, z_init, seq_len, vocab_size,
            target_property="gc", target_value=0.5,
            steps=10, lr=0.05,
        )
        self.assertEqual(z_opt.shape, (1, hidden_dim))
        # z should have changed
        self.assertFalse(torch.allclose(z_init, z_opt))

    def test_reconstruction_property(self):
        from perceptrome.model import PlasmidVAE
        from perceptrome.latent_strategies import gradient_optimize_latent

        seq_len, vocab_size, hidden_dim = 16, 4, 32
        model = PlasmidVAE(input_dim=seq_len * vocab_size, hidden_dim=hidden_dim)
        z_init = torch.randn(1, hidden_dim)
        z_opt = gradient_optimize_latent(
            model, z_init, seq_len, vocab_size,
            target_property="reconstruction", target_value=0.0,
            steps=5, lr=0.01,
        )
        self.assertEqual(z_opt.shape, (1, hidden_dim))


@unittest.skipIf(torch is None, "PyTorch not installed")
class LatentWalkTests(unittest.TestCase):
    def test_walk_shape(self):
        from perceptrome.latent_strategies import latent_walk
        z = torch.randn(1, 32)
        points = latent_walk(z, dim=0, steps=5, walk_range=2.0)
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertEqual(p.shape, (1, 32))

    def test_walk_only_varies_target_dim(self):
        from perceptrome.latent_strategies import latent_walk
        z = torch.zeros(1, 8)
        points = latent_walk(z, dim=3, steps=5, walk_range=2.0)
        for p in points:
            # All dims except 3 should be zero (same as center)
            for d in range(8):
                if d != 3:
                    self.assertAlmostEqual(p[0, d].item(), 0.0)

    def test_walk_range(self):
        from perceptrome.latent_strategies import latent_walk
        z = torch.zeros(1, 4)
        points = latent_walk(z, dim=0, steps=3, walk_range=3.0)
        # First point should be at -3.0, last at +3.0
        self.assertAlmostEqual(points[0][0, 0].item(), -3.0)
        self.assertAlmostEqual(points[-1][0, 0].item(), 3.0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class LatentWalkPCATests(unittest.TestCase):
    def test_pca_walk(self):
        from perceptrome.latent_strategies import latent_walk_pca
        # Create some latent vectors with known structure
        encoded = torch.randn(20, 16)
        points = latent_walk_pca(None, encoded, component=0, steps=5, walk_range=2.0)
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertEqual(p.shape, (1, 16))


@unittest.skipIf(torch is None, "PyTorch not installed")
class InpaintTests(unittest.TestCase):
    def test_inpaint_preserves_unmasked(self):
        from perceptrome.model import PlasmidVAE
        from perceptrome.latent_strategies import inpaint_latent, create_inpaint_mask

        seq_len, vocab_size, hidden_dim = 16, 4, 32
        model = PlasmidVAE(input_dim=seq_len * vocab_size, hidden_dim=hidden_dim)

        # Create a one-hot sequence
        x = torch.zeros(1, seq_len, vocab_size)
        x[0, :, 0] = 1.0  # all A's

        # Mask: regenerate positions 4-8, keep rest
        mask = create_inpaint_mask(seq_len, motif_positions=[(0, 4), (8, 16)])
        # mask should be True for positions 4-7 (regenerate), False elsewhere
        self.assertFalse(mask[0].item())
        self.assertTrue(mask[4].item())
        self.assertFalse(mask[8].item())

        result = inpaint_latent(model, x, mask, torch.device("cpu"), num_iterations=2)
        self.assertEqual(result.shape, (1, seq_len * vocab_size))

    def test_random_mask(self):
        from perceptrome.latent_strategies import create_inpaint_mask
        mask = create_inpaint_mask(100, mask_ratio=0.5)
        self.assertEqual(mask.shape[0], 100)
        # Roughly half should be True
        ratio = mask.float().mean().item()
        self.assertGreater(ratio, 0.2)
        self.assertLess(ratio, 0.8)


@unittest.skipIf(torch is None, "PyTorch not installed")
class EncodeSequenceTests(unittest.TestCase):
    def test_encode_shape(self):
        from perceptrome.model import PlasmidVAE
        from perceptrome.latent_strategies import encode_sequence

        model = PlasmidVAE(input_dim=64, hidden_dim=32)
        x = torch.randn(1, 64)
        mu = encode_sequence(model, x, torch.device("cpu"))
        self.assertEqual(mu.shape, (1, 32))


@unittest.skipIf(torch is None, "PyTorch not installed")
class GenerateLatentVectorsDispatcherTests(unittest.TestCase):
    def _make_model(self):
        from perceptrome.model import PlasmidVAE
        return PlasmidVAE(input_dim=64, hidden_dim=32)

    def test_random_strategy(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="random", latent_scale=1.0)
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"))
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].shape, (1, 32))

    def test_walk_strategy(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="walk", walk_steps=5, walk_dim=0, walk_range=2.0)
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"))
        self.assertEqual(len(vectors), 5)

    def test_gradient_strategy(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="gradient", optimize_property="gc",
                           optimize_target=0.5, optimize_steps=3, optimize_lr=0.01)
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"))
        self.assertEqual(len(vectors), 1)

    def test_interpolate_fallback_without_refs(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="interpolate", interpolate_steps=5)
        # Without references, should fallback to random
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"))
        self.assertEqual(len(vectors), 1)

    def test_interpolate_with_refs(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="interpolate", interpolate_steps=5, interpolate_method="slerp")
        refs = [torch.randn(1, 64), torch.randn(1, 64)]
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"),
                                          reference_sequences=refs)
        self.assertEqual(len(vectors), 5)

    def test_arithmetic_with_refs(self):
        from perceptrome.latent_strategies import LatentConfig, generate_latent_vectors
        model = self._make_model()
        cfg = LatentConfig(strategy="arithmetic", latent_scale=1.0)
        refs = [torch.randn(1, 64), torch.randn(1, 64), torch.randn(1, 64)]
        vectors = generate_latent_vectors(cfg, model, 32, torch.device("cpu"),
                                          reference_sequences=refs)
        self.assertEqual(len(vectors), 1)


@unittest.skipIf(torch is None, "PyTorch not installed")
class SamplingConfigTests(unittest.TestCase):
    """Test SamplingConfig and LatentConfig dataclass defaults."""

    def test_default_sampling(self):
        from perceptrome.sampling import default_sampling_config
        cfg = default_sampling_config()
        self.assertEqual(cfg.strategy, "temperature")
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.top_p, 1.0)

    def test_default_latent(self):
        from perceptrome.latent_strategies import default_latent_config
        cfg = default_latent_config()
        self.assertEqual(cfg.strategy, "random")
        self.assertEqual(cfg.latent_scale, 1.0)


if __name__ == "__main__":
    unittest.main()

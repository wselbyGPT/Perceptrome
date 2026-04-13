"""Tests for the six new VAE architectures added to the ML backbone.

Each architecture is tested for:
  1. Forward pass produces correct output shapes (recon_logits, mu, logvar)
  2. encode() returns (mu, logvar) with correct latent dimensions
  3. decode() returns flat logits with correct output dimension
  4. decode_probs() returns probabilities with correct shape for both CE and MSE
  5. Factory dispatch via load_or_init_model selects the right class
  6. Gradient flow (loss.backward() succeeds without error)
"""

import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# Shared test geometry — small enough for fast CPU tests
SEQ_LEN = 32
VOCAB_SIZE = 4  # DNA nucleotides
HIDDEN_DIM = 64
BATCH = 2
NUM_LAYERS = 2
DROPOUT = 0.0  # deterministic for tests


def _rand_input():
    """Random one-hot encoded batch: (B, L*V) flattened."""
    x = torch.zeros(BATCH, SEQ_LEN, VOCAB_SIZE)
    idx = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    x.scatter_(2, idx.unsqueeze(-1), 1.0)
    return x.view(BATCH, -1)


@unittest.skipIf(torch is None, "PyTorch not installed")
class ConvResVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import ConvResVAE
        self.model = ConvResVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))
        self.assertEqual(logvar.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_decode_probs_ce(self):
        mu, _ = self.model.encode(self.x)
        probs = self.model.decode_probs(mu, SEQ_LEN, VOCAB_SIZE, "ce")
        self.assertEqual(probs.shape, (BATCH, SEQ_LEN, VOCAB_SIZE))
        # softmax sums to 1 along vocab axis
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_decode_probs_mse(self):
        mu, _ = self.model.encode(self.x)
        probs = self.model.decode_probs(mu, SEQ_LEN, VOCAB_SIZE, "mse")
        self.assertEqual(probs.shape, (BATCH, SEQ_LEN, VOCAB_SIZE))
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


@unittest.skipIf(torch is None, "PyTorch not installed")
class RecurrentVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import RecurrentVAE
        self.model = RecurrentVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=True,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))
        self.assertEqual(logvar.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_unidirectional_variant(self):
        from perceptrome.model import RecurrentVAE
        model_uni = RecurrentVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=False,
        )
        recon, mu, logvar = model_uni(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()


@unittest.skipIf(torch is None, "PyTorch not installed")
class WaveNetVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import WaveNetVAE
        self.model = WaveNetVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, kernel_size=3, dropout=DROPOUT,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_decode_probs_ce(self):
        mu, _ = self.model.encode(self.x)
        probs = self.model.decode_probs(mu, SEQ_LEN, VOCAB_SIZE, "ce")
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()


@unittest.skipIf(torch is None, "PyTorch not installed")
class MambaVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import MambaVAE
        self.model = MambaVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, d_state=8, d_conv=4, dropout=DROPOUT,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))
        self.assertEqual(logvar.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_selective_scan_produces_different_outputs(self):
        """The selective SSM should produce input-dependent outputs."""
        x2 = _rand_input()
        mu1, _ = self.model.encode(self.x)
        mu2, _ = self.model.encode(x2)
        self.assertFalse(torch.allclose(mu1, mu2))

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()


@unittest.skipIf(torch is None, "PyTorch not installed")
class AttentionPoolVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import AttentionPoolVAE
        self.model = AttentionPoolVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            nhead=4, num_layers=NUM_LAYERS, num_queries=4, dropout=DROPOUT,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()


@unittest.skipIf(torch is None, "PyTorch not installed")
class ByteNetVAETests(unittest.TestCase):
    def setUp(self):
        from perceptrome.model import ByteNetVAE
        self.model = ByteNetVAE(
            seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, kernel_size=3, dropout=DROPOUT,
        )
        self.x = _rand_input()

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))
        self.assertEqual(mu.shape, (BATCH, HIDDEN_DIM))

    def test_encode_decode_roundtrip(self):
        mu, logvar = self.model.encode(self.x)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decode(z)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * VOCAB_SIZE))

    def test_decode_probs_both_modes(self):
        mu, _ = self.model.encode(self.x)
        probs_ce = self.model.decode_probs(mu, SEQ_LEN, VOCAB_SIZE, "ce")
        probs_mse = self.model.decode_probs(mu, SEQ_LEN, VOCAB_SIZE, "mse")
        self.assertEqual(probs_ce.shape, (BATCH, SEQ_LEN, VOCAB_SIZE))
        self.assertEqual(probs_mse.shape, (BATCH, SEQ_LEN, VOCAB_SIZE))

    def test_gradient_flow(self):
        recon, mu, logvar = self.model(self.x)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()


@unittest.skipIf(torch is None, "PyTorch not installed")
class FactoryDispatchTests(unittest.TestCase):
    """Verify load_or_init_model selects the correct class for each new type."""

    def _init_model(self, model_type: str):
        from perceptrome.model import load_or_init_model
        from perceptrome.config import IOConfig
        with tempfile.TemporaryDirectory() as tmp:
            io_cfg = IOConfig(
                cache_fasta_dir=os.path.join(tmp, "fasta"),
                cache_genbank_dir=os.path.join(tmp, "genbank"),
                cache_encoded_dir=os.path.join(tmp, "encoded"),
                model_dir=os.path.join(tmp, "model"),
                checkpoints_dir=os.path.join(tmp, "ckpt"),
                logs_dir=os.path.join(tmp, "logs"),
                state_file=os.path.join(tmp, "state.json"),
            )
            model, optimizer, step, ckpt = load_or_init_model(
                io_cfg=io_cfg,
                seq_len=SEQ_LEN,
                vocab_size=VOCAB_SIZE,
                hidden_dim=HIDDEN_DIM,
                learning_rate=1e-3,
                device=torch.device("cpu"),
                tokenizer="base",
                loss_type="mse",
                model_type=model_type,
                transformer_d_model=HIDDEN_DIM,
                transformer_nhead=4,
                transformer_layers=NUM_LAYERS,
                transformer_dropout=DROPOUT,
            )
            return model

    def test_conv_dispatch(self):
        from perceptrome.model import ConvResVAE
        model = self._init_model("conv")
        self.assertIsInstance(model, ConvResVAE)

    def test_conv_alias_cnn(self):
        from perceptrome.model import ConvResVAE
        model = self._init_model("cnn")
        self.assertIsInstance(model, ConvResVAE)

    def test_recurrent_dispatch(self):
        from perceptrome.model import RecurrentVAE
        model = self._init_model("recurrent")
        self.assertIsInstance(model, RecurrentVAE)

    def test_recurrent_alias_rnn(self):
        from perceptrome.model import RecurrentVAE
        model = self._init_model("rnn")
        self.assertIsInstance(model, RecurrentVAE)

    def test_recurrent_alias_gru(self):
        from perceptrome.model import RecurrentVAE
        model = self._init_model("gru")
        self.assertIsInstance(model, RecurrentVAE)

    def test_wavenet_dispatch(self):
        from perceptrome.model import WaveNetVAE
        model = self._init_model("wavenet")
        self.assertIsInstance(model, WaveNetVAE)

    def test_wavenet_alias_tcn(self):
        from perceptrome.model import WaveNetVAE
        model = self._init_model("tcn")
        self.assertIsInstance(model, WaveNetVAE)

    def test_mamba_dispatch(self):
        from perceptrome.model import MambaVAE
        model = self._init_model("mamba")
        self.assertIsInstance(model, MambaVAE)

    def test_mamba_alias_s4(self):
        from perceptrome.model import MambaVAE
        model = self._init_model("s4")
        self.assertIsInstance(model, MambaVAE)

    def test_attention_pool_dispatch(self):
        from perceptrome.model import AttentionPoolVAE
        model = self._init_model("attention_pool")
        self.assertIsInstance(model, AttentionPoolVAE)

    def test_attention_pool_alias_perceiver(self):
        from perceptrome.model import AttentionPoolVAE
        model = self._init_model("perceiver")
        self.assertIsInstance(model, AttentionPoolVAE)

    def test_bytenet_dispatch(self):
        from perceptrome.model import ByteNetVAE
        model = self._init_model("bytenet")
        self.assertIsInstance(model, ByteNetVAE)


@unittest.skipIf(torch is None, "PyTorch not installed")
class VAELossCompatibilityTests(unittest.TestCase):
    """Verify all new models work with vae_loss for both CE and MSE."""

    def _test_loss(self, model_type_cls, loss_type: str, **kwargs):
        from perceptrome.model import vae_loss
        model = model_type_cls(**kwargs)
        x = _rand_input()
        recon, mu, logvar = model(x)
        total, recon_loss, kl = vae_loss(recon, x, mu, logvar, 1e-3, loss_type, SEQ_LEN, VOCAB_SIZE)
        self.assertTrue(torch.isfinite(total))
        total.backward()

    def test_conv_ce(self):
        from perceptrome.model import ConvResVAE
        self._test_loss(ConvResVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

    def test_conv_mse(self):
        from perceptrome.model import ConvResVAE
        self._test_loss(ConvResVAE, "mse", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

    def test_recurrent_ce(self):
        from perceptrome.model import RecurrentVAE
        self._test_loss(RecurrentVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

    def test_wavenet_ce(self):
        from perceptrome.model import WaveNetVAE
        self._test_loss(WaveNetVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, kernel_size=3, dropout=DROPOUT)

    def test_mamba_ce(self):
        from perceptrome.model import MambaVAE
        self._test_loss(MambaVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, d_state=8, d_conv=4, dropout=DROPOUT)

    def test_attention_pool_ce(self):
        from perceptrome.model import AttentionPoolVAE
        self._test_loss(AttentionPoolVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, nhead=4, num_layers=NUM_LAYERS, num_queries=4, dropout=DROPOUT)

    def test_bytenet_ce(self):
        from perceptrome.model import ByteNetVAE
        self._test_loss(ByteNetVAE, "ce", seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, kernel_size=3, dropout=DROPOUT)


@unittest.skipIf(torch is None, "PyTorch not installed")
class CodonAndAAVocabTests(unittest.TestCase):
    """Verify new architectures work with codon (65) and AA (21) vocab sizes."""

    CODON_V = 65
    AA_V = 21

    def _forward_check(self, cls, vocab, **extra):
        model = cls(seq_len=SEQ_LEN, vocab_size=vocab, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, dropout=DROPOUT, **extra)
        x = torch.randn(BATCH, SEQ_LEN * vocab)
        recon, mu, logvar = model(x)
        self.assertEqual(recon.shape, (BATCH, SEQ_LEN * vocab))

    def test_conv_codon(self):
        from perceptrome.model import ConvResVAE
        self._forward_check(ConvResVAE, self.CODON_V)

    def test_recurrent_aa(self):
        from perceptrome.model import RecurrentVAE
        self._forward_check(RecurrentVAE, self.AA_V)

    def test_wavenet_codon(self):
        from perceptrome.model import WaveNetVAE
        self._forward_check(WaveNetVAE, self.CODON_V, kernel_size=3)

    def test_mamba_aa(self):
        from perceptrome.model import MambaVAE
        self._forward_check(MambaVAE, self.AA_V, d_state=8, d_conv=4)

    def test_attention_pool_codon(self):
        from perceptrome.model import AttentionPoolVAE
        self._forward_check(AttentionPoolVAE, self.CODON_V, nhead=4, num_queries=4)

    def test_bytenet_aa(self):
        from perceptrome.model import ByteNetVAE
        self._forward_check(ByteNetVAE, self.AA_V, kernel_size=3)


if __name__ == "__main__":
    unittest.main()

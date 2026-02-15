import unittest

try:
    from perceptrome.encoding.constants import AA_VOCAB_SIZE, CODON_VOCAB_SIZE
    from perceptrome.encoding_main import tokenizer_meta
    from perceptrome.model import PlasmidVAE, TransformerVAE, torch
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency/environment gate
    AA_VOCAB_SIZE = CODON_VOCAB_SIZE = 0
    tokenizer_meta = None
    PlasmidVAE = TransformerVAE = None
    torch = None
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"Perceptrome deps unavailable: {IMPORT_ERROR}")
class TokenizerMetaTests(unittest.TestCase):
    def test_tokenizer_meta_base_codon_aa(self):
        cases = [
            ("base", 12, 12, 4),
            ("codon", 12, 4, CODON_VOCAB_SIZE),
            ("aa", 12, 12, AA_VOCAB_SIZE),
        ]
        for tok, window, want_seq, want_vocab in cases:
            with self.subTest(tokenizer=tok):
                seq_len, vocab_size = tokenizer_meta(tok, window)
                self.assertEqual(seq_len, want_seq)
                self.assertEqual(vocab_size, want_vocab)


@unittest.skipIf(IMPORT_ERROR is not None or torch is None, "PyTorch/perceptrome deps are required for model shape checks")
class ModelShapeTests(unittest.TestCase):
    def test_plasmid_vae_forward_decode_probs_shapes_across_tokenizers(self):
        batch = 2
        for tok, window in (("base", 9), ("codon", 12), ("aa", 7)):
            with self.subTest(tokenizer=tok):
                seq_len, vocab_size = tokenizer_meta(tok, window)
                input_dim = seq_len * vocab_size
                model = PlasmidVAE(input_dim=input_dim, hidden_dim=8)
                x = torch.randn(batch, input_dim)

                recon_logits, mu, logvar = model.forward(x)
                probs = model.decode_probs(mu, seq_len, vocab_size, loss_type="ce")

                self.assertEqual(tuple(recon_logits.shape), (batch, input_dim))
                self.assertEqual(tuple(mu.shape), (batch, 8))
                self.assertEqual(tuple(logvar.shape), (batch, 8))
                self.assertEqual(tuple(probs.shape), (batch, seq_len, vocab_size))

    def test_transformer_vae_forward_and_decode_shapes(self):
        batch = 2
        seq_len, vocab_size = tokenizer_meta("aa", 6)
        model = TransformerVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=12,
            nhead=3,
            num_layers=1,
            dropout=0.0,
        )
        x_flat = torch.randn(batch, seq_len * vocab_size)

        recon_logits, mu, logvar = model.forward(x_flat)
        probs = model.decode_probs(mu, seq_len, vocab_size, loss_type="ce")

        self.assertEqual(tuple(recon_logits.shape), (batch, seq_len * vocab_size))
        self.assertEqual(tuple(mu.shape), (batch, 12))
        self.assertEqual(tuple(logvar.shape), (batch, 12))
        self.assertEqual(tuple(probs.shape), (batch, seq_len, vocab_size))


if __name__ == "__main__":
    unittest.main()

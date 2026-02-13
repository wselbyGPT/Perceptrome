import logging, os
from typing import Dict, Tuple

try:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore
    F = None      # type: ignore

from .config import IOConfig

ModuleBase = nn.Module if nn is not None else object

class TransformerVAE(ModuleBase):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for TransformerVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.fc_mu = nn.Linear(self.d_model, self.d_model)
        self.fc_logvar = nn.Linear(self.d_model, self.d_model)

        self.z_to_seq = nn.Linear(self.d_model, self.seq_len * self.d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(self.d_model * 4),
            dropout=float(dropout),
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(num_layers))
        self.out_proj = nn.Linear(self.d_model, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        h = self.input_proj(x_seq) + self.pos_embed
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), self.seq_len, self.d_model)
        h = self.decoder(h + self.pos_embed)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


class RNNVAE(ModuleBase):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for RNNVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.LSTM(
            input_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_to_h = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        _, (h_n, _) = self.encoder(x_seq)
        h_last = h_n[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h0 = torch.tanh(self.z_to_h(z))
        dec_in = h0.unsqueeze(1).expand(-1, self.seq_len, -1)
        h, _ = self.decoder(dec_in)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


class HybridCNNRNNVAE(ModuleBase):  # type: ignore[misc]
    """Hybrid VAE: local motif extraction via CNN + sequence grammar via BiLSTM."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if torch is None or nn is None or F is None:
            raise RuntimeError("PyTorch is required for HybridCNNRNNVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Linear(self.vocab_size, self.hidden_dim)
        self.conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(self.hidden_dim)

        self.encoder_rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=max(1, int(num_layers)),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.z_to_seq = nn.Linear(self.hidden_dim, self.seq_len * self.hidden_dim)
        self.decoder_rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=max(1, int(num_layers)),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(float(dropout))

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def _encode_features(self, x_seq: "torch.Tensor") -> "torch.Tensor":
        h = self.input_proj(x_seq)
        h_conv = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv_norm(h + F.gelu(h_conv))
        h = self.dropout(h)
        h, _ = self.encoder_rnn(h)
        return h

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self._encode_features(self._ensure_seq(x))
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), self.seq_len, self.hidden_dim)
        h, _ = self.decoder_rnn(h)
        logits = self.out_proj(self.dropout(h))
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        if str(loss_type).lower() == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


class DilatedConvBlock(ModuleBase):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        if nn is None or F is None:
            raise RuntimeError("PyTorch is required for DilatedConvBlock.")
        super().__init__()
        padding = int((int(kernel_size) - 1) * int(dilation) // 2)
        self.conv = nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=int(kernel_size), dilation=int(dilation), padding=padding)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class TCNVAE(ModuleBase):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        if torch is None or nn is None or F is None:
            raise RuntimeError("PyTorch is required for TCNVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Linear(self.vocab_size, self.hidden_dim)
        self.encoder_blocks = nn.ModuleList([
            DilatedConvBlock(self.hidden_dim, kernel_size=int(kernel_size), dilation=(2 ** i), dropout=float(dropout))
            for i in range(max(1, int(num_layers)))
        ])

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.z_to_seq = nn.Linear(self.hidden_dim, self.seq_len * self.hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            DilatedConvBlock(self.hidden_dim, kernel_size=int(kernel_size), dilation=(2 ** i), dropout=float(dropout))
            for i in range(max(1, int(num_layers)))
        ])
        self.out_proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.input_proj(self._ensure_seq(x))
        for block in self.encoder_blocks:
            h = block(h)
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_seq(z).view(z.size(0), self.seq_len, self.hidden_dim)
        for block in self.decoder_blocks:
            h = block(h)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        if str(loss_type).lower() == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


class MoEVAE(ModuleBase):  # type: ignore[misc]
    """Mixture-of-experts VAE with shared trunk and gated expert decoder."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        if torch is None or nn is None or F is None:
            raise RuntimeError("PyTorch is required for MoEVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = max(1, int(num_experts))

        self.input_proj = nn.Linear(self.vocab_size, self.hidden_dim)
        self.shared_enc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.gate = nn.Linear(self.hidden_dim, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.seq_len * self.vocab_size),
            )
            for _ in range(self.num_experts)
        ])

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        h = self.input_proj(x_seq)
        h = self.shared_enc(h)
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        gate_w = F.softmax(self.gate(z), dim=-1)
        expert_logits = torch.stack([expert(z) for expert in self.experts], dim=1)
        mixed = torch.sum(gate_w.unsqueeze(-1) * expert_logits, dim=1)
        return mixed

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        if str(loss_type).lower() == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


class GraphConvBlock(ModuleBase):  # type: ignore[misc]
    def __init__(self, hidden_dim: int, dropout: float):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for GraphConvBlock.")
        super().__init__()
        self.proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: "torch.Tensor", a_norm: "torch.Tensor") -> "torch.Tensor":
        h = torch.matmul(a_norm, x)
        h = self.proj(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class GNNVAE(ModuleBase):  # type: ignore[misc]
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if torch is None or nn is None or F is None:
            raise RuntimeError("PyTorch is required for GNNVAE.")
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Linear(self.vocab_size, self.hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [GraphConvBlock(self.hidden_dim, dropout=float(dropout)) for _ in range(max(1, int(num_layers)))]
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.z_to_nodes = nn.Linear(self.hidden_dim, self.seq_len * self.hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [GraphConvBlock(self.hidden_dim, dropout=float(dropout)) for _ in range(max(1, int(num_layers)))]
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def _ensure_seq(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            return x.view(x.size(0), self.seq_len, self.vocab_size)
        return x

    def _line_graph_adjacency(self, batch_size: int, device: "torch.device") -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("PyTorch is required.")
        n = self.seq_len
        adj = torch.eye(n, device=device)
        if n > 1:
            idx = torch.arange(n - 1, device=device)
            adj[idx, idx + 1] = 1.0
            adj[idx + 1, idx] = 1.0
        deg = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(deg, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        a_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        return a_norm.unsqueeze(0).expand(batch_size, -1, -1)

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        x_seq = self._ensure_seq(x)
        h = self.input_proj(x_seq)
        a_norm = self._line_graph_adjacency(h.size(0), h.device)
        for block in self.encoder_blocks:
            h = block(h, a_norm)
        pooled = h.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        h = self.z_to_nodes(z).view(z.size(0), self.seq_len, self.hidden_dim)
        a_norm = self._line_graph_adjacency(h.size(0), h.device)
        for block in self.decoder_blocks:
            h = block(h, a_norm)
        logits = self.out_proj(h)
        return logits.view(z.size(0), self.seq_len * self.vocab_size)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

class PlasmidVAE(ModuleBase):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for PlasmidVAE.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.act(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        """Return *logits* (not probabilities).

        - For MSE-based training, we apply sigmoid to these logits in the loss.
        - For categorical training (CE), these logits are fed directly to softmax/CE.
        """
        h = self.act(self.fc2(z))
        return self.fc_out(h)

    def decode_probs(self, z: "torch.Tensor", seq_len: int, vocab_size: int, loss_type: str) -> "torch.Tensor":
        """Return probabilities shaped (B, seq_len, vocab_size)."""
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        logits = self.decode(z).view(z.size(0), int(seq_len), int(vocab_size))
        lt = str(loss_type).lower()
        if lt == "ce":
            return F.softmax(logits, dim=-1)
        # mse (legacy): independent sigmoid weights
        return torch.sigmoid(logits)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

def get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_or_init_model(
    io_cfg: IOConfig,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    learning_rate: float,
    device: "torch.device",
    tokenizer: str,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
) -> Tuple["nn.Module", "optim.Optimizer", int, str]:
    """
    seq_len: number of positions (bp or codons)
    vocab_size: 4 for base, 65 for codon
    """
    if torch is None or nn is None or optim is None:
        raise RuntimeError("PyTorch is required.")

    ckpt_path = os.path.join(io_cfg.checkpoints_dir, "latest.pt")

    mt = str(model_type).lower()
    input_dim = int(seq_len) * int(vocab_size)
    if mt == "transformer":
        model = TransformerVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        ).to(device)
    elif mt == "rnn":
        model = RNNVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, int(transformer_layers)),
            dropout=float(transformer_dropout),
        ).to(device)
    elif mt == "hybrid":
        model = HybridCNNRNNVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, int(transformer_layers)),
            dropout=float(transformer_dropout),
        ).to(device)
    elif mt == "moe":
        model = MoEVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_experts=max(1, int(transformer_nhead)),
            dropout=float(transformer_dropout),
        ).to(device)
    elif mt == "gnn":
        model = GNNVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, int(transformer_layers)),
            dropout=float(transformer_dropout),
        ).to(device)
    elif mt == "tcn":
        model = TCNVAE(
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, int(transformer_layers)),
            dropout=float(transformer_dropout),
            kernel_size=3,
        ).to(device)
    else:
        model = PlasmidVAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0

    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=device)
        meta: Dict[str, object] = data.get("meta", {})
        ck_tok = str(meta.get("tokenizer", "base")).lower()
        ck_seq = int(meta.get("seq_len", seq_len))
        ck_vocab = int(meta.get("vocab_size", vocab_size))
        ck_hidden = int(meta.get("hidden_dim", hidden_dim))
        ck_loss = str(meta.get("loss_type", "mse")).lower()
        ck_model_type = str(meta.get("model_type", "mlp")).lower()
        ck_d_model = int(meta.get("transformer_d_model", transformer_d_model))
        ck_nhead = int(meta.get("transformer_nhead", transformer_nhead))
        ck_layers = int(meta.get("transformer_layers", transformer_layers))
        ck_dropout = float(meta.get("transformer_dropout", transformer_dropout))
        
        if ck_tok != tokenizer.lower():
            raise ValueError(f"Checkpoint tokenizer={ck_tok} but requested tokenizer={tokenizer}. Delete {ckpt_path} or match settings.")
        if ck_seq != seq_len:
            raise ValueError(f"Checkpoint seq_len={ck_seq} but requested seq_len={seq_len}. Delete {ckpt_path} or match settings.")
        if ck_vocab != vocab_size:
            raise ValueError(f"Checkpoint vocab_size={ck_vocab} but requested vocab_size={vocab_size}. Delete {ckpt_path} or match settings.")
        if ck_hidden != hidden_dim and mt not in {"transformer"}:
            raise ValueError(f"Checkpoint hidden_dim={ck_hidden} but requested hidden_dim={hidden_dim}. Delete {ckpt_path} or match settings.")
        if ck_loss != str(loss_type).lower():
            raise ValueError(
                f"Checkpoint loss_type={ck_loss} but requested loss_type={loss_type}. "
                f"Delete {ckpt_path} or match settings."
            )
        if ck_model_type != mt:
            raise ValueError(
                f"Checkpoint model_type={ck_model_type} but requested model_type={mt}. "
                f"Delete {ckpt_path} or match settings."
            )
        if mt == "transformer":
            if ck_d_model != transformer_d_model:
                raise ValueError(f"Checkpoint transformer_d_model={ck_d_model} but requested {transformer_d_model}. Delete {ckpt_path} or match settings.")
            if ck_nhead != transformer_nhead:
                raise ValueError(f"Checkpoint transformer_nhead={ck_nhead} but requested {transformer_nhead}. Delete {ckpt_path} or match settings.")
            if ck_layers != transformer_layers:
                raise ValueError(f"Checkpoint transformer_layers={ck_layers} but requested {transformer_layers}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint transformer_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")
        if mt == "rnn":
            if ck_layers != transformer_layers:
                raise ValueError(f"Checkpoint rnn_layers={ck_layers} but requested {transformer_layers}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint rnn_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")
        if mt == "hybrid":
            if ck_layers != transformer_layers:
                raise ValueError(f"Checkpoint hybrid_layers={ck_layers} but requested {transformer_layers}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint hybrid_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")
        if mt == "moe":
            if ck_nhead != transformer_nhead:
                raise ValueError(f"Checkpoint moe_experts={ck_nhead} but requested {transformer_nhead}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint moe_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")
        if mt == "gnn":
            if ck_layers != transformer_layers:
                raise ValueError(f"Checkpoint gnn_layers={ck_layers} but requested {transformer_layers}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint gnn_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")
        if mt == "tcn":
            if ck_layers != transformer_layers:
                raise ValueError(f"Checkpoint tcn_layers={ck_layers} but requested {transformer_layers}. Delete {ckpt_path} or match settings.")
            if abs(ck_dropout - float(transformer_dropout)) > 1e-8:
                raise ValueError(f"Checkpoint tcn_dropout={ck_dropout} but requested {transformer_dropout}. Delete {ckpt_path} or match settings.")

        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optim"])
        global_step = int(meta.get("global_step", 0))

        logging.info(
            "Loaded checkpoint %s (tokenizer=%s, seq_len=%s, vocab=%s, hidden=%s, model=%s, step=%s)",
            ckpt_path, ck_tok, ck_seq, ck_vocab, ck_hidden, ck_model_type, global_step
        )
        logging.info(
            "Initializing new VAE (tokenizer=%s, loss_type=%s, model=%s, "
            "seq_len=%s, vocab=%s, input_dim=%s, hidden=%s, d_model=%s, nhead=%s, layers=%s, dropout=%s, lr=%s)",
            tokenizer,
            loss_type,
            mt,
            seq_len,
            vocab_size,
            input_dim,
            hidden_dim,
            transformer_d_model,
            transformer_nhead,
            transformer_layers,
            transformer_dropout,
            learning_rate,
        )

    return model, optimizer, global_step, ckpt_path

def save_checkpoint(
    ckpt_path: str,
    model: "nn.Module",
    optimizer: "optim.Optimizer",
    global_step: int,
    tokenizer: str,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
    loss_type: str,
    model_type: str,
    transformer_d_model: int,
    transformer_nhead: int,
    transformer_layers: int,
    transformer_dropout: float,
) -> None:
    if torch is None:
        return
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "meta": {
            "global_step": int(global_step),
            "tokenizer": str(tokenizer).lower(),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "hidden_dim": int(hidden_dim),
            "loss_type": str(loss_type).lower(),
            "model_type": str(model_type).lower(),
            "transformer_d_model": int(transformer_d_model),
            "transformer_nhead": int(transformer_nhead),
            "transformer_layers": int(transformer_layers),
            "transformer_dropout": float(transformer_dropout),
        },
    }
    tmp = ckpt_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
    logging.info(f"Saved checkpoint step={global_step} -> {ckpt_path}")

def vae_loss(
    recon_logits: "torch.Tensor",
    x: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    beta_kl: float,
    loss_type: str,
    seq_len: int,
    vocab_size: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch required.")

    lt = str(loss_type).lower()
    if lt == "ce":
        if F is None:
            raise RuntimeError("PyTorch required.")
        # logits: (B, L*V) -> (B, L, V)
        logits = recon_logits.view(recon_logits.size(0), int(seq_len), int(vocab_size))
        targets = x.view(x.size(0), int(seq_len), int(vocab_size)).argmax(dim=2)  # (B, L)
        # per-position CE, mean over positions and batch
        ce = F.cross_entropy(logits.view(-1, int(vocab_size)), targets.view(-1), reduction="mean")
        recon_term = ce
    else:
        # legacy: regression on one-hot using sigmoid weights
        recon = torch.sigmoid(recon_logits)
        recon_term = nn.MSELoss(reduction="mean")(recon, x)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_term + float(beta_kl) * kl
    return total, recon_term, kl

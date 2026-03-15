from typing import Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class _ResidualConvBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.act(x + self.net(x))


class LocalCNNEncoder(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        seq_vocab_size: int,
        hidden_dim: int,
        channels: int = 128,
        kernels: Tuple[int, ...] = (3, 7, 15),
        dilation_cycle: Tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.input_proj = nn.Conv1d(int(seq_vocab_size), int(channels), kernel_size=1)
        blocks = []
        for i, k in enumerate(kernels):
            d = dilation_cycle[i % len(dilation_cycle)] if dilation_cycle else 1
            blocks.append(_ResidualConvBlock(int(channels), int(k), int(d), float(dropout)))
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Conv1d(int(channels), int(hidden_dim), kernel_size=1)

    def forward(self, seq_tokens: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        if seq_tokens.dim() != 3:
            raise ValueError("seq_tokens must be shaped (B, L, V)")
        x = seq_tokens.transpose(1, 2)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        token_embeddings = self.out_proj(x).transpose(1, 2)
        pooled = token_embeddings.mean(dim=1)
        return token_embeddings, pooled

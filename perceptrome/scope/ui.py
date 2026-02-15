import curses
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from ..encoding_main import compute_gc_from_encoded
from ..model import PlasmidVAE, get_device, load_or_init_model, vae_loss


class TerminalGradient:
    """ASCII + color gradient renderer for curses windows."""

    def __init__(self, stdscr, use_color: bool = True):
        self.palette = " .,:;irsXA253hMHGS#9B&@"
        self.stdscr = stdscr
        self.use_color = bool(use_color)
        self._pairs: list[int] = []
        if self.use_color:
            self._init_colors()

    def _init_colors(self) -> None:
        try:
            curses.start_color()
            if hasattr(curses, "use_default_colors"):
                curses.use_default_colors()
        except curses.error:
            self.use_color = False
            return

        can_change = False
        try:
            can_change = bool(curses.can_change_color())
        except curses.error:
            can_change = False

        if curses.COLORS >= 256 and can_change:
            gradient_ids = [196, 202, 208, 214, 220, 190, 118, 46, 39, 27, 21]
        elif curses.COLORS >= 16:
            gradient_ids = [1, 9, 3, 11, 2, 10, 6, 12, 4, 13]
        else:
            gradient_ids = [7]

        max_pairs = max(1, min(len(gradient_ids), curses.COLOR_PAIRS - 1))
        for i in range(max_pairs):
            pair_id = i + 1
            try:
                curses.init_pair(pair_id, gradient_ids[i], -1)
                self._pairs.append(pair_id)
            except curses.error:
                continue

        if not self._pairs:
            self.use_color = False

    def _attrs_for_value(self, value: float) -> int:
        if not self.use_color or not self._pairs:
            return curses.A_NORMAL
        v = max(0.0, min(1.0, float(value)))
        idx = int(v * (len(self._pairs) - 1))
        return curses.color_pair(self._pairs[idx])

    def glyph_for_value(self, value: float) -> str:
        v = max(0.0, min(1.0, float(value)))
        idx = int(v * (len(self.palette) - 1))
        return self.palette[idx]

    def draw_row(
        self,
        y: int,
        width: int,
        start_idx: int,
        end_idx: int,
        norm_values: np.ndarray,
    ) -> None:
        for col, wi in enumerate(range(start_idx, end_idx)):
            if col >= width - 1:
                break
            value = float(norm_values[wi])
            ch = self.glyph_for_value(value)
            attrs = self._attrs_for_value(value)
            try:
                self.stdscr.addch(y, col, ch, attrs)
            except curses.error:
                pass


def run_scope_ui(
    stdscr,
    accession: str,
    errors: np.ndarray,
    gc_values: np.ndarray,
    window_size: int,
    stride: int,
    fps: float,
    color: bool = True,
) -> None:
    """
    Curses-based genome scope:
      - Top strip: per-window reconstruction error
      - Second strip: per-window side metric (GC fraction in base/codon mode,
        hydrophobic fraction in AA mode)
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    num_windows = errors.shape[0]
    if num_windows == 0:
        stdscr.addstr(0, 0, "No windows to visualize (encoded array empty).")
        stdscr.refresh()
        import time
        time.sleep(2.0)
        return

    if gc_values.shape[0] != num_windows:
        raise ValueError(
            f"gc_values length {gc_values.shape[0]} != errors length {num_windows}"
        )

    gradient = TerminalGradient(stdscr, use_color=color)
    start_idx = 0

    import time
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        width = max(10, w - 2)
        end_idx = min(start_idx + width, num_windows)

        min_e = float(errors.min())
        max_e = float(errors.max())
        span_e = max(max_e - min_e, 1e-8)
        norm_err = (errors - min_e) / span_e

        min_gc = float(gc_values.min())
        max_gc = float(gc_values.max())
        span_gc = max(max_gc - min_gc, 1e-8)
        norm_gc = (gc_values - min_gc) / span_gc

        header = (
            f"GenomeScope — {accession}  windows={num_windows} "
            f"[q] quit  [←/→] scroll"
        )
        stdscr.addstr(0, 0, header[: w - 1])

        if h > 1:
            info_err = (
                f"ERROR  window_size={window_size} stride={stride} "
                f"min={min_e:.3g} max={max_e:.3g} "
                f"view={start_idx}-{end_idx - 1}"
            )
            stdscr.addstr(1, 0, info_err[: w - 1])

        if h > 2:
            info_gc = (
                f"METRIC min={min_gc:.3f} max={max_gc:.3f} "
                f"gradient={'ansi' if gradient.use_color else 'ascii'}"
            )
            stdscr.addstr(2, 0, info_gc[: w - 1])

        if h > 3:
            controls = "[q] quit   [←/→] scroll"
            stdscr.addstr(3, 0, controls[: w - 1])

        line_err_y = 5 if h > 5 else 0
        show_gc = h > line_err_y + 1
        line_gc_y = line_err_y + 1 if show_gc else None

        gradient.draw_row(line_err_y, w, start_idx, end_idx, norm_err)

        if show_gc and line_gc_y is not None:
            gradient.draw_row(line_gc_y, w, start_idx, end_idx, norm_gc)

        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key in (ord("q"), ord("Q")):
            break
        elif key == curses.KEY_LEFT:
            step = max(1, width // 2)
            start_idx = max(0, start_idx - step)
        elif key == curses.KEY_RIGHT:
            step = max(1, width // 2)
            if start_idx + width < num_windows:
                start_idx = min(num_windows - width, start_idx + step)

        time.sleep(max(0.0, 1.0 / max(fps, 1e-3)))


def compute_errors_with_model_and_tensor(
    model: PlasmidVAE,
    windows_tensor: "torch.Tensor",
    device: "torch.device",
    loss_type: str = "mse",
    seq_len: int = 0,
    vocab_size: int = 0,
) -> np.ndarray:
    """Compute per-window errors using posterior mean (mu) as z.

    - loss_type='mse': MSE between sigmoid(logits) and one-hot input
    - loss_type='ce': mean cross-entropy (NLL) per window
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install it with `pip install torch`."
        )

    model.eval()
    with torch.no_grad():
        wt = windows_tensor.to(device)
        if wt.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        N = wt.size(0)
        x_flat = wt.view(N, -1)
        mu, logvar = model.encode(x_flat)
        logits_flat = model.decode(mu)
        if str(loss_type).lower() == "ce":
            import torch.nn.functional as F
            if seq_len <= 0 or vocab_size <= 0:
                # Infer shape from tensor
                seq_len = int(wt.size(1))
                vocab_size = int(wt.size(2))
            logits3 = logits_flat.view(N, int(seq_len), int(vocab_size))
            targets = wt.argmax(dim=2)
            ce = F.cross_entropy(logits3.view(-1, int(vocab_size)), targets.view(-1), reduction="none")
            ce_w = ce.view(N, int(seq_len)).mean(dim=1)
            return ce_w.cpu().numpy().astype(np.float32)
        else:
            recon = torch.sigmoid(logits_flat).view_as(wt)
            mse = (recon - wt).pow(2).mean(dim=(1, 2))
            return mse.cpu().numpy().astype(np.float32)


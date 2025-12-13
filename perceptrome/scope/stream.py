from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import curses
except ImportError:
    curses = None  # type: ignore

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore

from ..model import PlasmidVAE
from .ui import compute_errors_with_model_and_tensor


@dataclass
class ScopeStreamContext:
    model: PlasmidVAE
    optimizer: "torch.optim.Optimizer"
    device: "torch.device"
    dataloader: DataLoader
    dataloader_iter: Any
    global_step: int
    last_total: float
    steps_target: int
    steps_done: int
    beta_kl: float
    kl_warmup_steps: int
    max_grad_norm: float
    loss_type: str
    seq_len: int
    vocab_size: int


def run_scope_stream_ui(
    stdscr,
    accession: str,
    windows_tensor: "torch.Tensor",
    gc_values: np.ndarray,
    window_size: int,
    stride: int,
    fps: float,
    update_every: int,
    ctx: ScopeStreamContext,
) -> None:
    """
    Live GenomeScope + VAE training.
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install it with `pip install torch`."
        )

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    num_windows = windows_tensor.size(0)
    if num_windows == 0:
        stdscr.addstr(0, 0, "No windows to visualize (encoded array empty).")
        stdscr.refresh()
        import time
        time.sleep(2.0)
        return

    if gc_values.shape[0] != num_windows:
        raise ValueError(
            f"gc_values length {gc_values.shape[0]} != num_windows {num_windows}"
        )

    palette = " .:-=+*#%@"
    start_idx = 0
    paused = False

    errors = compute_errors_with_model_and_tensor(
        ctx.model, windows_tensor, ctx.device,
        loss_type=ctx.loss_type, seq_len=ctx.seq_len, vocab_size=ctx.vocab_size,
    )

    import time
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        width = max(10, w - 2)
        end_idx = min(start_idx + width, num_windows)

        if errors.size > 0:
            min_e = float(errors.min())
            max_e = float(errors.max())
        else:
            min_e, max_e = 0.0, 1.0
        span_e = max(max_e - min_e, 1e-8)
        norm_err = (errors - min_e) / span_e if errors.size > 0 else errors

        min_gc = float(gc_values.min())
        max_gc = float(gc_values.max())
        span_gc = max(max_gc - min_gc, 1e-8)
        norm_gc = (gc_values - min_gc) / span_gc

        status = "PAUSED" if paused else "TRAINING"
        header = (
            f"GenomeScope STREAM — {accession}  windows={num_windows}  [{status}]"
        )
        stdscr.addstr(0, 0, header[: w - 1])

        if h > 1:
            info1 = (
                f"steps {ctx.steps_done}/{ctx.steps_target}  "
                f"(global={ctx.global_step})  total_loss={ctx.last_total:.6f}"
            )
            stdscr.addstr(1, 0, info1[: w - 1])

        if h > 2:
            info2 = (
                f"ERROR  min={min_e:.3g} max={max_e:.3g}  "
                f"METRIC min={min_gc:.3f} max={max_gc:.3f}"
            )
            stdscr.addstr(2, 0, info2[: w - 1])

        if h > 3:
            info3 = (
                f"window_size={window_size} stride={stride} "
                f"view={start_idx}-{end_idx - 1}  base_beta={ctx.beta_kl:.3g}"
            )
            stdscr.addstr(3, 0, info3[: w - 1])

        if h > 4:
            controls = "[q] quit  [SPACE] pause/resume  [←/→] scroll"
            stdscr.addstr(4, 0, controls[: w - 1])

        line_err_y = 6 if h > 6 else 0
        show_gc = h > line_err_y + 1
        line_gc_y = line_err_y + 1 if show_gc else None

        if errors.size > 0:
            for col, wi in enumerate(range(start_idx, end_idx)):
                if col >= w - 1:
                    break
                val = float(norm_err[wi])
                idx = int(val * (len(palette) - 1))
                ch = palette[idx]
                try:
                    stdscr.addch(line_err_y, col, ch)
                except curses.error:
                    pass

        if show_gc and line_gc_y is not None:
            for col, wi in enumerate(range(start_idx, end_idx)):
                if col >= w - 1:
                    break
                val = float(norm_gc[wi])
                idx = int(val * (len(palette) - 1))
                ch = palette[idx]
                try:
                    stdscr.addch(line_gc_y, col, ch)
                except curses.error:
                    pass

        if ctx.steps_done >= ctx.steps_target:
            msg = "Training complete — press [q] to exit."
            if h > line_err_y + 2:
                stdscr.addstr(line_err_y + 2, 0, msg[: w - 1])

        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord(" "),):
            paused = not paused
        elif key == curses.KEY_LEFT:
            step = max(1, width // 2)
            start_idx = max(0, start_idx - step)
        elif key == curses.KEY_RIGHT:
            step = max(1, width // 2)
            if start_idx + width < num_windows:
                start_idx = min(num_windows - width, start_idx + step)

        if (not paused) and (ctx.steps_done < ctx.steps_target):
            steps_this_frame = min(update_every, ctx.steps_target - ctx.steps_done)
            ctx.model.train()

            for _ in range(steps_this_frame):
                try:
                    (batch,) = next(ctx.dataloader_iter)
                except StopIteration:
                    ctx.dataloader_iter = iter(ctx.dataloader)
                    (batch,) = next(ctx.dataloader_iter)

                batch = batch.to(ctx.device)  # (B, L, V)
                B = batch.size(0)
                x_target_flat = batch.view(B, -1)
                x_in_flat = x_target_flat

                # KL annealing
                if ctx.kl_warmup_steps > 0:
                    warmup = min(
                        1.0, (ctx.global_step + 1) / float(ctx.kl_warmup_steps)
                    )
                    beta = ctx.beta_kl * warmup
                else:
                    beta = ctx.beta_kl

                ctx.optimizer.zero_grad(set_to_none=True)
                recon_logits, mu, logvar = ctx.model(x_in_flat)
                total_loss, recon_loss, kl_loss = vae_loss(
                    recon_logits, x_target_flat, mu, logvar, beta,
                    str(ctx.loss_type).lower(), int(ctx.seq_len), int(ctx.vocab_size)
                )
                total_loss.backward()

                if ctx.max_grad_norm and ctx.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        ctx.model.parameters(), ctx.max_grad_norm
                    )

                ctx.optimizer.step()

                ctx.steps_done += 1
                ctx.global_step += 1
                ctx.last_total = float(total_loss.item())

            errors = compute_errors_with_model_and_tensor(
                ctx.model, windows_tensor, ctx.device,
                loss_type=ctx.loss_type, seq_len=ctx.seq_len, vocab_size=ctx.vocab_size,
            )

        time.sleep(max(0.0, 1.0 / max(fps, 1e-3)))

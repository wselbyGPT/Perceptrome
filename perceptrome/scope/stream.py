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

from ..model import vae_loss
from .summary import ScopeSummaryAdapter, build_gradient_row, build_scope_summary_frame, normalize_values
from .ui import compute_errors_with_model_and_tensor


@dataclass
class ScopeStreamContext:
    # Architecture-agnostic model (MLP VAE, TransformerVAE, etc.)
    model: Any
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


@dataclass(frozen=True)
class ScopeStreamRenderModel:
    frame: Any
    status: str
    header_line: str
    progress_line: str
    distribution_line: str
    window_line: str
    controls_line: str
    error_row: str
    metric_row: str
    error_norm: np.ndarray
    metric_norm: np.ndarray


def compute_scope_stream_render_model(
    *,
    accession: str,
    errors: np.ndarray,
    metric_values: np.ndarray,
    window_size: int,
    stride: int,
    start_idx: int,
    width: int,
    paused: bool,
    ctx: ScopeStreamContext,
    adapter: ScopeSummaryAdapter,
    gradient_mode: str,
) -> ScopeStreamRenderModel:
    """Pure helper computing compact summary frame + formatted stream lines."""
    status = "PAUSED" if paused else "TRAINING"
    frame = adapter.publish(
        build_scope_summary_frame(
            accession=accession,
            errors=errors,
            metric_values=metric_values,
            window_size=window_size,
            stride=stride,
            start_idx=start_idx,
            width=width,
            status=status,
            steps_done=ctx.steps_done,
            steps_target=ctx.steps_target,
            global_step=ctx.global_step,
        )
    )
    error_norm = normalize_values(errors, frame.error_range)
    metric_norm = normalize_values(metric_values, frame.metric_range)
    return ScopeStreamRenderModel(
        frame=frame,
        status=status,
        header_line=f"GenomeScope STREAM — {accession}  windows={frame.num_windows}  [{status}]",
        progress_line=(
            f"steps {ctx.steps_done}/{ctx.steps_target}  (global={ctx.global_step})  "
            f"total_loss={ctx.last_total:.6f} progress={100.0 * frame.progress.ratio:.1f}%"
        ),
        distribution_line=(
            f"ERROR min={frame.error_range.min_value:.3g} max={frame.error_range.max_value:.3g} "
            f"bands={frame.error_bands.low}/{frame.error_bands.mid}/{frame.error_bands.high}  "
            f"METRIC min={frame.metric_range.min_value:.3f} max={frame.metric_range.max_value:.3f} "
            f"bands={frame.metric_bands.low}/{frame.metric_bands.mid}/{frame.metric_bands.high} "
            f"gradient={gradient_mode}"
        ),
        window_line=(
            f"window_size={window_size} stride={stride} "
            f"view={start_idx}-{frame.end_idx - 1}  base_beta={ctx.beta_kl:.3g}"
        ),
        controls_line="[q] quit  [SPACE] pause/resume  [←/→] scroll",
        error_row=build_gradient_row(error_norm, start_idx=start_idx, end_idx=frame.end_idx, width=width + 1),
        metric_row=build_gradient_row(metric_norm, start_idx=start_idx, end_idx=frame.end_idx, width=width + 1),
        error_norm=error_norm,
        metric_norm=metric_norm,
    )


def draw_scope_stream(stdscr: Any, model: ScopeStreamRenderModel, done: bool) -> None:
    """Thin curses renderer for stream scope model."""
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, model.header_line[: w - 1])
    if h > 1:
        stdscr.addstr(1, 0, model.progress_line[: w - 1])
    if h > 2:
        stdscr.addstr(2, 0, model.distribution_line[: w - 1])
    if h > 3:
        stdscr.addstr(3, 0, model.window_line[: w - 1])
    if h > 4:
        stdscr.addstr(4, 0, model.controls_line[: w - 1])

    line_err_y = 6 if h > 6 else 0
    show_metric = h > line_err_y + 1
    line_metric_y = line_err_y + 1 if show_metric else None
    try:
        stdscr.addstr(line_err_y, 0, model.error_row[: w - 1])
    except curses.error:
        pass
    if show_metric and line_metric_y is not None:
        try:
            stdscr.addstr(line_metric_y, 0, model.metric_row[: w - 1])
        except curses.error:
            pass
    if done and h > line_err_y + 2:
        msg = "Training complete — press [q] to exit."
        stdscr.addstr(line_err_y + 2, 0, msg[: w - 1])


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
    color: bool = True,
    adapter: ScopeSummaryAdapter | None = None,
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

    start_idx = 0
    paused = False
    adapter = adapter or ScopeSummaryAdapter()
    gradient_mode = "ansi" if bool(color and curses and curses.has_colors()) else "ascii"

    errors = compute_errors_with_model_and_tensor(
        ctx.model, windows_tensor, ctx.device,
        loss_type=ctx.loss_type, seq_len=ctx.seq_len, vocab_size=ctx.vocab_size,
    )

    import time
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        width = max(10, w - 2)
        model = compute_scope_stream_render_model(
            accession=accession,
            errors=errors,
            metric_values=gc_values,
            window_size=window_size,
            stride=stride,
            start_idx=start_idx,
            width=width,
            paused=paused,
            ctx=ctx,
            adapter=adapter,
            gradient_mode=gradient_mode,
        )
        draw_scope_stream(stdscr, model, done=ctx.steps_done >= ctx.steps_target)

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

import curses
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from ..encoding_main import compute_gc_from_encoded
from ..model import get_device, load_or_init_model, vae_loss


class TerminalGradient:
    """Render a normalized 0..1 value as a terminal gradient row."""

    def __init__(self, stdscr: Any, use_color: bool = True) -> None:
        self.stdscr = stdscr
        self.palette = " .:-=+*#%@"
        self.use_color = bool(use_color and curses.has_colors())

        if self.use_color:
            curses.start_color()
            curses.use_default_colors()
            # Use a compact gradient from blue->cyan->green->yellow->red.
            colors = [
                curses.COLOR_BLUE,
                curses.COLOR_CYAN,
                curses.COLOR_GREEN,
                curses.COLOR_YELLOW,
                curses.COLOR_RED,
            ]
            self._pair_ids = []
            for idx, fg in enumerate(colors, start=1):
                curses.init_pair(idx, fg, -1)
                self._pair_ids.append(idx)
        else:
            self._pair_ids = []

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
            val = float(norm_values[wi])
            val = min(1.0, max(0.0, val))
            idx = int(val * (len(self.palette) - 1))
            ch = self.palette[idx]

            try:
                if self.use_color and self._pair_ids:
                    cidx = int(val * (len(self._pair_ids) - 1))
                    self.stdscr.addch(y, col, ch, curses.color_pair(self._pair_ids[cidx]))
                else:
                    self.stdscr.addch(y, col, ch)
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

    palette = " .:-=+*#%@"
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
            info_gc = f"METRIC min={min_gc:.3f} max={max_gc:.3f}"
            stdscr.addstr(2, 0, info_gc[: w - 1])

        if h > 3:
            controls = "[q] quit   [←/→] scroll"
            stdscr.addstr(3, 0, controls[: w - 1])

        line_err_y = 5 if h > 5 else 0
        show_gc = h > line_err_y + 1
        line_gc_y = line_err_y + 1 if show_gc else None

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
    model: Any,
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


def load_bio_ast_visualization(accession: str) -> dict:
    """Load Bio-AST visualization artifacts for UI surfaces.

    Returns a payload with tree/graph JSON and a lightweight summary of
    node types, spans, and relationships.
    """
    import json
    import os

    from perceptrome.run_layout import ensure_run_layout, path_in_run

    layout = ensure_run_layout()
    base = path_in_run(layout, "artifacts", os.path.join("bio_ast", str(accession)))
    from perceptrome.encoding.bio_ast_export import export_filenames

    filenames = export_filenames()

    def _resolve_artifact_path(primary: str, legacy: str) -> str:
        primary_path = os.path.join(base, primary)
        if os.path.exists(primary_path):
            return primary_path
        return os.path.join(base, legacy)

    tree_path = _resolve_artifact_path(filenames["tree_json"], "tree_json.json")
    graph_path = _resolve_artifact_path(filenames["graph_json"], "graph_json.json")
    storage_map_path = _resolve_artifact_path(filenames["storage_map"], "storage_map.json")
    summary_path = _resolve_artifact_path(filenames["summary_json"], filenames["summary_json"])

    with open(tree_path, "r", encoding="utf-8") as handle:
        tree_payload = json.load(handle)
    with open(graph_path, "r", encoding="utf-8") as handle:
        graph_payload = json.load(handle)
    with open(storage_map_path, "r", encoding="utf-8") as handle:
        storage_map_payload = json.load(handle)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
    else:
        summary_payload = {
            "schema": "bio_ast_summary_v1",
            "accession": accession,
            "node_count": len(graph_payload.get("nodes", [])),
            "edge_count": len(graph_payload.get("edges", [])),
        }

    node_types = {}
    spans = []
    for node in graph_payload.get("nodes", []):
        ntype = str(node.get("node_type", "unknown"))
        node_types[ntype] = int(node_types.get(ntype, 0)) + 1
        span = node.get("span") if isinstance(node.get("span"), dict) else {}
        spans.append(
            {
                "id": node.get("id"),
                "node_type": ntype,
                "start": span.get("start"),
                "end": span.get("end"),
                "strand": span.get("strand"),
                "frame": span.get("frame"),
            }
        )

    relationships = [
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "relation": edge.get("relation"),
            "relation_type": edge.get("relation_type"),
        }
        for edge in graph_payload.get("edges", [])
    ]

    return {
        "accession": accession,
        "tree": tree_payload,
        "graph": graph_payload,
        "storage_map": storage_map_payload,
        "export_summary": summary_payload,
        "summary": {
            "node_types": node_types,
            "spans": spans,
            "relationships": relationships,
        },
    }

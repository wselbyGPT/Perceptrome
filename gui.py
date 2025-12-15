"""
Simple Tkinter GUI wrapper for common Perceptrome commands.

Provides a few quick actions to fetch, encode, train, and generate sequences
without using the CLI directly. Long-running jobs are dispatched to a worker
thread so the UI stays responsive.
"""

import argparse
import io
import threading
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import ttk
from tkinter import scrolledtext

from perceptrome.cli.commands import (
    cmd_fetch_one,
    cmd_encode_one,
    cmd_train_one,
    cmd_generate_plasmid,
    cmd_generate_protein,
)


def _optional_int(value: str):
    """Convert empty strings to None; otherwise cast to int."""
    value = value.strip()
    if not value:
        return None
    return int(value)


def _optional_float(value: str):
    value = value.strip()
    if not value:
        return None
    return float(value)


def _clean_str(value: str):
    value = value.strip()
    return value or None


class CommandRunner:
    """Runs CLI command functions on a background thread and streams output."""

    def __init__(self, log_widget: scrolledtext.ScrolledText, root: tk.Tk):
        self.log_widget = log_widget
        self.root = root
        self.lock = threading.Lock()
        self.running = False

    def _append(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")

    def _threadsafe_append(self, text: str) -> None:
        self.root.after(0, self._append, text)

    def run(self, label: str, func, args: dict) -> None:
        with self.lock:
            if self.running:
                self._threadsafe_append("Another command is already running; please wait...")
                return
            self.running = True

        def worker():
            buffer = io.StringIO()
            self._threadsafe_append(f"[START] {label}")
            try:
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    result = func(argparse.Namespace(**args))
            except Exception as exc:  # noqa: BLE001
                output = buffer.getvalue().strip()
                if output:
                    self._threadsafe_append(output)
                self._threadsafe_append(f"[ERROR] {label}: {exc}")
            else:
                output = buffer.getvalue().strip()
                if output:
                    self._threadsafe_append(output)
                self._threadsafe_append(f"[DONE] {label} (exit code {result})")
            finally:
                with self.lock:
                    self.running = False

        threading.Thread(target=worker, daemon=True).start()


class PerceptromeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Perceptrome GUI")
        root.geometry("860x700")

        self.config_path = tk.StringVar(value="config/stream_config.yaml")
        self._build_layout()

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Config path:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.config_path, width=60).pack(side=tk.LEFT, padx=6)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log = scrolledtext.ScrolledText(self.root, height=16, state="disabled")
        self.log.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        self.runner = CommandRunner(self.log, self.root)

        notebook.add(self._build_fetch_tab(notebook), text="Fetch")
        notebook.add(self._build_encode_tab(notebook), text="Encode")
        notebook.add(self._build_train_tab(notebook), text="Train")
        notebook.add(self._build_generate_tab(notebook), text="Generate")

    def _build_fetch_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)

        accession = tk.StringVar()
        source = tk.StringVar(value="fasta")
        force = tk.BooleanVar(value=False)

        self._labeled_entry(frame, "Accession", accession)
        self._labeled_combobox(frame, "Source", source, ["fasta", "genbank"])
        ttk.Checkbutton(frame, text="Force re-download", variable=force).pack(anchor=tk.W, pady=4)

        ttk.Button(
            frame,
            text="Fetch",
            command=lambda: self.runner.run(
                "Fetch accession",
                cmd_fetch_one,
                {
                    "config": self.config_path.get(),
                    "accession": accession.get().strip(),
                    "source": source.get(),
                    "force": bool(force.get()),
                },
            ),
        ).pack(anchor=tk.E, pady=10)

        return frame

    def _build_encode_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)

        accession = tk.StringVar()
        tokenizer = tk.StringVar(value="")
        window_size = tk.StringVar()
        stride = tk.StringVar()
        frame_offset = tk.StringVar()
        min_orf = tk.StringVar()
        source = tk.StringVar(value="")

        self._labeled_entry(frame, "Accession", accession)
        self._labeled_combobox(frame, "Tokenizer (blank=default)", tokenizer, ["", "base", "codon", "aa"])
        self._labeled_entry(frame, "Window size", window_size)
        self._labeled_entry(frame, "Stride", stride)
        self._labeled_entry(frame, "Frame offset (0/1/2)", frame_offset)
        self._labeled_entry(frame, "Min ORF AA (aa mode)", min_orf)
        self._labeled_combobox(frame, "Source (blank=auto)", source, ["", "fasta", "genbank"])

        ttk.Button(
            frame,
            text="Encode",
            command=lambda: self.runner.run(
                "Encode accession",
                cmd_encode_one,
                {
                    "config": self.config_path.get(),
                    "accession": accession.get().strip(),
                    "tokenizer": _clean_str(tokenizer.get()),
                    "window_size": _optional_int(window_size.get()),
                    "stride": _optional_int(stride.get()),
                    "frame_offset": _optional_int(frame_offset.get()),
                    "min_orf_aa": _optional_int(min_orf.get()),
                    "source": _clean_str(source.get()),
                },
            ),
        ).pack(anchor=tk.E, pady=10)

        return frame

    def _build_train_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)

        accession = tk.StringVar()
        steps = tk.StringVar()
        batch_size = tk.StringVar()
        window_size = tk.StringVar()
        stride = tk.StringVar()
        tokenizer = tk.StringVar(value="")
        frame_offset = tk.StringVar()
        min_orf = tk.StringVar()
        loss_type = tk.StringVar(value="")
        mask_prob = tk.StringVar()
        span_mask_prob = tk.StringVar()
        span_mask_len = tk.StringVar()
        reencode = tk.BooleanVar(value=False)

        self._labeled_entry(frame, "Accession", accession)
        self._labeled_entry(frame, "Steps", steps)
        self._labeled_entry(frame, "Batch size", batch_size)
        self._labeled_entry(frame, "Window size", window_size)
        self._labeled_entry(frame, "Stride", stride)
        self._labeled_combobox(frame, "Tokenizer (blank=default)", tokenizer, ["", "base", "codon", "aa"])
        self._labeled_entry(frame, "Frame offset (0/1/2)", frame_offset)
        self._labeled_entry(frame, "Min ORF AA (aa mode)", min_orf)
        self._labeled_combobox(frame, "Loss type (blank=default)", loss_type, ["", "mse", "ce"])
        self._labeled_entry(frame, "Mask prob (AA)", mask_prob)
        self._labeled_entry(frame, "Span mask prob (AA)", span_mask_prob)
        self._labeled_entry(frame, "Span mask len (AA)", span_mask_len)
        ttk.Checkbutton(frame, text="Re-encode before training", variable=reencode).pack(anchor=tk.W, pady=4)

        ttk.Button(
            frame,
            text="Train",
            command=lambda: self.runner.run(
                "Train accession",
                cmd_train_one,
                {
                    "config": self.config_path.get(),
                    "accession": accession.get().strip(),
                    "steps": _optional_int(steps.get()),
                    "batch_size": _optional_int(batch_size.get()),
                    "window_size": _optional_int(window_size.get()),
                    "stride": _optional_int(stride.get()),
                    "tokenizer": _clean_str(tokenizer.get()),
                    "frame_offset": _optional_int(frame_offset.get()),
                    "min_orf_aa": _optional_int(min_orf.get()),
                    "loss_type": _clean_str(loss_type.get()),
                    "mask_prob": _optional_float(mask_prob.get()),
                    "span_mask_prob": _optional_float(span_mask_prob.get()),
                    "span_mask_len": _optional_int(span_mask_len.get()),
                    "reencode": bool(reencode.get()),
                },
            ),
        ).pack(anchor=tk.E, pady=10)

        return frame

    def _build_generate_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)

        plasmid_length = tk.StringVar(value="10000")
        plasmid_windows = tk.StringVar()
        plasmid_window_size = tk.StringVar()
        plasmid_name = tk.StringVar(value="perceptrome_plasmid_1")
        plasmid_output = tk.StringVar(value="generated/novel_plasmid.fasta")
        plasmid_seed = tk.StringVar()
        plasmid_latent_scale = tk.StringVar(value="1.0")
        plasmid_temperature = tk.StringVar(value="1.0")
        plasmid_gc_bias = tk.StringVar(value="1.0")
        plasmid_tokenizer = tk.StringVar(value="")

        protein_length = tk.StringVar(value="600")
        protein_windows = tk.StringVar()
        protein_window = tk.StringVar()
        protein_name = tk.StringVar(value="perceptrome_protein_1")
        protein_output = tk.StringVar(value="generated/novel_protein.faa")
        protein_seed = tk.StringVar()
        protein_latent_scale = tk.StringVar(value="1.0")
        protein_temperature = tk.StringVar(value="1.0")
        protein_reject = tk.BooleanVar(value=False)
        protein_reject_tries = tk.StringVar(value="40")
        protein_reject_max_run = tk.StringVar(value="10")
        protein_reject_max_x_frac = tk.StringVar(value="0.15")

        section = ttk.LabelFrame(frame, text="Generate plasmid", padding=10)
        section.pack(fill=tk.BOTH, expand=False, padx=4, pady=6)

        self._labeled_combobox(section, "Tokenizer (blank=default)", plasmid_tokenizer, ["", "base", "codon", "aa"])
        self._labeled_entry(section, "Length (bp)", plasmid_length)
        self._labeled_entry(section, "Num windows", plasmid_windows)
        self._labeled_entry(section, "Window size", plasmid_window_size)
        self._labeled_entry(section, "Name", plasmid_name)
        self._labeled_entry(section, "Output path", plasmid_output)
        self._labeled_entry(section, "Seed", plasmid_seed)
        self._labeled_entry(section, "Latent scale", plasmid_latent_scale)
        self._labeled_entry(section, "Temperature", plasmid_temperature)
        self._labeled_entry(section, "GC bias", plasmid_gc_bias)

        ttk.Button(
            section,
            text="Generate plasmid",
            command=lambda: self.runner.run(
                "Generate plasmid",
                cmd_generate_plasmid,
                {
                    "config": self.config_path.get(),
                    "tokenizer": _clean_str(plasmid_tokenizer.get()),
                    "length_bp": _optional_int(plasmid_length.get()),
                    "num_windows": _optional_int(plasmid_windows.get()),
                    "window_size": _optional_int(plasmid_window_size.get()),
                    "name": plasmid_name.get().strip() or "perceptrome_plasmid_1",
                    "output": plasmid_output.get().strip(),
                    "seed": _optional_int(plasmid_seed.get()),
                    "latent_scale": _optional_float(plasmid_latent_scale.get()),
                    "temperature": _optional_float(plasmid_temperature.get()),
                    "gc_bias": _optional_float(plasmid_gc_bias.get()),
                },
            ),
        ).pack(anchor=tk.E, pady=8)

        section = ttk.LabelFrame(frame, text="Generate protein", padding=10)
        section.pack(fill=tk.BOTH, expand=False, padx=4, pady=6)

        self._labeled_entry(section, "Length (aa)", protein_length)
        self._labeled_entry(section, "Num windows", protein_windows)
        self._labeled_entry(section, "Window (aa)", protein_window)
        self._labeled_entry(section, "Name", protein_name)
        self._labeled_entry(section, "Output path", protein_output)
        self._labeled_entry(section, "Seed", protein_seed)
        self._labeled_entry(section, "Latent scale", protein_latent_scale)
        self._labeled_entry(section, "Temperature", protein_temperature)
        ttk.Checkbutton(section, text="Enable rejection sampling", variable=protein_reject).pack(anchor=tk.W, pady=4)
        self._labeled_entry(section, "Reject tries", protein_reject_tries)
        self._labeled_entry(section, "Reject max run", protein_reject_max_run)
        self._labeled_entry(section, "Reject max X frac", protein_reject_max_x_frac)

        ttk.Button(
            section,
            text="Generate protein",
            command=lambda: self.runner.run(
                "Generate protein",
                cmd_generate_protein,
                {
                    "config": self.config_path.get(),
                    "length_aa": _optional_int(protein_length.get()),
                    "num_windows": _optional_int(protein_windows.get()),
                    "window_aa": _optional_int(protein_window.get()),
                    "name": protein_name.get().strip() or "perceptrome_protein_1",
                    "output": protein_output.get().strip(),
                    "seed": _optional_int(protein_seed.get()),
                    "latent_scale": _optional_float(protein_latent_scale.get()),
                    "temperature": _optional_float(protein_temperature.get()),
                    "reject": bool(protein_reject.get()),
                    "reject_tries": _optional_int(protein_reject_tries.get()),
                    "reject_max_run": _optional_int(protein_reject_max_run.get()),
                    "reject_max_x_frac": _optional_float(protein_reject_max_x_frac.get()),
                },
            ),
        ).pack(anchor=tk.E, pady=8)

        return frame

    def _labeled_entry(self, parent, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=40).pack(side=tk.LEFT, padx=5)

    def _labeled_combobox(self, parent, label: str, variable: tk.StringVar, values: list[str]) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=variable, values=values, width=37, state="readonly").pack(side=tk.LEFT, padx=5)


def main() -> None:
    root = tk.Tk()
    PerceptromeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

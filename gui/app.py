from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .commands import import_help_message, load_commands, get_import_error
from .logging_panel import LogItem, LogPanel
from .runner import CommandRunner
from .settings import load_settings, save_settings
from .store import HistoryEntry, PresetHistoryStore
from .theme import apply_theme
from .tk_compat import filedialog, messagebox, scrolledtext, simpledialog, tk, ttk
from .utils import (
    ValidationError,
    coerce_to_string,
    date_ts,
    json_dumps,
    opt_float,
    opt_int,
    opt_str,
    req_str,
)
from .widgets import PresetsPanel, ToolTip


class PerceptromeGUI:
    def __init__(self, root: tk.Tk, *, dark: bool, geometry: str) -> None:
        self.root = root
        self.dark = dark
        self.settings = load_settings()
        self.store = PresetHistoryStore(self.settings)

        self.root.title("Perceptrome GUI")
        self.root.geometry(geometry or self.settings.get("geometry", "1020x780"))
        self.root.minsize(940, 700)

        apply_theme(root, dark=self.dark)

        self.config_path = tk.StringVar(value=self.settings.get("config_path", "config/stream_config.yaml"))
        self.status_var = tk.StringVar(value="Ready")

        # init lockables early (tabs add buttons during build)
        self._lockables: list[tk.Widget] = []

        # form variables (for loading presets/history)
        self.fetch_vars: Dict[str, tk.Variable] = {}
        self.encode_vars: Dict[str, tk.Variable] = {}
        self.train_vars: Dict[str, tk.Variable] = {}
        self.genp_vars: Dict[str, tk.Variable] = {}
        self.gena_vars: Dict[str, tk.Variable] = {}

        # generate focus: which section Enter runs
        self._gen_focus = "plasmid"  # or "protein"

        # history filters
        self.history_search_var = tk.StringVar(value="")
        self.history_cmd_filter_var = tk.StringVar(value="All")
        self._history_refresh_job = None

        self._build_menu()
        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # shortcuts
        self.root.bind("<Control-l>", lambda _e: self.log_clear_safe())
        self.root.bind("<Control-s>", lambda _e: self._save_log_dialog())
        self.root.bind("<Control-c>", lambda _e: self._copy_log())
        self.root.bind("<F5>", lambda _e: self._env_check())

        # new hotkeys
        self.root.bind("<Return>", self._hotkey_run_current, add=True)
        self.root.bind("<Control-Return>", self._hotkey_save_preset_current, add=True)

        self._ensure_commands_loaded(startup=True)
        self._refresh_history_tab()

    # ----------------------------
    # Small helpers
    # ----------------------------

    def _persist_settings(self) -> None:
        self.settings["config_path"] = self.config_path.get()
        try:
            self.settings["geometry"] = self.root.winfo_geometry()
        except Exception:
            pass
        save_settings(self.settings)

    def _register_lockable(self, widget: tk.Widget) -> None:
        self._lockables.append(widget)
        if hasattr(self, "runner"):
            self.runner.set_controls_to_lock(self._lockables)

    def command_title(self, command_key: str) -> str:
        return {
            "fetch_one": "Fetch",
            "encode_one": "Encode",
            "train_one": "Train",
            "gen_plasmid": "Generate plasmid",
            "gen_protein": "Generate protein",
        }.get(command_key, command_key)

    def _get_cmd(self, key: str):
        cmds = load_commands()
        fn = cmds.get(key)
        if fn is None:
            self._ensure_commands_loaded()
            fn = load_commands().get(key)
        return fn

    def _run_command(self, command_key: str, label: str, args: Dict[str, Any]) -> None:
        fn = self._get_cmd(command_key)
        if not fn:
            return
        self.store.add_history(ts=date_ts(), command_key=command_key, label=label, args=args)
        self._persist_settings()
        self._refresh_history_tab()
        self.runner.run(label, fn, args)

    def _set_var(self, var: tk.Variable, value: Any) -> None:
        if isinstance(var, tk.BooleanVar):
            var.set(bool(value))
        else:
            var.set(coerce_to_string(value))

    def _load_args_into_form(self, command_key: str, args: Dict[str, Any]) -> None:
        if "config" in args:
            self.config_path.set(coerce_to_string(args.get("config")))

        if command_key == "fetch_one":
            v = self.fetch_vars
            self._set_var(v["accession"], args.get("accession"))
            self._set_var(v["source"], args.get("source", "fasta"))
            self._set_var(v["force"], args.get("force", False))
            self.notebook.select(self.tab_fetch)
            return

        if command_key == "encode_one":
            v = self.encode_vars
            self._set_var(v["accession"], args.get("accession"))
            self._set_var(v["tokenizer"], args.get("tokenizer"))
            self._set_var(v["window_size"], args.get("window_size"))
            self._set_var(v["stride"], args.get("stride"))
            self._set_var(v["frame_offset"], args.get("frame_offset"))
            self._set_var(v["min_orf_aa"], args.get("min_orf_aa"))
            self._set_var(v["source"], args.get("source"))
            self.notebook.select(self.tab_encode)
            return

        if command_key == "train_one":
            v = self.train_vars
            for k in (
                "accession",
                "steps",
                "batch_size",
                "window_size",
                "stride",
                "tokenizer",
                "frame_offset",
                "min_orf_aa",
                "loss_type",
                "mask_prob",
                "span_mask_prob",
                "span_mask_len",
            ):
                self._set_var(v[k], args.get(k))
            self._set_var(v["reencode"], args.get("reencode", False))
            self.notebook.select(self.tab_train)
            return

        if command_key == "gen_plasmid":
            v = self.genp_vars
            for k in (
                "tokenizer",
                "length_bp",
                "num_windows",
                "window_size",
                "name",
                "output",
                "seed",
                "latent_scale",
                "temperature",
                "gc_bias",
            ):
                self._set_var(v[k], args.get(k))
            self._gen_focus = "plasmid"
            self.notebook.select(self.tab_generate)
            return

        if command_key == "gen_protein":
            v = self.gena_vars
            for k in (
                "length_aa",
                "num_windows",
                "window_aa",
                "name",
                "output",
                "seed",
                "latent_scale",
                "temperature",
                "reject_tries",
                "reject_max_run",
                "reject_max_x_frac",
            ):
                self._set_var(v[k], args.get(k))
            self._set_var(v["reject"], args.get("reject", False))
            self._gen_focus = "protein"
            self.notebook.select(self.tab_generate)
            return

    def _set_gen_focus(self, which: str) -> None:
        if which in ("plasmid", "protein"):
            self._gen_focus = which

    # ----------------------------
    # Hotkeys
    # ----------------------------

    def _focus_is_texty(self) -> bool:
        w = self.root.focus_get()
        if w is None:
            return False
        cls = w.winfo_class()
        if cls in ("Text", "Treeview"):
            return True
        if isinstance(w, tk.Text):
            return True
        return False

    def _current_action(self) -> Optional[Tuple[str, str, Callable[[], Dict[str, Any]]]]:
        """Return (command_key, label, collect_args) for the active tab."""
        tab = self.notebook.select()
        if tab == str(self.tab_fetch):
            return ("fetch_one", "Fetch accession", self._collect_fetch_args)
        if tab == str(self.tab_encode):
            return ("encode_one", "Encode accession", self._collect_encode_args)
        if tab == str(self.tab_train):
            return ("train_one", "Train accession", self._collect_train_args)
        if tab == str(self.tab_generate):
            if self._gen_focus == "protein":
                return ("gen_protein", "Generate protein", self._collect_gen_protein_args)
            return ("gen_plasmid", "Generate plasmid", self._collect_gen_plasmid_args)
        return None

    def _hotkey_run_current(self, _event: tk.Event) -> None:
        if self.runner.is_running():
            return
        if self._focus_is_texty():
            return
        action = self._current_action()
        if not action:
            return
        key, label, collect = action
        try:
            args = collect()
        except ValidationError as e:
            messagebox.showerror("Cannot run", str(e))
            return
        self._run_command(key, label, args)

    def _hotkey_save_preset_current(self, _event: tk.Event) -> None:
        if self.runner.is_running():
            return
        if self._focus_is_texty():
            return
        action = self._current_action()
        if not action:
            return
        key, _label, collect = action
        self._save_current_as_preset_for(key, collect)

    def _save_current_as_preset_for(self, command_key: str, collect: Callable[[], Dict[str, Any]]) -> None:
        try:
            args = collect()
        except ValidationError as e:
            messagebox.showerror("Cannot save preset", str(e))
            return
        default_name = f"{self.command_title(command_key)}_{time.strftime('%Y%m%d_%H%M%S')}"
        name = simpledialog.askstring("Preset name", "Name this preset:", initialvalue=default_name)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        existing = [p.name for p in self.store.list_presets(command_key)]
        if name in existing:
            if not messagebox.askyesno("Overwrite preset?", f"A preset named '{name}' already exists.\nOverwrite it?"):
                return
        self.store.add_or_replace_preset(command_key, name, args)
        self._persist_settings()
        self._refresh_all_preset_panels()
        self.runner.log_info(f"Saved preset '{name}' for {command_key}.")

    # ----------------------------
    # Menu
    # ----------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open config…", command=self._choose_config)
        m_file.add_separator()
        m_file.add_command(label="Export presets…", command=self._export_presets)
        m_file.add_command(label="Import presets…", command=self._import_presets)
        m_file.add_separator()
        m_file.add_command(label="Save log… (Ctrl+S)", command=self._save_log_dialog)
        m_file.add_command(label="Copy log (Ctrl+C)", command=self._copy_log)
        m_file.add_command(label="Clear log (Ctrl+L)", command=self.log_clear_safe)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=m_file)

        m_tools = tk.Menu(menubar, tearoff=0)
        m_tools.add_command(label="Environment check (F5)", command=self._env_check)
        m_tools.add_command(label="Install PyTorch (CPU) in this venv…", command=self._install_torch_cpu)
        m_tools.add_separator()
        m_tools.add_command(label="Export history…", command=self._export_history)
        m_tools.add_command(label="Clear history", command=self._clear_history)
        menubar.add_cascade(label="Tools", menu=m_tools)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=m_help)

        self.root.config(menu=menubar)

    # ----------------------------
    # Layout
    # ----------------------------

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        top = ttk.Frame(outer, style="Panel.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))

        top_inner = ttk.Frame(top, style="Panel.TFrame", padding=10)
        top_inner.pack(fill=tk.X)

        ttk.Label(top_inner, text="Config:", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        e_cfg = ttk.Entry(top_inner, textvariable=self.config_path, width=70)
        e_cfg.grid(row=0, column=1, sticky="we", padx=(8, 8))

        b_browse = ttk.Button(top_inner, text="Browse…", command=self._choose_config)
        b_browse.grid(row=0, column=2, sticky="e")

        b_check = ttk.Button(top_inner, text="Env check", command=self._env_check)
        b_check.grid(row=0, column=3, sticky="e", padx=(8, 0))
        ToolTip(b_check, "Runs basic checks (imports, torch, config existence, venv) and logs results. (F5)")

        top_inner.columnconfigure(1, weight=1)

        mid = ttk.PanedWindow(outer, orient=tk.VERTICAL)
        mid.pack(fill=tk.BOTH, expand=True)

        nb_frame = ttk.Frame(mid)
        log_frame = ttk.Frame(mid)

        mid.add(nb_frame, weight=6)
        mid.add(log_frame, weight=2)

        self.notebook = ttk.Notebook(nb_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.log = LogPanel(log_frame, dark=self.dark)
        self.log.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(outer)
        bottom.pack(fill=tk.X, pady=(10, 0))

        self.spinner = ttk.Progressbar(bottom, mode="indeterminate", length=140)
        self.spinner.pack(side=tk.RIGHT, padx=(8, 0))

        ttk.Label(bottom, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)

        # Runner must exist BEFORE building tabs (tab builders register buttons)
        self.runner = CommandRunner(self.root, self.log, self.status_var, self.spinner)

        # Base lockables
        self._lockables.clear()
        for w in (self.notebook, e_cfg, b_browse, b_check):
            self._register_lockable(w)

        # Build tabs
        self.tab_home = self._build_home_tab(self.notebook)
        self.tab_fetch = self._build_fetch_tab(self.notebook)
        self.tab_encode = self._build_encode_tab(self.notebook)
        self.tab_train = self._build_train_tab(self.notebook)
        self.tab_generate = self._build_generate_tab(self.notebook)
        self.tab_history = self._build_history_tab(self.notebook)

        self.notebook.add(self.tab_home, text="Home")
        self.notebook.add(self.tab_fetch, text="Fetch")
        self.notebook.add(self.tab_encode, text="Encode")
        self.notebook.add(self.tab_train, text="Train")
        self.notebook.add(self.tab_generate, text="Generate")
        self.notebook.add(self.tab_history, text="History")

        self.runner.set_controls_to_lock(self._lockables)

    # ----------------------------
    # Common row builders
    # ----------------------------

    def _row(self, parent: tk.Widget) -> ttk.Frame:
        r = ttk.Frame(parent)
        r.pack(fill=tk.X, pady=4)
        return r

    def _labeled_entry(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        *,
        width: int = 46,
        tooltip: str = "",
        on_focus: Optional[Callable[[], None]] = None,
    ) -> ttk.Entry:
        r = self._row(parent)
        ttk.Label(r, text=label, width=26, anchor="w").pack(side=tk.LEFT)
        entry = ttk.Entry(r, textvariable=var, width=width)
        entry.pack(side=tk.LEFT, padx=(6, 6))
        if tooltip:
            ToolTip(entry, tooltip)
        if on_focus is not None:
            entry.bind("<FocusIn>", lambda _e: on_focus(), add=True)
        return entry

    def _labeled_combo(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        values: list[str],
        *,
        width: int = 43,
        tooltip: str = "",
        on_focus: Optional[Callable[[], None]] = None,
    ) -> ttk.Combobox:
        r = self._row(parent)
        ttk.Label(r, text=label, width=26, anchor="w").pack(side=tk.LEFT)
        cb = ttk.Combobox(r, textvariable=var, values=values, width=width, state="readonly")
        cb.pack(side=tk.LEFT, padx=(6, 6))
        if tooltip:
            ToolTip(cb, tooltip)
        if on_focus is not None:
            cb.bind("<FocusIn>", lambda _e: on_focus(), add=True)
        return cb

    def _action_bar(self, parent: tk.Widget) -> ttk.Frame:
        ttk.Separator(parent).pack(fill=tk.X, pady=(10, 10))
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X)
        return bar

    # ----------------------------
    # Collect args (shared by run + preset save)
    # ----------------------------

    def _collect_fetch_args(self) -> Dict[str, Any]:
        v = self.fetch_vars
        cfg = req_str(self.config_path.get(), "Config path")
        acc = req_str(str(v["accession"].get()), "Accession")
        return {"config": cfg, "accession": acc.strip(), "source": str(v["source"].get() or "fasta"), "force": bool(v["force"].get())}

    def _collect_encode_args(self) -> Dict[str, Any]:
        v = self.encode_vars
        cfg = req_str(self.config_path.get(), "Config path")
        acc = req_str(str(v["accession"].get()), "Accession")
        return {
            "config": cfg,
            "accession": acc.strip(),
            "tokenizer": opt_str(str(v["tokenizer"].get())),
            "window_size": opt_int(str(v["window_size"].get()), "Window size", min_v=1),
            "stride": opt_int(str(v["stride"].get()), "Stride", min_v=1),
            "frame_offset": opt_int(str(v["frame_offset"].get()), "Frame offset", min_v=0, max_v=2),
            "min_orf_aa": opt_int(str(v["min_orf_aa"].get()), "Min ORF AA", min_v=1),
            "source": opt_str(str(v["source"].get())),
        }

    def _collect_train_args(self) -> Dict[str, Any]:
        v = self.train_vars
        cfg = req_str(self.config_path.get(), "Config path")
        acc = req_str(str(v["accession"].get()), "Accession")
        return {
            "config": cfg,
            "accession": acc.strip(),
            "steps": opt_int(str(v["steps"].get()), "Steps", min_v=1),
            "batch_size": opt_int(str(v["batch_size"].get()), "Batch size", min_v=1),
            "window_size": opt_int(str(v["window_size"].get()), "Window size", min_v=1),
            "stride": opt_int(str(v["stride"].get()), "Stride", min_v=1),
            "tokenizer": opt_str(str(v["tokenizer"].get())),
            "frame_offset": opt_int(str(v["frame_offset"].get()), "Frame offset", min_v=0, max_v=2),
            "min_orf_aa": opt_int(str(v["min_orf_aa"].get()), "Min ORF AA", min_v=1),
            "loss_type": opt_str(str(v["loss_type"].get())),
            "mask_prob": opt_float(str(v["mask_prob"].get()), "Mask prob", min_v=0.0, max_v=1.0),
            "span_mask_prob": opt_float(str(v["span_mask_prob"].get()), "Span mask prob", min_v=0.0, max_v=1.0),
            "span_mask_len": opt_int(str(v["span_mask_len"].get()), "Span mask len", min_v=1),
            "reencode": bool(v["reencode"].get()),
        }

    def _collect_gen_plasmid_args(self) -> Dict[str, Any]:
        v = self.genp_vars
        cfg = req_str(self.config_path.get(), "Config path")
        out = req_str(str(v["output"].get()), "Output path")
        name = str(v["name"].get()).strip() or "perceptrome_plasmid_1"
        return {
            "config": cfg,
            "tokenizer": opt_str(str(v["tokenizer"].get())),
            "length_bp": opt_int(str(v["length_bp"].get()), "Length (bp)", min_v=1),
            "num_windows": opt_int(str(v["num_windows"].get()), "Num windows", min_v=1),
            "window_size": opt_int(str(v["window_size"].get()), "Window size", min_v=1),
            "name": name,
            "output": out.strip(),
            "seed": opt_int(str(v["seed"].get()), "Seed"),
            "latent_scale": opt_float(str(v["latent_scale"].get()), "Latent scale", min_v=0.0),
            "temperature": opt_float(str(v["temperature"].get()), "Temperature", min_v=0.0),
            "gc_bias": opt_float(str(v["gc_bias"].get()), "GC bias", min_v=0.0),
        }

    def _collect_gen_protein_args(self) -> Dict[str, Any]:
        v = self.gena_vars
        cfg = req_str(self.config_path.get(), "Config path")
        out = req_str(str(v["output"].get()), "Output path")
        name = str(v["name"].get()).strip() or "perceptrome_protein_1"
        return {
            "config": cfg,
            "length_aa": opt_int(str(v["length_aa"].get()), "Length (aa)", min_v=1),
            "num_windows": opt_int(str(v["num_windows"].get()), "Num windows", min_v=1),
            "window_aa": opt_int(str(v["window_aa"].get()), "Window (aa)", min_v=1),
            "name": name,
            "output": out.strip(),
            "seed": opt_int(str(v["seed"].get()), "Seed"),
            "latent_scale": opt_float(str(v["latent_scale"].get()), "Latent scale", min_v=0.0),
            "temperature": opt_float(str(v["temperature"].get()), "Temperature", min_v=0.0),
            "reject": bool(v["reject"].get()),
            "reject_tries": opt_int(str(v["reject_tries"].get()), "Reject tries", min_v=1),
            "reject_max_run": opt_int(str(v["reject_max_run"].get()), "Reject max run", min_v=1),
            "reject_max_x_frac": opt_float(str(v["reject_max_x_frac"].get()), "Reject max X frac", min_v=0.0, max_v=1.0),
        }

    # ----------------------------
    # Tabs
    # ----------------------------

    def _build_home_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)

        hero = ttk.LabelFrame(frame, text="Perceptrome GUI", padding=12)
        hero.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(
            hero,
            text=(
                "Now with Presets + History (polished):\n"
                "  • Presets: ★ favorites (pin to top), click star to toggle, filter, duplicate\n"
                "  • History: search + command filter + save entry as preset\n\n"
                "Hotkeys:\n"
                "  Enter         Run current command tab\n"
                "  Ctrl+Enter    Save current form as preset\n"
                "  F5            Env check\n"
                "  Ctrl+L        Clear log\n"
                "  Ctrl+S        Save log\n"
                "  Ctrl+C        Copy log\n\n"
                "Generate tab:\n"
                "  Enter runs the section you last interacted with (plasmid vs protein).\n"
            ),
            justify=tk.LEFT,
        ).pack(anchor="w")

        return frame

    def _build_fetch_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)

        main = ttk.Frame(frame)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        accession = tk.StringVar()
        source = tk.StringVar(value="fasta")
        force = tk.BooleanVar(value=False)
        self.fetch_vars = {"accession": accession, "source": source, "force": force}

        self._labeled_entry(left, "Accession", accession, tooltip="NCBI accession. Required.")
        self._labeled_combo(left, "Source", source, ["fasta", "genbank"], tooltip="Which upstream format to fetch.")
        chk = ttk.Checkbutton(left, text="Force re-download", variable=force)
        chk.pack(anchor="w", pady=6)
        self._register_lockable(chk)

        bar = self._action_bar(left)
        btn = ttk.Button(bar, text="Fetch", command=lambda: self._run_command("fetch_one", "Fetch accession", self._collect_fetch_args()))
        btn.pack(side=tk.RIGHT)
        self._register_lockable(btn)

        PresetsPanel(
            right,
            gui=self,
            command_key="fetch_one",
            title="Presets",
            collect_args=self._collect_fetch_args,
            load_into_form=lambda a: self._load_args_into_form("fetch_one", a),
            save_preset_hotkey=lambda: self._save_current_as_preset_for("fetch_one", self._collect_fetch_args),
        ).pack(fill=tk.Y, expand=True)
        return frame

    def _build_encode_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)

        main = ttk.Frame(frame)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        accession = tk.StringVar()
        tokenizer = tk.StringVar(value="")
        window_size = tk.StringVar()
        stride = tk.StringVar()
        frame_offset = tk.StringVar()
        min_orf_aa = tk.StringVar()
        source = tk.StringVar(value="")

        self.encode_vars = {
            "accession": accession,
            "tokenizer": tokenizer,
            "window_size": window_size,
            "stride": stride,
            "frame_offset": frame_offset,
            "min_orf_aa": min_orf_aa,
            "source": source,
        }

        self._labeled_entry(left, "Accession", accession, tooltip="Required.")
        self._labeled_combo(left, "Tokenizer", tokenizer, ["", "base", "codon", "aa"], tooltip="Blank = default.")
        self._labeled_entry(left, "Window size", window_size, tooltip="Optional. Integer.")
        self._labeled_entry(left, "Stride", stride, tooltip="Optional. Integer.")
        self._labeled_entry(left, "Frame offset", frame_offset, tooltip="Optional. 0/1/2.")
        self._labeled_entry(left, "Min ORF AA", min_orf_aa, tooltip="Optional. AA mode.")
        self._labeled_combo(left, "Source (override)", source, ["", "fasta", "genbank"], tooltip="Blank = auto.")

        bar = self._action_bar(left)
        btn = ttk.Button(bar, text="Encode", command=lambda: self._run_command("encode_one", "Encode accession", self._collect_encode_args()))
        btn.pack(side=tk.RIGHT)
        self._register_lockable(btn)

        PresetsPanel(
            right,
            gui=self,
            command_key="encode_one",
            title="Presets",
            collect_args=self._collect_encode_args,
            load_into_form=lambda a: self._load_args_into_form("encode_one", a),
            save_preset_hotkey=lambda: self._save_current_as_preset_for("encode_one", self._collect_encode_args),
        ).pack(fill=tk.Y, expand=True)
        return frame

    def _build_train_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)

        main = ttk.Frame(frame)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        accession = tk.StringVar()
        steps = tk.StringVar()
        batch_size = tk.StringVar()
        window_size = tk.StringVar()
        stride = tk.StringVar()
        tokenizer = tk.StringVar(value="")
        frame_offset = tk.StringVar()
        min_orf_aa = tk.StringVar()
        loss_type = tk.StringVar(value="")
        mask_prob = tk.StringVar()
        span_mask_prob = tk.StringVar()
        span_mask_len = tk.StringVar()
        reencode = tk.BooleanVar(value=False)

        self.train_vars = {
            "accession": accession,
            "steps": steps,
            "batch_size": batch_size,
            "window_size": window_size,
            "stride": stride,
            "tokenizer": tokenizer,
            "frame_offset": frame_offset,
            "min_orf_aa": min_orf_aa,
            "loss_type": loss_type,
            "mask_prob": mask_prob,
            "span_mask_prob": span_mask_prob,
            "span_mask_len": span_mask_len,
            "reencode": reencode,
        }

        self._labeled_entry(left, "Accession", accession, tooltip="Required.")
        self._labeled_entry(left, "Steps", steps, tooltip="Optional. Integer.")
        self._labeled_entry(left, "Batch size", batch_size, tooltip="Optional. Integer.")
        self._labeled_entry(left, "Window size", window_size, tooltip="Optional. Integer.")
        self._labeled_entry(left, "Stride", stride, tooltip="Optional. Integer.")
        self._labeled_combo(left, "Tokenizer", tokenizer, ["", "base", "codon", "aa"], tooltip="Blank = default.")
        self._labeled_entry(left, "Frame offset", frame_offset, tooltip="Optional. 0/1/2.")
        self._labeled_entry(left, "Min ORF AA", min_orf_aa, tooltip="Optional. AA mode.")
        self._labeled_combo(left, "Loss type", loss_type, ["", "mse", "ce"], tooltip="Blank = default.")
        self._labeled_entry(left, "Mask prob (AA)", mask_prob, tooltip="Optional. Float in [0..1].")
        self._labeled_entry(left, "Span mask prob (AA)", span_mask_prob, tooltip="Optional. Float in [0..1].")
        self._labeled_entry(left, "Span mask len (AA)", span_mask_len, tooltip="Optional. Integer.")
        chk = ttk.Checkbutton(left, text="Re-encode before training", variable=reencode)
        chk.pack(anchor="w", pady=6)
        self._register_lockable(chk)

        bar = self._action_bar(left)
        btn = ttk.Button(bar, text="Train", command=lambda: self._run_command("train_one", "Train accession", self._collect_train_args()))
        btn.pack(side=tk.RIGHT)
        self._register_lockable(btn)

        PresetsPanel(
            right,
            gui=self,
            command_key="train_one",
            title="Presets",
            collect_args=self._collect_train_args,
            load_into_form=lambda a: self._load_args_into_form("train_one", a),
            save_preset_hotkey=lambda: self._save_current_as_preset_for("train_one", self._collect_train_args),
        ).pack(fill=tk.Y, expand=True)
        return frame

    def _build_generate_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)
        outer = ttk.Frame(frame)
        outer.pack(fill=tk.BOTH, expand=True)

        # plasmid row
        row1 = ttk.Frame(outer)
        row1.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        left1 = ttk.LabelFrame(row1, text="Generate plasmid", padding=12)
        left1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right1 = ttk.Frame(row1)
        right1.pack(side=tk.RIGHT, fill=tk.Y)

        p_tokenizer = tk.StringVar(value="")
        p_length_bp = tk.StringVar(value="10000")
        p_num_windows = tk.StringVar()
        p_window_size = tk.StringVar()
        p_name = tk.StringVar(value="perceptrome_plasmid_1")
        p_output = tk.StringVar(value="generated/novel_plasmid.fasta")
        p_seed = tk.StringVar()
        p_latent_scale = tk.StringVar(value="1.0")
        p_temperature = tk.StringVar(value="1.0")
        p_gc_bias = tk.StringVar(value="1.0")

        self.genp_vars = {
            "tokenizer": p_tokenizer,
            "length_bp": p_length_bp,
            "num_windows": p_num_windows,
            "window_size": p_window_size,
            "name": p_name,
            "output": p_output,
            "seed": p_seed,
            "latent_scale": p_latent_scale,
            "temperature": p_temperature,
            "gc_bias": p_gc_bias,
        }

        pf = lambda: self._set_gen_focus("plasmid")
        self._labeled_combo(left1, "Tokenizer", p_tokenizer, ["", "base", "codon", "aa"], tooltip="Blank = default.", on_focus=pf)
        self._labeled_entry(left1, "Length (bp)", p_length_bp, tooltip="Optional. Integer.", on_focus=pf)
        self._labeled_entry(left1, "Num windows", p_num_windows, tooltip="Optional. Integer.", on_focus=pf)
        self._labeled_entry(left1, "Window size", p_window_size, tooltip="Optional. Integer.", on_focus=pf)
        self._labeled_entry(left1, "Name", p_name, tooltip="Optional.", on_focus=pf)
        self._labeled_entry(left1, "Output path", p_output, tooltip="Required.", on_focus=pf)
        self._labeled_entry(left1, "Seed", p_seed, tooltip="Optional. Integer.", on_focus=pf)
        self._labeled_entry(left1, "Latent scale", p_latent_scale, tooltip="Optional. Float.", on_focus=pf)
        self._labeled_entry(left1, "Temperature", p_temperature, tooltip="Optional. Float.", on_focus=pf)
        self._labeled_entry(left1, "GC bias", p_gc_bias, tooltip="Optional. Float.", on_focus=pf)

        bar1 = self._action_bar(left1)
        b1 = ttk.Button(bar1, text="Generate plasmid", command=lambda: self._run_command("gen_plasmid", "Generate plasmid", self._collect_gen_plasmid_args()))
        b1.pack(side=tk.RIGHT)
        self._register_lockable(b1)

        PresetsPanel(
            right1,
            gui=self,
            command_key="gen_plasmid",
            title="Presets",
            collect_args=self._collect_gen_plasmid_args,
            load_into_form=lambda a: self._load_args_into_form("gen_plasmid", a),
            save_preset_hotkey=lambda: self._save_current_as_preset_for("gen_plasmid", self._collect_gen_plasmid_args),
        ).pack(fill=tk.Y, expand=True)

        # protein row
        row2 = ttk.Frame(outer)
        row2.pack(fill=tk.BOTH, expand=True)
        left2 = ttk.LabelFrame(row2, text="Generate protein", padding=12)
        left2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right2 = ttk.Frame(row2)
        right2.pack(side=tk.RIGHT, fill=tk.Y)

        a_length_aa = tk.StringVar(value="600")
        a_num_windows = tk.StringVar()
        a_window_aa = tk.StringVar()
        a_name = tk.StringVar(value="perceptrome_protein_1")
        a_output = tk.StringVar(value="generated/novel_protein.faa")
        a_seed = tk.StringVar()
        a_latent_scale = tk.StringVar(value="1.0")
        a_temperature = tk.StringVar(value="1.0")
        a_reject = tk.BooleanVar(value=False)
        a_reject_tries = tk.StringVar(value="40")
        a_reject_max_run = tk.StringVar(value="10")
        a_reject_max_x_frac = tk.StringVar(value="0.15")

        self.gena_vars = {
            "length_aa": a_length_aa,
            "num_windows": a_num_windows,
            "window_aa": a_window_aa,
            "name": a_name,
            "output": a_output,
            "seed": a_seed,
            "latent_scale": a_latent_scale,
            "temperature": a_temperature,
            "reject": a_reject,
            "reject_tries": a_reject_tries,
            "reject_max_run": a_reject_max_run,
            "reject_max_x_frac": a_reject_max_x_frac,
        }

        af = lambda: self._set_gen_focus("protein")
        self._labeled_entry(left2, "Length (aa)", a_length_aa, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Num windows", a_num_windows, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Window (aa)", a_window_aa, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Name", a_name, tooltip="Optional.", on_focus=af)
        self._labeled_entry(left2, "Output path", a_output, tooltip="Required.", on_focus=af)
        self._labeled_entry(left2, "Seed", a_seed, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Latent scale", a_latent_scale, tooltip="Optional. Float.", on_focus=af)
        self._labeled_entry(left2, "Temperature", a_temperature, tooltip="Optional. Float.", on_focus=af)
        chk = ttk.Checkbutton(left2, text="Enable rejection sampling", variable=a_reject, command=af)
        chk.pack(anchor="w", pady=6)
        self._register_lockable(chk)

        self._labeled_entry(left2, "Reject tries", a_reject_tries, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Reject max run", a_reject_max_run, tooltip="Optional. Integer.", on_focus=af)
        self._labeled_entry(left2, "Reject max X frac", a_reject_max_x_frac, tooltip="Optional. Float in [0..1].", on_focus=af)

        bar2 = self._action_bar(left2)
        b2 = ttk.Button(bar2, text="Generate protein", command=lambda: self._run_command("gen_protein", "Generate protein", self._collect_gen_protein_args()))
        b2.pack(side=tk.RIGHT)
        self._register_lockable(b2)

        PresetsPanel(
            right2,
            gui=self,
            command_key="gen_protein",
            title="Presets",
            collect_args=self._collect_gen_protein_args,
            load_into_form=lambda a: self._load_args_into_form("gen_protein", a),
            save_preset_hotkey=lambda: self._save_current_as_preset_for("gen_protein", self._collect_gen_protein_args),
        ).pack(fill=tk.Y, expand=True)

        return frame

    # ----------------------------
    # History tab (filters + save-as-preset)
    # ----------------------------

    def _build_history_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14)

        top = ttk.Frame(frame)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Search:").pack(side=tk.LEFT)
        e_search = ttk.Entry(top, textvariable=self.history_search_var, width=26)
        e_search.pack(side=tk.LEFT, padx=(6, 10))
        ToolTip(e_search, "Filters by label/command/args JSON substring.")

        ttk.Label(top, text="Command:").pack(side=tk.LEFT)
        cb = ttk.Combobox(
            top,
            textvariable=self.history_cmd_filter_var,
            state="readonly",
            width=20,
            values=["All", "Fetch", "Encode", "Train", "Generate plasmid", "Generate protein"],
        )
        cb.pack(side=tk.LEFT)

        def schedule_refresh(*_args: Any) -> None:
            if self._history_refresh_job:
                try:
                    self.root.after_cancel(self._history_refresh_job)
                except Exception:
                    pass
            self._history_refresh_job = self.root.after(120, self._refresh_history_tab)

        self.history_search_var.trace_add("write", schedule_refresh)
        self.history_cmd_filter_var.trace_add("write", schedule_refresh)

        btns = ttk.Frame(top)
        btns.pack(side=tk.RIGHT)

        b_rerun = ttk.Button(btns, text="Re-run", command=self._history_rerun_selected)
        b_load = ttk.Button(btns, text="Load into form", command=self._history_load_selected)
        b_preset = ttk.Button(btns, text="Save as preset…", command=self._history_save_as_preset)
        b_copy = ttk.Button(btns, text="Copy args JSON", command=self._history_copy_selected)
        b_del = ttk.Button(btns, text="Delete", command=self._history_delete_selected)

        for b in (b_rerun, b_load, b_preset, b_copy, b_del):
            b.pack(side=tk.LEFT, padx=(0, 8))
            self._register_lockable(b)

        self._register_lockable(e_search)
        self._register_lockable(cb)

        ttk.Separator(frame).pack(fill=tk.X, pady=(10, 10))

        self.history_tree = ttk.Treeview(frame, columns=("ts", "cmd", "label"), show="headings", height=12)
        self.history_tree.heading("ts", text="Timestamp")
        self.history_tree.heading("cmd", text="Command")
        self.history_tree.heading("label", text="Label")
        self.history_tree.column("ts", width=160, anchor="w")
        self.history_tree.column("cmd", width=170, anchor="w")
        self.history_tree.column("label", width=560, anchor="w")
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        self._register_lockable(self.history_tree)

        self.history_tree.bind("<<TreeviewSelect>>", lambda _e: self._history_show_details())

        ttk.Label(frame, text="Selected record:").pack(anchor="w", pady=(10, 4))
        self.history_details = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD)
        self.history_details.pack(fill=tk.BOTH, expand=False)
        self.history_details.configure(state="disabled")

        return frame

    def _history_matches_filters(self, h: HistoryEntry) -> bool:
        q = (self.history_search_var.get() or "").strip().lower()
        cmd_filter = self.history_cmd_filter_var.get() or "All"

        if cmd_filter != "All":
            want = {
                "Fetch": "fetch_one",
                "Encode": "encode_one",
                "Train": "train_one",
                "Generate plasmid": "gen_plasmid",
                "Generate protein": "gen_protein",
            }.get(cmd_filter, "")
            if want and h.command_key != want:
                return False

        if not q:
            return True

        blob = (h.label + " " + h.command_key + " " + json_dumps(h.args)).lower()
        return q in blob

    def _refresh_history_tab(self) -> None:
        if not hasattr(self, "history_tree"):
            return
        self.history_tree.delete(*self.history_tree.get_children())
        hist = self.store.list_history()
        for h in hist:
            if not self._history_matches_filters(h):
                continue
            self.history_tree.insert("", tk.END, iid=str(h.id), values=(h.ts, self.command_title(h.command_key), h.label))
        self._history_show_details()

    def _history_selected_id(self) -> Optional[int]:
        if not hasattr(self, "history_tree"):
            return None
        sel = self.history_tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def _history_get_by_id(self, entry_id: int) -> Optional[HistoryEntry]:
        for h in self.store.list_history():
            if h.id == entry_id:
                return h
        return None

    def _history_show_details(self) -> None:
        if not hasattr(self, "history_details"):
            return
        entry_id = self._history_selected_id()
        text = ""
        if entry_id is not None:
            h = self._history_get_by_id(entry_id)
            if h:
                text = json_dumps({"id": h.id, "ts": h.ts, "command_key": h.command_key, "label": h.label, "args": h.args})

        self.history_details.configure(state="normal")
        self.history_details.delete("1.0", tk.END)
        if text:
            self.history_details.insert(tk.END, text)
        self.history_details.configure(state="disabled")

    def _history_rerun_selected(self) -> None:
        entry_id = self._history_selected_id()
        if entry_id is None:
            messagebox.showinfo("Select an entry", "Select a history entry to re-run.")
            return
        h = self._history_get_by_id(entry_id)
        if not h:
            return
        self._run_command(h.command_key, f"Rerun: {h.label}", h.args)

    def _history_load_selected(self) -> None:
        entry_id = self._history_selected_id()
        if entry_id is None:
            messagebox.showinfo("Select an entry", "Select a history entry to load.")
            return
        h = self._history_get_by_id(entry_id)
        if not h:
            return
        self._load_args_into_form(h.command_key, h.args)
        self.runner.log_info(f"Loaded history entry into form: {h.label}")

    def _history_save_as_preset(self) -> None:
        entry_id = self._history_selected_id()
        if entry_id is None:
            messagebox.showinfo("Select an entry", "Select a history entry first.")
            return
        h = self._history_get_by_id(entry_id)
        if not h:
            return
        default_name = h.label.replace(":", "_").replace("/", "_").strip()[:60] or f"preset_{entry_id}"
        name = simpledialog.askstring("Save as preset", "Preset name:", initialvalue=default_name)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        existing = [p.name for p in self.store.list_presets(h.command_key)]
        if name in existing:
            if not messagebox.askyesno("Overwrite preset?", f"A preset named '{name}' already exists.\nOverwrite it?"):
                return
        self.store.add_or_replace_preset(h.command_key, name, dict(h.args))
        self._persist_settings()
        self._refresh_all_preset_panels()
        self.runner.log_info(f"Saved history entry as preset '{name}' for {h.command_key}.")

    def _history_copy_selected(self) -> None:
        entry_id = self._history_selected_id()
        if entry_id is None:
            messagebox.showinfo("Select an entry", "Select a history entry first.")
            return
        h = self._history_get_by_id(entry_id)
        if not h:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(json_dumps(h.args))
            self.runner.log_info("History args copied to clipboard.")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Copy failed", f"Could not copy args:\n{e}")

    def _history_delete_selected(self) -> None:
        entry_id = self._history_selected_id()
        if entry_id is None:
            messagebox.showinfo("Select an entry", "Select a history entry to delete.")
            return
        if not messagebox.askyesno("Delete history entry?", "Delete the selected history entry?"):
            return
        self.store.delete_history_by_id(entry_id)
        self._persist_settings()
        self._refresh_history_tab()

    # ----------------------------
    # Command import status
    # ----------------------------

    def _ensure_commands_loaded(self, *, startup: bool = False) -> bool:
        cmds = load_commands()
        if cmds:
            if not startup:
                self.runner.log_info("Perceptrome commands loaded successfully.")
            return True

        err = get_import_error()
        if err is not None:
            self.log.append(LogItem("ERROR", "Perceptrome import failed — commands are unavailable."))
            self.log.append(LogItem("STDERR", import_help_message(err)))
        else:
            self.log.append(LogItem("ERROR", "Perceptrome commands unavailable (unknown reason)."))
        return False

    # ----------------------------
    # Tools / dialogs
    # ----------------------------

    def log_clear_safe(self) -> None:
        if hasattr(self, "runner") and self.runner.is_running():
            messagebox.showinfo("Busy", "A command is running. Please wait before clearing the log.")
            return
        self.log.clear()

    def _choose_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Perceptrome config YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.config_path.set(path)
            self.runner.log_info(f"Config set to: {path}")
            self._persist_settings()

    def _save_log_dialog(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(self.log.get_all(), encoding="utf-8")
            self.runner.log_info(f"Saved log to: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", f"Could not save log:\n{e}")

    def _copy_log(self) -> None:
        try:
            text = self.log.get_all()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.runner.log_info("Log copied to clipboard.")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Copy failed", f"Could not copy log:\n{e}")

    def _env_check(self) -> None:
        self.log.append(LogItem("START", "START  Environment check"))

        def worker() -> None:
            q = self.runner.q
            q.put(LogItem("INFO", f"Python: {sys.version.splitlines()[0]}"))
            q.put(LogItem("INFO", f"Executable: {sys.executable}"))
            q.put(LogItem("INFO", f"CWD: {os.getcwd()}"))
            q.put(LogItem("INFO", f"Venv: {os.environ.get('VIRTUAL_ENV') or '(none detected)'}"))

            cfg = self.config_path.get().strip()
            if cfg:
                q.put(LogItem("INFO", f"Config path: {cfg}"))
                q.put(LogItem("INFO", f"Config exists: {Path(cfg).exists()}"))
            else:
                q.put(LogItem("ERROR", "Config path is empty."))

            cmds = load_commands()
            if cmds:
                q.put(LogItem("DONE", "DONE   Perceptrome commands import: OK"))
            else:
                err = get_import_error()
                if err:
                    q.put(LogItem("ERROR", "ERROR  Perceptrome commands import: FAILED"))
                    q.put(LogItem("STDERR", import_help_message(err)))
                else:
                    q.put(LogItem("ERROR", "ERROR  Perceptrome commands import: FAILED (unknown)"))

            try:
                import torch  # type: ignore

                q.put(LogItem("DONE", f"DONE   torch import: OK (torch {torch.__version__})"))
                try:
                    cuda = torch.cuda.is_available()
                    q.put(LogItem("INFO", f"torch.cuda.is_available(): {cuda}"))
                    if cuda:
                        q.put(LogItem("INFO", f"CUDA device: {torch.cuda.get_device_name(0)}"))
                except Exception as e:  # noqa: BLE001
                    q.put(LogItem("STDERR", f"CUDA query failed: {e!r}"))
            except Exception as e:  # noqa: BLE001
                q.put(LogItem("ERROR", f"ERROR  torch import failed: {type(e).__name__}: {e}"))
                q.put(LogItem("INFO", "Install CPU torch with:"))
                q.put(LogItem("INFO", "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"))

            q.put(LogItem("DONE", "DONE   Environment check complete"))

        threading.Thread(target=worker, daemon=True).start()

    def _install_torch_cpu(self) -> None:
        if messagebox.askyesno(
            "Install PyTorch (CPU)",
            "This will run pip to install CPU PyTorch in your CURRENT Python environment.\n\nProceed?",
        ) is False:
            return

        if self.runner.is_running():
            messagebox.showinfo("Busy", "A command is running. Please wait before installing.")
            return

        self.log.append(LogItem("START", "START  Installing PyTorch (CPU) via pip"))

        def worker() -> None:
            q = self.runner.q
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
            q.put(LogItem("INFO", "Running: " + " ".join(cmd)))
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert p.stdout is not None
                for line in p.stdout:
                    q.put(LogItem("STDOUT", line.rstrip("\n")))
                rc = p.wait()
                if rc == 0:
                    q.put(LogItem("DONE", "DONE   PyTorch install completed successfully."))
                else:
                    q.put(LogItem("ERROR", f"ERROR  PyTorch install failed (exit code {rc})."))
            except Exception as e:  # noqa: BLE001
                q.put(LogItem("ERROR", f"ERROR  pip install failed: {type(e).__name__}: {e}"))
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                q.put(LogItem("STDERR", tb))

        threading.Thread(target=worker, daemon=True).start()

    def _about(self) -> None:
        messagebox.showinfo(
            "About Perceptrome GUI",
            "Perceptrome GUI\n\n"
            "A lightweight Tkinter/ttk front-end for Perceptrome commands.\n"
            "• Streaming logs\n"
            "• Lazy imports\n"
            "• Presets + History (favorites + hotkeys)\n"
            "• Environment check + Torch install helper\n",
        )

    # ----------------------------
    # Import/export presets + history
    # ----------------------------

    def _refresh_all_preset_panels(self) -> None:
        def walk(w: tk.Widget) -> None:
            for c in w.winfo_children():
                if isinstance(c, PresetsPanel):
                    c.refresh()
                walk(c)

        walk(self.notebook)

    def _export_presets(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export presets to JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = self.store.export_presets_only()
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.runner.log_info(f"Exported presets to: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export failed", f"Could not export presets:\n{e}")

    def _import_presets(self) -> None:
        path = filedialog.askopenfilename(
            title="Import presets JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Import failed", f"Could not read JSON:\n{e}")
            return

        replace = messagebox.askyesno(
            "Import presets",
            "Replace existing presets?\n\nYes = replace\nNo = merge (replace-by-name within each command)",
        )
        try:
            self.store.import_presets_only(payload, replace=bool(replace))
            self._persist_settings()
            self.runner.log_info(f"Imported presets from: {path}")
            self._refresh_all_preset_panels()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Import failed", f"Could not import presets:\n{e}")

    def _export_history(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export history to JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = {"version": 2, "history": self.settings.get("history", [])}
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.runner.log_info(f"Exported history to: {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export failed", f"Could not export history:\n{e}")

    def _clear_history(self) -> None:
        if not messagebox.askyesno("Clear history?", "Clear all history entries?"):
            return
        self.store.clear_history()
        self._persist_settings()
        self._refresh_history_tab()

    # ----------------------------
    # Shutdown / persistence
    # ----------------------------

    def _on_close(self) -> None:
        self._persist_settings()
        self.root.destroy()


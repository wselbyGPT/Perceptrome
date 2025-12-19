from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from .store import Preset
from .tk_compat import messagebox, simpledialog, tk, ttk
from .utils import fmt_epoch, json_dumps

if TYPE_CHECKING:
    from .app import PerceptromeGUI


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add=True)
        widget.bind("<Leave>", self._hide, add=True)

    def _show(self, _event: tk.Event) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            self.tip,
            text=self.text,
            justify=tk.LEFT,
            background="#111827",
            foreground="#e5e7eb",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=6,
            font=("TkDefaultFont", 9),
        )
        lbl.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


class PresetsPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        gui: "PerceptromeGUI",
        command_key: str,
        title: str,
        collect_args: Callable[[], Dict[str, Any]],
        load_into_form: Callable[[Dict[str, Any]], None],
        save_preset_hotkey: Callable[[], None],
    ) -> None:
        super().__init__(parent, text=title, padding=10)
        self.gui = gui
        self.command_key = command_key
        self.collect_args = collect_args
        self.load_into_form = load_into_form
        self.save_preset_hotkey = save_preset_hotkey

        self.filter_var = tk.StringVar(value="")
        self.favs_only_var = tk.BooleanVar(value=False)

        fr = ttk.Frame(self)
        fr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(fr, text="Filter:").pack(side=tk.LEFT)
        e = ttk.Entry(fr, textvariable=self.filter_var, width=18)
        e.pack(side=tk.LEFT, padx=(6, 0))
        b_clear = ttk.Button(fr, text="×", width=3, command=self._clear_filter)
        b_clear.pack(side=tk.LEFT, padx=(6, 10))
        ToolTip(e, "Type to filter preset names.")

        chk = ttk.Checkbutton(fr, text="★ only", variable=self.favs_only_var, command=self.refresh)
        chk.pack(side=tk.LEFT)
        ToolTip(chk, "Show favorites only.")

        self.filter_var.trace_add("write", lambda *_: self.refresh())

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("fav", "name", "last", "uses"), show="headings", height=12)
        self.tree.heading("fav", text="★")
        self.tree.heading("name", text="Name")
        self.tree.heading("last", text="Last used")
        self.tree.heading("uses", text="Uses")
        self.tree.column("fav", width=36, anchor="center")
        self.tree.column("name", width=200, anchor="w")
        self.tree.column("last", width=120, anchor="w")
        self.tree.column("uses", width=50, anchor="e")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.bind("<Double-1>", lambda _e: self._run_selected())
        self.tree.bind("<Button-1>", self._maybe_toggle_star, add=True)
        self.tree.bind("<space>", lambda _e: self._toggle_star_selected())
        self.tree.bind("<f>", lambda _e: self._toggle_star_selected())

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, pady=(10, 0))

        b_save = ttk.Button(btns, text="Save current…", command=self._save_current)
        b_star = ttk.Button(btns, text="Toggle ★", command=self._toggle_star_selected)
        b_run = ttk.Button(btns, text="Run", command=self._run_selected)
        b_load = ttk.Button(btns, text="Load", command=self._load_selected)
        b_dup = ttk.Button(btns, text="Duplicate…", command=self._duplicate_selected)
        b_rename = ttk.Button(btns, text="Rename…", command=self._rename_selected)
        b_del = ttk.Button(btns, text="Delete", command=self._delete_selected)
        b_copy = ttk.Button(btns, text="Copy args", command=self._copy_selected_args)

        b_save.grid(row=0, column=0, sticky="we")
        b_run.grid(row=0, column=1, sticky="we", padx=(8, 0))
        b_star.grid(row=1, column=0, sticky="we", pady=(8, 0))
        b_load.grid(row=1, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        b_dup.grid(row=2, column=0, sticky="we", pady=(8, 0))
        b_rename.grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        b_del.grid(row=3, column=0, sticky="we", pady=(8, 0))
        b_copy.grid(row=3, column=1, sticky="we", padx=(8, 0), pady=(8, 0))

        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        for b in (b_save, b_star, b_run, b_load, b_dup, b_rename, b_del, b_copy, b_clear, chk):
            self.gui._register_lockable(b)
        self.gui._register_lockable(e)
        self.gui._register_lockable(self.tree)

        self.tree.bind("<Control-Return>", lambda _e: self.save_preset_hotkey(), add=True)
        e.bind("<Control-Return>", lambda _e: self.save_preset_hotkey(), add=True)

        self.refresh()

    def _clear_filter(self) -> None:
        self.filter_var.set("")

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        presets = self.gui.store.list_presets(self.command_key)
        q = (self.filter_var.get() or "").strip().lower()
        favs_only = bool(self.favs_only_var.get())
        for p in presets:
            if favs_only and not p.favorite:
                continue
            if q and q not in p.name.lower():
                continue
            star = "★" if p.favorite else "☆"
            self.tree.insert(
                "",
                tk.END,
                iid=p.name,
                values=(star, p.name, fmt_epoch(p.last_used), str(p.use_count)),
            )

    def _selected_name(self) -> Optional[str]:
        sel = self.tree.selection()
        if not sel:
            return None
        return str(sel[0])

    def _maybe_toggle_star(self, event: tk.Event) -> None:
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        if col != "#1":
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self.tree.selection_set(row)
        self._toggle_star_selected()

    def _toggle_star_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        new_state = self.gui.store.toggle_preset_favorite(self.command_key, name)
        if new_state is None:
            return
        self.gui._persist_settings()
        self.refresh()
        self.gui.runner.log_info(f"Preset '{name}' favorite = {new_state}")

    def _save_current(self) -> None:
        self.save_preset_hotkey()

    def _run_selected(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset to run.")
            return
        preset = next((p for p in self.gui.store.list_presets(self.command_key) if p.name == name), None)
        if not preset:
            return
        self.gui.store.touch_preset_used(self.command_key, name)
        self.gui._persist_settings()
        self.refresh()
        self.gui._run_command(self.command_key, f"{self.gui.command_title(self.command_key)} (preset: {name})", preset.args)

    def _load_selected(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset to load.")
            return
        preset = next((p for p in self.gui.store.list_presets(self.command_key) if p.name == name), None)
        if not preset:
            return
        self.load_into_form(preset.args)
        self.gui.runner.log_info(f"Loaded preset '{name}' into the form.")

    def _duplicate_selected(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset to duplicate.")
            return
        preset = next((p for p in self.gui.store.list_presets(self.command_key) if p.name == name), None)
        if not preset:
            return
        new = simpledialog.askstring("Duplicate preset", "New name:", initialvalue=f"{name}_copy")
        if not new:
            return
        new = new.strip()
        if not new:
            return
        existing = [p.name for p in self.gui.store.list_presets(self.command_key)]
        if new in existing:
            messagebox.showerror("Name exists", f"A preset named '{new}' already exists.")
            return
        self.gui.store.add_or_replace_preset(self.command_key, new, dict(preset.args))
        self.gui._persist_settings()
        self.refresh()
        self.gui.runner.log_info(f"Duplicated preset '{name}' -> '{new}'.")

    def _rename_selected(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset to rename.")
            return
        new = simpledialog.askstring("Rename preset", f"New name for '{name}':")
        if not new:
            return
        new = new.strip()
        if not new:
            return
        existing = [p.name for p in self.gui.store.list_presets(self.command_key)]
        if new in existing:
            messagebox.showerror("Name exists", f"A preset named '{new}' already exists.")
            return
        self.gui.store.rename_preset(self.command_key, name, new)
        self.gui._persist_settings()
        self.refresh()
        self.gui.runner.log_info(f"Renamed preset '{name}' -> '{new}'.")

    def _delete_selected(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset to delete.")
            return
        if not messagebox.askyesno("Delete preset?", f"Delete preset '{name}'?"):
            return
        self.gui.store.delete_preset(self.command_key, name)
        self.gui._persist_settings()
        self.refresh()
        self.gui.runner.log_info(f"Deleted preset '{name}'.")

    def _copy_selected_args(self) -> None:
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Select a preset", "Choose a preset first.")
            return
        preset = next((p for p in self.gui.store.list_presets(self.command_key) if p.name == name), None)
        if not preset:
            return
        try:
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(json_dumps(preset.args))
            self.gui.runner.log_info("Preset args copied to clipboard.")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Copy failed", f"Could not copy args:\n{e}")

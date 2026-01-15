from __future__ import annotations

from .tk_compat import tk, ttk


def apply_theme(root: tk.Tk, dark: bool) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    if dark:
        bg = "#0b0d14"
        panel = "#0f111a"
        fg = "#e6e6e6"
        border = "#1f2335"
        accent = "#3b82f6"
        select_bg = "#1f3c73"
        select_fg = "#f8fafc"
    else:
        bg = "#f7f7fb"
        panel = "#ffffff"
        fg = "#111111"
        border = "#d0d4dd"
        accent = "#2563eb"
        select_bg = "#dbeafe"
        select_fg = "#0b1f4a"

    root.configure(background=bg)

    style.configure("TFrame", background=bg)
    style.configure("Panel.TFrame", background=panel)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("Panel.TLabel", background=panel, foreground=fg)
    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 8))
    style.configure("TLabelframe", background=bg, foreground=fg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)

    style.configure("TEntry", fieldbackground=panel, foreground=fg)
    style.configure("TCombobox", fieldbackground=panel, foreground=fg)
    style.map("TCombobox", fieldbackground=[("readonly", panel)], foreground=[("readonly", fg)])

    style.configure("TButton", padding=(12, 6), font=("TkDefaultFont", 10))
    style.configure("Accent.TButton", padding=(12, 6), font=("TkDefaultFont", 10), foreground="#ffffff", background=accent)
    style.map(
        "Accent.TButton",
        background=[("pressed", accent), ("active", accent)],
        foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
    )

    style.configure(
        "Themed.Treeview",
        rowheight=24,
        fieldbackground=panel,
        background=panel,
        foreground=fg,
        font=("TkDefaultFont", 10),
    )
    style.configure(
        "Themed.Treeview.Heading",
        background=bg,
        foreground=fg,
        font=("TkDefaultFont", 10, "bold"),
    )
    style.map(
        "Themed.Treeview",
        background=[("selected", select_bg)],
        foreground=[("selected", select_fg)],
    )

    style.configure("Status.TLabel", background=bg, foreground=fg)
    style.configure("TProgressbar", troughcolor=border)

    return style

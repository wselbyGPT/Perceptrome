from __future__ import annotations

from .tk_compat import tk, ttk

CURRENT_PALETTE = {
    "bg": "#0b0d14",
    "panel": "#0f111a",
    "fg": "#e6e6e6",
    "border": "#1f2335",
    "tooltip_bg": "#111827",
    "tooltip_fg": "#e5e7eb",
}


def get_theme_palette() -> dict[str, str]:
    return dict(CURRENT_PALETTE)


def apply_theme(root: tk.Tk, dark: bool) -> ttk.Style:
    global CURRENT_PALETTE
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
        tooltip_bg = "#111827"
        tooltip_fg = "#e5e7eb"
    else:
        bg = "#f7f7fb"
        panel = "#ffffff"
        fg = "#111111"
        border = "#d0d4dd"
        tooltip_bg = "#f9fafb"
        tooltip_fg = "#111827"

    CURRENT_PALETTE = {
        "bg": bg,
        "panel": panel,
        "fg": fg,
        "border": border,
        "tooltip_bg": tooltip_bg,
        "tooltip_fg": tooltip_fg,
    }

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

    style.configure("Status.TLabel", background=bg, foreground=fg)
    style.configure("TProgressbar", troughcolor=border)

    return style

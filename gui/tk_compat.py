from __future__ import annotations

# Tkinter imports (fail fast with a helpful message)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        "Tkinter is not available in this Python. On Ubuntu/Debian:\n"
        "  sudo apt-get update && sudo apt-get install -y python3-tk\n"
        f"\nOriginal error: {e!r}"
    ) from e

__all__ = ["tk", "ttk", "filedialog", "messagebox", "scrolledtext", "simpledialog"]

from __future__ import annotations

import io
import queue
from dataclasses import dataclass
from typing import Any

from .tk_compat import scrolledtext, tk
from .utils import ts


@dataclass
class LogItem:
    kind: str  # "INFO" | "STDOUT" | "STDERR" | "START" | "DONE" | "ERROR"
    text: str


class QueueWriter(io.TextIOBase):
    """File-like writer that streams text into a queue, line-by-line."""

    def __init__(self, q: "queue.Queue[LogItem]", kind: str) -> None:
        self.q = q
        self.kind = kind
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() != "":
                self.q.put(LogItem(self.kind, line))
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.q.put(LogItem(self.kind, self._buf.rstrip("\n")))
        self._buf = ""


class LogPanel:
    def __init__(self, parent: tk.Widget, *, dark: bool) -> None:
        self.text = scrolledtext.ScrolledText(parent, height=14, wrap=tk.WORD)
        self.text.configure(state="disabled")
        self.dark = dark
        self._configure_colors()

    def _configure_colors(self) -> None:
        if self.dark:
            bg = "#0f111a"
            fg = "#e6e6e6"
            ins = "#1f2335"
        else:
            bg = "#ffffff"
            fg = "#111111"
            ins = "#e8e8ee"

        self.text.configure(
            background=bg,
            foreground=fg,
            insertbackground=fg,
            selectbackground=ins,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=ins,
            highlightcolor=ins,
        )

        self.text.tag_configure("INFO", spacing3=2)
        self.text.tag_configure("STDOUT", spacing3=2)
        self.text.tag_configure("STDERR", spacing3=2, foreground="#ff6b6b" if self.dark else "#b00020")
        self.text.tag_configure("START", spacing3=4, font=("TkDefaultFont", 10, "bold"))
        self.text.tag_configure(
            "DONE",
            spacing3=4,
            font=("TkDefaultFont", 10, "bold"),
            foreground="#6ee7b7" if self.dark else "#0a7a3c",
        )
        self.text.tag_configure(
            "ERROR",
            spacing3=4,
            font=("TkDefaultFont", 10, "bold"),
            foreground="#ff6b6b" if self.dark else "#b00020",
        )

    def pack(self, **kwargs: Any) -> None:
        self.text.pack(**kwargs)

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.configure(state="disabled")

    def get_all(self) -> str:
        return self.text.get("1.0", tk.END)

    def append(self, item: LogItem) -> None:
        self.text.configure(state="normal")
        prefix = f"[{ts()}] "
        tag = item.kind if item.kind in ("INFO", "STDOUT", "STDERR", "START", "DONE", "ERROR") else "INFO"
        self.text.insert(tk.END, prefix + item.text + "\n", tag)
        self.text.see(tk.END)
        self.text.configure(state="disabled")

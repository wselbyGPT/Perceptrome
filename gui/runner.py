from __future__ import annotations

import argparse
import queue
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

from .commands import CommandFunc
from .logging_panel import LogItem, LogPanel, QueueWriter
from .tk_compat import tk, ttk


class CommandRunner:
    def __init__(self, root: tk.Tk, log: LogPanel, status_var: tk.StringVar, spinner: ttk.Progressbar) -> None:
        self.root = root
        self.log = log
        self.status_var = status_var
        self.spinner = spinner

        self.q: "queue.Queue[LogItem]" = queue.Queue()
        self._running_lock = threading.Lock()
        self._running = False
        self._controls_to_lock: list[tk.Widget] = []

        self._pump_logs()

    def set_controls_to_lock(self, widgets: list[tk.Widget]) -> None:
        self._controls_to_lock = widgets

    def is_running(self) -> bool:
        with self._running_lock:
            return self._running

    def _set_running(self, value: bool) -> None:
        with self._running_lock:
            self._running = value

    def _lock_controls(self, locked: bool) -> None:
        for w in self._controls_to_lock:
            try:
                w.configure(state=("disabled" if locked else "normal"))
            except Exception:
                pass

    def _busy(self, label: str, busy: bool) -> None:
        def apply() -> None:
            self.status_var.set(label)
            if busy:
                try:
                    self.spinner.start(12)
                except Exception:
                    pass
            else:
                try:
                    self.spinner.stop()
                except Exception:
                    pass

        self.root.after(0, apply)

    def _pump_logs(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                self.log.append(item)
        except queue.Empty:
            pass
        self.root.after(60, self._pump_logs)

    def log_info(self, text: str) -> None:
        self.q.put(LogItem("INFO", text))

    def run(self, label: str, func: CommandFunc, args: Dict[str, Any]) -> None:
        if self.is_running():
            self.q.put(LogItem("ERROR", "Another command is already running. Please wait for it to finish."))
            return

        self._set_running(True)
        self._lock_controls(True)
        self._busy(f"Running: {label}", True)
        self.q.put(LogItem("START", f"START  {label}"))
        self.q.put(LogItem("INFO", f"Args: {args}"))

        def worker() -> None:
            out = QueueWriter(self.q, "STDOUT")
            err = QueueWriter(self.q, "STDERR")
            exit_code: Optional[int] = None
            failed: Optional[BaseException] = None

            try:
                with redirect_stdout(out), redirect_stderr(err):
                    exit_code = func(argparse.Namespace(**args))
            except BaseException as e:  # noqa: BLE001
                failed = e
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                self.q.put(LogItem("ERROR", f"ERROR  {label}: {type(e).__name__}: {e}"))
                self.q.put(LogItem("STDERR", tb))
            finally:
                try:
                    out.flush()
                    err.flush()
                except Exception:
                    pass

                if failed is None:
                    self.q.put(LogItem("DONE", f"DONE   {label} (exit code {exit_code})"))

                self._set_running(False)
                self.root.after(0, lambda: self._lock_controls(False))
                self._busy("Ready", False)

        threading.Thread(target=worker, daemon=True).start()

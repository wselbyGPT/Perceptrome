import re
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess, QTimer


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


@dataclass
class ProcState:
    proc: Optional[QProcess] = None
    buf: str = ""
    kill_timer: Optional[QTimer] = None
    busy: bool = False
    stop_requested: bool = False
    started: bool = False


class ProcessRunner(QObject):
    """
    Minimal QProcess runner:
      - runs command via: bash -lc "<cmd>"
      - merges stdout+stderr
      - streams output lines to callbacks
      - supports terminate -> kill escalation
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = ProcState()

    def is_running(self) -> bool:
        return self.state.proc is not None and self.state.proc.state() != QProcess.NotRunning

    def was_stop_requested(self) -> bool:
        return self.state.stop_requested

    def has_started(self) -> bool:
        return self.state.started

    def start(
        self,
        cmd: str,
        workdir: str,
        on_started: Callable[[], None],
        on_line: Callable[[str], None],
        on_finished: Callable[[int, str], None],
        on_error: Callable[[str], None],
    ) -> bool:
        cmd = (cmd or "").strip()
        if not cmd:
            on_error("Empty command.")
            return False
        if self.is_running():
            on_error("Process already running.")
            return False

        p = QProcess(self)
        p.setWorkingDirectory(workdir if workdir else ".")
        p.setProcessChannelMode(QProcess.MergedChannels)

        self.state.stop_requested = False
        self.state.started = False
        self.state.buf = ""

        def _started():
            self.state.started = True
            on_started()

        p.started.connect(_started)

        def _read_ready():
            data = bytes(p.readAllStandardOutput()).decode(errors="replace")
            if not data:
                return
            data = strip_ansi(data)
            self._feed(data, on_line)

        p.readyReadStandardOutput.connect(_read_ready)

        def _finished(exit_code: int, exit_status):
            _read_ready()
            if self.state.stop_requested:
                status = "stopped"
            else:
                status = "ok" if exit_code == 0 else f"exit_code={exit_code}"
            on_finished(exit_code, status)
            self.state.proc = None

        p.finished.connect(_finished)

        def _error(err):
            msg = p.errorString() or str(err)
            on_error(msg)

        p.errorOccurred.connect(_error)

        p.start("bash", ["-lc", cmd])

        self.state.proc = p
        return True

    def stop(self, on_line: Optional[Callable[[str], None]]) -> None:
        if not self.is_running():
            return
        p = self.state.proc
        if p is None:
            return

        self.state.stop_requested = True
        if on_line:
            on_line("[gui] Sending terminate()...\n")
        p.terminate()

        kt = QTimer(self)
        kt.setSingleShot(True)

        def _kill():
            if self.is_running():
                if on_line:
                    on_line("[gui] terminate() timed out; sending kill()...\n")
                p.kill()

        kt.timeout.connect(_kill)
        kt.start(2000)
        self.state.kill_timer = kt

    def _feed(self, text: str, on_line: Callable[[str], None]) -> None:
        self.state.buf += text
        if not self.state.buf:
            return

        parts = self.state.buf.splitlines(keepends=True)
        if parts and not parts[-1].endswith("\n"):
            self.state.buf = parts[-1]
            parts = parts[:-1]
        else:
            self.state.buf = ""

        for chunk in parts:
            on_line(chunk)

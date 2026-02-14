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
    stop_requested: bool = False
    kill_sent: bool = False
    last_error: Optional[str] = None


def _process_error_to_text(err: QProcess.ProcessError) -> str:
    names = {
        QProcess.FailedToStart: "failed_to_start",
        QProcess.Crashed: "crashed",
        QProcess.Timedout: "timed_out",
        QProcess.WriteError: "write_error",
        QProcess.ReadError: "read_error",
        QProcess.UnknownError: "unknown_error",
    }
    return names.get(err, f"process_error_{int(err)}")


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
        self.state.kill_sent = False
        self.state.last_error = None

        # wire signals
        p.started.connect(on_started)

        def _read_ready():
            data = bytes(p.readAllStandardOutput()).decode(errors="replace")
            if not data:
                return
            data = strip_ansi(data)
            self._feed(data, on_line)

        p.readyReadStandardOutput.connect(_read_ready)

        def _finished(exit_code: int, exit_status):
            # flush remaining
            _read_ready()
            if self.state.kill_timer is not None:
                self.state.kill_timer.stop()
                self.state.kill_timer.deleteLater()
                self.state.kill_timer = None

            if self.state.stop_requested:
                status = "timed_out" if self.state.kill_sent else "cancelled"
            elif exit_status == QProcess.CrashExit:
                status = "failed"
            elif self.state.last_error:
                status = f"failed ({self.state.last_error})"
            else:
                status = "ok" if exit_code == 0 else f"failed (exit_code={exit_code})"

            on_finished(exit_code, status)

            self.state.proc = None
            self.state.buf = ""
            self.state.stop_requested = False
            self.state.kill_sent = False
            self.state.last_error = None

        p.finished.connect(_finished)

        def _error(err: QProcess.ProcessError):
            msg = _process_error_to_text(err)
            self.state.last_error = msg
            on_error(msg)

        p.errorOccurred.connect(_error)

        # start through bash so user can paste normal CLI commands
        p.start("bash", ["-lc", cmd])

        self.state.proc = p
        self.state.buf = ""
        return True

    def stop(self, on_line: Optional[Callable[[str], None]] = None) -> bool:
        if not self.is_running():
            return False
        p = self.state.proc
        if p is None:
            return False

        if self.state.stop_requested:
            return False

        self.state.stop_requested = True

        if on_line:
            on_line("[gui] Sending terminate()...\n")
        p.terminate()

        if self.state.kill_timer is not None:
            self.state.kill_timer.stop()
            self.state.kill_timer.deleteLater()

        # escalate to kill if it doesn't stop quickly
        kt = QTimer(self)
        kt.setSingleShot(True)

        def _kill():
            if self.is_running():
                self.state.kill_sent = True
                if on_line:
                    on_line("[gui] terminate() timed out; sending kill()...\n")
                p.kill()

        kt.timeout.connect(_kill)
        kt.start(2000)
        self.state.kill_timer = kt
        return True

    def _feed(self, text: str, on_line: Callable[[str], None]) -> None:
        # Keep line boundaries neat; preserve partial line across reads
        self.state.buf += text
        if not self.state.buf:
            return

        # split into lines, keep trailing partial
        parts = self.state.buf.splitlines(keepends=True)
        if parts and not parts[-1].endswith("\n"):
            self.state.buf = parts[-1]
            parts = parts[:-1]
        else:
            self.state.buf = ""

        for chunk in parts:
            on_line(chunk)

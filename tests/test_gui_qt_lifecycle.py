import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from gui_qt.app import PerceptromeQt
    QT_AVAILABLE = True
except ModuleNotFoundError:
    QT_AVAILABLE = False


def _history_rows(win):
    rows = []
    for idx in range(win.history_table.rowCount()):
        action = win.history_table.item(idx, 1).text()
        details = win.history_table.item(idx, 2).text()
        rows.append((action, details))
    return rows


def _wait_until(app, predicate, timeout_ms: int = 7000):
    end = time.time() + (timeout_ms / 1000.0)
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return predicate()


@unittest.skipUnless(QT_AVAILABLE, "PySide6 is required for GUI lifecycle tests")
class TestGuiQtLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.win = PerceptromeQt()

    def tearDown(self):
        self.win.shutdown()
        self.app.processEvents()
        self.win.close()
        self.app.processEvents()

    def test_train_stop_marks_stopping_and_cancelled(self):
        self.win.train_cmd.setText('python -c "import time; print(\"run\", flush=True); time.sleep(5)"')
        self.win._train_start()

        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_train_stop.isEnabled(), timeout_ms=2500))
        self.win._train_stop()
        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_train_start.isEnabled(), timeout_ms=5000))

        history = _history_rows(self.win)
        self.assertTrue(any(action == "train_stopping" for action, _ in history))
        self.assertTrue(any(action == "train_cancelled" for action, _ in history))

    def test_train_stop_timeout_marks_timed_out(self):
        self.win.train_cmd.setText("bash -lc 'trap \"\" TERM; echo run; sleep 10'")
        self.win._train_start()

        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_train_stop.isEnabled(), timeout_ms=2500))
        self.win._train_stop()
        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_train_start.isEnabled(), timeout_ms=7000))

        history = _history_rows(self.win)
        self.assertTrue(any(action == "train_timed_out" for action, _ in history))

    def test_train_failure_marks_failed(self):
        self.win.train_cmd.setText("python -c 'import sys; sys.exit(3)'")
        self.win._train_start()

        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_train_start.isEnabled(), timeout_ms=3000))
        history = _history_rows(self.win)
        self.assertTrue(any(action == "train_failed" for action, _ in history))

    def test_shutdown_while_running_requests_stop_without_exceptions(self):
        self.win.gen_cmd.setText('python -c "import time; print(\"run\", flush=True); time.sleep(5)"')
        self.win._gen_start()

        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_gen_stop.isEnabled(), timeout_ms=2500))
        self.win.shutdown()
        self.assertTrue(_wait_until(self.app, lambda: self.win.btn_generate.isEnabled(), timeout_ms=6000))

        history = _history_rows(self.win)
        self.assertTrue(any(action == "generate_stopping" for action, _ in history))
        self.assertTrue(any(action in {"generate_cancelled", "generate_timed_out"} for action, _ in history))


if __name__ == "__main__":
    unittest.main()

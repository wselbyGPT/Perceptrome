import math
import os
import re
import shlex
import sys
from datetime import datetime

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont, QTextCursor, QPainter, QPen, QColor, QPdfWriter
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QComboBox
)

from .theme import apply_dark_mode
from .runner import ProcessRunner
from perceptrome.config import load_full_config, extract_configs
from perceptrome.io_utils import ensure_dirs
from perceptrome.ncbi_fetch import fetch_fasta


PCT_RE = re.compile(r"(\d{1,3})\s*%")
EPOCH_RE = re.compile(r"(?:epoch|Epoch)\s*[:=]?\s*(\d+)\s*/\s*(\d+)")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PerceptromeQt(QMainWindow):
    TRAIN_TOKENIZERS = ["base", "codon", "aa"]
    GENERATE_TOKENIZERS = ["base", "codon"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("perceptrome")
        self.resize(980, 680)

        self.settings = QSettings("Perceptrome", "GuiQt")

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        self.setCentralWidget(container)

        logo = QLabel("perceptrome")
        logo.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        f = QFont()
        f.setPointSize(28)
        f.setBold(True)
        logo.setFont(f)
        logo.setStyleSheet("QLabel { padding: 10px 0px; letter-spacing: 1px; }")

        self.tabs = QTabWidget()
        root.addWidget(logo)
        root.addWidget(self.tabs, 1)

        # runners
        self.train_runner = ProcessRunner(self)
        self.gen_runner = ProcessRunner(self)

        self._closing = False

        # Build tabs
        self.tab_home = self._build_home_tab()
        self.tab_train = self._build_train_tab()
        self.tab_generate = self._build_generate_tab()
        self.tab_view = self._build_view_tab()
        self.tab_history = self._build_history_tab()

        self.tabs.addTab(self.tab_home, "Home / Config")
        self.tabs.addTab(self.tab_train, "Train")
        self.tabs.addTab(self.tab_generate, "Generate")
        self.tabs.addTab(self.tab_view, "View")
        self.tabs.addTab(self.tab_history, "History")

        self._load_config()

    # -------------------------
    # Tabs
    # -------------------------
    def _build_home_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()

        self.cfg_project_dir = QLineEdit()
        self.cfg_stream_yaml = QLineEdit()
        self.cfg_dataset_list = QLineEdit()

        self.cfg_epochs = QSpinBox()
        self.cfg_epochs.setRange(1, 1_000_000)
        self.cfg_epochs.setValue(10)

        self.cfg_batch = QSpinBox()
        self.cfg_batch.setRange(1, 1_000_000)
        self.cfg_batch.setValue(256)

        self.cfg_lr = QDoubleSpinBox()
        self.cfg_lr.setDecimals(6)
        self.cfg_lr.setRange(0.0, 10.0)
        self.cfg_lr.setSingleStep(0.0005)
        self.cfg_lr.setValue(0.001)

        form.addRow("Project dir:", self.cfg_project_dir)
        form.addRow("stream_config.yaml:", self.cfg_stream_yaml)
        form.addRow("Dataset list file:", self.cfg_dataset_list)
        form.addRow("Epochs:", self.cfg_epochs)
        form.addRow("Batch size:", self.cfg_batch)
        form.addRow("Learning rate:", self.cfg_lr)

        btn_row = QHBoxLayout()
        self.btn_save_cfg = QPushButton("Save config")
        self.btn_go_train = QPushButton("Go to Train tab")
        btn_row.addWidget(self.btn_save_cfg)
        btn_row.addWidget(self.btn_go_train)
        btn_row.addStretch(1)

        self.cfg_status = QLabel("Config not saved yet.")

        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.cfg_status)
        layout.addStretch(1)

        self.btn_save_cfg.clicked.connect(self._save_config)
        self.btn_go_train.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tab_train))
        return w

    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()
        self.train_dataset = QLineEdit()
        self.train_dataset.setPlaceholderText("config/plasmids_10.txt")

        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 1_000_000)
        self.train_epochs.setValue(10)

        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 1_000_000)
        self.train_batch.setValue(256)

        self.train_tokenizer = QComboBox()
        self.train_tokenizer.addItems(self.TRAIN_TOKENIZERS)

        self.train_raw_toggle = QCheckBox("Advanced: use raw command")

        self.train_cmd = QLineEdit()
        self.train_cmd.setPlaceholderText('Example: python stream_train.py stream --catalog config/plasmids_10.txt')
        self.train_cmd.setMinimumHeight(32)
        self.train_cmd.setStyleSheet("QLineEdit { font-family: monospace; }")
        self.train_cmd.hide()

        self.train_error = QLabel("")
        self.train_error.setStyleSheet("QLabel { color: #ff7272; }")

        form.addRow("Dataset / catalog:", self.train_dataset)
        form.addRow("Epochs:", self.train_epochs)
        form.addRow("Batch size:", self.train_batch)
        form.addRow("Tokenizer:", self.train_tokenizer)
        form.addRow("", self.train_raw_toggle)
        form.addRow("Raw command:", self.train_cmd)
        form.addRow("", self.train_error)

        btn_row = QHBoxLayout()
        self.btn_train_help = QPushButton("Help")
        self.btn_train_start = QPushButton("Start")
        self.btn_train_stop = QPushButton("Stop")
        self.btn_train_stop.setEnabled(False)
        btn_row.addWidget(self.btn_train_help)
        btn_row.addWidget(self.btn_train_start)
        btn_row.addWidget(self.btn_train_stop)
        btn_row.addStretch(1)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("Live training output will appear here...")

        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.train_progress)
        layout.addWidget(self.train_log, 1)

        self.btn_train_help.clicked.connect(self._train_help)
        self.btn_train_start.clicked.connect(self._train_start)
        self.btn_train_stop.clicked.connect(self._train_stop)
        self.train_raw_toggle.toggled.connect(lambda checked: self.train_cmd.setVisible(checked))
        return w

    def _build_generate_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()
        self.gen_tokenizer = QComboBox()
        self.gen_tokenizer.addItems(self.GENERATE_TOKENIZERS)

        self.gen_output = QLineEdit()
        self.gen_output.setPlaceholderText("generated/novel_plasmid.fasta")

        self.gen_raw_toggle = QCheckBox("Advanced: use raw command")

        self.gen_cmd = QLineEdit()
        self.gen_cmd.setPlaceholderText('Example: python stream_train.py generate-plasmid --output generated/novel_plasmid.fasta')
        self.gen_cmd.setMinimumHeight(32)
        self.gen_cmd.setStyleSheet("QLineEdit { font-family: monospace; }")
        self.gen_cmd.hide()

        self.gen_error = QLabel("")
        self.gen_error.setStyleSheet("QLabel { color: #ff7272; }")

        form.addRow("Tokenizer:", self.gen_tokenizer)
        form.addRow("Output path:", self.gen_output)
        form.addRow("", self.gen_raw_toggle)
        form.addRow("Raw command:", self.gen_cmd)
        form.addRow("", self.gen_error)

        btn_row = QHBoxLayout()
        self.btn_gen_help = QPushButton("Help")
        self.btn_generate = QPushButton("Start")
        self.btn_gen_stop = QPushButton("Stop")
        self.btn_gen_stop.setEnabled(False)
        btn_row.addWidget(self.btn_gen_help)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_gen_stop)
        btn_row.addStretch(1)

        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 100)
        self.gen_progress.setValue(0)

        self.gen_out = QPlainTextEdit()
        self.gen_out.setReadOnly(True)
        self.gen_out.setPlaceholderText("Live generate output will appear here...")

        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.gen_progress)
        layout.addWidget(self.gen_out, 1)

        self.btn_gen_help.clicked.connect(self._gen_help)
        self.btn_generate.clicked.connect(self._gen_start)
        self.btn_gen_stop.clicked.connect(self._gen_stop)
        self.gen_raw_toggle.toggled.connect(lambda checked: self.gen_cmd.setVisible(checked))
        return w

    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Action", "Details"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)

        btn_row = QHBoxLayout()
        self.btn_clear_history = QPushButton("Clear history")
        btn_row.addWidget(self.btn_clear_history)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)
        layout.addWidget(self.history_table, 1)

        self.btn_clear_history.clicked.connect(self._clear_history)
        return w

    def _build_view_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        source_group = QGroupBox("Genome source")
        source_layout = QFormLayout(source_group)
        self.view_accession = QLineEdit()
        self.view_accession.setPlaceholderText("Example: NC_000913.3")
        self.view_fasta_path = QLineEdit()
        self.view_fasta_path.setPlaceholderText("generated/novel_plasmid.fasta")
        source_layout.addRow("Genome accession:", self.view_accession)
        source_layout.addRow("FASTA path:", self.view_fasta_path)

        output_group = QGroupBox("PDF output")
        output_layout = QFormLayout(output_group)
        self.view_pdf_path = QLineEdit()
        self.view_pdf_path.setPlaceholderText("generated/circular_genome.pdf")
        self.view_title = QLineEdit()
        self.view_title.setPlaceholderText("Optional title override")
        output_layout.addRow("Output PDF:", self.view_pdf_path)
        output_layout.addRow("Title:", self.view_title)

        btn_row = QHBoxLayout()
        self.btn_view_generate = QPushButton("Generate PDF")
        self.btn_view_open = QPushButton("Open PDF")
        btn_row.addWidget(self.btn_view_generate)
        btn_row.addWidget(self.btn_view_open)
        btn_row.addStretch(1)

        self.view_log = QPlainTextEdit()
        self.view_log.setReadOnly(True)
        self.view_log.setPlaceholderText("PDF generation status will appear here...")

        self.view_pdf_doc = QPdfDocument(self)
        self.view_pdf = QPdfView()
        self.view_pdf.setDocument(self.view_pdf_doc)
        self.view_pdf.setZoomMode(QPdfView.ZoomMode.FitInView)

        layout.addWidget(source_group)
        layout.addWidget(output_group)
        layout.addLayout(btn_row)
        layout.addWidget(self.view_log)
        layout.addWidget(self.view_pdf, 1)

        self.btn_view_generate.clicked.connect(self._view_generate_pdf)
        self.btn_view_open.clicked.connect(self._view_open_pdf)
        return w

    # -------------------------
    # Config persistence
    # -------------------------
    def _save_config(self):
        self.settings.setValue("project_dir", self.cfg_project_dir.text().strip())
        self.settings.setValue("stream_yaml", self.cfg_stream_yaml.text().strip())
        self.settings.setValue("dataset_list", self.cfg_dataset_list.text().strip())
        self.settings.setValue("epochs", int(self.cfg_epochs.value()))
        self.settings.setValue("batch", int(self.cfg_batch.value()))
        self.settings.setValue("lr", float(self.cfg_lr.value()))
        self.settings.setValue("train_dataset", self.train_dataset.text().strip())
        self.settings.setValue("train_epochs", int(self.train_epochs.value()))
        self.settings.setValue("train_batch", int(self.train_batch.value()))
        self.settings.setValue("train_tokenizer", self.train_tokenizer.currentText())
        self.settings.setValue("train_raw_enabled", bool(self.train_raw_toggle.isChecked()))
        self.settings.setValue("train_cmd", self.train_cmd.text().strip())

        self.settings.setValue("gen_tokenizer", self.gen_tokenizer.currentText())
        self.settings.setValue("gen_output", self.gen_output.text().strip())
        self.settings.setValue("gen_raw_enabled", bool(self.gen_raw_toggle.isChecked()))
        self.settings.setValue("gen_cmd", self.gen_cmd.text().strip())
        self.settings.setValue("view_accession", self.view_accession.text().strip())
        self.settings.setValue("view_fasta_path", self.view_fasta_path.text().strip())
        self.settings.setValue("view_pdf_path", self.view_pdf_path.text().strip())
        self.settings.setValue("view_title", self.view_title.text().strip())
        self.settings.sync()

        self.cfg_status.setText(f"Saved at {now_str()}")
        self._add_history("save_config", "ok")

    def _load_config(self):
        self.cfg_project_dir.setText(self.settings.value("project_dir", "."))
        self.cfg_stream_yaml.setText(self.settings.value("stream_yaml", "stream_config.yaml"))
        self.cfg_dataset_list.setText(self.settings.value("dataset_list", "config/plasmids_10.txt"))
        self.cfg_epochs.setValue(int(self.settings.value("epochs", 10)))
        self.cfg_batch.setValue(int(self.settings.value("batch", 256)))
        self.cfg_lr.setValue(float(self.settings.value("lr", 0.001)))

        self.train_dataset.setText(self.settings.value("train_dataset", self.settings.value("dataset_list", "config/plasmids_10.txt")))
        self.train_epochs.setValue(int(self.settings.value("train_epochs", self.settings.value("epochs", 10))))
        self.train_batch.setValue(int(self.settings.value("train_batch", self.settings.value("batch", 256))))
        self._set_combo_text(self.train_tokenizer, self.settings.value("train_tokenizer", "base"), default="base")
        self.train_raw_toggle.setChecked(self._bool_setting("train_raw_enabled", False))
        self.train_cmd.setText(self.settings.value("train_cmd", "python stream_train.py --help"))

        self._set_combo_text(self.gen_tokenizer, self.settings.value("gen_tokenizer", "base"), default="base")
        self.gen_output.setText(self.settings.value("gen_output", "generated/novel_plasmid.fasta"))
        self.gen_raw_toggle.setChecked(self._bool_setting("gen_raw_enabled", False))
        self.gen_cmd.setText(self.settings.value("gen_cmd", "python stream_train.py --help"))
        self._load_last_success_presets()
        self.view_accession.setText(self.settings.value("view_accession", ""))
        self.view_fasta_path.setText(self.settings.value("view_fasta_path", "generated/novel_plasmid.fasta"))
        self.view_pdf_path.setText(self.settings.value("view_pdf_path", "generated/circular_genome.pdf"))
        self.view_title.setText(self.settings.value("view_title", ""))

        self.cfg_status.setText("Loaded saved config (if any).")

    # -------------------------
    # Helpers
    # -------------------------
    def _workdir(self) -> str:
        wd = self.cfg_project_dir.text().strip()
        return wd if wd else "."

    def _bool_setting(self, key: str, default: bool) -> bool:
        raw = self.settings.value(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _set_combo_text(self, combo: QComboBox, value: str, default: str = ""):
        target = str(value or "").strip().lower()
        idx = combo.findText(target)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        fallback = combo.findText(default)
        combo.setCurrentIndex(fallback if fallback >= 0 else 0)

    def _build_train_command(self) -> str:
        dataset = self.train_dataset.text().strip()
        tokenizer = self.train_tokenizer.currentText().strip().lower()
        parts = [
            "python", "stream_train.py", "stream",
            "--catalog", dataset,
            "--max-epochs", str(self.train_epochs.value()),
            "--batch-size", str(self.train_batch.value()),
            "--tokenizer", tokenizer,
        ]
        return " ".join(shlex.quote(p) for p in parts)

    def _build_gen_command(self) -> str:
        tokenizer = self.gen_tokenizer.currentText().strip().lower()
        output_path = self.gen_output.text().strip()
        parts = [
            "python", "stream_train.py", "generate-plasmid",
            "--tokenizer", tokenizer,
            "--output", output_path,
        ]
        return " ".join(shlex.quote(p) for p in parts)

    def _validate_train_inputs(self) -> str:
        if self.train_raw_toggle.isChecked():
            if not self.train_cmd.text().strip():
                return "Raw command is required when advanced mode is enabled."
            return ""

        if not self.train_dataset.text().strip():
            return "Dataset / catalog path is required."
        return ""

    def _validate_gen_inputs(self) -> str:
        if self.gen_raw_toggle.isChecked():
            if not self.gen_cmd.text().strip():
                return "Raw command is required when advanced mode is enabled."
            return ""

        if not self.gen_output.text().strip():
            return "Output path is required."
        return ""

    def _save_last_success_train_preset(self):
        self.settings.setValue("last_success/train_dataset", self.train_dataset.text().strip())
        self.settings.setValue("last_success/train_epochs", int(self.train_epochs.value()))
        self.settings.setValue("last_success/train_batch", int(self.train_batch.value()))
        self.settings.setValue("last_success/train_tokenizer", self.train_tokenizer.currentText().strip().lower())
        self.settings.sync()

    def _save_last_success_gen_preset(self):
        self.settings.setValue("last_success/gen_tokenizer", self.gen_tokenizer.currentText().strip().lower())
        self.settings.setValue("last_success/gen_output", self.gen_output.text().strip())
        self.settings.sync()

    def _load_last_success_presets(self):
        train_dataset = self.settings.value("last_success/train_dataset", "")
        if train_dataset:
            self.train_dataset.setText(str(train_dataset))
            self.train_epochs.setValue(int(self.settings.value("last_success/train_epochs", self.train_epochs.value())))
            self.train_batch.setValue(int(self.settings.value("last_success/train_batch", self.train_batch.value())))
            self._set_combo_text(self.train_tokenizer, self.settings.value("last_success/train_tokenizer", self.train_tokenizer.currentText()), default="base")

        gen_output = self.settings.value("last_success/gen_output", "")
        if gen_output:
            self.gen_output.setText(str(gen_output))
            self._set_combo_text(self.gen_tokenizer, self.settings.value("last_success/gen_tokenizer", self.gen_tokenizer.currentText()), default="base")

    def _append_log(self, box: QPlainTextEdit, text: str, max_lines: int = 5000):
        if not isValid(box) or self._closing:
            return
        box.moveCursor(QTextCursor.End)
        box.insertPlainText(text)
        # trim occasionally
        if box.blockCount() > max_lines:
            cur = box.textCursor()
            cur.movePosition(cur.Start)
            for _ in range(box.blockCount() - max_lines):
                cur.select(cur.LineUnderCursor)
                cur.removeSelectedText()
                cur.deleteChar()  # newline
            box.setTextCursor(cur)
        box.verticalScrollBar().setValue(box.verticalScrollBar().maximum())

    def _set_busy(self, bar: QProgressBar, busy: bool):
        if not isValid(bar) or self._closing:
            return
        if busy:
            bar.setRange(0, 0)  # indeterminate
        else:
            bar.setRange(0, 100)

    def _maybe_update_progress(self, bar: QProgressBar, line: str):
        m = PCT_RE.search(line)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                if bar.minimum() == 0 and bar.maximum() == 0:
                    self._set_busy(bar, False)
                bar.setValue(v)
                return

        m2 = EPOCH_RE.search(line)
        if m2:
            cur = int(m2.group(1))
            tot = int(m2.group(2))
            if tot > 0:
                v = int((cur / tot) * 100)
                v = max(0, min(100, v))
                if bar.minimum() == 0 and bar.maximum() == 0:
                    self._set_busy(bar, False)
                bar.setValue(v)

    # -------------------------
    # Train actions (real QProcess)
    # -------------------------
    def _train_help(self):
        self.train_raw_toggle.setChecked(True)
        self.train_cmd.setText("python stream_train.py stream --help")

    def _train_start(self):
        err = self._validate_train_inputs()
        self.train_error.setText(err)
        if err:
            return

        cmd = self.train_cmd.text().strip() if self.train_raw_toggle.isChecked() else self._build_train_command()
        wd = self._workdir()

        self.train_log.clear()
        self._append_log(self.train_log, f"[{now_str()}] [train] start\n$ {cmd}\n\n")
        self._set_busy(self.train_progress, True)
        self.train_progress.setValue(0)

        def on_started():
            self.btn_train_start.setEnabled(False)
            self.btn_train_stop.setEnabled(True)
            self._add_history("train_start", cmd)

        def on_line(s: str):
            self._append_log(self.train_log, s)
            self._maybe_update_progress(self.train_progress, s)

        def on_finished(exit_code: int, status: str):
            self._set_busy(self.train_progress, False)
            self.train_progress.setValue(100 if exit_code == 0 else 0)
            self._append_log(self.train_log, f"\n[{now_str()}] [train] finished ({status})\n")
            self.btn_train_start.setEnabled(True)
            self.btn_train_stop.setEnabled(False)
            if exit_code == 0:
                self._save_last_success_train_preset()
            self._add_history("train_done", status)

        def on_error(msg: str):
            self._set_busy(self.train_progress, False)
            self._append_log(self.train_log, f"\n[{now_str()}] [train] ERROR: {msg}\n")
            self.btn_train_start.setEnabled(True)
            self.btn_train_stop.setEnabled(False)
            self._add_history("train_error", msg)

        ok = self.train_runner.start(cmd, wd, on_started, on_line, on_finished, on_error)
        if not ok:
            on_error("Failed to start process.")

    def _train_stop(self):
        self.train_runner.stop(lambda s: self._append_log(self.train_log, s))
        self._add_history("train_stop", "requested")
        self.btn_train_stop.setEnabled(False)

    # -------------------------
    # Generate actions (real QProcess)
    # -------------------------
    def _gen_help(self):
        self.gen_raw_toggle.setChecked(True)
        self.gen_cmd.setText("python stream_train.py generate-plasmid --help")

    def _gen_start(self):
        err = self._validate_gen_inputs()
        self.gen_error.setText(err)
        if err:
            return

        cmd = self.gen_cmd.text().strip() if self.gen_raw_toggle.isChecked() else self._build_gen_command()
        wd = self._workdir()

        self.gen_out.clear()
        self._append_log(self.gen_out, f"[{now_str()}] [generate] start\n$ {cmd}\n\n")
        self._set_busy(self.gen_progress, True)
        self.gen_progress.setValue(0)

        def on_started():
            self.btn_generate.setEnabled(False)
            self.btn_gen_stop.setEnabled(True)
            self._add_history("generate_start", cmd)

        def on_line(s: str):
            self._append_log(self.gen_out, s)
            self._maybe_update_progress(self.gen_progress, s)

        def on_finished(exit_code: int, status: str):
            self._set_busy(self.gen_progress, False)
            self.gen_progress.setValue(100 if exit_code == 0 else 0)
            self._append_log(self.gen_out, f"\n[{now_str()}] [generate] finished ({status})\n")
            self.btn_generate.setEnabled(True)
            self.btn_gen_stop.setEnabled(False)
            if exit_code == 0:
                self._save_last_success_gen_preset()
            self._add_history("generate_done", status)

        def on_error(msg: str):
            self._set_busy(self.gen_progress, False)
            self._append_log(self.gen_out, f"\n[{now_str()}] [generate] ERROR: {msg}\n")
            self.btn_generate.setEnabled(True)
            self.btn_gen_stop.setEnabled(False)
            self._add_history("generate_error", msg)

        ok = self.gen_runner.start(cmd, wd, on_started, on_line, on_finished, on_error)
        if not ok:
            on_error("Failed to start process.")

    def _gen_stop(self):
        self.gen_runner.stop(lambda s: self._append_log(self.gen_out, s))
        self._add_history("generate_stop", "requested")
        self.btn_gen_stop.setEnabled(False)

    # -------------------------
    # View actions
    # -------------------------
    def _read_fasta_sequence(self, path: str) -> str:
        seq_parts = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    continue
                seq_parts.append(line)
        return "".join(seq_parts)

    def _resolve_genome_sequence(self) -> tuple[str, str]:
        accession = self.view_accession.text().strip()
        fasta_path = self.view_fasta_path.text().strip()
        if accession:
            cfg_path = self.cfg_stream_yaml.text().strip() or "stream_config.yaml"
            cfg = load_full_config(cfg_path)
            ncbi_cfg, _, io_cfg = extract_configs(cfg)
            ensure_dirs(io_cfg)
            fasta_path = fetch_fasta(accession, io_cfg, ncbi_cfg, force=False)
            seq = self._read_fasta_sequence(fasta_path)
            return seq, f"accession {accession}"
        if fasta_path:
            seq = self._read_fasta_sequence(fasta_path)
            return seq, f"fasta {fasta_path}"
        raise ValueError("Provide a genome accession or FASTA path.")

    def _write_circular_pdf(self, seq: str, output_path: str, title: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = QPdfWriter(output_path)
        writer.setResolution(150)

        painter = QPainter(writer)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            rect = painter.viewport()
            side = int(min(rect.width(), rect.height()) * 0.72)
            cx = rect.width() // 2
            cy = rect.height() // 2
            radius = side // 2

            seq_len = len(seq)
            gc = 0.0
            if seq_len:
                gc = (seq.count("G") + seq.count("C")) / float(seq_len)

            painter.setPen(QPen(QColor("#68d5ff"), 4))
            painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

            painter.setPen(QPen(QColor("#3b3f46"), 1))
            for i in range(12):
                angle = (i / 12.0) * 2.0 * math.pi
                x_outer = cx + int(radius * 1.02 * math.cos(angle))
                y_outer = cy + int(radius * 1.02 * math.sin(angle))
                x_inner = cx + int(radius * 0.92 * math.cos(angle))
                y_inner = cy + int(radius * 0.92 * math.sin(angle))
                painter.drawLine(x_inner, y_inner, x_outer, y_outer)

            painter.setPen(QPen(QColor("#d9d9d9"), 2))
            painter.drawText(cx - radius, cy - radius - 40, radius * 2, 30, Qt.AlignCenter, title or "Circular genome view")

            info = f"Length: {seq_len:,} bp    GC: {gc * 100:.2f}%"
            painter.setPen(QPen(QColor("#a8b0b8"), 1))
            painter.drawText(cx - radius, cy + radius + 12, radius * 2, 30, Qt.AlignCenter, info)
        finally:
            painter.end()

    def _view_generate_pdf(self):
        self.view_log.clear()
        try:
            seq, source = self._resolve_genome_sequence()
            output_path = self.view_pdf_path.text().strip() or "generated/circular_genome.pdf"
            title = self.view_title.text().strip() or f"Circular genome ({source})"
            self._append_log(self.view_log, f"[{now_str()}] Generating PDF from {source}\n")
            self._write_circular_pdf(seq, output_path, title)
            self._append_log(self.view_log, f"[{now_str()}] Saved PDF -> {output_path}\n")
            self._add_history("view_pdf", output_path)
            self._load_pdf(output_path)
        except Exception as exc:
            self._append_log(self.view_log, f"[{now_str()}] ERROR: {exc}\n")

    def _load_pdf(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        status = self.view_pdf_doc.load(path)
        if status != QPdfDocument.Error.None_:
            raise RuntimeError(f"Failed to load PDF (status {status})")

    def _view_open_pdf(self):
        path = self.view_pdf_path.text().strip()
        if not path:
            self._append_log(self.view_log, f"[{now_str()}] ERROR: No PDF path set.\n")
            return
        try:
            self._load_pdf(path)
            self._append_log(self.view_log, f"[{now_str()}] Loaded PDF -> {path}\n")
        except Exception as exc:
            self._append_log(self.view_log, f"[{now_str()}] ERROR: {exc}\n")

    # -------------------------
    # History
    # -------------------------
    def _add_history(self, action: str, details: str):
        r = self.history_table.rowCount()
        self.history_table.insertRow(r)
        self.history_table.setItem(r, 0, QTableWidgetItem(now_str()))
        self.history_table.setItem(r, 1, QTableWidgetItem(action))
        self.history_table.setItem(r, 2, QTableWidgetItem(details))
        self.history_table.scrollToBottom()

    def _clear_history(self):
        self.history_table.setRowCount(0)
        self._add_history("history_cleared", "")



    def shutdown(self):
        # Called on app quit/close to prevent "QProcess destroyed while running"
        if self._closing:
            return
        self._closing = True
        try:
            self.train_runner.stop(None)
        except Exception:
            pass
        try:
            self.gen_runner.stop(None)
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
def main():
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    win = PerceptromeQt()
    app.aboutToQuit.connect(win.shutdown)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

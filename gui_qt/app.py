import json
import math
import os
import re
import shlex
import sys
import shutil
from glob import glob
from datetime import datetime

from PySide6.QtCore import Qt, QSettings, QEvent, QPoint
from PySide6.QtGui import QFont, QTextCursor, QPainter, QPen, QColor, QPdfWriter, QPolygon, QFontMetrics
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox,
    QMessageBox, QCheckBox
)

from .theme import apply_dark_mode
from .runner import ProcessRunner
from perceptrome.config import load_full_config, extract_configs
from perceptrome.io_utils import ensure_dirs
from perceptrome.io_utils import read_catalog, write_catalog, select_unique_accessions
from perceptrome.ncbi_fetch import fetch_fasta, fetch_genbank
from perceptrome.encoding.parse import parse_genbank_dna
from perceptrome.encoding.genbank_features import parse_cds_features_from_genbank, CDSFeature
from perceptrome.jobs import JobSpec
from .job_api import build_generate_plasmid_spec, build_stream_spec


PCT_RE = re.compile(r"(\d{1,3})\s*%")
EPOCH_RE = re.compile(r"(?:epoch|Epoch)\s*[:=]?\s*(\d+)\s*/\s*(\d+)")
MODEL_TYPES = ["mlp", "transformer", "ssm"]
DATASET_SOURCE_OPTIONS = [
    ("Plasmids", "accessions/plasmid_accessions.txt"),
    ("Viruses", "accessions/viruses_accessions.txt"),
    ("Eukaryotes", "accessions/eukaryote_accessions.txt"),
    ("Bacteria", "accessions/bacteria_accessions.txt"),
    ("Archaea", "accessions/archaea_accessions.txt"),
    ("Metagenomes", "accessions/metagenome_accessions.txt"),
    ("Chloroplast", "accessions/chloroplast_accessions.txt"),
    ("Mitochondrion", "accessions/mitochondrion_accessions.txt"),
    ("Synthetic construct", "accessions/synthetic_construct_accessions.txt"),
    ("Viroid", "accessions/viroid_accessions.txt"),
]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PerceptromeQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("perceptrome")
        self.resize(980, 680)

        self.settings = QSettings("Perceptrome", "GuiQt")

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        self.setCentralWidget(container)

        logo = QLabel("perceptrome")
        logo.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        logo.setObjectName("Logo")
        f = QFont()
        f.setPointSize(30)
        f.setBold(True)
        logo.setFont(f)
        logo.setStyleSheet("QLabel { padding: 12px 0px; letter-spacing: 1.5px; }")

        self.tabs = QTabWidget()
        root.addWidget(logo)
        root.addWidget(self.tabs, 1)

        # runners
        self.train_runner = ProcessRunner(self)
        self.gen_runner = ProcessRunner(self)

        self._closing = False
        self.pdf_window = GenomePdfWindow(self.settings, self)

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

        settings_group = QGroupBox("Project settings")
        settings_form = QFormLayout(settings_group)

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

        settings_form.addRow("Project dir:", self.cfg_project_dir)
        settings_form.addRow("stream_config.yaml:", self.cfg_stream_yaml)
        settings_form.addRow("Dataset list file:", self.cfg_dataset_list)
        settings_form.addRow("Epochs:", self.cfg_epochs)
        settings_form.addRow("Batch size:", self.cfg_batch)
        settings_form.addRow("Learning rate:", self.cfg_lr)

        dataset_group = QGroupBox("Custom dataset builder")
        dataset_layout = QVBoxLayout(dataset_group)

        build_form = QFormLayout()
        self.ds_source = QComboBox()
        for label, rel_path in DATASET_SOURCE_OPTIONS:
            self.ds_source.addItem(label, rel_path)
        self.ds_count = QSpinBox()
        self.ds_count.setRange(1, 1_000_000)
        self.ds_count.setValue(100)
        self.ds_shuffle = QCheckBox("Shuffle inside each category")
        self.ds_shuffle.setChecked(True)
        build_form.addRow("Category:", self.ds_source)
        build_form.addRow("Count:", self.ds_count)
        build_form.addRow("", self.ds_shuffle)

        ds_btn_row = QHBoxLayout()
        self.btn_ds_add = QPushButton("Add category quota")
        self.btn_ds_remove = QPushButton("Remove selected")
        ds_btn_row.addWidget(self.btn_ds_add)
        ds_btn_row.addWidget(self.btn_ds_remove)
        ds_btn_row.addStretch(1)

        self.ds_table = QTableWidget(0, 3)
        self.ds_table.setHorizontalHeaderLabels(["Category", "Source", "Count"])
        self.ds_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.ds_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.ds_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ds_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ds_table.setEditTriggers(QTableWidget.NoEditTriggers)

        output_form = QFormLayout()
        self.ds_output_path = QLineEdit("config/custom_dataset.txt")
        output_form.addRow("Output catalog:", self.ds_output_path)

        ds_create_row = QHBoxLayout()
        self.btn_ds_create = QPushButton("Create dataset list")
        self.ds_status = QLabel("Define one or more category quotas and create a list.")
        ds_create_row.addWidget(self.btn_ds_create)
        ds_create_row.addStretch(1)

        dataset_layout.addLayout(build_form)
        dataset_layout.addLayout(ds_btn_row)
        dataset_layout.addWidget(self.ds_table)
        dataset_layout.addLayout(output_form)
        dataset_layout.addLayout(ds_create_row)
        dataset_layout.addWidget(self.ds_status)

        btn_row = QHBoxLayout()
        self.btn_save_cfg = QPushButton("Save config")
        self.btn_go_train = QPushButton("Go to Train tab")
        btn_row.addWidget(self.btn_save_cfg)
        btn_row.addWidget(self.btn_go_train)
        btn_row.addStretch(1)

        self.cfg_status = QLabel("Config not saved yet.")

        layout.addWidget(settings_group)
        layout.addWidget(dataset_group, 1)
        layout.addLayout(btn_row)
        layout.addWidget(self.cfg_status)

        self.btn_save_cfg.clicked.connect(self._save_config)
        self.btn_go_train.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tab_train))
        self.btn_ds_add.clicked.connect(self._dataset_add_quota)
        self.btn_ds_remove.clicked.connect(self._dataset_remove_selected)
        self.btn_ds_create.clicked.connect(self._dataset_create_catalog)
        return w

    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        train_group = QGroupBox("Training setup")
        train_form = QFormLayout(train_group)
        self.train_model_type = QComboBox()
        self.train_model_type.addItems(MODEL_TYPES)
        self.train_model_type.setToolTip("Select the neural network backbone for stream training")
        train_form.addRow("Neural network:", self.train_model_type)

        self.train_cmd = QLineEdit()
        self.train_cmd.setPlaceholderText("perceptrome --config config/stream_config.yaml stream --catalog config/custom_dataset.txt --model-type mlp")
        self.train_cmd.setMinimumHeight(36)
        self.train_cmd.setStyleSheet("QLineEdit { font-family: monospace; }")
        train_form.addRow("Training command:", self.train_cmd)

        btn_row = QHBoxLayout()
        self.btn_train_help = QPushButton("Help")
        self.btn_train_rebuild = QPushButton("Build command")
        self.btn_train_start = QPushButton("Start")
        self.btn_train_stop = QPushButton("Stop")
        self.btn_train_stop.setEnabled(False)
        btn_row.addWidget(self.btn_train_help)
        btn_row.addWidget(self.btn_train_rebuild)
        btn_row.addWidget(self.btn_train_start)
        btn_row.addWidget(self.btn_train_stop)
        btn_row.addStretch(1)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("Live training output will appear here...")

        layout.addWidget(train_group)
        layout.addLayout(btn_row)
        layout.addWidget(self.train_progress)
        layout.addWidget(self.train_log, 1)

        self.btn_train_help.clicked.connect(self._train_help)
        self.btn_train_rebuild.clicked.connect(self._refresh_train_command)
        self.btn_train_start.clicked.connect(self._train_start)
        self.btn_train_stop.clicked.connect(self._train_stop)
        self.train_model_type.currentTextChanged.connect(self._refresh_train_command)
        return w

    def _build_generate_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gen_group = QGroupBox("Generation setup")
        gen_form = QFormLayout(gen_group)
        self.gen_model_picker = QComboBox()
        self.gen_model_picker.setToolTip("Select a trained checkpoint to use for generation")
        gen_form.addRow("Trained model:", self.gen_model_picker)

        self.gen_cmd = QLineEdit()
        self.gen_cmd.setPlaceholderText("perceptrome --config config/stream_config.yaml generate-plasmid --length-bp 10000")
        self.gen_cmd.setMinimumHeight(36)
        self.gen_cmd.setStyleSheet("QLineEdit { font-family: monospace; }")
        gen_form.addRow("Generate command:", self.gen_cmd)

        btn_row = QHBoxLayout()
        self.btn_gen_help = QPushButton("Help")
        self.btn_gen_refresh_models = QPushButton("Refresh models")
        self.btn_gen_rebuild = QPushButton("Build command")
        self.btn_generate = QPushButton("Start")
        self.btn_gen_stop = QPushButton("Stop")
        self.btn_gen_stop.setEnabled(False)
        btn_row.addWidget(self.btn_gen_help)
        btn_row.addWidget(self.btn_gen_refresh_models)
        btn_row.addWidget(self.btn_gen_rebuild)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_gen_stop)
        btn_row.addStretch(1)

        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 100)
        self.gen_progress.setValue(0)

        self.gen_out = QPlainTextEdit()
        self.gen_out.setReadOnly(True)
        self.gen_out.setPlaceholderText("Live generate output will appear here...")

        layout.addWidget(gen_group)
        layout.addLayout(btn_row)
        layout.addWidget(self.gen_progress)
        layout.addWidget(self.gen_out, 1)

        self.btn_gen_help.clicked.connect(self._gen_help)
        self.btn_gen_refresh_models.clicked.connect(self._refresh_trained_model_list)
        self.btn_gen_rebuild.clicked.connect(self._refresh_generate_command)
        self.btn_generate.clicked.connect(self._gen_start)
        self.btn_gen_stop.clicked.connect(self._gen_stop)
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
        self.view_render_mode = QComboBox()
        self.view_render_mode.addItems(["Circular", "Linear"])
        self.view_pdf_path = QLineEdit()
        self.view_pdf_path.setPlaceholderText("generated/circular_genome.pdf")
        self.view_title = QLineEdit()
        self.view_title.setPlaceholderText("Optional title override")
        output_layout.addRow("Render mode:", self.view_render_mode)
        output_layout.addRow("Output PDF:", self.view_pdf_path)
        output_layout.addRow("Title:", self.view_title)

        btn_row = QHBoxLayout()
        self.btn_view_generate = QPushButton("Generate PDF")
        self.btn_view_open = QPushButton("Open PDF Window")
        btn_row.addWidget(self.btn_view_generate)
        btn_row.addWidget(self.btn_view_open)
        btn_row.addStretch(1)

        self.view_log = QPlainTextEdit()
        self.view_log.setReadOnly(True)
        self.view_log.setPlaceholderText("PDF generation status will appear here...")

        layout.addWidget(source_group)
        layout.addWidget(output_group)
        layout.addLayout(btn_row)
        layout.addWidget(self.view_log, 1)

        self.btn_view_generate.clicked.connect(self._view_generate_pdf)
        self.btn_view_open.clicked.connect(self._view_open_pdf)
        self.view_render_mode.currentTextChanged.connect(self._view_on_render_mode_changed)
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
        self.settings.setValue("train_cmd", self.train_cmd.text().strip())
        self.settings.setValue("train_model_type", self.train_model_type.currentText().strip())
        self.settings.setValue("gen_cmd", self.gen_cmd.text().strip())
        self.settings.setValue("gen_model", self.gen_model_picker.currentData())
        self.settings.setValue("dataset_builder_output", self.ds_output_path.text().strip())
        self.settings.setValue("dataset_builder_shuffle", bool(self.ds_shuffle.isChecked()))
        self.settings.setValue("view_accession", self.view_accession.text().strip())
        self.settings.setValue("view_fasta_path", self.view_fasta_path.text().strip())
        self.settings.setValue("view_render_mode", self.view_render_mode.currentText().strip())
        self.settings.setValue("view_pdf_path", self.view_pdf_path.text().strip())
        self.settings.setValue("view_title", self.view_title.text().strip())
        self.pdf_window.save_window_settings()
        self.settings.sync()

        self.cfg_status.setText(f"Saved at {now_str()}")
        self._add_history("save_config", "ok")

    def _load_config(self):
        self.cfg_project_dir.setText(self.settings.value("project_dir", "."))
        self.cfg_stream_yaml.setText(self.settings.value("stream_yaml", "config/stream_config.yaml"))
        self.cfg_dataset_list.setText(self.settings.value("dataset_list", "config/plasmids_10.txt"))
        self.cfg_epochs.setValue(int(self.settings.value("epochs", 10)))
        self.cfg_batch.setValue(int(self.settings.value("batch", 256)))
        self.cfg_lr.setValue(float(self.settings.value("lr", 0.001)))

        model_type = self.settings.value("train_model_type", "mlp")
        idx = self.train_model_type.findText(str(model_type))
        self.train_model_type.setCurrentIndex(idx if idx >= 0 else 0)

        self.train_cmd.setText(self.settings.value("train_cmd", ""))
        self.gen_cmd.setText(self.settings.value("gen_cmd", ""))
        self.ds_output_path.setText(self.settings.value("dataset_builder_output", "config/custom_dataset.txt"))
        self.ds_shuffle.setChecked(bool(self.settings.value("dataset_builder_shuffle", True, type=bool)))

        self.view_accession.setText(self.settings.value("view_accession", ""))
        self.view_fasta_path.setText(self.settings.value("view_fasta_path", "generated/novel_plasmid.fasta"))
        render_mode = self.settings.value("view_render_mode", "Circular")
        mode_idx = self.view_render_mode.findText(render_mode)
        self.view_render_mode.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        default_pdf = self._default_pdf_output_for_mode(self.view_render_mode.currentText())
        self.view_pdf_path.setText(self.settings.value("view_pdf_path", default_pdf))
        self.view_title.setText(self.settings.value("view_title", ""))
        self.pdf_window.load_window_settings()
        if self.settings.contains("pdf_last_opened_path") and not self.view_pdf_path.text().strip():
            self.view_pdf_path.setText(self.settings.value("pdf_last_opened_path", ""))

        self._refresh_train_command()
        self._refresh_trained_model_list()
        saved_model = self.settings.value("gen_model", "")
        if saved_model:
            model_idx = self.gen_model_picker.findData(saved_model)
            if model_idx >= 0:
                self.gen_model_picker.setCurrentIndex(model_idx)
        self._refresh_generate_command()

        self.cfg_status.setText("Loaded saved config (if any).")

    def _workdir(self) -> str:
        wd = self.cfg_project_dir.text().strip()
        return wd if wd else "."

    def _refresh_train_command(self):
        self.train_cmd.setText(self._job_spec_command(self._build_train_job_spec()))

    def _refresh_generate_command(self):
        self.gen_cmd.setText(self._job_spec_command(self._build_generate_job_spec()))

    def _dataset_add_quota(self):
        src = self.ds_source.currentData()
        if not src:
            return
        row = self.ds_table.rowCount()
        self.ds_table.insertRow(row)
        self.ds_table.setItem(row, 0, QTableWidgetItem(self.ds_source.currentText()))
        self.ds_table.setItem(row, 1, QTableWidgetItem(str(src)))
        self.ds_table.setItem(row, 2, QTableWidgetItem(str(int(self.ds_count.value()))))

    def _dataset_remove_selected(self):
        rows = sorted({i.row() for i in self.ds_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.ds_table.removeRow(r)

    def _dataset_create_catalog(self):
        if self.ds_table.rowCount() == 0:
            QMessageBox.warning(self, "No dataset quotas", "Add at least one category quota first.")
            return

        category_quotas = []
        category_candidates = {}
        wd = self._workdir()

        try:
            for row in range(self.ds_table.rowCount()):
                name_item = self.ds_table.item(row, 0)
                source_item = self.ds_table.item(row, 1)
                count_item = self.ds_table.item(row, 2)
                if name_item is None or source_item is None or count_item is None:
                    continue
                name = name_item.text().strip().lower().replace(" ", "_")
                rel_source = source_item.text().strip()
                source = rel_source if os.path.isabs(rel_source) else os.path.join(wd, rel_source)
                count = int(count_item.text().strip())
                accessions = read_catalog(source)
                category_quotas.append((name, count))
                category_candidates[name] = accessions

            selected = select_unique_accessions(
                category_quotas,
                category_candidates,
                shuffle_within_category=bool(self.ds_shuffle.isChecked()),
            )
            output = self.ds_output_path.text().strip() or "config/custom_dataset.txt"
            output_path = output if os.path.isabs(output) else os.path.join(wd, output)
            write_catalog(
                output_path,
                selected,
                header=[
                    f"Generated at {now_str()}",
                    f"Quotas: {', '.join([f'{c}:{q}' for c, q in category_quotas])}",
                ],
            )
            rel = os.path.relpath(output_path, wd)
            self.cfg_dataset_list.setText(rel)
            self.ds_status.setText(f"Created {len(selected)} accessions -> {rel}")
            self._add_history("dataset_create", f"{len(selected)} -> {rel}")
            self._refresh_train_command()
        except Exception as exc:
            QMessageBox.critical(self, "Dataset creation failed", str(exc))
            self.ds_status.setText(f"Failed: {exc}")

    def _refresh_trained_model_list(self):
        wd = self._workdir()
        candidates = []
        for pattern in ("model/checkpoints/*.pt", "model/*.pt"):
            candidates.extend(glob(os.path.join(wd, pattern)))
        unique_sorted = sorted(set(candidates))

        self.gen_model_picker.blockSignals(True)
        self.gen_model_picker.clear()
        if not unique_sorted:
            self.gen_model_picker.addItem("No checkpoints found (using default latest.pt)", "")
        else:
            for path in unique_sorted:
                rel = os.path.relpath(path, wd)
                self.gen_model_picker.addItem(rel, path)
        self.gen_model_picker.blockSignals(False)

    def _activate_selected_model_checkpoint(self) -> str:
        selected = self.gen_model_picker.currentData()
        if not selected:
            return ""
        wd = self._workdir()
        latest = os.path.join(wd, "model", "checkpoints", "latest.pt")
        selected_path = str(selected)
        if os.path.abspath(selected_path) == os.path.abspath(latest):
            return f"[gui] using checkpoint: {os.path.relpath(selected_path, wd)}"
        os.makedirs(os.path.dirname(latest), exist_ok=True)
        shutil.copy2(selected_path, latest)
        return (
            f"[gui] activated checkpoint {os.path.relpath(selected_path, wd)} "
            f"-> {os.path.relpath(latest, wd)}"
        )

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

    def _build_train_job_spec(self) -> JobSpec:
        cfg = self.cfg_stream_yaml.text().strip() or "config/stream_config.yaml"
        catalog = self.cfg_dataset_list.text().strip() or "config/plasmids_10.txt"
        model_type = self.train_model_type.currentText().strip() or "mlp"
        return build_stream_spec(cfg, catalog, model_type, int(self.cfg_epochs.value()), int(self.cfg_batch.value()))

    def _build_generate_job_spec(self) -> JobSpec:
        cfg = self.cfg_stream_yaml.text().strip() or "config/stream_config.yaml"
        return build_generate_plasmid_spec(cfg, 10000, "generated/novel_plasmid.fasta")

    def _job_spec_command(self, spec: JobSpec) -> str:
        payload = json.dumps({"kind": spec.kind, "config_path": spec.config_path, "params": spec.params})
        return f"perceptrome --config {shlex.quote(spec.config_path)} run-job-spec --spec-json {shlex.quote(payload)}"

    # -------------------------
    # Train actions (real QProcess)
    # -------------------------
    def _train_help(self):
        self.train_cmd.setText("perceptrome --help")

    def _train_start(self):
        cmd = self._job_spec_command(self._build_train_job_spec())
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
            self._add_history("train_done", status)
            self._refresh_trained_model_list()

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
        self.gen_cmd.setText("perceptrome --help")

    def _gen_start(self):
        cmd = self._job_spec_command(self._build_generate_job_spec())
        wd = self._workdir()

        checkpoint_note = self._activate_selected_model_checkpoint()

        self.gen_out.clear()
        self._append_log(self.gen_out, f"[{now_str()}] [generate] start\n$ {cmd}\n\n")
        if checkpoint_note:
            self._append_log(self.gen_out, checkpoint_note + "\n")
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

    def _resolve_genome_sequence(self) -> tuple[str, str, str]:
        accession = self.view_accession.text().strip()
        fasta_path = self.view_fasta_path.text().strip()
        genbank_path = ""

        def _genbank_candidates_from_fasta(path: str) -> list[str]:
            base, ext = os.path.splitext(path)
            candidates = [
                path,
                f"{base}.gb",
                f"{base}.gbk",
                f"{base}.genbank",
            ]
            if ext.lower() in (".fa", ".fna", ".fasta"):
                stem = os.path.basename(base)
                candidates.append(os.path.join("cache", "genbank", f"{stem}.gb"))
            return candidates

        def _first_existing(paths: list[str]) -> str:
            for p in paths:
                if p and os.path.exists(p):
                    return p
            return ""

        if accession:
            cfg_path = self.cfg_stream_yaml.text().strip() or "stream_config.yaml"
            cfg = load_full_config(cfg_path)
            ncbi_cfg, _, io_cfg = extract_configs(cfg)
            ensure_dirs(io_cfg)
            fasta_path = fetch_fasta(accession, io_cfg, ncbi_cfg, force=False)
            genbank_path = fetch_genbank(accession, io_cfg, ncbi_cfg, force=False)
            seq = self._read_fasta_sequence(fasta_path)
            return seq, f"accession {accession}", genbank_path
        if fasta_path:
            genbank_path = _first_existing(_genbank_candidates_from_fasta(fasta_path))
            if fasta_path.lower().endswith((".gb", ".gbk", ".genbank")):
                seq = parse_genbank_dna(fasta_path)
                return seq, f"genbank {fasta_path}", fasta_path

            seq = self._read_fasta_sequence(fasta_path)
            return seq, f"fasta {fasta_path}", genbank_path
        raise ValueError("Provide a genome accession or FASTA path.")

    def _default_pdf_output_for_mode(self, mode: str) -> str:
        return "generated/linear_genome.pdf" if mode.lower() == "linear" else "generated/circular_genome.pdf"

    def _default_title_for_mode(self, mode: str, source: str) -> str:
        view_name = "Linear genome" if mode.lower() == "linear" else "Circular genome"
        return f"{view_name} ({source})"

    def _view_on_render_mode_changed(self, mode: str):
        mode = mode or "Circular"
        path = self.view_pdf_path.text().strip()
        circular_default = self._default_pdf_output_for_mode("Circular")
        linear_default = self._default_pdf_output_for_mode("Linear")
        if not path or path in {circular_default, linear_default}:
            self.view_pdf_path.setText(self._default_pdf_output_for_mode(mode))

    def _draw_annotation_pages(
        self,
        painter: QPainter,
        writer: QPdfWriter,
        title: str,
        features: list[CDSFeature] | None,
    ) -> None:
        writer.newPage()
        rect = painter.viewport()
        margin_x = int(rect.width() * 0.06)
        top_margin = int(rect.height() * 0.06)
        content_w = max(200, rect.width() - (2 * margin_x))
        bottom_limit = int(rect.height() * 0.94)

        header_font = QFont("Sans Serif", 13)
        body_font = QFont("Sans Serif", 8)
        painter.setFont(header_font)
        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        painter.drawText(margin_x, top_margin, content_w, 28, Qt.AlignLeft | Qt.AlignVCenter, "Annotations")

        painter.setFont(QFont("Sans Serif", 9))
        painter.setPen(QPen(QColor("#a8b0b8"), 1))
        painter.drawText(margin_x, top_margin + 26, content_w, 22, Qt.AlignLeft | Qt.AlignVCenter, title)

        y = top_margin + 58
        if not features:
            painter.setFont(QFont("Sans Serif", 10))
            painter.setPen(QPen(QColor("#b8c0c8"), 1))
            painter.drawText(
                margin_x,
                y,
                content_w,
                80,
                Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                "No annotations available (GenBank CDS metadata not found for this genome source).",
            )
            return

        cols = [
            ("Gene/Locus", 0.15),
            ("Product", 0.37),
            ("Coordinates", 0.16),
            ("Strand", 0.08),
            ("Protein length", 0.10),
            ("Translation source", 0.14),
        ]

        col_widths = [max(55, int(content_w * frac)) for _, frac in cols]
        width_over = sum(col_widths) - content_w
        if width_over > 0:
            col_widths[-1] = max(55, col_widths[-1] - width_over)

        def draw_header(y_pos: int) -> int:
            painter.setFont(QFont("Sans Serif", 8, QFont.Bold))
            painter.setPen(QPen(QColor("#59636e"), 1))
            painter.drawLine(margin_x, y_pos, margin_x + content_w, y_pos)
            x = margin_x
            for (name, _), w in zip(cols, col_widths):
                painter.drawText(x + 4, y_pos + 3, w - 8, 18, Qt.AlignLeft | Qt.AlignVCenter, name)
                x += w
            painter.drawLine(margin_x, y_pos + 20, margin_x + content_w, y_pos + 20)
            return y_pos + 24

        y = draw_header(y)
        painter.setFont(body_font)
        fm = QFontMetrics(body_font)

        for feat in features:
            values = [
                feat.gene_or_locus_tag,
                feat.product,
                f"{feat.start}..{feat.end}",
                "+" if feat.strand >= 0 else "-",
                str(feat.protein_length),
                feat.translation_source,
            ]

            row_height = 20
            for value, width in zip(values, col_widths):
                text_h = fm.boundingRect(0, 0, width - 8, 1000, Qt.TextWordWrap, value or "-").height()
                row_height = max(row_height, text_h + 8)

            if y + row_height > bottom_limit:
                writer.newPage()
                rect = painter.viewport()
                margin_x = int(rect.width() * 0.06)
                content_w = max(200, rect.width() - (2 * margin_x))
                bottom_limit = int(rect.height() * 0.94)
                y = int(rect.height() * 0.06)
                painter.setFont(header_font)
                painter.setPen(QPen(QColor("#d9d9d9"), 1))
                painter.drawText(margin_x, y, content_w, 28, Qt.AlignLeft | Qt.AlignVCenter, "Annotations (continued)")
                y += 34
                y = draw_header(y)
                painter.setFont(body_font)

            painter.setPen(QPen(QColor("#3b3f46"), 1))
            painter.drawLine(margin_x, y + row_height, margin_x + content_w, y + row_height)
            x = margin_x
            painter.setPen(QPen(QColor("#b8c0c8"), 1))
            for value, width in zip(values, col_widths):
                painter.drawText(x + 4, y + 4, width - 8, row_height - 6, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, value or "-")
                x += width
            y += row_height

    def _write_circular_pdf(self, seq: str, output_path: str, title: str, features: list[CDSFeature] | None = None) -> None:
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

            if features:
                palette = ["#91ff9d", "#fcbf49", "#ff6f91", "#a0c4ff", "#cdb4db", "#8be9fd"]
                for idx, feat in enumerate(features):
                    a0 = 360.0 * ((feat.start - 1) / max(seq_len, 1))
                    a1 = 360.0 * (feat.end / max(seq_len, 1))
                    span = max(1.0, a1 - a0)
                    color = QColor(palette[idx % len(palette)])
                    painter.setPen(QPen(color, 6))
                    painter.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, int(-a0 * 16), int(-span * 16))

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

            painter.setPen(QPen(QColor("#a8b0b8"), 1))
            legend_msg = (
                f"CDS features: {len(features)} (see annotation table pages)"
                if features
                else "CDS features: none (see annotation table page)"
            )
            painter.drawText(cx - radius, cy + radius + 40, radius * 2, 24, Qt.AlignCenter, legend_msg)

            self._draw_annotation_pages(painter, writer, title, features)
        finally:
            painter.end()

    def _write_linear_pdf(self, seq: str, output_path: str, title: str, features: list[CDSFeature] | None = None) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = QPdfWriter(output_path)
        writer.setResolution(150)

        painter = QPainter(writer)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            rect = painter.viewport()
            seq_len = max(1, len(seq))
            gc = (seq.count("G") + seq.count("C")) / float(seq_len)

            left = int(rect.width() * 0.08)
            right = int(rect.width() * 0.92)
            baseline_y = int(rect.height() * 0.45)
            width = max(1, right - left)

            painter.setPen(QPen(QColor("#d9d9d9"), 2))
            painter.drawText(left, 70, width, 30, Qt.AlignCenter, title or "Linear genome view")

            painter.setPen(QPen(QColor("#68d5ff"), 3))
            painter.drawLine(left, baseline_y, right, baseline_y)

            tick_vals = sorted(set([1, seq_len // 4, seq_len // 2, (3 * seq_len) // 4, seq_len]))
            for tick_bp in tick_vals:
                x = left + int(((tick_bp - 1) / max(seq_len - 1, 1)) * width)
                painter.setPen(QPen(QColor("#58616b"), 1))
                painter.drawLine(x, baseline_y - 8, x, baseline_y + 8)
                painter.setPen(QPen(QColor("#a8b0b8"), 1))
                painter.drawText(x - 40, baseline_y + 14, 80, 20, Qt.AlignHCenter | Qt.AlignTop, f"{tick_bp:,}")

            if features:
                lane_height = 14
                lane_gap = 6
                track_gap = 36
                lane_count = 4
                track_shift = [-(track_gap // 2), +(track_gap // 2)]
                palette = ["#91ff9d", "#fcbf49", "#ff6f91", "#a0c4ff", "#cdb4db", "#8be9fd"]
                lane_last_end = [[0] * lane_count for _ in range(2)]

                for idx, feat in enumerate(sorted(features, key=lambda f: (f.start, f.end))):
                    track_idx = 0 if feat.strand >= 0 else 1
                    lane_idx = 0
                    for i in range(lane_count):
                        if feat.start > lane_last_end[track_idx][i]:
                            lane_idx = i
                            break
                    lane_last_end[track_idx][lane_idx] = feat.end

                    x0 = left + int(((feat.start - 1) / max(seq_len - 1, 1)) * width)
                    x1 = left + int(((feat.end - 1) / max(seq_len - 1, 1)) * width)
                    if x1 < x0:
                        x0, x1 = x1, x0
                    x1 = max(x1, x0 + 2)

                    lane_sign = -1 if track_idx == 0 else 1
                    y = baseline_y + track_shift[track_idx] + lane_sign * (lane_idx * (lane_height + lane_gap))

                    color = QColor(palette[idx % len(palette)])
                    painter.setPen(QPen(color, 1))
                    painter.setBrush(color)

                    arrow_len = min(12, max(5, x1 - x0))
                    if feat.strand >= 0:
                        points = [
                            (x0, y - lane_height // 2),
                            (x1 - arrow_len, y - lane_height // 2),
                            (x1, y),
                            (x1 - arrow_len, y + lane_height // 2),
                            (x0, y + lane_height // 2),
                        ]
                    else:
                        points = [
                            (x1, y - lane_height // 2),
                            (x0 + arrow_len, y - lane_height // 2),
                            (x0, y),
                            (x0 + arrow_len, y + lane_height // 2),
                            (x1, y + lane_height // 2),
                        ]
                    painter.drawPolygon(QPolygon([QPoint(p[0], p[1]) for p in points]))

            info = f"Length: {len(seq):,} bp    GC: {gc * 100:.2f}%"
            painter.setPen(QPen(QColor("#a8b0b8"), 1))
            painter.drawText(left, baseline_y + 90, width, 24, Qt.AlignCenter, info)

            self._draw_annotation_pages(painter, writer, title, features)
        finally:
            painter.end()

    def _view_generate_pdf(self):
        self.view_log.clear()
        try:
            seq, source, genbank_path = self._resolve_genome_sequence()
            mode = self.view_render_mode.currentText().strip() or "Circular"
            output_path = self.view_pdf_path.text().strip() or self._default_pdf_output_for_mode(mode)
            title = self.view_title.text().strip() or self._default_title_for_mode(mode, source)
            features: list[CDSFeature] = []
            if genbank_path:
                try:
                    features = parse_cds_features_from_genbank(genbank_path)
                    self._append_log(self.view_log, f"[{now_str()}] Loaded {len(features)} CDS features from {genbank_path}\n")
                except Exception as exc:
                    self._append_log(self.view_log, f"[{now_str()}] WARNING: Failed to parse CDS features from {genbank_path}: {exc}\n")
            if not genbank_path:
                self._append_log(self.view_log, f"[{now_str()}] WARNING: Genome source does not include GenBank feature metadata; annotation table will show no annotations available.\n")
            elif not features:
                self._append_log(self.view_log, f"[{now_str()}] WARNING: GenBank source had no CDS features; annotation table will show no annotations available.\n")
            self._append_log(self.view_log, f"[{now_str()}] Generating {mode} PDF from {source}\n")
            if mode.lower() == "linear":
                self._write_linear_pdf(seq, output_path, title, features=features)
            else:
                self._write_circular_pdf(seq, output_path, title, features=features)
            self._append_log(self.view_log, f"[{now_str()}] Saved {mode} PDF -> {output_path}\n")
            self._add_history("view_pdf", output_path)
            self._load_pdf(output_path, mode=mode)
            self._show_pdf_window()
        except Exception as exc:
            self._append_log(self.view_log, f"[{now_str()}] ERROR: {exc}\n")

    def _load_pdf(self, path: str, mode: str | None = None):
        self.pdf_window.set_render_mode(mode or self.view_render_mode.currentText())
        self.pdf_window.load_pdf(path)
        self.view_pdf_path.setText(path)

    def _show_pdf_window(self):
        self.pdf_window.show()
        self.pdf_window.raise_()
        self.pdf_window.activateWindow()

    def _view_open_pdf(self):
        path = self.view_pdf_path.text().strip() or self.settings.value("pdf_last_opened_path", "")
        if not path:
            self._append_log(self.view_log, f"[{now_str()}] ERROR: No PDF path set.\n")
            return
        try:
            self._load_pdf(path)
            self._show_pdf_window()
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
        self.pdf_window.save_window_settings()
        self.settings.sync()
        # Intentionally pass None during shutdown: logs may already be tearing down,
        # but stop() must still terminate/kill child processes safely.
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



class GenomePdfWindow(QMainWindow):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Genome PDF")
        self.resize(900, 650)

        root = QWidget()
        layout = QVBoxLayout(root)

        self.pdf_doc = QPdfDocument(self)
        self.pdf_view = QPdfView()
        self.pdf_view.setDocument(self.pdf_doc)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.pdf_view.installEventFilter(self)
        self.pdf_view.setFocusPolicy(Qt.StrongFocus)

        self._pdf_drag_active = False
        self._pdf_drag_last_pos = None
        self._pdf_render_mode = "Circular"

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(32)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(32)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("100%")
        zoom_row.addWidget(QLabel("Zoom:"))
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.zoom_label)

        pan_row = QHBoxLayout()
        self.btn_pan_left = QPushButton("◀")
        self.btn_pan_right = QPushButton("▶")
        pan_row.addWidget(QLabel("Pan X:"))
        pan_row.addWidget(self.btn_pan_left)
        pan_row.addWidget(self.btn_pan_right)
        pan_row.addStretch(1)

        layout.addLayout(zoom_row)
        layout.addLayout(pan_row)
        layout.addWidget(self.pdf_view, 1)
        self.setCentralWidget(root)

        self.zoom_slider.valueChanged.connect(self._view_zoom_changed)
        self.btn_zoom_out.clicked.connect(lambda: self._step_view_zoom(-10))
        self.btn_zoom_in.clicked.connect(lambda: self._step_view_zoom(10))
        self.btn_pan_left.clicked.connect(lambda: self._pan_horizontal(-120))
        self.btn_pan_right.clicked.connect(lambda: self._pan_horizontal(120))

    def set_render_mode(self, mode: str):
        self._pdf_render_mode = (mode or "Circular").strip().lower().title()

    def load_pdf(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        status = self.pdf_doc.load(path)
        if status != QPdfDocument.Error.None_:
            raise RuntimeError(f"Failed to load PDF (status {status})")
        self.settings.setValue("pdf_last_opened_path", path)

    def zoom_percent(self) -> int:
        return int(self.zoom_slider.value())

    def set_zoom_percent(self, value: int):
        self.zoom_slider.setValue(max(25, min(400, int(value))))

    def _view_zoom_changed(self, value: int):
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.pdf_view.setZoomFactor(value / 100.0)
        self.zoom_label.setText(f"{value}%")

    def _step_view_zoom(self, delta: int):
        self.zoom_slider.setValue(max(25, min(400, self.zoom_slider.value() + delta)))

    def _is_linear_mode(self) -> bool:
        return self._pdf_render_mode.lower() == "linear"

    def _reset_zoom_fit_width(self):
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.zoom_label.setText("Fit width")

    def _pan_horizontal(self, delta: int):
        hbar = self.pdf_view.horizontalScrollBar()
        hbar.setValue(hbar.value() + delta)

    def save_window_settings(self):
        self.settings.setValue("pdf_window_zoom", self.zoom_percent())
        self.settings.setValue("pdf_window_geometry", self.saveGeometry())

    def load_window_settings(self):
        self.set_zoom_percent(int(self.settings.value("pdf_window_zoom", 100)))
        geometry = self.settings.value("pdf_window_geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def eventFilter(self, watched, event):
        if watched is self.pdf_view:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                self._pdf_drag_active = True
                self._pdf_drag_last_pos = pos
                self.pdf_view.setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self._pdf_drag_active:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                delta = pos - self._pdf_drag_last_pos
                self._pdf_drag_last_pos = pos
                hbar = self.pdf_view.horizontalScrollBar()
                vbar = self.pdf_view.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                if self._is_linear_mode() and not (event.modifiers() & Qt.ShiftModifier):
                    delta_y = 0
                else:
                    delta_y = delta.y()
                vbar.setValue(vbar.value() - delta_y)
                return True
            if event.type() == QEvent.MouseButtonRelease and self._pdf_drag_active:
                self._pdf_drag_active = False
                self._pdf_drag_last_pos = None
                self.pdf_view.setCursor(Qt.ArrowCursor)
                return True
            if event.type() == QEvent.KeyPress:
                key = event.key()
                mods = event.modifiers()
                if key in (Qt.Key_Plus, Qt.Key_Equal):
                    self._step_view_zoom(10)
                    return True
                if key == Qt.Key_Minus:
                    self._step_view_zoom(-10)
                    return True
                if key == Qt.Key_Left:
                    self._pan_horizontal(-120)
                    return True
                if key == Qt.Key_Right:
                    self._pan_horizontal(120)
                    return True
                if self._is_linear_mode() and key == Qt.Key_0 and (mods & Qt.ControlModifier):
                    self._reset_zoom_fit_width()
                    return True
        return super().eventFilter(watched, event)


def main():
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    win = PerceptromeQt()
    app.aboutToQuit.connect(win.shutdown)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

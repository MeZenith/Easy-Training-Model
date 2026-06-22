"""Export / Deploy page — GGUF 导出 + Ollama 部署"""

import logging
import os
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.exporter import Exporter, ExportWorker
from core.model_manager import ModelManager
from core.ollama_deployer import OllamaDeployer
from core.services.export_service import detect_format, find_gguf, find_safetensors_dir
from core.services.train_service import list_loras_for_combo

logger = logging.getLogger("EasyTinking")

QUANT_INFO = {
    "Q4_K_M": "~2 GB (4-bit)", "Q8_0": "~3.2 GB (8-bit)",
    "F16": "~6 GB (lossless)",
}


class ExportPage(QWidget):
    export_finished = Signal(dict)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._export_worker = None

        self._exporter = Exporter(config.get("export_dir", ""))
        self._deployer = OllamaDeployer()
        self._models_dir = config.get("download_dir", os.path.join(config.workspace, "models"))
        self._lora_dir = os.path.join(config.workspace, "lora")

        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    # ── UI setup ──────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel()
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        self._title_label = title

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left.setLayout(self._build_export_panel())
        splitter.addWidget(left)

        right = QWidget()
        right.setLayout(self._build_deploy_panel())
        splitter.addWidget(right)
        splitter.setSizes([480, 480])

        self._refresh_texts()

    def _build_export_panel(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        # ── Model / LoRA / Path ──
        g = QGroupBox()
        f = QFormLayout(g)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(200)
        f.addRow("Base Model:", self._model_combo)

        self._lora_combo = QComboBox()
        f.addRow("LoRA:", self._lora_combo)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(self._config.get("export_dir", ""))
        self._dir_edit.setReadOnly(True)
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse_btn = QPushButton("...")
        dir_row.addWidget(self._dir_browse_btn)
        f.addRow("Export Dir:", dir_row)

        self._name_edit = QLineEdit()
        f.addRow("Name:", self._name_edit)

        layout.addWidget(g)
        self._cfg_group = g

        # ── Formats ──
        g2 = QGroupBox()
        fl = QVBoxLayout(g2)
        self._fmt_16bit = QCheckBox("16-bit Full Model (~6 GB)")
        self._fmt_q4 = QCheckBox("GGUF Q4_K_M (~2 GB, 4-bit)")
        self._fmt_q4.setChecked(True)
        self._fmt_q8 = QCheckBox("GGUF Q8_0 (~3.2 GB, 8-bit)")
        self._fmt_f16 = QCheckBox("GGUF F16 (~6 GB, lossless)")
        self._fmt_lora = QCheckBox("LoRA Adapter Only (~120 MB)")
        for cb in [self._fmt_16bit, self._fmt_q4, self._fmt_q8, self._fmt_f16, self._fmt_lora]:
            fl.addWidget(cb)
        layout.addWidget(g2)
        self._fmt_group = g2

        # ── Export button + Progress ──
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Start Export")
        self._export_btn.setObjectName("primaryBtn")
        self._export_btn.setMinimumHeight(36)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self._status_label)

        # ── Log ──
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumHeight(120)
        self._log_output.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self._log_output)

        return layout

    def _build_deploy_panel(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)

        # ── Ollama status ──
        status_row = QHBoxLayout()
        self._ollama_status_label = QLabel()
        self._ollama_detect_btn = QPushButton("Detect")
        status_row.addWidget(self._ollama_status_label, 1)
        status_row.addWidget(self._ollama_detect_btn)
        layout.addLayout(status_row)

        # ── Deploy config ──
        g = QGroupBox()
        f = QFormLayout(g)

        self._export_selector = QComboBox()
        f.addRow("Export:", self._export_selector)

        self._gguf_path_label = QLabel()
        self._gguf_path_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self._gguf_path_label.setWordWrap(True)
        f.addRow("Path:", self._gguf_path_label)

        self._ollama_name_edit = QLineEdit(self._config.get("ollama_name", "my-model"))
        f.addRow("Model Name:", self._ollama_name_edit)

        self._system_prompt_edit = QTextEdit()
        self._system_prompt_edit.setMaximumHeight(80)
        self._system_prompt_edit.setPlaceholderText("You are a helpful assistant.")
        f.addRow("System:", self._system_prompt_edit)

        btn_row = QHBoxLayout()
        self._deploy_btn = QPushButton("Create Model")
        self._deploy_btn.setObjectName("primaryBtn")
        self._run_btn = QPushButton("Run in Terminal")
        btn_row.addWidget(self._deploy_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        f.addRow(btn_row)
        layout.addWidget(g)
        self._deploy_group = g

        # ── Exported models table ──
        g2 = QGroupBox()
        el = QVBoxLayout(g2)
        self._export_table = QTableWidget()
        self._export_table.setColumnCount(4)
        self._export_table.setHorizontalHeaderLabels(["Name", "Format", "Size", "Time"])
        self._export_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._export_table.setSelectionBehavior(QTableWidget.SelectRows)
        el.addWidget(self._export_table)

        eb = QHBoxLayout()
        self._export_delete_btn = QPushButton("Delete")
        self._export_open_btn = QPushButton("Open Dir")
        eb.addWidget(self._export_delete_btn)
        eb.addWidget(self._export_open_btn)
        eb.addStretch()
        el.addLayout(eb)
        layout.addWidget(g2)
        self._exports_list_group = g2

        # ── Ollama models table ──
        g3 = QGroupBox()
        ol = QVBoxLayout(g3)
        self._ollama_table = QTableWidget()
        self._ollama_table.setColumnCount(3)
        self._ollama_table.setHorizontalHeaderLabels(["Name", "Size", "Modified"])
        self._ollama_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ollama_table.setSelectionBehavior(QTableWidget.SelectRows)
        ol.addWidget(self._ollama_table)

        ob = QHBoxLayout()
        self._ollama_delete_btn = QPushButton("Delete")
        self._ollama_run_btn = QPushButton("Run")
        ob.addWidget(self._ollama_delete_btn)
        ob.addWidget(self._ollama_run_btn)
        ob.addStretch()
        ol.addLayout(ob)
        layout.addWidget(g3)
        self._ollama_group = g3

        return layout

    # ── Signal connections ───────────────────────────────────

    def _connect_signals(self):
        self._export_btn.clicked.connect(self._on_export)
        self._dir_browse_btn.clicked.connect(self._on_browse_dir)
        self._lora_combo.currentIndexChanged.connect(self._on_lora_changed)
        self._export_selector.currentIndexChanged.connect(self._on_export_selected)

        self._deploy_btn.clicked.connect(self._on_deploy)
        self._run_btn.clicked.connect(self._on_run_ollama)
        self._ollama_detect_btn.clicked.connect(self._detect_ollama)

        self._export_delete_btn.clicked.connect(self._delete_export)
        self._export_open_btn.clicked.connect(self._open_export_dir)
        self._ollama_delete_btn.clicked.connect(self._delete_ollama_model)
        self._ollama_run_btn.clicked.connect(self._on_run_ollama)

    # ── Model / LoRA loading ────────────────────────────────

    def _load_models(self):
        self._model_combo.clear()
        mgr = ModelManager(self._models_dir)
        for m in mgr.list_downloaded_models():
            self._model_combo.addItem(m["name"], userData=m["path"])

    def _load_loras(self):
        self._lora_combo.clear()
        loras = list_loras_for_combo(self._config.workspace)
        for item in loras:
            self._lora_combo.addItem(
                item["display"],
                userData={"lora_path": item["lora_path"], "model_path": item["model_path"]}
            )
        if not loras:
            self._lora_combo.addItem("(No LoRA found)", userData={})

    def _on_lora_changed(self, _index):
        data = self._lora_combo.currentData() or {}
        lora_path = data.get("lora_path", "")
        if lora_path:
            self._name_edit.setText(os.path.basename(lora_path))

    # ── Export logic ────────────────────────────────────────

    def _on_export(self):
        model_path = self._model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "Warning", "Please select a base model first")
            return

        export_name = self._name_edit.text().strip() or "exported_model"
        formats = []
        if self._fmt_16bit.isChecked():
            formats.append("16bit")
        if self._fmt_q4.isChecked():
            formats.append("gguf_Q4_K_M")
        if self._fmt_q8.isChecked():
            formats.append("gguf_Q8_0")
        if self._fmt_f16.isChecked():
            formats.append("gguf_F16")
        if self._fmt_lora.isChecked():
            formats.append("lora_only")

        if not formats:
            QMessageBox.warning(self, "Warning", "Please select at least one export format")
            return

        lora_data = self._lora_combo.currentData() or {}
        lora_path = lora_data.get("lora_path", "")

        self._export_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._log_output.clear()
        self._status_label.setText("Starting export...")

        export_dir = self._dir_edit.text() or self._config.workspace + "/exports"

        def on_progress(pct, desc):
            self._progress_bar.setValue(pct)
            self._status_label.setText(desc)

        def on_finished(result):
            self._export_btn.setEnabled(True)
            self._progress_bar.setValue(100)
            self._status_label.setText("Export complete!")
            self._refresh_exports_list()
            self._refresh_export_selector()
            self.export_finished.emit(result)
            errors = result.get("errors", [])
            if errors:
                self._append_log(f"Warnings: {errors}")

        def on_error(code, detail):
            self._export_btn.setEnabled(True)
            self._status_label.setText(f"Error: {detail}")
            self._append_log(f"ERROR [{code}]: {detail}")

        worker = ExportWorker(
            lora_path=lora_path,
            model_path=model_path,
            export_dir=export_dir,
            export_name=export_name,
            formats=formats,
        )
        worker.signals.progress.connect(on_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        worker.signals.log.connect(self._append_log)
        self._export_worker = worker
        worker.start()

    def _append_log(self, msg):
        t = time.strftime("%H:%M:%S")
        self._log_output.append(f"[{t}] {msg}")

    def _on_browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if d:
            self._dir_edit.setText(d)
            self._config.set("export_dir", d)

    # ── Exported models ─────────────────────────────────────

    def _refresh_exports_list(self):
        self._export_table.setRowCount(0)
        exports = self._exporter.list_exports()
        for exp in exports:
            r = self._export_table.rowCount()
            self._export_table.insertRow(r)
            self._export_table.setItem(r, 0, QTableWidgetItem(exp["name"]))
            fmt = detect_format(exp["path"])
            self._export_table.setItem(r, 1, QTableWidgetItem(fmt))
            size_mb = exp["size"] / (1024 * 1024) if exp["size"] > 0 else 0
            self._export_table.setItem(r, 2, QTableWidgetItem(f"{size_mb:.0f} MB"))
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(exp["export_time"])) if exp["export_time"] else ""
            self._export_table.setItem(r, 3, QTableWidgetItem(ts))

    def _refresh_export_selector(self):
        self._export_selector.clear()
        exports = self._exporter.list_exports()
        for exp in exports:
            gguf_files = exp.get("gguf_files", [])
            if gguf_files:
                for gf in gguf_files:
                    label = f"{exp['name']} / {os.path.basename(gf)}"
                    self._export_selector.addItem(label, userData={"dir": exp["path"], "gguf": gf})
            else:
                self._export_selector.addItem(exp["name"], userData={"dir": exp["path"], "gguf": ""})

    def _on_export_selected(self, _index):
        data = self._export_selector.currentData() or {}
        full_dir = data.get("dir", "")
        gguf = data.get("gguf", "")
        if gguf:
            self._gguf_path_label.setText(gguf)
            self._ollama_name_edit.setText(os.path.splitext(os.path.basename(gguf))[0])
        elif full_dir:
            found = find_gguf(full_dir)
            self._gguf_path_label.setText(found or full_dir)

    def _delete_export(self):
        row = self._export_table.currentRow()
        if row < 0:
            return
        name = self._export_table.item(row, 0).text()
        path = os.path.join(self._config.get("export_dir", ""), name)
        reply = QMessageBox.question(self, "Confirm", f"Delete export '{name}'?")
        if reply == QMessageBox.Yes:
            import shutil as _shutil
            _shutil.rmtree(path, ignore_errors=True)
            self._refresh_exports_list()
            self._refresh_export_selector()

    def _open_export_dir(self):
        row = self._export_table.currentRow()
        if row < 0:
            return
        name = self._export_table.item(row, 0).text()
        path = os.path.join(self._config.get("export_dir", ""), name)
        if os.path.isdir(path):
            os.startfile(path)

    # ── Ollama ──────────────────────────────────────────────

    def _detect_ollama(self):
        if self._deployer.is_installed():
            ver = self._deployer.get_version()
            self._ollama_status_label.setText(f"[OK] Ollama {ver}")
            self._ollama_status_label.setStyleSheet("color: #22c55e;")
            self._deploy_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
        else:
            self._ollama_status_label.setText("Ollama not installed")
            self._ollama_status_label.setStyleSheet("color: #ef4444;")
            self._deploy_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
        self._refresh_ollama_list()

    def _on_deploy(self):
        data = self._export_selector.currentData() or {}
        gguf_path = data.get("gguf", "")
        full_dir = data.get("dir", "")

        if gguf_path:
            model_path = gguf_path
        elif full_dir:
            model_path = find_gguf(full_dir) or find_safetensors_dir(full_dir)
        else:
            QMessageBox.warning(self, "Warning", "Please select an export first")
            return

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {model_path}")
            return

        ollama_name = self._ollama_name_edit.text().strip()
        if not ollama_name:
            QMessageBox.warning(self, "Warning", "Please enter a model name")
            return

        is_dir = os.path.isdir(model_path)
        system_prompt = self._system_prompt_edit.toPlainText().strip()
        modelfile = self._deployer.generate_modelfile(
            model_path=model_path, model_name=ollama_name,
            system_prompt=system_prompt, is_directory=is_dir,
        )

        self._deploy_btn.setEnabled(False)
        success, output = self._deployer.create_model(
            modelfile_content=modelfile, model_name=ollama_name,
            working_dir=self._config.workspace,
        )
        self._deploy_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", f"Model '{ollama_name}' created!")
            self._refresh_ollama_list()
        else:
            QMessageBox.critical(self, "Error", f"Deploy failed: {output}")

    def _on_run_ollama(self):
        ollama_name = self._ollama_name_edit.text().strip()

        if not ollama_name:
            # Try selected ollama table row
            row = self._ollama_table.currentRow()
            if row >= 0:
                ollama_name = self._ollama_table.item(row, 0).text()

        if not ollama_name:
            QMessageBox.warning(self, "Warning", "Please enter or select a model name")
            return

        success, msg = self._deployer.run_model(ollama_name)
        if not success:
            QMessageBox.critical(self, "Error", msg)

    def _refresh_ollama_list(self):
        self._ollama_table.setRowCount(0)
        if not self._deployer.is_installed():
            return
        models = self._deployer.list_models()
        for m in models:
            r = self._ollama_table.rowCount()
            self._ollama_table.insertRow(r)
            self._ollama_table.setItem(r, 0, QTableWidgetItem(m.get("name", "")))
            self._ollama_table.setItem(r, 1, QTableWidgetItem(m.get("size", "")))
            self._ollama_table.setItem(r, 2, QTableWidgetItem(m.get("modified", "")))

    def _delete_ollama_model(self):
        row = self._ollama_table.currentRow()
        if row < 0:
            return
        name = self._ollama_table.item(row, 0).text()
        reply = QMessageBox.question(self, "Confirm", f"Delete Ollama model '{name}'?")
        if reply == QMessageBox.Yes:
            success, msg = self._deployer.delete_model(name)
            if not success:
                QMessageBox.critical(self, "Error", msg)
            self._refresh_ollama_list()

    # ── i18n refresh ────────────────────────────────────────

    def _refresh_texts(self):
        t = self._i18n.t
        self._title_label.setText(t("nav.export"))
        self._cfg_group.setTitle(t("export.lora_adapter"))
        self._fmt_group.setTitle(t("export.format"))
        self._fmt_16bit.setText(t("format.16bit"))
        self._fmt_q4.setText(t("format.gguf_q4"))
        self._fmt_q8.setText(t("format.gguf_q8"))
        self._fmt_f16.setText(t("format.gguf_f16"))
        self._fmt_lora.setText(t("format.lora_only"))
        self._export_btn.setText(t("common.save"))
        self._exports_list_group.setTitle(t("export.name"))
        self._export_open_btn.setText(t("common.browse"))
        self._export_delete_btn.setText(t("common.delete"))
        self._deploy_group.setTitle(t("deploy.create"))
        self._ollama_group.setTitle("Ollama Models")
        self._deploy_btn.setText(t("deploy.create"))
        self._run_btn.setText(t("deploy.run"))
        self._ollama_detect_btn.setText(t("deploy.detect"))
        self._ollama_delete_btn.setText(t("common.delete"))
        self._ollama_run_btn.setText(t("deploy.run"))
        self._export_table.setHorizontalHeaderLabels([
            t("common.name"), t("common.format"), t("common.size"), t("common.time")
        ])
        self._ollama_table.setHorizontalHeaderLabels(["Name", "Size", "Modified"])
        self._detect_ollama()

    def showEvent(self, event):
        super().showEvent(event)
        self._load_models()
        self._load_loras()
        self._refresh_exports_list()
        self._refresh_export_selector()
        self._detect_ollama()

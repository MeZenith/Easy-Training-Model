"""Export / Deploy page - Full implementation"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
import time

from core.exporter import Exporter, ExportWorker
from core.trainer import ProcessTrainer
from core.ollama_deployer import OllamaDeployer
from core.error_handler import friendly_error_message


class ExportPage(QWidget):
    export_finished = Signal(dict)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._exporter = Exporter(config.get("export_dir", ""))
        self._deployer = OllamaDeployer()
        self._export_worker = None
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)

        # ===== LoRA selection =====
        lora_group = QGroupBox()
        lora_form = QFormLayout(lora_group)
        self._lora_combo = QComboBox()
        lora_form.addRow("LoRA:", self._lora_combo)
        self._lora_info_label = QLabel()
        self._lora_info_label.setObjectName("label-secondary")
        self._lora_info_label.setWordWrap(True)
        lora_form.addRow("", self._lora_info_label)
        scroll_layout.addWidget(lora_group)
        self._lora_group = lora_group

        # ===== Export formats =====
        format_group = QGroupBox()
        format_layout = QVBoxLayout(format_group)
        self._fmt_16bit = QCheckBox(self._i18n.t("format.16bit"))
        self._fmt_q4 = QCheckBox(self._i18n.t("format.gguf_q4"))
        self._fmt_q8 = QCheckBox(self._i18n.t("format.gguf_q8"))
        self._fmt_q8.setChecked(True)
        self._fmt_f16 = QCheckBox(self._i18n.t("format.gguf_f16"))
        self._fmt_lora = QCheckBox(self._i18n.t("format.lora_only"))
        for cb in [self._fmt_16bit, self._fmt_q4, self._fmt_q8, self._fmt_f16, self._fmt_lora]:
            format_layout.addWidget(cb)
        scroll_layout.addWidget(format_group)
        self._format_group = format_group

        # ===== Export name and directory =====
        name_dir_row = QHBoxLayout()
        name_form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Model name")
        name_form.addRow(self._i18n.t("export.name") + ":", self._name_edit)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setText(self._config.get("export_dir", ""))
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse_btn = QPushButton()
        dir_row.addWidget(self._dir_browse_btn)
        name_form.addRow(self._i18n.t("export.dir") + ":", dir_row)

        name_dir_row.addLayout(name_form)
        scroll_layout.addLayout(name_dir_row)

        # ===== Export button and progress =====
        export_row = QHBoxLayout()
        self._export_btn = QPushButton()
        self._export_btn.setObjectName("primaryBtn")
        self._export_btn.setMinimumHeight(36)
        export_row.addWidget(self._export_btn)

        self._open_dir_btn = QPushButton()
        self._open_dir_btn.setEnabled(False)
        export_row.addWidget(self._open_dir_btn)
        scroll_layout.addLayout(export_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        scroll_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel()
        self._progress_label.setObjectName("label-muted")
        scroll_layout.addWidget(self._progress_label)

        # ===== Exported models list =====
        exports_group = QGroupBox()
        exports_layout = QVBoxLayout(exports_group)
        self._export_table = QTableWidget()
        self._export_table.setColumnCount(4)
        self._export_table.setHorizontalHeaderLabels([
            self._i18n.t("common.name"),
            self._i18n.t("common.format"),
            self._i18n.t("common.size"),
            self._i18n.t("common.time")
        ])
        self._export_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._export_table.setAlternatingRowColors(True)
        exports_layout.addWidget(self._export_table)
        scroll_layout.addWidget(exports_group)
        self._exports_group = exports_group

        # ===== Ollama deployment =====
        deploy_group = QGroupBox()
        deploy_layout = QFormLayout(deploy_group)

        self._ollama_status_label = QLabel()
        deploy_layout.addRow("Ollama:", self._ollama_status_label)

        self._detect_btn = QPushButton()
        self._detect_btn.clicked.connect(self._detect_ollama)
        deploy_layout.addRow(self._detect_btn)

        self._ollama_name_edit = QLineEdit()
        self._ollama_name_edit.setText(self._config.get("ollama_name", "my-model"))
        deploy_layout.addRow(self._i18n.t("deploy.model_name") + ":", self._ollama_name_edit)

        self._system_prompt_edit = QTextEdit()
        self._system_prompt_edit.setMaximumHeight(80)
        identity_name = self._config.get("model_identity.name", "")
        identity_creator = self._config.get("model_identity.creator", "")
        identity_desc = self._config.get("model_identity.description", "")
        default_prompt = ""
        if identity_name:
            default_prompt = f"You are {identity_name}"
            if identity_creator:
                default_prompt += f", created by {identity_creator}"
            if identity_desc:
                default_prompt += f". {identity_desc}"
        self._system_prompt_edit.setPlainText(default_prompt)
        deploy_layout.addRow(self._i18n.t("deploy.system_prompt") + ":", self._system_prompt_edit)

        deploy_btn_row = QHBoxLayout()
        self._deploy_btn = QPushButton()
        self._deploy_btn.setObjectName("primaryBtn")
        deploy_btn_row.addWidget(self._deploy_btn)

        self._run_btn = QPushButton()
        deploy_btn_row.addWidget(self._run_btn)
        deploy_layout.addRow(deploy_btn_row)

        self._wizard_label = QLabel()
        self._wizard_label.setWordWrap(True)
        self._wizard_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;"
            " color: #8b949e; padding: 8px;"
            " background-color: rgba(255,255,255,0.03); border-radius: 4px;"
        )
        self._wizard_label.setText(
            f"[1] {self._i18n.t('export.wizard_1')}\n"
            f"[2] {self._i18n.t('export.wizard_2')}\n"
            f"[3] {self._i18n.t('export.wizard_3')}\n"
            f"[4] {self._i18n.t('export.wizard_4')}\n"
            f"[5] {self._i18n.t('export.wizard_5')}\n"
            f"[6] {self._i18n.t('export.wizard_6')}"
        )
        deploy_layout.addRow("", self._wizard_label)

        scroll_layout.addWidget(deploy_group)
        self._deploy_group = deploy_group

        scroll_layout.addStretch()

        from PySide6.QtWidgets import QScrollArea, QFrame
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        self._refresh_texts()
        self._detect_ollama()

    def _connect_signals(self):
        self._export_btn.clicked.connect(self._on_export)
        self._dir_browse_btn.clicked.connect(self._on_browse_dir)
        self._open_dir_btn.clicked.connect(self._on_open_dir)
        self._lora_combo.currentIndexChanged.connect(self._on_lora_changed)
        self._deploy_btn.clicked.connect(self._on_deploy)
        self._run_btn.clicked.connect(self._on_run)

    def _load_loras(self):
        """Load trained LoRA adapter list"""
        self._lora_combo.clear()
        trainer = ProcessTrainer(self._config.workspace)
        loras = trainer.list_loras()
        for lora in loras:
            meta = lora.get("metadata", {})
            model_path = meta.get("model_path", "")
            name = lora["name"]
            if model_path:
                self._lora_combo.addItem(f"{name} -> {os.path.basename(model_path)}",
                                         userData={"lora_path": lora["path"], "model_path": model_path})
            else:
                self._lora_combo.addItem(name,
                                         userData={"lora_path": lora["path"], "model_path": ""})

        if not loras:
            self._lora_combo.addItem("(No LoRA found)", userData={})

    def _on_lora_changed(self, index):
        """LoRA selection changed"""
        data = self._lora_combo.currentData() or {}
        lora_path = data.get("lora_path", "")
        model_path = data.get("model_path", "")
        if lora_path:
            meta_path = os.path.join(lora_path, "metadata.json")
            if os.path.isfile(meta_path):
                import json
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    info_lines = []
                    info_lines.append(f"Model: {meta.get('model_path', 'N/A')}")
                    info_lines.append(f"Loss: {meta.get('initial_loss', 0):.4f} -> {meta.get('final_loss', 0):.4f}")
                    info_lines.append(f"Time: {meta.get('elapsed_seconds', 0):.1f}s")
                    self._lora_info_label.setText("\n".join(info_lines))
                    # Auto-set export name
                    self._name_edit.setText(os.path.basename(lora_path))
                except Exception:
                    self._lora_info_label.setText(lora_path)

    def _on_export(self):
        """Start export"""
        data = self._lora_combo.currentData() or {}
        lora_path = data.get("lora_path", "")
        model_path = data.get("model_path", "")

        if not lora_path:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("export.select_lora"))
            return

        export_name = self._name_edit.text().strip()
        if not export_name:
            export_name = "exported_model"

        # Collect selected formats
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
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("export.format"))
            return

        if not model_path:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("error.no_model"))
            return

        self._export_btn.setEnabled(False)
        self._progress_bar.setValue(0)

        export_dir = self._dir_edit.text()
        self._exporter = Exporter(export_dir)

        self._export_worker = self._exporter.start_export(
            lora_path=lora_path,
            model_path=model_path,
            export_name=export_name,
            formats=formats,
            on_progress=self._on_export_progress,
            on_finished=self._on_export_finished,
            on_error=self._on_export_error,
        )

    def _on_export_progress(self, percent: int, desc: str):
        self._progress_bar.setValue(percent)
        self._progress_label.setText(desc)

    def _on_export_finished(self, result: dict):
        self._export_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._progress_label.setText(self._i18n.t("export.complete"))
        self._open_dir_btn.setEnabled(True)
        self._refresh_exports()
        self.export_finished.emit(result)

    def _on_export_error(self, error_code: str, detail: str):
        self._export_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        QMessageBox.critical(self, self._i18n.t("common.error"), detail)

    def _on_browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, self._i18n.t("export.dir"))
        if dir_path:
            self._dir_edit.setText(dir_path)
            self._config.set("export_dir", dir_path)

    def _on_open_dir(self):
        export_dir = os.path.join(self._dir_edit.text(), self._name_edit.text().strip())
        if os.path.isdir(export_dir):
            os.startfile(export_dir)

    def _refresh_exports(self):
        """Refresh exported models list"""
        self._export_table.setRowCount(0)
        exports = self._exporter.list_exports()
        for exp in exports:
            row = self._export_table.rowCount()
            self._export_table.insertRow(row)
            self._export_table.setItem(row, 0, QTableWidgetItem(exp["name"]))
            size_mb = exp["size"] / (1024 * 1024) if exp["size"] > 0 else 0
            self._export_table.setItem(row, 1, QTableWidgetItem(f"{exp['file_count']} files"))
            self._export_table.setItem(row, 2, QTableWidgetItem(f"{size_mb:.1f} MB"))
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(exp["export_time"])) if exp["export_time"] else ""
            self._export_table.setItem(row, 3, QTableWidgetItem(t))

    def _detect_ollama(self):
        """Detect if Ollama is installed"""
        if self._deployer.is_installed():
            ver = self._deployer.get_version()
            self._ollama_status_label.setText(f"[OK] {ver}")
            self._ollama_status_label.setStyleSheet("color: #22c55e;")
            self._deploy_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
        else:
            self._ollama_status_label.setText(self._i18n.t("deploy.not_installed"))
            self._ollama_status_label.setStyleSheet("color: #ef4444;")
            self._deploy_btn.setEnabled(False)
            self._run_btn.setEnabled(False)

    def _on_deploy(self):
        """Deploy to Ollama"""
        if not self._deployer.is_installed():
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("deploy.install_hint"))
            return

        ollama_name = self._ollama_name_edit.text().strip()
        if not ollama_name:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("deploy.model_name"))
            return

        export_name = self._name_edit.text().strip() or "exported_model"
        export_dir = os.path.join(self._dir_edit.text(), export_name)
        if not os.path.isdir(export_dir):
            QMessageBox.warning(self, self._i18n.t("common.error"),
                                self._i18n.t("export.no_gguf"))
            return

        # 1. Find GGUF file first
        model_path = self._find_gguf(export_dir)

        # 2. No GGUF -> find safetensors directory
        if not model_path:
            safetensors_dir = self._find_safetensors_dir(export_dir)
            if safetensors_dir:
                model_path = safetensors_dir
            else:
                QMessageBox.warning(self, self._i18n.t("common.error"),
                                    self._i18n.t("export.no_gguf"))
                return

        # 3. Check if it's HuggingFace format (directory contains config.json)
        is_hf_format = os.path.isdir(model_path) and os.path.isfile(
            os.path.join(model_path, "config.json")
        )

        system_prompt = self._system_prompt_edit.toPlainText().strip()
        modelfile = self._deployer.generate_modelfile(
            model_path=model_path,
            model_name=ollama_name,
            system_prompt=system_prompt,
            is_directory=is_hf_format,
        )

        self._deploy_btn.setEnabled(False)
        success, output = self._deployer.create_model(
            modelfile_content=modelfile,
            model_name=ollama_name,
            working_dir=self._config.workspace,
        )
        self._deploy_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, self._i18n.t("common.success"),
                                    self._i18n.t("export.deploy_success") + f" '{ollama_name}'")
        else:
            QMessageBox.critical(self, self._i18n.t("common.error"),
                                 self._i18n.t("export.deploy_fail") + f": {output}")

    def _find_gguf(self, directory: str) -> str:
        """Recursively find GGUF file"""
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".gguf"):
                    return os.path.join(root, f)
        return ""

    def _find_safetensors_dir(self, directory: str) -> str:
        """Find directory containing config.json + safetensors (HuggingFace format)"""
        # Check current directory
        if os.path.isfile(os.path.join(directory, "config.json")):
            for f in os.listdir(directory):
                if f.endswith(".safetensors"):
                    return directory
        # Check subdirectories (e.g. model_16bit/)
        for entry in os.listdir(directory):
            sub = os.path.join(directory, entry)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
                for f in os.listdir(sub):
                    if f.endswith(".safetensors"):
                        return sub
        return ""

    def _on_run(self):
        """Run Ollama model"""
        ollama_name = self._ollama_name_edit.text().strip()
        if not ollama_name:
            return
        success, msg = self._deployer.run_model(ollama_name)
        if not success:
            QMessageBox.critical(self, self._i18n.t("common.error"), msg)

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.export"))
        self._lora_group.setTitle(self._i18n.t("export.lora_adapter"))
        self._format_group.setTitle(self._i18n.t("export.format"))
        self._fmt_16bit.setText(self._i18n.t("format.16bit"))
        self._fmt_q4.setText(self._i18n.t("format.gguf_q4"))
        self._fmt_q8.setText(self._i18n.t("format.gguf_q8"))
        self._fmt_f16.setText(self._i18n.t("format.gguf_f16"))
        self._fmt_lora.setText(self._i18n.t("format.lora_only"))
        self._export_btn.setText(self._i18n.t("common.save"))
        self._open_dir_btn.setText(self._i18n.t("common.browse"))
        self._exports_group.setTitle(self._i18n.t("export.name"))
        self._deploy_group.setTitle(self._i18n.t("deploy.create"))
        self._detect_btn.setText(self._i18n.t("deploy.detect"))
        self._deploy_btn.setText(self._i18n.t("deploy.create"))
        self._run_btn.setText(self._i18n.t("deploy.run"))
        self._dir_browse_btn.setText(self._i18n.t("common.browse"))
        self._name_edit.setPlaceholderText(self._i18n.t("export.name"))
        self._export_table.setHorizontalHeaderLabels([
            self._i18n.t("common.name"),
            self._i18n.t("common.format"),
            self._i18n.t("common.size"),
            self._i18n.t("common.time")
        ])
        self._detect_ollama()

    def showEvent(self, event):
        super().showEvent(event)
        self._load_loras()
        self._refresh_exports()

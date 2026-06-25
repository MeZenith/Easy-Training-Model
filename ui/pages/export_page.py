import logging
import os
import time

from PySide6.QtCore import Qt, QThread, Signal
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

from core.exporter import Exporter
from core.exporter_process import ProcessExporter
from core.model_manager import ModelManager
from core.ollama_deployer import OllamaDeployer
from core.services.export_service import detect_format, find_gguf, find_safetensors_dir
from core.services.train_service import list_loras_for_combo

logger = logging.getLogger("EasyTinking")


class _DeployWorker(QThread):
    #Ollama模型创建后台线程

    finished = Signal(bool, str)

    def __init__(self, deployer, modelfile_content, model_name, working_dir, parent=None):
        super().__init__(parent)
        self._deployer = deployer
        self._modelfile = modelfile_content
        self._model_name = model_name
        self._working_dir = working_dir

    def run(self):
        try:
            success, output = self._deployer.create_model(
                modelfile_content=self._modelfile,
                model_name=self._model_name,
                working_dir=self._working_dir,
            )
            self.finished.emit(success, output)
        except Exception as e:
            self.finished.emit(False, str(e))


class ExportPage(QWidget):
    #导出页：GGUF导出 + Ollama部署

    export_finished = Signal(dict)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._exporter_process = None

        export_dir = config.get("export_dir", "")
        if not export_dir:
            export_dir = os.path.join(config.workspace, "exports")
            config.set("export_dir", export_dir)
        self._exporter = Exporter(export_dir)
        self._deployer = OllamaDeployer()
        self._models_dir = config.get("download_dir", os.path.join(config.workspace, "models"))

        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self._title_label)

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

        #配置区
        g = QGroupBox()
        f = QFormLayout(g)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(200)
        self._model_label = QLabel()
        f.addRow(self._model_label, self._model_combo)

        self._lora_combo = QComboBox()
        self._lora_label = QLabel()
        f.addRow(self._lora_label, self._lora_combo)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(self._config.get("export_dir", ""))
        self._dir_edit.setReadOnly(True)
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse_btn = QPushButton()
        dir_row.addWidget(self._dir_browse_btn)
        self._dir_label = QLabel()
        f.addRow(self._dir_label, dir_row)

        self._name_edit = QLineEdit()
        self._name_label = QLabel()
        f.addRow(self._name_label, self._name_edit)

        layout.addWidget(g)
        self._cfg_group = g

        #导出格式选择
        g2 = QGroupBox()
        fl = QVBoxLayout(g2)
        self._fmt_fp32 = QCheckBox()
        self._fmt_f16 = QCheckBox()
        self._fmt_f16.setChecked(True)
        self._fmt_q8 = QCheckBox()
        self._fmt_q4 = QCheckBox()
        self._fmt_16bit = QCheckBox()
        self._fmt_lora = QCheckBox()
        for cb in [self._fmt_fp32, self._fmt_f16, self._fmt_q8, self._fmt_q4, self._fmt_16bit, self._fmt_lora]:
            fl.addWidget(cb)
        layout.addWidget(g2)
        self._fmt_group = g2

        #导出按钮
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton()
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

        #日志输出
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

        #Ollama状态
        status_row = QHBoxLayout()
        self._ollama_status_label = QLabel()
        self._ollama_detect_btn = QPushButton()
        status_row.addWidget(self._ollama_status_label, 1)
        status_row.addWidget(self._ollama_detect_btn)
        layout.addLayout(status_row)

        #部署配置
        g = QGroupBox()
        f = QFormLayout(g)

        self._export_selector = QComboBox()
        self._export_pick_label = QLabel()
        f.addRow(self._export_pick_label, self._export_selector)

        self._gguf_path_label = QLabel()
        self._gguf_path_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self._gguf_path_label.setWordWrap(True)
        self._deploy_path_label = QLabel()
        f.addRow(self._deploy_path_label, self._gguf_path_label)

        self._ollama_name_edit = QLineEdit(self._config.get("ollama_name", "my-model"))
        self._deploy_name_label = QLabel()
        f.addRow(self._deploy_name_label, self._ollama_name_edit)

        self._system_prompt_edit = QTextEdit()
        self._system_prompt_edit.setMaximumHeight(80)
        self._system_prompt_edit.setPlaceholderText("You are a helpful assistant.")
        self._deploy_system_label = QLabel()
        f.addRow(self._deploy_system_label, self._system_prompt_edit)

        btn_row = QHBoxLayout()
        self._deploy_btn = QPushButton()
        self._deploy_btn.setObjectName("primaryBtn")
        self._run_btn = QPushButton()
        btn_row.addWidget(self._deploy_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        f.addRow(btn_row)
        layout.addWidget(g)
        self._deploy_group = g

        #已导出列表
        g2 = QGroupBox()
        el = QVBoxLayout(g2)
        self._export_table = QTableWidget()
        self._export_table.setColumnCount(4)
        self._export_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._export_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        el.addWidget(self._export_table)

        eb = QHBoxLayout()
        self._export_delete_btn = QPushButton()
        self._export_open_btn = QPushButton()
        eb.addWidget(self._export_delete_btn)
        eb.addWidget(self._export_open_btn)
        eb.addStretch()
        el.addLayout(eb)
        layout.addWidget(g2)
        self._exports_list_group = g2

        #Ollama模型列表
        g3 = QGroupBox()
        ol = QVBoxLayout(g3)
        self._ollama_table = QTableWidget()
        self._ollama_table.setColumnCount(3)
        self._ollama_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ollama_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ol.addWidget(self._ollama_table)

        ob = QHBoxLayout()
        self._ollama_delete_btn = QPushButton()
        self._ollama_run_btn = QPushButton()
        ob.addWidget(self._ollama_delete_btn)
        ob.addWidget(self._ollama_run_btn)
        ob.addStretch()
        ol.addLayout(ob)
        layout.addWidget(g3)
        self._ollama_group = g3

        return layout

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
                item["display"], userData={"lora_path": item["lora_path"], "model_path": item["model_path"]}
            )
        if not loras:
            self._lora_combo.addItem("(No LoRA found)", userData={})

    def _on_lora_changed(self, _index):
        data = self._lora_combo.currentData() or {}
        lora_path = data.get("lora_path", "")
        if lora_path:
            self._name_edit.setText(os.path.basename(lora_path))

    def _on_export(self):
        #开始导出
        t = self._i18n.t
        model_path = self._model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, t("common.warning"), t("export.select_model"))
            return

        export_name = self._name_edit.text().strip() or "exported_model"
        formats = []
        if self._fmt_16bit.isChecked():
            formats.append("16bit")
        if self._fmt_fp32.isChecked():
            formats.append("gguf_FP32")
        if self._fmt_f16.isChecked():
            formats.append("gguf_F16")
        if self._fmt_q8.isChecked():
            formats.append("gguf_Q8_0")
        if self._fmt_q4.isChecked():
            formats.append("gguf_Q4_K_M")
        if self._fmt_lora.isChecked():
            formats.append("lora_only")

        if not formats:
            QMessageBox.warning(self, t("common.warning"), t("export.select_format"))
            return

        lora_data = self._lora_combo.currentData() or {}
        lora_path = lora_data.get("lora_path", "")
        lora_model = lora_data.get("model_path", "")
        if lora_path and lora_model:
            model_path = lora_model
        export_dir = self._dir_edit.text()

        self._export_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._log_output.clear()
        self._status_label.setText(t("export.exporting"))

        cfg = {
            "model_path": model_path,
            "lora_path": lora_path,
            "out_dir": os.path.join(export_dir, export_name),
            "formats": formats,
        }

        self._exporter_process = ProcessExporter(self)
        self._exporter_process.progress.connect(self._on_export_progress)
        self._exporter_process.finished.connect(self._on_export_finished)
        self._exporter_process.error.connect(self._on_export_error)
        self._exporter_process.log_message.connect(self._append_log)
        self._exporter_process.start_export(cfg)

    def _on_export_progress(self, pct, desc):
        self._progress_bar.setValue(pct)
        self._status_label.setText(desc)

    def _on_export_finished(self, result):
        t = self._i18n.t
        self._export_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText(t("export.complete"))
        self._refresh_exports_list()
        self._refresh_export_selector()
        self.export_finished.emit(result)
        errors = result.get("errors", [])
        if errors:
            self._append_log(f"Warnings: {errors}")

    def _on_export_error(self, code, detail):
        t = self._i18n.t
        self._export_btn.setEnabled(True)
        self._status_label.setText(f"{t('common.error')}: {detail}")
        self._append_log(f"ERROR [{code}]: {detail}")

    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self._log_output.append(f"[{ts}] {msg}")

    def _on_browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if d:
            self._dir_edit.setText(d)
            self._config.set("export_dir", d)

    def _refresh_exports_list(self):
        self._export_table.setRowCount(0)
        exports = self._exporter.list_exports()
        for exp in exports:
            r = self._export_table.rowCount()
            self._export_table.insertRow(r)
            self._export_table.setItem(r, 0, QTableWidgetItem(exp["name"]))
            self._export_table.setItem(r, 1, QTableWidgetItem(detect_format(exp["path"])))
            size_mb = exp["size"] / (1024 * 1024) if exp["size"] > 0 else 0
            self._export_table.setItem(r, 2, QTableWidgetItem(f"{size_mb:.0f} MB"))
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(exp["export_time"])) if exp["export_time"] else ""
            self._export_table.setItem(r, 3, QTableWidgetItem(ts))

    def _refresh_export_selector(self):
        self._export_selector.clear()
        exports = self._exporter.list_exports()
        for exp in exports:
            gguf_files = exp.get("gguf_files", []) or []
            if gguf_files:
                for gf in gguf_files:
                    self._export_selector.addItem(
                        f"{exp['name']} / {os.path.basename(gf)}", userData={"dir": exp["path"], "gguf": gf}
                    )
            else:
                self._export_selector.addItem(exp["name"], userData={"dir": exp["path"], "gguf": ""})

    def _on_export_selected(self, _index):
        data = self._export_selector.currentData() or {}
        gguf = data.get("gguf", "")
        if gguf:
            self._gguf_path_label.setText(gguf)
            self._ollama_name_edit.setText(os.path.splitext(os.path.basename(gguf))[0])
        else:
            full_dir = data.get("dir", "")
            found = find_gguf(full_dir) if full_dir else ""
            self._gguf_path_label.setText(found or full_dir)

    def _delete_export(self):
        t = self._i18n.t
        row = self._export_table.currentRow()
        if row < 0:
            return
        name = self._export_table.item(row, 0).text()
        reply = QMessageBox.question(self, t("common.warning"), t("export.delete_confirm"))
        if reply == QMessageBox.Yes:
            import shutil
            shutil.rmtree(os.path.join(self._config.get("export_dir", ""), name), ignore_errors=True)
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

    def _detect_ollama(self):
        #检测ollama安装状态
        t = self._i18n.t
        if self._deployer.is_installed():
            ver = self._deployer.get_version()
            self._ollama_status_label.setText(f"[OK] Ollama {ver}")
            self._ollama_status_label.setStyleSheet("color: #22c55e;")
            self._deploy_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
        else:
            self._ollama_status_label.setText(t("deploy.not_installed"))
            self._ollama_status_label.setStyleSheet("color: #ef4444;")
            self._deploy_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
        self._refresh_ollama_list()

    def _on_deploy(self):
        #部署到ollama
        t = self._i18n.t
        data = self._export_selector.currentData() or {}
        gguf_path = data.get("gguf", "")
        full_dir = data.get("dir", "")

        if gguf_path:
            model_path = gguf_path
        elif full_dir:
            model_path = find_gguf(full_dir) or find_safetensors_dir(full_dir)
        else:
            QMessageBox.warning(self, t("common.warning"), t("export.select_export"))
            return

        if not os.path.exists(model_path):
            QMessageBox.warning(self, t("common.warning"), t("export.model_missing"))
            return

        ollama_name = self._ollama_name_edit.text().strip()
        if not ollama_name:
            QMessageBox.warning(self, t("common.warning"), t("export.enter_name"))
            return

        is_dir = os.path.isdir(model_path)
        system_prompt = self._system_prompt_edit.toPlainText().strip()
        modelfile = self._deployer.generate_modelfile(
            model_path=model_path,
            model_name=ollama_name,
            system_prompt=system_prompt,
            is_directory=is_dir,
        )

        self._deploy_btn.setEnabled(False)
        self._status_label.setText(t("deploy.create") + "...")

        self._deploy_thread = _DeployWorker(
            self._deployer,
            modelfile,
            ollama_name,
            self._config.workspace,
            self,
        )
        self._deploy_thread.finished.connect(self._on_deploy_finished)
        self._deploy_thread.start()

    def _on_deploy_finished(self, success, output):
        t = self._i18n.t
        self._deploy_btn.setEnabled(True)
        self._status_label.setText("")

        if success:
            QMessageBox.information(
                self, t("common.success"), t("export.deploy_success").format(self._ollama_name_edit.text().strip())
            )
            self._refresh_ollama_list()
        else:
            logger.error("Ollama deploy failed: %s", output)
            QMessageBox.critical(self, t("common.error"), f"{t('export.deploy_fail')}:\n{output}")

    def _on_run_ollama(self):
        t = self._i18n.t
        ollama_name = self._ollama_name_edit.text().strip()
        if not ollama_name:
            row = self._ollama_table.currentRow()
            if row >= 0:
                ollama_name = self._ollama_table.item(row, 0).text()
        if not ollama_name:
            return
        success, msg = self._deployer.run_model(ollama_name)
        if not success:
            QMessageBox.critical(self, t("common.error"), msg)

    def _refresh_ollama_list(self):
        self._ollama_table.setRowCount(0)
        if not self._deployer.is_installed():
            return
        for m in self._deployer.list_models():
            r = self._ollama_table.rowCount()
            self._ollama_table.insertRow(r)
            self._ollama_table.setItem(r, 0, QTableWidgetItem(m.get("name", "")))
            self._ollama_table.setItem(r, 1, QTableWidgetItem(m.get("size", "")))
            self._ollama_table.setItem(r, 2, QTableWidgetItem(m.get("modified", "")))

    def _delete_ollama_model(self):
        t = self._i18n.t
        row = self._ollama_table.currentRow()
        if row < 0:
            return
        name = self._ollama_table.item(row, 0).text()
        reply = QMessageBox.question(self, t("common.warning"), f"{t('common.delete')} '{name}'?")
        if reply == QMessageBox.Yes:
            success, msg = self._deployer.delete_model(name)
            if not success:
                QMessageBox.critical(self, t("common.error"), msg)
            self._refresh_ollama_list()

    def _refresh_texts(self):
        t = self._i18n.t
        self._title_label.setText(t("nav.export"))
        self._cfg_group.setTitle(t("export.lora_adapter"))
        self._model_label.setText(t("export.base_model") + ":")
        self._lora_label.setText(t("train.lora_path") + ":")
        self._dir_label.setText(t("export.dir") + ":")
        self._name_label.setText(t("export.name") + ":")
        self._dir_browse_btn.setText(t("common.browse"))
        self._fmt_group.setTitle(t("export.format"))
        self._fmt_16bit.setText(t("format.16bit"))
        self._fmt_fp32.setText(t("format.gguf_fp32"))
        self._fmt_f16.setText(t("format.gguf_f16"))
        self._fmt_q8.setText(t("format.gguf_q8"))
        self._fmt_q4.setText(t("format.gguf_q4"))
        self._fmt_lora.setText(t("format.lora_only"))
        self._export_btn.setText(t("export.start"))
        self._exports_list_group.setTitle(t("export.exports_title"))
        self._export_delete_btn.setText(t("common.delete"))
        self._export_open_btn.setText(t("export.open_dir"))
        self._export_pick_label.setText(t("export.name") + ":")
        self._deploy_path_label.setText(t("export.dir") + ":")
        self._deploy_name_label.setText(t("deploy.model_name") + ":")
        self._deploy_system_label.setText(t("deploy.system_prompt") + ":")
        self._deploy_group.setTitle(t("deploy.create"))
        self._ollama_group.setTitle(t("export.ollama_title"))
        self._deploy_btn.setText(t("deploy.create"))
        self._run_btn.setText(t("deploy.run"))
        self._ollama_detect_btn.setText(t("deploy.detect"))
        self._ollama_delete_btn.setText(t("common.delete"))
        self._ollama_run_btn.setText(t("deploy.run"))
        self._export_table.setHorizontalHeaderLabels(
            [t("common.name"), t("common.format"), t("common.size"), t("common.time")]
        )
        self._ollama_table.setHorizontalHeaderLabels([t("common.name"), t("common.size"), t("common.time")])
        self._system_prompt_edit.setPlaceholderText("You are a helpful assistant.")
        self._detect_ollama()

    def showEvent(self, event):
        super().showEvent(event)
        self._load_models()
        self._load_loras()
        self._refresh_exports_list()
        self._refresh_export_selector()
        self._detect_ollama()

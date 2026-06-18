"""Training Center - Config page + Monitor page dual layout"""

import os
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QScrollArea, QFrame, QProgressBar,
    QMessageBox, QTextEdit, QStackedWidget, QListWidget, QListWidgetItem,
    QGridLayout, QSizePolicy, QSpacerItem,
)
from PySide6.QtCore import Qt, Signal, QTimer

from core.trainer import ProcessTrainer
from core.error_handler import friendly_error_message
from core.model_manager import ModelManager
from core.data_manager import DataManager
from ui.components.loss_chart import LossChart
from ui.components.gpu_monitor import GPUMonitor

logger = logging.getLogger("EasyTinking")


class TrainPage(QWidget):
    training_started = Signal()
    training_finished = Signal(dict)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._trainer = ProcessTrainer(config.workspace)
        self._is_training = False
        self._paused = False
        self._loss_values = []
        self._setup_ui()
        self._connect_signals()
        self._attach_trainer_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    # ================ UI Setup ================

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_config_page())
        self._stack.addWidget(self._build_monitor_page())
        main_layout.addWidget(self._stack, 1)

        self._load_models()
        self._apply_preset("standard")
        self._refresh_texts()

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        # Row 1: Model + Dataset + Identity
        top_row = QHBoxLayout()

        model_group = QGroupBox()
        model_group.setObjectName("model_group")
        mf = QFormLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(250)
        mf.addRow(self._i18n.t("nav.model") + ":", self._model_combo)
        self._model_info_label = QLabel()
        self._model_info_label.setObjectName("label-muted")
        self._model_info_label.setStyleSheet("font-size: 12px;")
        mf.addRow("", self._model_info_label)
        top_row.addWidget(model_group)
        self._model_group = model_group

        data_group = QGroupBox()
        data_group.setObjectName("data_group")
        dl = QVBoxLayout(data_group)
        self._dataset_list = QListWidget()
        self._dataset_list.setSelectionMode(QListWidget.NoSelection)
        self._dataset_list.setMaximumHeight(150)
        dl.addWidget(self._dataset_list)
        self._total_data_label = QLabel()
        self._total_data_label.setObjectName("label-secondary")
        self._total_data_label.setStyleSheet("font-size: 12px;")
        dl.addWidget(self._total_data_label)
        top_row.addWidget(data_group)
        self._data_group = data_group

        identity_group = QGroupBox()
        identity_group.setObjectName("identity_group")
        idf = QFormLayout(identity_group)
        self._identity_btn = QPushButton()
        self._identity_btn.clicked.connect(self._on_edit_identity)
        idf.addRow(self._i18n.t("train.identity_name") + ":", self._identity_btn)
        top_row.addWidget(identity_group)
        self._identity_group = identity_group

        cl.addLayout(top_row)

        # Row 2: Preset + Training Params + Advanced Params
        params_row = QHBoxLayout()

        preset_group = QGroupBox()
        preset_group.setObjectName("preset_group")
        pf = QFormLayout(preset_group)
        self._preset_combo = QComboBox()
        pf.addRow(self._i18n.t("train.preset") + ":", self._preset_combo)
        self._est_vram_label = QLabel()
        self._est_vram_label.setObjectName("label-muted")
        pf.addRow(self._i18n.t("train.mem_estimate") + ":", self._est_vram_label)
        params_row.addWidget(preset_group)
        self._preset_group = preset_group

        train_params_group = QGroupBox()
        train_params_group.setObjectName("train_params_group")
        tpf = QFormLayout(train_params_group)

        self._lora_rank_spin = QSpinBox()
        self._lora_rank_spin.setRange(1, 128)
        self._lora_rank_spin.setValue(16)
        tpf.addRow(self._i18n.t("train.lora_rank") + ":", self._lora_rank_spin)

        self._lora_alpha_spin = QSpinBox()
        self._lora_alpha_spin.setRange(1, 256)
        self._lora_alpha_spin.setValue(16)
        tpf.addRow(self._i18n.t("train.lora_alpha") + ":", self._lora_alpha_spin)

        self._lora_dropout_spin = QDoubleSpinBox()
        self._lora_dropout_spin.setRange(0, 0.5)
        self._lora_dropout_spin.setDecimals(2)
        self._lora_dropout_spin.setSingleStep(0.05)
        tpf.addRow(self._i18n.t("train.lora_dropout") + ":", self._lora_dropout_spin)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 20)
        self._epochs_spin.setValue(3)
        tpf.addRow(self._i18n.t("train.epochs") + ":", self._epochs_spin)

        self._batch_combo = QComboBox()
        self._batch_combo.addItems(["1", "2", "4", "8"])
        tpf.addRow(self._i18n.t("train.batch_size") + ":", self._batch_combo)

        self._grad_accum_combo = QComboBox()
        self._grad_accum_combo.addItems(["1", "2", "4", "8"])
        tpf.addRow(self._i18n.t("train.grad_accum") + ":", self._grad_accum_combo)

        self._advanced_toggle = QPushButton("+ " + self._i18n.t("train.more_params"))
        self._advanced_toggle.setObjectName("primaryBtn")
        self._advanced_toggle.setMaximumHeight(28)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        tpf.addRow("", self._advanced_toggle)

        params_row.addWidget(train_params_group)
        self._train_params_group = train_params_group

        more_params_group = QGroupBox()
        more_params_group.setObjectName("more_params_group")
        more_params_group.setVisible(False)
        mpf = QFormLayout(more_params_group)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.00001, 0.001)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setSingleStep(0.00001)
        self._lr_spin.setValue(0.0002)
        mpf.addRow(self._i18n.t("train.lr") + ":", self._lr_spin)

        self._scheduler_combo = QComboBox()
        self._scheduler_combo.addItems(["cosine", "linear", "constant"])
        mpf.addRow(self._i18n.t("train.scheduler") + ":", self._scheduler_combo)

        self._warmup_spin = QSpinBox()
        self._warmup_spin.setRange(0, 50)
        self._warmup_spin.setValue(5)
        mpf.addRow(self._i18n.t("train.warmup") + ":", self._warmup_spin)

        self._max_seq_combo = QComboBox()
        self._max_seq_combo.addItems(["512", "1024", "2048", "4096"])
        self._max_seq_combo.setCurrentText("2048")
        mpf.addRow(self._i18n.t("train.max_seq") + ":", self._max_seq_combo)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(["adamw_8bit", "adamw"])
        mpf.addRow(self._i18n.t("train.optimizer") + ":", self._optimizer_combo)

        self._quant_combo = QComboBox()
        self._quant_combo.addItems(["4bit", "8bit", "none"])
        mpf.addRow(self._i18n.t("train.quantization") + ":", self._quant_combo)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(3407)
        mpf.addRow(self._i18n.t("train.seed") + ":", self._seed_spin)

        params_row.addWidget(more_params_group)
        self._more_params_group = more_params_group

        cl.addLayout(params_row)

        # Row 3: Pre-check + GPU
        check_row = QHBoxLayout()

        check_group = QGroupBox()
        check_group.setObjectName("check_group")
        ckl = QVBoxLayout(check_group)
        self._check_model_label = QLabel()
        self._check_data_label = QLabel()
        self._check_vram_label = QLabel()
        self._check_disk_label = QLabel()
        for lbl in [self._check_model_label, self._check_data_label,
                    self._check_vram_label, self._check_disk_label]:
            ckl.addWidget(lbl)
        self._run_check_btn = QPushButton()
        self._run_check_btn.clicked.connect(self._run_pre_check)
        ckl.addWidget(self._run_check_btn)
        check_row.addWidget(check_group)
        self._check_group = check_group

        gpu_group = QGroupBox()
        gpu_group.setObjectName("gpu_group")
        gl = QVBoxLayout(gpu_group)
        self._gpu_monitor = GPUMonitor()
        gl.addWidget(self._gpu_monitor)
        check_row.addWidget(gpu_group)
        self._gpu_group = gpu_group

        cl.addLayout(check_row)

        # Start training button
        self._start_btn = QPushButton()
        self._start_btn.setObjectName("primaryBtn")
        self._start_btn.setMinimumHeight(44)
        self._start_btn.setStyleSheet(
            "QPushButton#primaryBtn { font-size: 15px; font-weight: bold; }"
        )
        cl.addWidget(self._start_btn)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return page

    def _build_monitor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Title + training info
        self._monitor_title = QLabel()
        self._monitor_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self._monitor_title)

        self._monitor_info = QLabel()
        self._monitor_info.setObjectName("label-secondary")
        self._monitor_info.setWordWrap(True)
        self._monitor_info.setStyleSheet("font-size: 13px; padding: 4px 0;")
        layout.addWidget(self._monitor_info)

        # Control buttons
        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton()
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setMinimumWidth(100)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        self._back_btn = QPushButton()
        self._back_btn.setMinimumHeight(36)
        self._back_btn.setVisible(False)
        btn_row.addWidget(self._back_btn)
        layout.addLayout(btn_row)

        # Loss chart
        self._loss_chart = LossChart()
        self._loss_chart.setMinimumHeight(300)
        layout.addWidget(self._loss_chart, 1)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        self._step_label = QLabel()
        self._step_label.setObjectName("label-secondary")
        self._step_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._step_label)

        # Result panel
        self._result_panel = QWidget()
        self._result_panel.setVisible(False)
        result_layout = QVBoxLayout(self._result_panel)
        result_layout.setContentsMargins(0, 8, 0, 0)
        result_layout.setSpacing(4)

        result_title = QLabel()
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #22c55e;")
        result_title.setText(self._i18n.t("train.training_complete"))
        result_layout.addWidget(result_title)
        self._result_title_label = result_title

        self._result_grid = QGridLayout()
        self._result_grid.setSpacing(6)
        result_layout.addLayout(self._result_grid)

        layout.addWidget(self._result_panel)
        return page

    # ================ Signals ================

    def _connect_signals(self):
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._back_btn.clicked.connect(self._on_back)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._dataset_list.itemChanged.connect(lambda: self._update_data_label())
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

    def _attach_trainer_signals(self):
        self._trainer.progress.connect(self._on_train_progress)
        self._trainer.finished.connect(self._on_train_finished)
        self._trainer.error.connect(self._on_train_error)
        self._trainer.log_message.connect(self._on_train_log)
        self._trainer.metric.connect(self._on_metric)

    # ================ Data Loading ================

    def _load_datasets(self):
        self._dataset_list.clear()
        data_dir = os.path.join(self._config.workspace, "data")
        self._data_manager = DataManager(data_dir)
        for name in sorted(self._data_manager.list_names()):
            ds = self._data_manager.get(name)
            if ds:
                item = QListWidgetItem(f"{name} ({ds.count})")
                item.setData(Qt.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self._dataset_list.addItem(item)
        self._update_data_label()

    def _update_data_label(self):
        total = 0
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item and item.checkState() == Qt.Checked:
                ds = self._data_manager.get(item.data(Qt.UserRole))
                if ds:
                    total += ds.count
        self._total_data_label.setText(f"{self._i18n.t('data.count')}: {total}")

    def _load_models(self):
        self._model_combo.clear()
        download_dir = self._config.get("download_dir", "")
        if not download_dir:
            download_dir = os.path.join(self._config.workspace, "models")
        manager = ModelManager(download_dir, self._config.get("hf_mirror", ""))
        for m in manager.list_downloaded_models():
            if m["status"] == "ok":
                self._model_combo.addItem(f"{m['name']} ({m['params']})", m["path"])

    def _toggle_advanced(self):
        visible = not self._more_params_group.isVisible()
        self._more_params_group.setVisible(visible)
        prefix = "- " if visible else "+ "
        self._advanced_toggle.setText(prefix + self._i18n.t("train.more_params"))

    # ================ Presets ================

    def _on_preset_changed(self, index):
        data = self._preset_combo.currentData()
        if data:
            self._apply_preset(data)

    def _apply_preset(self, preset: str):
        presets = self._config.get("train_presets", {})
        if preset not in presets:
            return
        p = presets[preset]
        self._lora_rank_spin.setValue(p.get("lora_rank", 16))
        self._epochs_spin.setValue(p.get("epochs", 3))
        self._batch_combo.setCurrentText(str(p.get("batch_size", 1)))
        self._lr_spin.setValue(p.get("learning_rate", 0.0002))
        if "max_seq_length" in p:
            self._max_seq_combo.setCurrentText(str(p["max_seq_length"]))
        vram_estimates = {"quick": "~5 GB", "standard": "~6 GB", "fine": "~7 GB", "custom": "-"}
        self._est_vram_label.setText(vram_estimates.get(preset, "-"))

    # ================ Config ================

    def _on_model_changed(self, index):
        model_path = self._model_combo.currentData()
        if model_path:
            valid, missing = ModelManager.validate_model(model_path)
            if valid:
                self._model_info_label.setText(f"[OK] {model_path}")
            else:
                self._model_info_label.setText(f"[!] Missing: {', '.join(missing)}")
        else:
            self._model_info_label.setText("")

    def _build_train_config(self) -> dict:
        return {
            "model_path": self._model_combo.currentData() or "",
            "data": [],
            "dataset_names": [],
            "lora_name": self._config.get("model_identity.name", "") or "untitled",
            "lora_rank": self._lora_rank_spin.value(),
            "lora_alpha": self._lora_alpha_spin.value(),
            "lora_dropout": self._lora_dropout_spin.value(),
            "target_modules": self._config.get("default_train_params.target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]),
            "epochs": self._epochs_spin.value(),
            "batch_size": int(self._batch_combo.currentText()),
            "grad_accum": int(self._grad_accum_combo.currentText()),
            "learning_rate": self._lr_spin.value(),
            "lr_scheduler": self._scheduler_combo.currentText(),
            "warmup_steps": self._warmup_spin.value(),
            "max_seq_length": int(self._max_seq_combo.currentText()),
            "optimizer": self._optimizer_combo.currentText(),
            "weight_decay": self._config.get("default_train_params.weight_decay", 0.01),
            "seed": self._seed_spin.value(),
            "quantization": self._quant_combo.currentText(),
        }

    # ================ Training Controls ================

    def _on_start(self):
        config = self._build_train_config()
        if not config["model_path"]:
            QMessageBox.warning(self, self._i18n.t("common.warning"), self._i18n.t("error.no_model"))
            return

        self._config.set("last_train.model_path", config["model_path"])
        self._config.set("last_train.lora_rank", config["lora_rank"])
        self._config.set("last_train.lora_alpha", config["lora_alpha"])
        self._config.set("last_train.epochs", config["epochs"])
        self._config.set("last_train.batch_size", config["batch_size"])
        self._config.set("last_train.learning_rate", config["learning_rate"])
        self._config.set("last_train.max_seq_length", config["max_seq_length"])

        all_data = []
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item and item.checkState() == Qt.Checked:
                ds = self._data_manager.get(item.data(Qt.UserRole))
                if ds:
                    all_data.extend(ds.data)
                    config["dataset_names"].append(f"{item.data(Qt.UserRole)} ({ds.count})")

        identity_name = self._config.get("model_identity.name", "")
        if identity_name:
            identity_data = DataManager.generate_identity_data(
                name=identity_name,
                creator=self._config.get("model_identity.creator", ""),
                description=self._config.get("model_identity.description", ""),
            )
            all_data.extend(identity_data)
            config["dataset_names"].append(f"identity ({len(identity_data)})")

        config["data"] = all_data
        if not config["data"]:
            QMessageBox.warning(self, self._i18n.t("common.warning"), self._i18n.t("error.no_data"))
            return

        self._is_training = True
        self._loss_values = []

        model_name = os.path.basename(config["model_path"])
        self._monitor_title.setText(f"[T] {model_name}")
        self._monitor_info.setText(
            f"{self._i18n.t('data.count')}: {len(config['data'])}, "
            f"Epoch: {config['epochs']}, Batch: {config['batch_size']}"
        )
        self._step_label.setText(self._i18n.t("common.loading"))
        self._progress_bar.setValue(0)
        self._loss_chart.clear()
        self._result_panel.setVisible(False)
        self._back_btn.setVisible(False)

        self._stack.setCurrentIndex(1)
        self._trainer.start_training(config)
        self.training_started.emit()

    def _on_stop(self):
        reply = QMessageBox.question(
            self, self._i18n.t("common.confirm"),
            self._i18n.t("train.stop") + "?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._trainer.stop_training()
            self._step_label.setText(self._i18n.t("train.stop"))
            QMessageBox.information(self, self._i18n.t("common.success"), "Training stopped")

    def _on_back(self):
        self._stack.setCurrentIndex(0)

    # ================ Training Callbacks ================

    def _on_train_progress(self, percent: int, desc: str):
        self._progress_bar.setValue(percent)
        self._step_label.setText(desc)

    def _on_train_log(self, msg: str):
        logger.info(msg)

    def _on_metric(self, data: dict):
        step = data["step"]
        loss_val = data["loss"]
        lr_val = data.get("lr", 0)
        self._loss_values.append(loss_val)
        self._loss_chart.add_point(step, loss_val, lr_val)

    def _on_train_finished(self, result: dict):
        self._is_training = False
        self._progress_bar.setValue(100)
        self._step_label.setText(self._i18n.t("train.complete"))
        self._loss_chart.update_data(self._loss_values)
        self._show_result(result)
        self._back_btn.setVisible(True)
        self.training_finished.emit(result)

    def _show_result(self, r: dict):
        self._result_panel.setVisible(True)
        self._result_title_label.setText(self._i18n.t("train.training_complete"))

        # Clear old data
        while self._result_grid.count():
            item = self._result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def add_row(col, row, label, value):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888; font-size: 12px;")
            val = QLabel(str(value))
            val.setStyleSheet("font-weight: bold; font-size: 13px;")
            self._result_grid.addWidget(lbl, row, col * 2)
            self._result_grid.addWidget(val, row, col * 2 + 1)

        data = [
            (0, 0, self._i18n.t("test.gen_time_label"), f"{r.get('elapsed_seconds', 0):.1f} s"),
            (0, 1, self._i18n.t("train.epochs"), str(r.get("epochs", "-"))),
            (0, 2, self._i18n.t("train.loss_final"), f"{r.get('final_loss', 0):.4f}"),
            (0, 3, self._i18n.t("train.initial_loss"), f"{r.get('initial_loss', 0):.4f}"),
            (0, 4, self._i18n.t("train.loss_drop"), f"{r.get('loss_drop_pct', 0):.1f}%"),
            (1, 0, self._i18n.t("train.lr"), f"{r.get('learning_rate', 0):.6f}"),
            (1, 1, self._i18n.t("train.lora_rank"), f"r={r.get('lora_rank', '-')} a={r.get('lora_alpha', '-')}"),
            (1, 2, self._i18n.t("train.batch_size"), f"{r.get('batch_size', '-')} x {r.get('grad_accum', '-')}"),
            (1, 3, self._i18n.t("train.total_samples"), str(r.get("total_samples", "-"))),
            (1, 4, self._i18n.t("train.base_model"), os.path.basename(r.get("model_path", "")) or "-"),
        ]
        for col, row, label, value in data:
            add_row(col, row, label, value)

    def _on_train_error(self, error_code: str, detail: str):
        self._is_training = False
        key = f"error.{error_code.lower().replace('err_', '')}"
        text = self._i18n.t(key)
        if text == key:
            text = detail
        QMessageBox.critical(self, self._i18n.t("common.error"), text)
        self._back_btn.setVisible(True)

    # ================ Identity Dialog ================

    def _on_edit_identity(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QTextEdit, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(self._i18n.t("train.identity"))
        fl = QFormLayout(dlg)
        name = QLineEdit(self._config.get("model_identity.name", ""))
        creator = QLineEdit(self._config.get("model_identity.creator", ""))
        desc = QTextEdit()
        desc.setMaximumHeight(60)
        desc.setText(self._config.get("model_identity.description", ""))
        fl.addRow(self._i18n.t("train.identity_name") + ":", name)
        fl.addRow(self._i18n.t("train.identity_creator") + ":", creator)
        fl.addRow(self._i18n.t("train.identity_desc") + ":", desc)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        fl.addRow(btns)
        if dlg.exec() == QDialog.Accepted:
            self._config.set("model_identity.name", name.text().strip())
            self._config.set("model_identity.creator", creator.text().strip())
            self._config.set("model_identity.description", desc.toPlainText().strip())
            self._update_identity_label()

    def _update_identity_label(self):
        name = self._config.get("model_identity.name", "")
        self._identity_btn.setText(name or self._i18n.t("train.identity"))

    # ================ Pre-check ================

    def _run_pre_check(self):
        all_pass = True
        model_path = self._model_combo.currentData()
        if model_path:
            valid, missing = ModelManager.validate_model(model_path)
            self._check_model_label.setText(
                f"[+] {self._i18n.t('train.check_model')}" if valid
                else f"[-] Missing: {', '.join(missing)}")
            self._check_model_label.setStyleSheet(
                "color: #22c55e;" if valid else "color: #ef4444;")
            if not valid:
                all_pass = False
        else:
            self._check_model_label.setText(f"[-] {self._i18n.t('error.no_model')}")
            self._check_model_label.setStyleSheet("color: #ef4444;")
            all_pass = False

        self._check_data_label.setText(f"[~] {self._i18n.t('train.check_data')}")
        self._check_data_label.setStyleSheet("color: #f59e0b;")

        from utils.gpu_info import get_gpu_info
        gpus = get_gpu_info()
        if gpus:
            g = gpus[0]
            pct = int(g["vram_used_mb"] / g["vram_total_mb"] * 100) if g["vram_total_mb"] > 0 else 0
            ok = pct < 90
            self._check_vram_label.setText(
                f"[{'+' if ok else '-'}] {self._i18n.t('train.check_vram')}: "
                f"{g['vram_free_mb']}/{g['vram_total_mb']} MB free")
            self._check_vram_label.setStyleSheet("color: #22c55e;" if ok else "color: #ef4444;")
            if not ok:
                all_pass = False
        else:
            self._check_vram_label.setText("[-] No GPU detected")
            self._check_vram_label.setStyleSheet("color: #ef4444;")

        import shutil
        try:
            usage = shutil.disk_usage(self._config.workspace)
            free_gb = usage.free / (1024 ** 3)
            ok = free_gb > 5
            self._check_disk_label.setText(
                f"[{'+' if ok else '-'}] {self._i18n.t('train.check_disk')}: {free_gb:.1f} GB free")
            self._check_disk_label.setStyleSheet("color: #22c55e;" if ok else "color: #ef4444;")
        except OSError:
            self._check_disk_label.setText("[~] Disk check skipped")
        return all_pass

    # ================ Refresh ================

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.train"))
        self._model_group.setTitle(self._i18n.t("nav.model"))
        self._data_group.setTitle(self._i18n.t("train.datasets_used"))
        self._identity_group.setTitle(self._i18n.t("train.identity"))
        self._preset_group.setTitle(self._i18n.t("train.preset"))
        self._train_params_group.setTitle(self._i18n.t("train.config"))
        self._more_params_group.setTitle(self._i18n.t("train.more_params"))
        self._check_group.setTitle(self._i18n.t("train.check_before"))
        self._run_check_btn.setText(self._i18n.t("train.check_before"))
        self._start_btn.setText(self._i18n.t("train.start"))
        self._stop_btn.setText(self._i18n.t("train.stop"))
        self._gpu_group.setTitle(self._i18n.t("train.gpu_mem"))
        self._back_btn.setText(self._i18n.t("train.back"))
        self._update_identity_label()

        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem(self._i18n.t("train.preset.quick"), "quick")
        self._preset_combo.addItem(self._i18n.t("train.preset.standard"), "standard")
        self._preset_combo.addItem(self._i18n.t("train.preset.fine"), "fine")
        self._preset_combo.addItem(self._i18n.t("train.preset.custom"), "custom")
        self._preset_combo.setCurrentIndex(1)  # standard
        self._preset_combo.blockSignals(False)
        self._apply_preset("standard")

    def showEvent(self, event):
        super().showEvent(event)
        self._load_models()
        self._load_datasets()
        self._restore_last_config()

    def _restore_last_config(self):
        last = self._config.get("last_train")
        if not last:
            return
        model_path = last.get("model_path", "")
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model_path:
                self._model_combo.setCurrentIndex(i)
                break
        if last.get("lora_rank"):
            self._lora_rank_spin.setValue(last["lora_rank"])
        if last.get("lora_alpha"):
            self._lora_alpha_spin.setValue(last["lora_alpha"])
        if last.get("epochs"):
            self._epochs_spin.setValue(last["epochs"])
        if last.get("batch_size"):
            self._batch_combo.setCurrentText(str(last["batch_size"]))
        if last.get("learning_rate"):
            self._lr_spin.setValue(last["learning_rate"])
        if last.get("max_seq_length"):
            self._max_seq_combo.setCurrentText(str(last["max_seq_length"]))

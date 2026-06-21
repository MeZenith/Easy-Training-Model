"""Training config panel — model/dataset selection, parameters, presets, pre-check"""

import logging
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.data_manager import DataManager
from core.model_manager import ModelManager
from ui.components.gpu_monitor import GPUMonitor

logger = logging.getLogger("EasyTinking")


class TrainConfigPanel(QWidget):
    """Training configuration panel — model, dataset, LoRA params, presets, pre-checks

    Signals:
        start_requested: user clicked start training — caller should collect config and start
    """

    start_requested = Signal()

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._data_manager = None
        self._setup_ui()
        self._connect_signals()
        self.load_models()
        self._apply_preset("standard")
        self.refresh_texts()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
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
        idf = QFormLayout(identity_group)
        self._identity_btn = QPushButton()
        self._identity_btn.clicked.connect(self._on_edit_identity)
        idf.addRow(self._i18n.t("train.identity_name") + ":", self._identity_btn)

        self._lora_name_edit = QLineEdit()
        self._lora_name_edit.setPlaceholderText(self._i18n.t("export.name"))
        idf.addRow(self._i18n.t("export.name") + ":", self._lora_name_edit)
        top_row.addWidget(identity_group)
        self._identity_group = identity_group

        cl.addLayout(top_row)

        # Row 2: Preset + Training Params + Advanced Params
        params_row = QHBoxLayout()

        preset_group = QGroupBox()
        pf = QFormLayout(preset_group)
        self._preset_combo = QComboBox()
        pf.addRow(self._i18n.t("train.preset") + ":", self._preset_combo)
        self._est_vram_label = QLabel()
        self._est_vram_label.setObjectName("label-muted")
        pf.addRow(self._i18n.t("train.mem_estimate") + ":", self._est_vram_label)
        params_row.addWidget(preset_group)
        self._preset_group = preset_group

        train_params_group = QGroupBox()
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

        self._advanced_toggle = QPushButton(self._i18n.t("train.more_params_show"))
        self._advanced_toggle.setObjectName("primaryBtn")
        self._advanced_toggle.setMaximumHeight(28)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        tpf.addRow("", self._advanced_toggle)

        params_row.addWidget(train_params_group)
        self._train_params_group = train_params_group

        more_params_group = QGroupBox()
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
        gl = QVBoxLayout(gpu_group)
        self._gpu_monitor = GPUMonitor(parent=self, i18n=self._i18n)
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
        self._start_btn.clicked.connect(self.start_requested.emit)
        cl.addWidget(self._start_btn)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _connect_signals(self):
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._dataset_list.itemChanged.connect(lambda: self._update_data_label())
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

    # ---- Public API ----

    def get_train_config(self) -> dict:
        """Build training configuration dictionary from UI values"""
        return {
            "model_path": self._model_combo.currentData() or "",
            "data": [],
            "dataset_names": [],
            "lora_name": self._lora_name_edit.text().strip() or self._config.get("model_identity.name", "") or "untitled",
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

    def get_checked_datasets(self) -> list:
        """Get checked datasets with identity data appended"""
        all_data = []
        dataset_names = []
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item and item.checkState() == Qt.Checked:
                ds = self._data_manager.get(item.data(Qt.UserRole))
                if ds:
                    all_data.extend(ds.data)
                    dataset_names.append(f"{item.data(Qt.UserRole)} ({ds.count})")

        identity_name = self._config.get("model_identity.name", "")
        if identity_name:
            identity_data = DataManager.generate_identity_data(
                name=identity_name,
                creator=self._config.get("model_identity.creator", ""),
                description=self._config.get("model_identity.description", ""),
            )
            all_data.extend(identity_data)
            dataset_names.append(f"identity ({len(identity_data)})")

        return all_data, dataset_names

    def load_models(self):
        """Refresh model combo from disk"""
        self._model_combo.clear()
        download_dir = self._config.get("download_dir", "")
        if not download_dir:
            download_dir = os.path.join(self._config.workspace, "models")
        manager = ModelManager(download_dir, self._config.get("hf_mirror", ""))
        for m in manager.list_downloaded_models():
            if m["status"] == "ok":
                self._model_combo.addItem(f"{m['name']} ({m['params']})", m["path"])

    def load_datasets(self):
        """Refresh dataset list"""
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

    def restore_last_config(self):
        """Restore last training config from saved state"""
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

    def run_pre_check(self) -> bool:
        """Run all pre-flight checks, return True if all pass"""
        return self._run_pre_check()

    # ---- Internal helpers ----

    def _update_data_label(self):
        total = 0
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item and item.checkState() == Qt.Checked:
                ds = self._data_manager.get(item.data(Qt.UserRole))
                if ds:
                    total += ds.count
        self._total_data_label.setText(f"{self._i18n.t('data.count')}: {total}")

    def _toggle_advanced(self):
        visible = not self._more_params_group.isVisible()
        self._more_params_group.setVisible(visible)
        prefix = self._i18n.t("train.more_params_hide") if visible else self._i18n.t("train.more_params_show")
        self._advanced_toggle.setText(prefix)

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

    def _on_edit_identity(self):
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

    def _run_pre_check(self) -> bool:
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
            threshold = self._config.get("ui_constants.training.vram_margin_pct", 90)
            ok = pct < threshold
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
            min_free = self._config.get("ui_constants.training.disk_min_free_gb", 5)
            ok = free_gb > min_free
            self._check_disk_label.setText(
                f"[{'+' if ok else '-'}] {self._i18n.t('train.check_disk')}: {free_gb:.1f} GB free")
            self._check_disk_label.setStyleSheet("color: #22c55e;" if ok else "color: #ef4444;")
        except OSError:
            self._check_disk_label.setText("[~] Disk check skipped")
        return all_pass

    def refresh_texts(self):
        """Update all UI text for language switch"""
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
        self._gpu_group.setTitle(self._i18n.t("train.gpu_mem"))
        self._update_identity_label()

        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem(self._i18n.t("train.preset.quick"), "quick")
        self._preset_combo.addItem(self._i18n.t("train.preset.standard"), "standard")
        self._preset_combo.addItem(self._i18n.t("train.preset.fine"), "fine")
        self._preset_combo.addItem(self._i18n.t("train.preset.custom"), "custom")
        self._preset_combo.setCurrentIndex(1)
        self._preset_combo.blockSignals(False)
        self._apply_preset("standard")

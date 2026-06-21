"""Model Management page - Full implementation"""

import logging
import os
import shutil

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.error_handler import friendly_error_message
from core.model_manager import BUILTIN_MODELS, ModelManager
from core.trainer import ProcessTrainer
from core.workers.download_worker import DownloadWorker
from ui.components.model_card import ModelCard

logger = logging.getLogger("EasyTinking")


class ModelPage(QWidget):
    model_selected = Signal(str)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._manager = None
        self._download_worker = None
        self._custom_model_id = ""
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title row
        title_row = QHBoxLayout()
        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        title_row.addWidget(self._title_label)
        title_row.addStretch()
        self._refresh_btn = QPushButton()
        title_row.addWidget(self._refresh_btn)
        layout.addLayout(title_row)

        # Main content area
        content_split = QHBoxLayout()
        layout.addLayout(content_split, 1)

        # Left: Downloaded models list
        left_panel = QVBoxLayout()
        content_split.addLayout(left_panel, 3)

        self._downloaded_label = QLabel()
        self._downloaded_label.setObjectName("label-secondary")
        left_panel.addWidget(self._downloaded_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._model_list_container = QWidget()
        self._model_list_layout = QVBoxLayout(self._model_list_container)
        self._model_list_layout.setAlignment(Qt.AlignTop)
        self._model_list_layout.setSpacing(8)
        scroll.setWidget(self._model_list_container)
        left_panel.addWidget(scroll, 1)

        # Right: Operations panel
        right_panel = QVBoxLayout()
        content_split.addLayout(right_panel, 2)

        # Download area
        download_group = QGroupBox()
        download_layout = QVBoxLayout(download_group)

        # Built-in model selection
        form_row = QFormLayout()
        self._builtin_combo = QComboBox()
        for m in BUILTIN_MODELS:
            self._builtin_combo.addItem(f"{m['name']} ({m['params']}, ~{m['size_gb']}GB)", m["id"])
        self._builtin_combo.addItem(self._i18n.t("model.custom_id"), "custom")
        form_row.addRow(self._i18n.t("model.builtin_list"), self._builtin_combo)
        download_layout.addLayout(form_row)

        # Download directory
        download_layout.addWidget(QLabel(self._i18n.t("model.select_dir")))
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(self._config.get("download_dir", ""))
        self._dir_edit.setReadOnly(True)
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse_btn = QPushButton()
        dir_row.addWidget(self._dir_browse_btn)
        download_layout.addLayout(dir_row)

        # Mirror source
        mirror_row = QHBoxLayout()
        self._mirror_edit = QLineEdit()
        self._mirror_edit.setText(self._config.get("hf_mirror", "https://hf-mirror.com"))
        self._mirror_edit.setPlaceholderText("https://hf-mirror.com")
        mirror_row.addWidget(QLabel(self._i18n.t("model.mirror")), 0)
        mirror_row.addWidget(self._mirror_edit, 1)
        download_layout.addLayout(mirror_row)

        # Download button and progress
        self._download_btn = QPushButton(self._i18n.t("model.download"))
        self._download_btn.setObjectName("primaryBtn")
        self._download_btn.setMinimumHeight(36)
        download_layout.addWidget(self._download_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        download_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel()
        self._progress_label.setObjectName("label-muted")
        download_layout.addWidget(self._progress_label)

        right_panel.addWidget(download_group)
        self._download_group = download_group

        # Model detail area
        detail_group = QGroupBox()
        detail_layout = QVBoxLayout(detail_group)
        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(200)
        self._detail_text.setPlaceholderText(self._i18n.t("model.not_found"))
        detail_layout.addWidget(self._detail_text)
        right_panel.addWidget(detail_group)
        self._detail_group = detail_group

        # LoRA management
        lora_group = QGroupBox()
        lora_layout = QVBoxLayout(lora_group)
        self._lora_list = QListWidget()
        self._lora_list.setMaximumHeight(120)
        lora_layout.addWidget(self._lora_list)
        self._lora_delete_btn = QPushButton()
        self._lora_delete_btn.setObjectName("trashBtn")
        self._lora_delete_btn.clicked.connect(self._on_delete_lora)
        lora_layout.addWidget(self._lora_delete_btn)
        right_panel.addWidget(lora_group)
        self._lora_group = lora_group

        right_panel.addStretch()

        self._refresh_texts()

    def _connect_signals(self):
        self._download_btn.clicked.connect(self._on_download)
        self._dir_browse_btn.clicked.connect(self._on_browse_dir)
        self._builtin_combo.currentIndexChanged.connect(self._on_combo_changed)
        self._refresh_btn.clicked.connect(self._refresh_model_list)
        self._mirror_edit.editingFinished.connect(self._on_mirror_changed)

    def _init_manager(self):
        """Lazy-init ModelManager"""
        if self._manager is None:
            download_dir = self._config.get("download_dir", "")
            if not download_dir:
                download_dir = os.path.join(self._config.workspace, "models")
            hf_mirror = self._config.get("hf_mirror", "")
            self._manager = ModelManager(download_dir, hf_mirror)

    def _on_combo_changed(self, index):
        data = self._builtin_combo.currentData()
        if data == "custom":
            model_id, ok = QInputDialog.getText(
                self, self._i18n.t("model.builtin_list"),
                self._i18n.t("model.custom_id") + ":"
            )
            if ok and model_id.strip():
                self._custom_model_id = model_id.strip()
            # Switch back to first item
            if self._builtin_combo.count() > 0:
                self._builtin_combo.setCurrentIndex(0)

    def _on_browse_dir(self):
        """Select download directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, self._i18n.t("model.select_dir")
        )
        if dir_path:
            self._dir_edit.setText(dir_path)
            self._config.set("download_dir", dir_path)
            if self._manager:
                self._manager.download_dir = dir_path

    def _on_mirror_changed(self):
        """Mirror source changed"""
        mirror = self._mirror_edit.text().strip()
        self._config.set("hf_mirror", mirror)
        if self._manager:
            self._manager.hf_mirror = mirror

    def _on_download(self):
        """Start model download"""
        self._init_manager()

        data = self._builtin_combo.currentData()
        if data == "custom":
            model_id, ok = QInputDialog.getText(
                self, self._i18n.t("model.builtin_list"),
                self._i18n.t("model.custom_id") + ":"
            )
            if not ok or not model_id.strip():
                return
            model_id = model_id.strip()
        else:
            model_id = data

        download_dir = self._dir_edit.text()
        hf_mirror = self._mirror_edit.text().strip()

        # Check disk space
        try:
            disk_usage = shutil.disk_usage(download_dir)
            free_gb = disk_usage.free / (1024 ** 3)
            min_free = self._config.get("ui_constants.training.disk_min_free_gb", 5)
            if free_gb < min_free:
                QMessageBox.warning(self, self._i18n.t("common.error"),
                                    self._i18n.t("error.disk_full"))
                return
        except OSError:
            logger.warning(f"Failed to check disk space for {download_dir}")
            pass

        # Check if already downloaded
        model_name = model_id.split("/")[-1]
        model_path = os.path.join(download_dir, model_name)
        if os.path.isdir(model_path):
            valid, _ = ModelManager.validate_model(model_path)
            if valid:
                QMessageBox.information(self, self._i18n.t("common.success"),
                                        self._i18n.t("model.downloaded"))
                self._refresh_model_list()
                return

        # Start download thread
        self._download_btn.setEnabled(False)
        self._download_btn.setText(self._i18n.t("model.downloading"))
        self._progress_bar.setValue(0)

        self._download_worker = DownloadWorker(model_id, download_dir, hf_mirror)
        self._download_worker.signals.progress.connect(self._on_download_progress)
        self._download_worker.signals.finished.connect(self._on_download_finished)
        self._download_worker.signals.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_progress(self, percent: int, desc: str):
        """Download progress callback"""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(desc)

    def _on_download_finished(self, result: dict):
        """Download complete callback"""
        self._download_btn.setEnabled(True)
        self._download_btn.setText(self._i18n.t("model.download"))
        self._progress_bar.setValue(100)
        self._progress_label.setText(self._i18n.t("model.downloaded"))
        self._refresh_model_list()

    def _on_download_error(self, error_code: str, detail: str):
        """Download error callback"""
        self._download_btn.setEnabled(True)
        self._download_btn.setText(self._i18n.t("model.download"))
        self._progress_bar.setValue(0)
        msg = friendly_error_message(Exception(detail), self._i18n.t)
        QMessageBox.critical(self, self._i18n.t("common.error"), msg)

    def _refresh_model_list(self):
        """Refresh downloaded models list"""
        self._init_manager()
        # Clear existing cards
        while self._model_list_layout.count():
            item = self._model_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        models = self._manager.list_downloaded_models()
        if not models:
            empty_label = QLabel(self._i18n.t("model.not_found"))
            empty_label.setObjectName("label-muted")
            empty_label.setAlignment(Qt.AlignCenter)
            self._model_list_layout.addWidget(empty_label)
            return

        for m in models:
            size_mb = m["size_bytes"] / (1024 * 1024) if m["size_bytes"] > 0 else 0
            if size_mb > 1024:
                size_str = f"{size_mb / 1024:.1f} GB"
            else:
                size_str = f"{size_mb:.0f} MB"

            status_str = "[OK]" if m["status"] == "ok" else m["status"]
            card = ModelCard(
                name=m["name"],
                params=m["params"],
                size=size_str,
                status=status_str,
                path=m["path"],
            )
            card.clicked.connect(self._show_model_detail)
            card.delete_requested.connect(self._on_delete_model)
            card.load_requested.connect(self._on_load_model)
            self._model_list_layout.addWidget(card)

        self._downloaded_label.setText(
            f"{self._i18n.t('model.downloaded')} ({len(models)})"
        )

    def _show_model_detail(self, model_path: str):
        detail = ModelManager.get_model_detail(model_path)
        lines = []
        lines.append(f"{self._i18n.t('model.path')}: {detail['path']}")
        valid_text = "Yes" if detail['valid'] else "No"
        lines.append(f"Valid: {valid_text}")
        if detail['missing']:
            lines.append(f"Missing: {', '.join(detail['missing'])}")

        cfg = detail.get("config", {})
        if cfg:
            for key in ["model_type", "architectures", "hidden_size",
                        "num_hidden_layers", "num_attention_heads",
                        "intermediate_size", "max_position_embeddings"]:
                if key in cfg:
                    lines.append(f"{key}: {cfg[key]}")

        lines.append(f"\nFiles ({len(detail['files'])}):")
        for f in detail["files"][:10]:
            sz = f["size"]
            if sz > 1024 * 1024:
                lines.append(f"  {f['name']}: {sz / (1024*1024):.1f} MB")
            else:
                lines.append(f"  {f['name']}: {sz / 1024:.0f} KB")

        self._detail_text.setPlainText("\n".join(lines))
        self.model_selected.emit(model_path)

    def _on_delete_model(self, model_path: str):
        """Delete model"""
        reply = QMessageBox.question(
            self, self._i18n.t("common.confirm"),
            self._i18n.t("model.delete_confirm"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._init_manager()
            if self._manager.delete_model(model_path):
                self._refresh_model_list()
                self._detail_text.clear()

    def _on_load_model(self, model_path: str):
        """Load model — persist selection to config"""
        self._config.set("last_state.selected_model", model_path)
        self.model_selected.emit(model_path)

    def _refresh_loras(self):
        """Refresh trained LoRA adapters list"""
        self._lora_list.clear()
        trainer = ProcessTrainer(self._config.workspace)
        loras = trainer.list_loras()
        for lora in loras:
            meta = lora.get("metadata", {})
            model_name = os.path.basename(meta.get("model_path", "")) if meta.get("model_path") else "?"
            self._lora_list.addItem(f"{lora['name']}  [{model_name}]")
        if not loras:
            self._lora_list.addItem("(no trained LoRA)")

    def _on_delete_lora(self):
        """Delete selected LoRA adapter"""
        item = self._lora_list.currentItem()
        if not item or item.text().startswith("("):
            return
        import shutil
        lora_name = item.text().split("  [")[0] if "  [" in item.text() else item.text()
        lora_path = os.path.join(self._config.workspace, "lora", lora_name)
        if not os.path.isdir(lora_path):
            return
        reply = QMessageBox.question(
            self, self._i18n.t("common.confirm"),
            f"{self._i18n.t('model.delete_confirm')}\n\n{lora_name}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                shutil.rmtree(lora_path)
                self._refresh_loras()
            except OSError as e:
                QMessageBox.critical(self, self._i18n.t("common.error"), str(e))

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.model"))
        self._refresh_btn.setText(self._i18n.t("common.refresh"))
        self._downloaded_label.setText(self._i18n.t("model.downloaded"))
        self._download_group.setTitle(self._i18n.t("model.download"))
        self._detail_group.setTitle(self._i18n.t("model.path"))
        self._dir_browse_btn.setText(self._i18n.t("common.browse"))
        self._detail_text.setPlaceholderText(self._i18n.t("model.not_found"))
        self._download_btn.setText(self._i18n.t("model.download"))
        self._lora_group.setTitle(self._i18n.t("export.lora_adapter"))
        self._lora_delete_btn.setText(self._i18n.t("common.delete"))

    def showEvent(self, event):
        """Refresh list on page display"""
        super().showEvent(event)
        self._refresh_model_list()
        self._refresh_loras()

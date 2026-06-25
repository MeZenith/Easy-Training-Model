import logging
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.trainer import ProcessTrainer
from ui.pages.train.config_panel import TrainConfigPanel
from ui.pages.train.monitor_panel import TrainMonitorPanel

logger = logging.getLogger("EasyTinking")


class TrainPage(QWidget):
    #训练页 — 配置面板和监控面板双页切换

    training_started = Signal()
    training_finished = Signal(dict)

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._trainer = ProcessTrainer(config.workspace)
        self._is_training = False

        self._setup_ui()
        self._connect_signals()
        self._attach_trainer_signals()
        self._i18n.language_changed.connect(self._refresh_texts)
        self._refresh_texts()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._config_panel = TrainConfigPanel(self._config, self._i18n)
        self._monitor_panel = TrainMonitorPanel(self._config, self._i18n)
        self._stack.addWidget(self._config_panel)
        self._stack.addWidget(self._monitor_panel)
        main_layout.addWidget(self._stack, 1)

    def _connect_signals(self):
        self._config_panel.start_requested.connect(self._on_start)
        self._monitor_panel.stop_requested.connect(self._on_stop)
        self._monitor_panel.back_requested.connect(self._on_back)

    def _attach_trainer_signals(self):
        self._trainer.progress.connect(self._monitor_panel.update_progress)
        self._trainer.finished.connect(self._on_train_finished)
        self._trainer.error.connect(self._on_train_error)
        self._trainer.log_message.connect(lambda msg: logger.info(msg))
        self._trainer.metric.connect(self._on_metric)

    def _on_start(self):
        #开始训练
        config = self._config_panel.get_train_config()
        if not config["model_path"]:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("error.no_model"))
            return

        #保存配置给下次恢复
        self._config.set("last_train.model_path", config["model_path"])
        self._config.set("last_train.lora_rank", config["lora_rank"])
        self._config.set("last_train.lora_alpha", config["lora_alpha"])
        self._config.set("last_train.epochs", config["epochs"])
        self._config.set("last_train.batch_size", config["batch_size"])
        self._config.set("last_train.learning_rate", config["learning_rate"])
        self._config.set("last_train.max_seq_length", config["max_seq_length"])

        all_data, dataset_names = self._config_panel.get_checked_datasets()
        config["data"] = all_data
        config["dataset_names"] = dataset_names

        if not config["data"]:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("error.no_data"))
            return

        min_samples = self._config.get("ui_constants.training.data_min_samples", 10)
        if len(config["data"]) < min_samples:
            reply = QMessageBox.warning(
                self, self._i18n.t("common.warning"),
                self._i18n.t("train.data_too_few").format(len(config["data"])),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self._is_training = True

        model_name = os.path.basename(config["model_path"])
        info_text = (
            f"{self._i18n.t('data.count')}: {len(config['data'])}, "
            f"Epoch: {config['epochs']}, Batch: {config['batch_size']}"
        )
        self._monitor_panel.set_training_state(model_name, info_text)

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
            QMessageBox.information(self, self._i18n.t("common.success"),
                                    self._i18n.t("train.stopped"))

    def _on_back(self):
        self._stack.setCurrentIndex(0)

    def _on_metric(self, data: dict):
        self._monitor_panel.add_metric(
            data["step"], data["loss"], data.get("lr", 0)
        )

    def _on_train_finished(self, result: dict):
        self._is_training = False
        self._monitor_panel.set_finished(result)
        self.training_finished.emit(result)

    def _on_train_error(self, error_code: str, detail: str):
        self._is_training = False
        self._monitor_panel.set_failed(error_code, detail)
        key = f"error.{error_code.lower().replace('err_', '')}"
        text = self._i18n.t(key)
        if text == key:
            text = detail
        QMessageBox.critical(self, self._i18n.t("common.error"), text)

    def _refresh_texts(self):
        self._config_panel.refresh_texts()
        self._monitor_panel.refresh_texts()

    def showEvent(self, event):
        super().showEvent(event)
        self._config_panel.load_models()
        self._config_panel.load_datasets()
        self._config_panel.restore_last_config()

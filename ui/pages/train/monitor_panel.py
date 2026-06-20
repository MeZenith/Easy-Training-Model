"""Training monitor panel — loss chart, progress bar, result display"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGridLayout,
)
from PySide6.QtCore import Signal

from ui.components.loss_chart import LossChart


class TrainMonitorPanel(QWidget):
    """Training monitor panel with loss chart, progress bar, and result grid

    Signals:
        stop_requested: user clicked stop training
        back_requested: user clicked back to config
    """

    stop_requested = Signal()
    back_requested = Signal()

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._loss_values = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self._title = QLabel()
        self._title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self._title)

        self._info = QLabel()
        self._info.setObjectName("label-secondary")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("font-size: 13px; padding: 4px 0;")
        layout.addWidget(self._info)

        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton()
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setMinimumWidth(100)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()

        self._back_btn = QPushButton()
        self._back_btn.setMinimumHeight(36)
        self._back_btn.setVisible(False)
        self._back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(self._back_btn)
        layout.addLayout(btn_row)

        self._loss_chart = LossChart(parent=self, i18n=self._i18n)
        self._loss_chart.setMinimumHeight(300)
        layout.addWidget(self._loss_chart, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        self._step_label = QLabel()
        self._step_label.setObjectName("label-secondary")
        self._step_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._step_label)

        self._result_panel = QWidget()
        self._result_panel.setVisible(False)
        result_layout = QVBoxLayout(self._result_panel)
        result_layout.setContentsMargins(0, 8, 0, 0)
        result_layout.setSpacing(4)

        self._result_title_label = QLabel()
        self._result_title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #22c55e;")
        result_layout.addWidget(self._result_title_label)

        self._result_grid = QGridLayout()
        self._result_grid.setSpacing(6)
        result_layout.addLayout(self._result_grid)

        layout.addWidget(self._result_panel)

    def set_training_state(self, model_name: str, info_text: str):
        """Initialize monitor panel for new training session"""
        self._loss_values = []
        self._title.setText(self._i18n.t("train.training_label").format(model_name))
        self._info.setText(info_text)
        self._step_label.setText(self._i18n.t("common.loading"))
        self._progress_bar.setValue(0)
        self._loss_chart.clear()
        self._result_panel.setVisible(False)
        self._back_btn.setVisible(False)

    def update_progress(self, pct: int, desc: str):
        """Update progress bar and step label"""
        self._progress_bar.setValue(pct)
        self._step_label.setText(desc)

    def add_metric(self, step: int, loss: float, lr: float = 0):
        """Add a loss data point"""
        self._loss_values.append(loss)
        self._loss_chart.add_point(step, loss, lr)

    def set_finished(self, result: dict):
        """Training complete — update chart and show results"""
        self._progress_bar.setValue(100)
        self._step_label.setText(self._i18n.t("train.complete"))
        self._loss_chart.update_data(self._loss_values)
        self._show_result(result)
        self._back_btn.setVisible(True)

    def set_failed(self, error_code: str, detail: str):
        """Training failed — show error state"""
        self._back_btn.setVisible(True)

    def _show_result(self, r: dict):
        """Display training result metrics"""
        self._result_panel.setVisible(True)
        self._result_title_label.setText(self._i18n.t("train.training_complete"))

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

    def refresh_texts(self):
        """Update all translatable text"""
        self._stop_btn.setText(self._i18n.t("train.stop"))
        self._back_btn.setText(self._i18n.t("train.back"))
        self._result_title_label.setText(self._i18n.t("train.training_complete"))

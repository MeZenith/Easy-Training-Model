"""GPU 显存监控条组件"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer

from utils.gpu_info import get_gpu_info


class GPUMonitor(QWidget):
    """GPU 显存实时监控条"""

    def __init__(self, parent=None, i18n=None):
        super().__init__(parent)
        self._i18n = i18n
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        txt_gpu = "GPU:" if i18n is None else i18n.t("component.gpu_label")
        self._label = QLabel(txt_gpu)
        self._label.setObjectName("label-secondary")
        self._label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(12)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, 1)

        txt_na = "N/A" if self._i18n is None else self._i18n.t("component.na")
        self._value_label = QLabel(txt_na)
        self._value_label.setObjectName("label-secondary")
        self._value_label.setStyleSheet("font-size: 12px;")
        self._value_label.setMinimumWidth(80)
        layout.addWidget(self._value_label)

        # 每3秒自动刷新
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)
        self.refresh()

    def refresh(self):
        """刷新 GPU 信息"""
        try:
            gpus = get_gpu_info()
            if gpus:
                g = gpus[0]
                pct = int(g["vram_used_mb"] / g["vram_total_mb"] * 100) if g["vram_total_mb"] > 0 else 0
                self._bar.setValue(pct)
                self._value_label.setText(
                    f"{g['vram_used_mb']}/{g['vram_total_mb']} MB"
                )
            else:
                self._bar.setValue(0)
                txt_na = "N/A" if self._i18n is None else self._i18n.t("component.na")
                self._value_label.setText(txt_na)
        except Exception:
            self._bar.setValue(0)
            txt_err = "Error" if self._i18n is None else self._i18n.t("common.error")
            self._value_label.setText(txt_err)

    def stop(self):
        """停止定时器"""
        self._timer.stop()

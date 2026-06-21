"""自定义进度条组件"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget


class CustomProgressBar(QWidget):
    """带百分比和描述文本的进度条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        layout.addWidget(self._bar, 1)

        self._label = QLabel("0%")
        self._label.setMinimumWidth(40)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setObjectName("label-secondary")
        layout.addWidget(self._label)

        self._desc = QLabel()
        self._desc.setObjectName("label-muted")
        layout.addWidget(self._desc)

    def set_value(self, percent: int, description: str = ""):
        self._bar.setValue(percent)
        self._label.setText(f"{percent}%")
        if description:
            self._desc.setText(description)

    def reset(self):
        self._bar.setValue(0)
        self._label.setText("0%")
        self._desc.setText("")

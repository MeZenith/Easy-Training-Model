"""模型卡片组件"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal


class ModelCard(QFrame):
    """模型信息卡片"""

    clicked = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, name: str, params: str, size: str,
                 status: str = "", path: str = "", parent=None):
        super().__init__(parent)
        self._model_name = name
        self._model_path = path
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        top_row.addWidget(name_label)
        top_row.addStretch()

        self._status_label = QLabel(status)
        self._status_label.setObjectName("label-muted")
        self._status_label.setStyleSheet("font-size: 12px;")
        top_row.addWidget(self._status_label)

        layout.addLayout(top_row)

        bottom_row = QHBoxLayout()
        params_label = QLabel(params)
        params_label.setObjectName("label-secondary")
        params_label.setStyleSheet("font-size: 12px;")
        bottom_row.addWidget(params_label)

        size_label = QLabel(size)
        size_label.setObjectName("label-secondary")
        size_label.setStyleSheet("font-size: 12px;")
        bottom_row.addWidget(size_label)
        bottom_row.addStretch()

        self._delete_btn = QPushButton("X")
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setObjectName("trashBtn")
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._model_path))
        bottom_row.addWidget(self._delete_btn)

        layout.addLayout(bottom_row)

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_path(self):
        return self._model_path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._model_path)
        super().mousePressEvent(event)

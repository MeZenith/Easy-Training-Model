from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ModelCard(QFrame):
    #模型信息卡片

    clicked = Signal(str)
    delete_requested = Signal(str)
    load_requested = Signal(str)

    def __init__(self, name: str, params: str, size: str,
                 status: str = "", path: str = "", parent=None):
        super().__init__(parent)
        self._model_name = name
        self._model_path = path
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(72)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(2)

        #第一行：名字 + 状态
        top_row = QHBoxLayout()
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_row.addWidget(name_label)
        top_row.addStretch()

        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-size: 11px;")
        if status == "ok":
            self._status_label.setText("[OK]")
            self._status_label.setStyleSheet("font-size: 11px; color: #3fb950;")
        elif status:
            self._status_label.setText(f"[{status}]")
            self._status_label.setStyleSheet("font-size: 11px; color: #f85149;")
        top_row.addWidget(self._status_label)

        layout.addLayout(top_row)

        #第二行：参数+大小 + 操作按钮
        bottom_row = QHBoxLayout()
        detail_label = QLabel(f"{params}  ·  {size}")
        detail_label.setObjectName("label-secondary")
        detail_label.setStyleSheet("font-size: 11px;")
        bottom_row.addWidget(detail_label)
        bottom_row.addStretch()

        self._load_btn = QPushButton("Load")
        self._load_btn.setFixedSize(48, 20)
        self._load_btn.setObjectName("primaryBtn")
        self._load_btn.setStyleSheet("font-size: 10px; padding: 0;")
        self._load_btn.clicked.connect(lambda: self.load_requested.emit(self._model_path))
        bottom_row.addWidget(self._load_btn)

        self._delete_btn = QPushButton("X")
        self._delete_btn.setFixedSize(20, 20)
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
        #点击卡片
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._model_path)
        super().mousePressEvent(event)

"""主题与样式管理"""

import os
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication


def _load_qss(filename: str) -> str:
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "assess", filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""


_DARK_QSS = _load_qss("professional_theme.qss")

_LIGHT_OVERRIDE = """
QWidget { background-color: #ffffff; color: #1e1e1e; }
QMainWindow { background-color: #ffffff; }

#titleBar { background-color: #e8e8e8; border-bottom: 1px solid #d0d0d0; }
#titleBar QLabel { color: #1e1e1e; }
#minBtn, #maxBtn, #closeBtn { color: #666666; }
#minBtn:hover, #maxBtn:hover { background-color: #cccccc; color: #000000; }
#closeBtn:hover { background-color: #e81123; color: #ffffff; }

#sidebar { background-color: #f0f0f0; border-right: 1px solid #d0d0d0; }
#sidebarNav { background-color: transparent; }
#sidebarNav::item { color: #666666; font-size: 15px; }
#sidebarNav::item:hover { background-color: #e0e0e0; color: #000000; }
#sidebarNav::item:selected { background-color: #dce8f4; color: #1a6fb5; border-left: 3px solid #1a6fb5; }
#sidebarCollapse { color: #888888; background: transparent; }
#sidebarCollapse:hover { color: #1a6fb5; }

QPushButton { background-color: #f0f0f0; color: #1e1e1e; border-color: #c0c0c0; }
QPushButton:hover { background-color: #e0e0e0; }
#primaryBtn { background-color: #1a6fb5; color: #ffffff; border-color: #1a6fb5; }
#primaryBtn:hover { background-color: #1f7ec4; }
#dangerBtn { background-color: #cc3333; color: #ffffff; }

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff; color: #1e1e1e; border-color: #c0c0c0;
    padding: 2px 6px;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #1a6fb5;
}

/* SpinBox buttons light */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    border-left: 1px solid #c0c0c0; border-bottom: 1px solid #c0c0c0; background-color: #e8e8e8;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    border-left: 1px solid #c0c0c0; background-color: #e8e8e8;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #d0d0d0; }

QComboBox { background-color: #ffffff; color: #1e1e1e; border-color: #c0c0c0; }
QComboBox:focus { border-color: #1a6fb5; }
QComboBox QAbstractItemView { background-color: #ffffff; border-color: #c0c0c0; }

/* 复选框 -- 黑边框确保可见 */
QCheckBox::indicator {
    border: 2px solid #000000; background-color: #ffffff;
}
QCheckBox::indicator:checked { background-color: #1a6fb5; border-color: #1a6fb5; }

QGroupBox { border-color: #c0c0c0; color: #888888; }
QGroupBox::title { color: #1a6fb5; }

QTableWidget { background-color: #ffffff; gridline-color: #e0e0e0; border-color: #c0c0c0; }
QHeaderView::section { background-color: #f0f0f0; color: #888888; }

QProgressBar { background-color: #e0e0e0; color: #666666; }
QProgressBar::chunk { background-color: #1a6fb5; }

QScrollBar::handle:vertical { background: #c0c0c0; }
QScrollBar::handle:vertical:hover { background: #a0a0a0; }
QScrollBar::handle:horizontal { background: #c0c0c0; }

QStatusBar { background-color: #f0f0f0; color: #888888; border-top-color: #c0c0c0; }

QListWidget { background-color: #ffffff; border-color: #c0c0c0; }
QListWidget::item:selected { background-color: #1a6fb5; color: #ffffff; }

QSlider::groove:horizontal { background: #e0e0e0; }
QSlider::handle:horizontal { background: #1a6fb5; }
QSlider::sub-page:horizontal { background: #1a6fb5; }

/* 日志/等宽文字区域深色 */
QPlainTextEdit { color: #000000; }
QTextEdit { color: #000000; }
#logDisplay { color: #000000; }
"""


class ThemeManager(QObject):
    theme_changed = Signal(str)
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def apply_theme(self, app: QApplication, theme: str):
        """应用主题 — 深色基础 + 浅色覆盖"""
        if theme == "light":
            app.setStyleSheet(_DARK_QSS + _LIGHT_OVERRIDE)
        else:
            app.setStyleSheet(_DARK_QSS)
        self.theme_changed.emit(theme)

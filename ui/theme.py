"""主题与样式管理"""

import os
import sys
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication


def _load_qss(filename: str) -> str:
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if getattr(sys, "frozen", False):
            base = sys._MEIPASS
        path = os.path.join(base, "assess", filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""


_DARK_QSS = _load_qss("professional_theme.qss") or """
QWidget { background:#0d1117; color:#c9d1d9; font-family:"Segoe UI","Microsoft YaHei",sans-serif; font-size:13px; }
QMainWindow { background:#0d1117; border-radius:10px; }
#titleBar { background:#161b22; border-bottom:1px solid #21262d; min-height:36px; }
#titleBar QLabel { color:#c9d1d9; }
#minBtn,#maxBtn,#closeBtn { background:transparent; color:#8b949e; border:none; border-radius:4px; font-size:15px; min-width:32px; min-height:28px; }
#minBtn:hover,#maxBtn:hover { background:#30363d; color:#f0f6fc; }
#closeBtn:hover { background:#f85149; color:#fff; }
#sidebar { background:#161b22; border-right:1px solid #21262d; }
#sidebarNav { background:transparent; border:none; }
#sidebarNav::item { color:#8b949e; padding:14px 20px; min-height:48px; border-left:3px solid transparent; font-size:15px; }
#sidebarNav::item:hover { background:#1c2128; color:#f0f6fc; }
#sidebarNav::item:selected { background:#1c2128; color:#3fb950; border-left:3px solid #3fb950; }
QPushButton { background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; padding:6px 16px; font-size:12px; }
QPushButton:hover { background:#30363d; }
#primaryBtn { background:#238636; color:#fff; border-color:#238636; font-weight:bold; }
#primaryBtn:hover { background:#2ea043; }
#dangerBtn { background:#da3633; color:#fff; }
#dangerBtn:hover { background:#f85149; }
#trashBtn { background:transparent!important; border:1px solid #f85149!important; color:#f85149!important; }
QLineEdit,QTextEdit,QSpinBox,QDoubleSpinBox { background:#0d1117; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; padding:2px 6px; }
QLineEdit:focus,QTextEdit:focus { border:1px solid #3fb950; }
QComboBox { background:#0d1117; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; }
QGroupBox { border:1px solid #21262d; border-radius:6px; margin-top:14px; padding-top:16px; color:#8b949e; }
QGroupBox::title { color:#3fb950; }
QTableWidget { background:#0d1117; gridline-color:#21262d; }
QProgressBar { background:#161b22; border:none; border-radius:3px; height:6px; }
QProgressBar::chunk { background:#3fb950; border-radius:3px; }
QScrollBar:vertical { background:transparent; width:6px; }
QScrollBar::handle:vertical { background:#30363d; border-radius:3px; }
QStatusBar { background:#161b22; color:#8b949e; border-top:1px solid #21262d; }
QListWidget { background:#0d1117; border:1px solid #21262d; }
QSlider::groove:horizontal { background:#21262d; border-radius:3px; height:4px; }
QSlider::handle:horizontal { background:#3fb950; width:14px; height:14px; border-radius:7px; }
QCheckBox::indicator { border:2px solid #484f58; border-radius:3px; background:#0d1117; }
QCheckBox::indicator:checked { background:#3fb950; border-color:#3fb950; }
"""

_LIGHT_QSS = _load_qss("light_theme.qss") or _DARK_QSS


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
        """应用主题 — 深色/浅色切换"""
        if theme == "light":
            app.setStyleSheet(_LIGHT_QSS)
        else:
            app.setStyleSheet(_DARK_QSS)
        self.theme_changed.emit(theme)

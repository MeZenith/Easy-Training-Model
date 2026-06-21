"""主题与样式管理 — 加载 QSS 文件应用深色/浅色主题"""

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


_DARK_QSS = _load_qss("professional_theme.qss")
_LIGHT_QSS = _load_qss("light_theme.qss") or _DARK_QSS


class ThemeManager(QObject):
    """主题管理器 (单例) — 深色/浅色切换

    Signals:
        theme_changed(str): 切换后发出主题名 ("dark" / "light")
    """
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
        if theme == "light":
            app.setStyleSheet(_LIGHT_QSS)
        else:
            app.setStyleSheet(_DARK_QSS)
        self.theme_changed.emit(theme)

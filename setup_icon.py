"""应用图标设置工具 —— 任务栏/标题栏/Alt+Tab/托盘 全覆盖

调用方式:
    from setup_icon import setup_app_icon, set_window_icon
    setup_app_icon()          # QApplication 创建之前
    app = QApplication(sys.argv)
    set_window_icon(app)      # QApplication 创建之后
    set_window_icon(window)   # 窗口也设一份
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("EasyTinking")

APP_NAME = "EasyTinking"
COMPANY = "BlueCornerStudio"
VERSION = "1.0"
ICON_FILE = "icon.ico"


def get_resource_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = Path(__file__).parent.resolve()
    return str(Path(base_path) / relative_path)


def find_icon_file() -> str:
    candidates = [
        f"res/{ICON_FILE}",
        f"resources/{ICON_FILE}",
        ICON_FILE,
        "res/icon.png",
        "icon.png",
    ]
    for candidate in candidates:
        path = get_resource_path(candidate)
        if os.path.exists(path):
            return path
    return None


def set_appusermodelid():
    if sys.platform != "win32":
        return
    app_id = f"{COMPANY}.{APP_NAME}.{VERSION}"
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        logger.warning("Failed to set AppUserModelID")


_icon_path = None


def setup_app_icon():
    global _icon_path
    _icon_path = find_icon_file()
    set_appusermodelid()


def set_window_icon(app_or_window):
    from PySide6.QtGui import QIcon
    global _icon_path
    if _icon_path is None:
        _icon_path = find_icon_file()
    if _icon_path and os.path.exists(_icon_path):
        icon = QIcon(_icon_path)
        app_or_window.setWindowIcon(icon)

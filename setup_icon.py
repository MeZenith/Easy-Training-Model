import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("EasyTraining")

APP_NAME = "EasyTraining"
COMPANY = "BlueCornerStudio"
VERSION = "1.0"
ICON_FILE = "icon.ico"


def get_resource_path(relative_path: str) -> str:
    #获取资源文件的绝对路径，兼容PyInstaller打包
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = Path(__file__).parent.resolve()
    return str(Path(base_path) / relative_path)


def find_icon_file() -> str:
    #找图标文件，支持ico和png
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
    #设置Windows任务栏应用ID（Win下图标不显示时用的）
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
    #QApplication创建之前调用
    global _icon_path
    _icon_path = find_icon_file()
    set_appusermodelid()


def set_window_icon(app_or_window):
    #给窗口设图标
    from PySide6.QtGui import QIcon
    global _icon_path
    if _icon_path is None:
        _icon_path = find_icon_file()
    if _icon_path and os.path.exists(_icon_path):
        icon = QIcon(_icon_path)
        app_or_window.setWindowIcon(icon)

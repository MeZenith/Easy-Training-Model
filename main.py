"""Easy Tinking - 一站式 AI 模型微调工具

程序入口
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from setup_icon import setup_app_icon, set_window_icon
from core.config import AppConfig
from utils.i18n import I18n
from utils.logger import setup_logger
from ui.theme import ThemeManager
from ui.app import MainWindow, global_exception_handler


def main():
    setup_app_icon()

    config = AppConfig()

    logger = setup_logger(config.workspace)
    logger.info("Easy Tinking starting...")

    i18n = I18n.instance()
    lang = config.get("language", "")
    if not lang:
        lang = i18n.detect_system_language()
        config.set("language", lang)
    i18n.load_language(lang)

    app = QApplication(sys.argv)
    app.setApplicationName("Easy Tinking")
    app.setOrganizationName("BlueCornerStudio")

    set_window_icon(app)

    sys.excepthook = global_exception_handler

    theme_manager = ThemeManager.instance()
    theme = config.get("theme", "dark")
    theme_manager.apply_theme(app, theme)

    window = MainWindow(config, i18n)
    set_window_icon(window)
    window.show()

    logger.info("Easy Tinking started successfully")

    exit_code = app.exec()
    logger.info("Easy Tinking shutting down...")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

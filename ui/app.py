"""主窗口与导航"""

import os
import sys
import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QStatusBar, QFrame, QSizePolicy,
    QMessageBox, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon, QFont, QKeySequence, QShortcut

from core.config import AppConfig
from core.error_handler import friendly_error_message
from utils.i18n import I18n
from ui.theme import ThemeManager
from ui.pages.model_page import ModelPage
from ui.pages.data_page import DataPage
from ui.pages.train import TrainPage
from ui.pages.export_page import ExportPage
from ui.pages.test_page import TestPage
from ui.pages.settings_page import SettingsPage
from ui.pages.logs_page import LogsPage
from utils.gpu_info import get_gpu_info
from assess.nav_icons import ICON_MAP

logger = logging.getLogger("EasyTinking")


# 导航项定义: (page_id, i18n_key)
NAV_ITEMS = [
    ("model", "nav.model"),
    ("data", "nav.data"),
    ("train", "nav.train"),
    ("export", "nav.export"),
    ("test", "nav.test"),
    ("---", ""),
    ("settings", "nav.settings"),
    ("logs", "nav.logs"),
]


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, i18n: I18n):
        super().__init__()
        self._config = config
        self._i18n = i18n
        self._nav_buttons = {}
        self._sidebar_expanded = True
        self._drag_pos = None

        self._init_window()
        self._init_ui()
        self._apply_saved_state()

        self._i18n.language_changed.connect(self._on_language_changed)
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)

    def _init_window(self):
        self.setObjectName("mainWindow")
        self.setWindowTitle(self._i18n.t("app.title") + " - " + self._i18n.t("app.subtitle"))
        self.setMinimumSize(QSize(
            self._config.get("ui_constants.window_min_width", 1024),
            self._config.get("ui_constants.window_min_height", 640)
        ))
        w = self._config.get("window.width", 1280)
        h = self._config.get("window.height", 800)
        self.resize(w, h)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ===== 自定义标题栏 =====
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(self._config.get("ui_constants.title_bar_height", 40))
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(12, 0, 8, 0)
        tb_layout.setSpacing(4)

        self._title_icon = QLabel()
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "icon.png")
        if os.path.isfile(icon_path):
            self._title_icon.setPixmap(QIcon(icon_path).pixmap(20, 20))
        tb_layout.addWidget(self._title_icon)

        self._title_text = QLabel(self._i18n.t("app.title"))
        self._title_text.setStyleSheet("font-size: 13px; font-weight: 600; color: #8b949e;")
        tb_layout.addWidget(self._title_text)
        tb_layout.addStretch()

        for btn_info in [("minBtn", "__"), ("maxBtn", "[]"), ("closeBtn", "X")]:
            btn = QPushButton(btn_info[1])
            btn.setObjectName(btn_info[0])
            btn.setFixedSize(32, 28)
            btn.clicked.connect(lambda checked, a=btn_info[0]: self._on_title_btn(a))
            tb_layout.addWidget(btn)

        self._title_bar = title_bar
        main_layout.addWidget(title_bar)

        # ===== 主内容 =====
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        # 侧边栏
        self._sidebar = QWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 8, 0, 8)
        sidebar_layout.setSpacing(2)

        # 导航列表
        self._nav_list = QListWidget()
        self._nav_list.setObjectName("sidebarNav")
        self._nav_list.setSpacing(0)
        self._nav_list.setIconSize(QSize(18, 18))
        for page_id, i18n_key in NAV_ITEMS:
            if page_id == "---":
                continue
            item = QListWidgetItem(f"  {self._i18n.t(i18n_key)}")
            item.setData(Qt.UserRole, page_id)
            if page_id in ICON_MAP:
                item.setIcon(ICON_MAP[page_id]())
            self._nav_list.addItem(item)
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)
        sidebar_layout.addWidget(self._nav_list, 1)

        sidebar_layout.addStretch()

        # 折叠按钮 + 版权
        self._collapse_btn = QPushButton("<>")
        self._collapse_btn.setObjectName("sidebarCollapse")
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.clicked.connect(self._toggle_sidebar)
        sidebar_layout.addWidget(self._collapse_btn)

        copyright_label = QLabel(self._i18n.t("app.copyright"))
        copyright_label.setStyleSheet("color: #484f58; font-size: 10px; padding: 8px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(copyright_label)

        content_row.addWidget(self._sidebar)

        # 页面堆栈
        self._stack = QStackedWidget()
        self._pages = {
            "model": ModelPage(self._config, self._i18n),
            "data": DataPage(self._config, self._i18n),
            "train": TrainPage(self._config, self._i18n),
            "export": ExportPage(self._config, self._i18n),
            "test": TestPage(self._config, self._i18n),
            "settings": SettingsPage(self._config, self._i18n),
            "logs": LogsPage(self._config, self._i18n),
        }
        for page_id in ["model", "data", "train", "export", "test", "settings", "logs"]:
            self._stack.addWidget(self._pages[page_id])
        content_row.addWidget(self._stack, 1)
        main_layout.addLayout(content_row, 1)

        # 状态栏
        self._statusbar = QStatusBar()
        self._statusbar.setObjectName("statusbar")
        self.setStatusBar(self._statusbar)
        self._gpu_label = QLabel()
        self._copyright_label = QLabel(self._i18n.t("app.company"))
        self._version_label = QLabel("v1.0")
        for lbl in [self._gpu_label, self._copyright_label, self._version_label]:
            lbl.setStyleSheet("font-size: 12px; color: #8b949e;")
        self._statusbar.addPermanentWidget(self._gpu_label, 1)
        self._statusbar.addPermanentWidget(self._copyright_label)
        self._statusbar.addPermanentWidget(self._version_label)
        self._update_statusbar()
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_statusbar)
        self._gpu_timer.start(5000)

        last_page = self._config.get("last_state.current_page", "train")
        self._switch_page(last_page)
        self._setup_shortcuts()

    # ===== 标题栏按钮 =====
    def _on_title_btn(self, action):
        if action == "closeBtn":
            self.close()
        elif action == "maxBtn":
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        elif action == "minBtn":
            self.showMinimized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().y() < 40:
            self._drag_pos = event.globalPosition().toPoint()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.position().y() < 40:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        super().mouseDoubleClickEvent(event)

    def _on_nav_changed(self, row):
        item = self._nav_list.item(row)
        if item:
            self._switch_page(item.data(Qt.UserRole))

    def _switch_page(self, page_id: str):
        if page_id in self._pages:
            self._stack.setCurrentWidget(self._pages[page_id])
            for i in range(self._nav_list.count()):
                item = self._nav_list.item(i)
                if item and item.data(Qt.UserRole) == page_id:
                    self._nav_list.setCurrentRow(i)
                    break
            self._config.set("last_state.current_page", page_id)

    def _toggle_sidebar(self):
        self._sidebar_expanded = not self._sidebar_expanded
        w_expanded = self._config.get("ui_constants.sidebar_width_expanded", 200)
        w_collapsed = self._config.get("ui_constants.sidebar_width_collapsed", 56)
        if self._sidebar_expanded:
            self._sidebar.setFixedWidth(w_expanded)
            self._collapse_btn.setText("\u25c0")
            for i in range(self._nav_list.count()):
                item = self._nav_list.item(i)
                if item:
                    pid = item.data(Qt.UserRole)
                    for pi, ik in NAV_ITEMS:
                        if pi == pid:
                            item.setText(f"  {self._i18n.t(ik)}")
                            break
        else:
            self._sidebar.setFixedWidth(w_collapsed)
            self._collapse_btn.setText("\u25b6")
            for i in range(self._nav_list.count()):
                item = self._nav_list.item(i)
                if item:
                    item.setText("")

    def _update_statusbar(self):
        try:
            gpus = get_gpu_info()
            if gpus:
                g = gpus[0]
                pct = int(g["vram_used_mb"] / g["vram_total_mb"] * 100) if g["vram_total_mb"] > 0 else 0
                text = f"GPU: {g['vram_used_mb']}/{g['vram_total_mb']} MB [{pct}%]  {g['name']}"
            else:
                text = "GPU: N/A"
        except Exception:
            text = "GPU: N/A"
        self._gpu_label.setText(text)

    def _on_language_changed(self):
        for i in range(self._nav_list.count()):
            item = self._nav_list.item(i)
            if item:
                pid = item.data(Qt.UserRole)
                for pi, ik in NAV_ITEMS:
                    if pi == pid and ik:
                        item.setText(f"  {self._i18n.t(ik)}")
                        break

    @staticmethod
    def _on_theme_changed(theme: str):
        pass

    def _apply_saved_state(self):
        pass

    def _setup_shortcuts(self):
        shortcuts = {
            "Ctrl+1": "model", "Ctrl+2": "data", "Ctrl+3": "train",
            "Ctrl+4": "export", "Ctrl+5": "test", "Ctrl+6": "settings", "Ctrl+7": "logs",
        }
        for key, page_id in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda pid=page_id: self._switch_page(pid))

    def closeEvent(self, event):
        geo = self.geometry()
        self._config.set("window.width", geo.width())
        self._config.set("window.height", geo.height())
        self._config.set("window.maximized", self.isMaximized())
        logger.info("Easy Tinking closing")
        super().closeEvent(event)


def global_exception_handler(exc_type, exc_value, exc_tb):
    """全局未捕获异常处理，显示友好提示"""
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
    msg = friendly_error_message(exc_value)
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            QMessageBox.critical(None, "Error", msg)
            return
    except Exception as ex:
        logger.warning(f"Could not show error dialog in exception handler: {ex}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)

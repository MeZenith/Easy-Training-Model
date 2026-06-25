import logging
import os
import sys

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from assess.nav_icons import ICON_MAP
from core.config import AppConfig
from core.error_handler import friendly_error_message
from ui.pages.data_page import DataPage
from ui.pages.export_page import ExportPage
from ui.pages.logs_page import LogsPage
from ui.pages.model_page import ModelPage
from ui.pages.settings_page import SettingsPage
from ui.pages.test_page import TestPage
from ui.pages.train import TrainPage
from ui.theme import ThemeManager
from utils.gpu_info import get_gpu_info
from utils.i18n import I18n

logger = logging.getLogger("EasyTinking")


#导航项定义
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
    #主窗口

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
        #初始化窗口属性
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

        #设图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        #Win11圆角窗口
        self._apply_round_corners()

    def _apply_round_corners(self):
        #调Windows DWM API让Win11窗口变圆角
        try:
            import ctypes
            corner_pref = 33  # DWMWA_WINDOW_CORNER_PREFERENCE
            round_corners = 2  # DWMWCP_ROUND
            hwnd = int(self.winId())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                corner_pref,
                ctypes.byref(ctypes.c_int(round_corners)),
                ctypes.sizeof(ctypes.c_int),
            )
        except Exception:
            logger.warning("Failed to apply rounded corners via DWM")
            pass

    def _init_ui(self):
        #构建整个窗口UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        #自定义标题栏
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

        #最小化/最大化/关闭按钮
        for btn_info in [("minBtn", "__"), ("maxBtn", "[]"), ("closeBtn", "X")]:
            btn = QPushButton(btn_info[1])
            btn.setObjectName(btn_info[0])
            btn.setFixedSize(32, 28)
            btn.clicked.connect(lambda checked, a=btn_info[0]: self._on_title_btn(a))
            tb_layout.addWidget(btn)

        self._title_bar = title_bar
        main_layout.addWidget(title_bar)

        #主内容区域
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        #侧边栏
        self._sidebar = QWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 8, 0, 8)
        sidebar_layout.setSpacing(2)

        #导航列表
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

        #折叠按钮
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

        #页面栈
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

        #状态栏（底部）
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

    def _on_title_btn(self, action):
        #标题栏按钮事件
        if action == "closeBtn":
            self.close()
        elif action == "maxBtn":
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        elif action == "minBtn":
            self.showMinimized()

    #窗口拖拽（无边框窗口用手拖标题栏移动）
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

    #双击标题栏最大化/还原
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
        #切换页面
        if page_id in self._pages:
            self._stack.setCurrentWidget(self._pages[page_id])
            for i in range(self._nav_list.count()):
                item = self._nav_list.item(i)
                if item and item.data(Qt.UserRole) == page_id:
                    self._nav_list.blockSignals(True)
                    self._nav_list.setCurrentRow(i)
                    self._nav_list.blockSignals(False)
                    break
            self._config.set("last_state.current_page", page_id)

    def _toggle_sidebar(self):
        #折叠/展开侧边栏
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
        #刷新状态栏GPU信息
        try:
            gpus = get_gpu_info()
            if gpus:
                g = gpus[0]
                pct = int(g["vram_used_mb"] / g["vram_total_mb"] * 100) if g["vram_total_mb"] > 0 else 0
                text = f"GPU: {g['vram_used_mb']}/{g['vram_total_mb']} MB [{pct}%]  {g['name']}"
            else:
                text = "GPU: N/A"
        except Exception:
            logger.warning("Failed to query GPU info for status bar")
            text = "GPU: N/A"
        self._gpu_label.setText(text)

    def _on_language_changed(self):
        #语言切换后刷新导航文字
        for i in range(self._nav_list.count()):
            item = self._nav_list.item(i)
            if item:
                pid = item.data(Qt.UserRole)
                for pi, ik in NAV_ITEMS:
                    if pi == pid and ik:
                        item.setText(f"  {self._i18n.t(ik)}")
                        break

    def _on_theme_changed(self, theme: str):
        #主题切换回调（ThemeManager已处理样式，这里预留后续操作）
        pass

    def _apply_saved_state(self):
        #恢复上次窗口状态（预留）
        pass

    def _setup_shortcuts(self):
        #设置键盘快捷键 Ctrl+数字切换页面
        shortcuts = {
            "Ctrl+1": "model", "Ctrl+2": "data", "Ctrl+3": "train",
            "Ctrl+4": "export", "Ctrl+5": "test", "Ctrl+6": "settings", "Ctrl+7": "logs",
        }
        for key, page_id in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda pid=page_id: self._switch_page(pid))

    def closeEvent(self, event):
        #关闭前保存窗口状态
        geo = self.geometry()
        self._config.set("window.width", geo.width())
        self._config.set("window.height", geo.height())
        self._config.set("window.maximized", self.isMaximized())
        logger.info("Easy Tinking closing")
        super().closeEvent(event)


def global_exception_handler(exc_type, exc_value, exc_tb):
    #全局未捕获异常处理
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

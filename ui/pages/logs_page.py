import logging
import os

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LogHandler(QObject):
    #线程安全的日志处理器，通过Signal把日志发到主线程

    entry_received = Signal(str, int)

    def __init__(self):
        super().__init__()
        self._handler = logging.Handler()

    def setup(self, fmt: str = "", datefmt: str = "%H:%M:%S"):
        #把Python logging输出转发到Signal
        self._handler = logging.Handler()
        self._handler.emit = self._emit_record
        if fmt:
            self._handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger = logging.getLogger("EasyTraining")
        logger.addHandler(self._handler)

    def _emit_record(self, record):
        try:
            msg = self._handler.format(record)
            self.entry_received.emit(msg, record.levelno)
        except Exception as e:
            import sys
            print(f"LogHandler format error: {e}", file=sys.stderr)

    def remove(self):
        logger = logging.getLogger("EasyTraining")
        logger.removeHandler(self._handler)


class LogsPage(QWidget):
    #日志查看页

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._log_handler = None
        self._all_entries = []
        self._auto_scroll = True
        self._setup_ui()
        self._connect_signals()
        self._init_log_handler()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        #工具栏
        toolbar = QHBoxLayout()

        self._level_combo = QComboBox()
        self._level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_label = QLabel()
        toolbar.addWidget(self._level_label)
        toolbar.addWidget(self._level_combo)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search...")
        toolbar.addWidget(self._search_edit, 1)

        self._auto_scroll_btn = QPushButton()
        self._auto_scroll_btn.setCheckable(True)
        self._auto_scroll_btn.setChecked(True)
        toolbar.addWidget(self._auto_scroll_btn)

        self._export_btn = QPushButton()
        self._clear_btn = QPushButton()
        self._clear_btn.setObjectName("dangerBtn")
        toolbar.addWidget(self._export_btn)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        #日志显示区
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setObjectName("logDisplay")
        layout.addWidget(self._log_text, 1)

        self._refresh_texts()

    def _connect_signals(self):
        self._level_combo.currentIndexChanged.connect(self._apply_filter)
        self._search_edit.textChanged.connect(self._apply_filter)
        self._export_btn.clicked.connect(self._on_export)
        self._clear_btn.clicked.connect(self._on_clear)
        self._auto_scroll_btn.toggled.connect(self._on_toggle_auto_scroll)

    def _init_log_handler(self):
        self._log_handler = LogHandler()
        self._log_handler.entry_received.connect(self._on_log_entry)
        self._log_handler.setup(
            "[%(asctime)s] %(levelname)-7s %(message)s"
        )
        self._load_existing_logs()

    def _load_existing_logs(self):
        #加载已有的日志文件
        log_dir = os.path.join(self._config.workspace, "logs")
        log_path = os.path.join(log_dir, "app.log")
        if os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            level = self._parse_level(line)
                            self._all_entries.append((line, level))
                self._apply_filter()
            except OSError:
                pass

    @staticmethod
    def _parse_level(line: str) -> int:
        #从日志行里判断等级
        if "CRITICAL" in line:
            return logging.CRITICAL
        elif "ERROR" in line:
            return logging.ERROR
        elif "WARNING" in line:
            return logging.WARNING
        elif "INFO" in line:
            return logging.INFO
        elif "DEBUG" in line:
            return logging.DEBUG
        return logging.DEBUG

    def _on_log_entry(self, message: str, level: int):
        #收到新日志
        self._all_entries.append((message, level))
        level_filter = self._level_combo.currentText()
        if self._should_show(message, level, level_filter, self._search_edit.text()):
            self._append_colored(message, level)

    def _should_show(self, message: str, level: int,
                     level_filter: str, keyword: str) -> bool:
        #判断这条日志要不要显示
        level_map = {
            "ALL": logging.DEBUG,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        min_level = level_map.get(level_filter, logging.DEBUG)
        if level < min_level:
            return False
        if keyword and keyword.lower() not in message.lower():
            return False
        return True

    def _append_colored(self, message: str, level: int):
        #带颜色显示日志
        color_map = {
            logging.ERROR: "#ef4444",
            logging.WARNING: "#f59e0b",
            logging.CRITICAL: "#ef4444",
        }
        color = color_map.get(level, "#e8e8f0")
        self._log_text.append(
            f"<span style='color:{color};'>{message}</span>"
        )
        if self._auto_scroll:
            from PySide6.QtGui import QTextCursor
            cursor = self._log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._log_text.setTextCursor(cursor)

    @Slot()
    def _apply_filter(self):
        self._log_text.clear()
        level_filter = self._level_combo.currentText()
        keyword = self._search_edit.text().strip()

        for message, level in self._all_entries:
            if self._should_show(message, level, level_filter, keyword):
                self._append_colored(message, level)

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self._i18n.t("logs.export"),
            "easytinking.log", "Text Files (*.txt *.log)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._log_text.toPlainText())
            except OSError as e:
                QMessageBox.critical(self, self._i18n.t("common.error"), str(e))

    def _on_clear(self):
        reply = QMessageBox.question(
            self, self._i18n.t("common.confirm"),
            f"{self._i18n.t('logs.clear')}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._log_text.clear()
            self._all_entries.clear()

    def _on_toggle_auto_scroll(self, checked: bool):
        self._auto_scroll = checked
        self._auto_scroll_btn.setText(
            self._i18n.t("logs.auto_scroll_on") if checked else self._i18n.t("logs.auto_scroll_off")
        )

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("logs.title"))
        self._level_label.setText(self._i18n.t("logs.level_label") + ":")
        self._export_btn.setText(self._i18n.t("logs.export"))
        self._clear_btn.setText(self._i18n.t("logs.clear"))
        self._search_edit.setPlaceholderText(self._i18n.t("common.search"))
        self._auto_scroll_btn.setText(
            self._i18n.t("logs.auto_scroll_on") if self._auto_scroll else self._i18n.t("logs.auto_scroll_off")
        )

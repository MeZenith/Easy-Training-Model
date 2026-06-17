"""系统设置页"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QFormLayout, QFileDialog,
    QScrollArea, QFrame, QTextEdit
)
from PySide6.QtCore import Qt


class SettingsPage(QWidget):
    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        form = QVBoxLayout(content)
        form.setSpacing(20)

        # 语言设置
        lang_group = QGroupBox()
        lang_form = QFormLayout(lang_group)
        self._lang_combo = QComboBox()
        for lang in self._i18n.available_languages():
            self._lang_combo.addItem(lang.upper(), lang)
        current = self._config.get("language", "zh")
        idx = self._lang_combo.findData(current)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        lang_form.addRow(self._i18n.t("settings.language"), self._lang_combo)
        form.addWidget(lang_group)
        self._lang_group = lang_group

        # 主题设置
        theme_group = QGroupBox()
        theme_form = QFormLayout(theme_group)
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.addItem("Light", "light")
        current_theme = self._config.get("theme", "dark")
        idx = self._theme_combo.findData(current_theme)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)
        theme_form.addRow(self._i18n.t("settings.theme"), self._theme_combo)
        form.addWidget(theme_group)
        self._theme_group = theme_group

        # 工作目录
        ws_group = QGroupBox()
        ws_form = QFormLayout(ws_group)
        ws_row = QHBoxLayout()
        self._ws_edit = QLineEdit()
        self._ws_edit.setText(self._config.get("workspace", ""))
        self._ws_edit.setReadOnly(True)
        ws_row.addWidget(self._ws_edit, 1)
        self._ws_browse_btn = QPushButton()
        ws_row.addWidget(self._ws_browse_btn)
        ws_form.addRow(self._i18n.t("settings.workspace"), ws_row)
        form.addWidget(ws_group)
        self._ws_group = ws_group

        # HuggingFace 镜像
        hf_group = QGroupBox()
        hf_form = QFormLayout(hf_group)
        self._hf_edit = QLineEdit()
        self._hf_edit.setText(self._config.get("hf_mirror", "https://hf-mirror.com"))
        hf_form.addRow(self._i18n.t("settings.hf_mirror"), self._hf_edit)
        form.addWidget(hf_group)
        self._hf_group = hf_group

        # 代理设置
        proxy_group = QGroupBox()
        proxy_form = QFormLayout(proxy_group)
        self._proxy_http_edit = QLineEdit()
        self._proxy_http_edit.setText(self._config.get("proxy_http", ""))
        self._proxy_http_edit.setPlaceholderText("http://127.0.0.1:7890")
        proxy_form.addRow("HTTP:", self._proxy_http_edit)

        self._proxy_socks5_edit = QLineEdit()
        self._proxy_socks5_edit.setText(self._config.get("proxy_socks5", ""))
        self._proxy_socks5_edit.setPlaceholderText("socks5://127.0.0.1:7891")
        proxy_form.addRow("SOCKS5:", self._proxy_socks5_edit)
        form.addWidget(proxy_group)
        self._proxy_group = proxy_group

        # 系统信息
        sys_group = QGroupBox()
        sys_layout = QVBoxLayout(sys_group)
        self._sys_info_text = QTextEdit()
        self._sys_info_text.setReadOnly(True)
        self._sys_info_text.setStyleSheet("font-size: 13px;")

        self._gpu_info_text = QTextEdit()
        self._gpu_info_text.setReadOnly(True)
        self._gpu_info_text.setStyleSheet("font-size: 13px;")
        self._load_gpu_info()
        sys_layout.addWidget(self._gpu_info_text)
        form.addWidget(sys_group)
        self._sys_group = sys_group

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _connect_signals(self):
        self._lang_combo.currentIndexChanged.connect(self._on_language_changed)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._ws_browse_btn.clicked.connect(self._on_browse_workspace)
        self._hf_edit.editingFinished.connect(self._on_hf_mirror_changed)
        self._proxy_http_edit.editingFinished.connect(self._on_proxy_changed)
        self._proxy_socks5_edit.editingFinished.connect(self._on_proxy_changed)

    def _on_language_changed(self, index):
        lang = self._lang_combo.currentData()
        if lang:
            self._config.set("language", lang)
            self._i18n.load_language(lang, force=True)

    def _on_theme_changed(self, index):
        theme = self._theme_combo.currentData()
        if theme:
            self._config.set("theme", theme)
            from ui.theme import ThemeManager
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                ThemeManager.instance().apply_theme(app, theme)

    def _on_browse_workspace(self):
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, self._i18n.t("settings.workspace"))
        if dir_path:
            self._ws_edit.setText(dir_path)
            self._config.set("workspace", dir_path)

    def _on_hf_mirror_changed(self):
        self._config.set("hf_mirror", self._hf_edit.text().strip())

    def _on_proxy_changed(self):
        self._config.set("proxy_http", self._proxy_http_edit.text().strip())
        self._config.set("proxy_socks5", self._proxy_socks5_edit.text().strip())

    def _load_sys_info(self):
        try:
            from utils.system_info import get_system_info
            info = get_system_info()
            lines = []
            for k, v in info.items():
                lines.append(f"{k}: {v}")
            self._sys_info_text.setPlainText("\n".join(lines))
        except Exception:
            self._sys_info_text.setPlainText("N/A")

    def _load_gpu_info(self):
        try:
            from utils.gpu_info import get_gpu_info, get_cuda_version
            gpus = get_gpu_info()
            cuda_ver = get_cuda_version()
            lines = []
            for g in gpus:
                lines.append(
                    f"GPU {g['index']}: {g['name']} | "
                    f"VRAM: {g['vram_used_mb']}/{g['vram_total_mb']} MB | "
                    f"Temp: {g['temperature_c']}C | "
                    f"Driver: {g['driver_version']}"
                )
            if cuda_ver:
                lines.append(f"CUDA: {cuda_ver}")
            if not gpus:
                lines.append("No GPU detected")
            self._gpu_info_text.setPlainText("\n".join(lines))
        except Exception:
            self._gpu_info_text.setPlainText("N/A")

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.settings"))
        self._lang_group.setTitle(self._i18n.t("settings.language"))
        self._theme_group.setTitle(self._i18n.t("settings.theme"))
        self._ws_group.setTitle(self._i18n.t("settings.workspace"))
        self._ws_browse_btn.setText(self._i18n.t("common.browse"))
        self._hf_group.setTitle(self._i18n.t("settings.hf_mirror"))
        self._proxy_group.setTitle(self._i18n.t("settings.proxy"))
        self._sys_group.setTitle(self._i18n.t("settings.sys_info"))

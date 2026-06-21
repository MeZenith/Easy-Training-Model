import json
import locale as sys_locale
import logging
import os

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("EasyTinking")


class I18n(QObject):
    """国际化管理器，支持动态切换语言并通知 UI 刷新"""

    language_changed = Signal()

    _instance = None

    def __init__(self, locale_dir: str = ""):
        super().__init__()
        if not locale_dir:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            locale_dir = os.path.join(base, "locale")
        self._locale_dir = locale_dir
        self._current_lang = "zh"
        self._translations = {}
        self._fallback = {}
        self._load_fallback()

    @classmethod
    def instance(cls, locale_dir: str = "") -> "I18n":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(locale_dir)
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例（测试用）"""
        cls._instance = None

    def _load_fallback(self):
        """加载英文作为回退语言"""
        en_path = os.path.join(self._locale_dir, "en.json")
        try:
            if os.path.isfile(en_path):
                with open(en_path, "r", encoding="utf-8-sig") as f:
                    self._fallback = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._fallback = {}

    def detect_system_language(self) -> str:
        """检测系统语言，返回 'zh' 或 'en'"""
        try:
            lang = sys_locale.getdefaultlocale()[0] or ""
            if lang.startswith("zh"):
                return "zh"
        except Exception:
            logger.warning("Failed to detect system language")
            pass
        return "en"

    def load_language(self, lang: str, force: bool = False):
        """加载指定语言

        Args:
            lang: 语言代码如 'zh', 'en'
            force: 强制重新加载并发射信号（即使语言相同）
        """
        if not force and lang == self._current_lang and self._translations:
            return
        path = os.path.join(self._locale_dir, f"{lang}.json")
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8-sig") as f:
                    self._translations = json.load(f)
                self._current_lang = lang
            else:
                self._translations = {}
        except (json.JSONDecodeError, OSError):
            self._translations = {}
        self.language_changed.emit()

    @property
    def current_language(self) -> str:
        return self._current_lang

    def t(self, key: str, **kwargs) -> str:
        """获取翻译文本，支持格式化参数

        用法:
            i18n.t("error.dep_missing", name="torch")
        """
        text = self._translations.get(key)
        if text is None:
            text = self._fallback.get(key, key)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                pass
        return text

    def available_languages(self) -> list:
        """返回可用语言列表"""
        langs = []
        for fname in os.listdir(self._locale_dir):
            if fname.endswith(".json"):
                langs.append(fname[:-5])
        return sorted(langs) if langs else ["zh", "en"]

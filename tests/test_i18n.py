"""Tests for utils/i18n.py — internationalization manager"""

import os
import tempfile
import json
import pytest

from utils.i18n import I18n


class TestI18n:
    def setup_method(self):
        self.locale_dir = tempfile.mkdtemp()
        I18n.reset()
        with open(os.path.join(self.locale_dir, "zh.json"), "w", encoding="utf-8") as f:
            json.dump({
                "nav.model": "模型",
                "nav.data": "数据",
                "greeting": "你好 {name}",
            }, f, ensure_ascii=False)
        with open(os.path.join(self.locale_dir, "en.json"), "w", encoding="utf-8") as f:
            json.dump({
                "nav.model": "Model",
                "nav.data": "Data",
                "greeting": "Hello {name}",
            }, f, ensure_ascii=False)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.locale_dir, ignore_errors=True)
        I18n.reset()

    def test_load_zh_returns_chinese(self):
        i18n = I18n.instance(self.locale_dir)
        i18n.load_language("zh")
        assert i18n.t("nav.model") == "模型"
        assert i18n.t("nav.data") == "数据"

    def test_load_en_returns_english(self):
        i18n = I18n.instance(self.locale_dir)
        i18n.load_language("en")
        assert i18n.t("nav.model") == "Model"

    def test_fallback_to_key_when_missing(self):
        i18n = I18n.instance(self.locale_dir)
        i18n.load_language("zh")
        assert i18n.t("nonexistent_key") == "nonexistent_key"

    def test_format_params(self):
        i18n = I18n.instance(self.locale_dir)
        i18n.load_language("zh", force=True)
        assert i18n.t("greeting", name="World") == "你好 World"

    def test_switch_language(self):
        i18n = I18n.instance(self.locale_dir)
        i18n.load_language("zh")
        assert i18n.t("nav.model") == "模型"
        i18n.load_language("en", force=True)
        assert i18n.t("nav.model") == "Model"

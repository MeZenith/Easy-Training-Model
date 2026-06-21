"""Tests for core/config.py — AppConfig thread-safe JSON configuration"""

import os
import tempfile

from core.config import AppConfig


class TestAppConfig:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "config.json")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_config_returns_defaults(self):
        config = AppConfig(self.config_path)
        assert config.get("language") == "zh"
        assert config.get("theme") == "dark"

    def test_set_and_get(self):
        config = AppConfig(self.config_path)
        config.set("language", "en")
        assert config.get("language") == "en"

    def test_dot_path_set_get(self):
        config = AppConfig(self.config_path)
        config.set("window.width", 1920)
        assert config.get("window.width") == 1920

    def test_persist_and_reload(self):
        config = AppConfig(self.config_path)
        config.set("theme", "light")
        config2 = AppConfig(self.config_path)
        assert config2.get("theme") == "light"

    def test_reset_defaults(self):
        config = AppConfig(self.config_path)
        config.set("language", "en")
        config.reset_defaults("language")
        assert config.get("language") == "zh"

    def test_deep_merge_preserves_unset(self):
        config = AppConfig(self.config_path)
        config.set("window.width", 1920)
        assert config.get("window.height") == 800

    def test_workspace_auto_resolve(self):
        config = AppConfig(self.config_path)
        ws = config.workspace
        assert "workspace" in ws or ws.endswith("workspace")
        assert os.path.isdir(ws)

    def test_ui_constants_accessible(self):
        config = AppConfig(self.config_path)
        assert config.get("ui_constants.training.vram_margin_pct") == 90
        assert config.get("ui_constants.window_min_width") == 1024

import json
import logging
import os
import sys
import threading
from copy import deepcopy

logger = logging.getLogger("EasyTraining")

_DEFAULT_CONFIG = {
    "language": "zh",
    "theme": "dark",
    "workspace": "",
    "hf_mirror": "https://hf-mirror.com",
    "proxy_http": "",
    "proxy_socks5": "",
    "download_dir": "",
    "export_dir": "",
    "default_train_params": {
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 4,
        "learning_rate": 0.0002,
        "lr_scheduler": "cosine",
        "warmup_steps": 5,
        "max_seq_length": 2048,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "seed": 3407,
        "quantization": "8bit"
    },
    "train_presets": {
        "quick": {"lora_rank": 8, "epochs": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seq_length": 1024},
        "standard": {"lora_rank": 16, "epochs": 3, "batch_size": 1, "learning_rate": 0.0002, "max_seq_length": 1024},
        "fine": {"lora_rank": 32, "epochs": 5, "batch_size": 1, "learning_rate": 0.0001, "max_seq_length": 2048}
    },
    "model_identity": {
        "name": "",
        "creator": "",
        "description": ""
    },
    "ollama_name": "my-model",
    "window": {
        "width": 1280,
        "height": 800,
        "x": 100,
        "y": 100,
        "maximized": False
    },
    "last_state": {
        "selected_model": "",
        "selected_datasets": [],
        "selected_lora": "",
        "current_page": "train"
    },
    "ui_constants": {
        "title_bar_height": 40,
        "sidebar_width_expanded": 200,
        "sidebar_width_collapsed": 56,
        "window_min_width": 1024,
        "window_min_height": 640,
        "window_default_width": 1280,
        "window_default_height": 800,
        "training": {
            "vram_margin_pct": 90,
            "disk_min_free_gb": 5,
            "data_min_samples": 10,
            "data_recommend_samples": 50,
        },
        "slider_defaults": {
            "temperature": 70,
            "top_p": 90,
            "top_k": 50,
            "repetition_penalty": 110,
            "presence_penalty": 0,
            "max_tokens": 1024,
        },
    }
}


class AppConfig:
    #应用配置管理，支持点号路径取值，线程安全

    def __init__(self, config_path: str = ""):
        if not config_path:
            workspace = self._resolve_workspace()
            config_path = os.path.join(workspace, "config.json")
        self._config_path = config_path
        self._lock = threading.Lock()
        self._data = deepcopy(_DEFAULT_CONFIG)
        self._ensure_dirs()
        self._load()
        self._fixup_workspace()

    def _fixup_workspace(self):
        #修复因为目录移动/重命名导致的workspace路径不一致
        resolved = self._resolve_workspace()
        saved = self._data.get("workspace", "")
        if saved and os.path.normpath(saved) != os.path.normpath(resolved):
            if not os.path.isdir(saved):
                self._data["workspace"] = resolved
                self._data["download_dir"] = os.path.join(resolved, "models")
                self._data["export_dir"] = os.path.join(resolved, "exports")
                self._ensure_dirs()
                self.save()

    def _resolve_workspace(self) -> str:
        #解析工作目录，兼容PyInstaller打包
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "workspace")

    def _ensure_dirs(self):
        #确保工作目录和子目录都存在
        ws = self._data.get("workspace") or self._resolve_workspace()
        if not self._data["workspace"]:
            self._data["workspace"] = ws
        if not self._data.get("download_dir"):
            self._data["download_dir"] = os.path.join(ws, "models")
        if not self._data.get("export_dir"):
            self._data["export_dir"] = os.path.join(ws, "exports")
        for subdir in ["models", "data", "lora", "exports", "logs"]:
            os.makedirs(os.path.join(ws, subdir), exist_ok=True)
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)

    def _load(self):
        #从文件加载配置
        try:
            if os.path.isfile(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._deep_merge(self._data, saved)
        except json.JSONDecodeError as e:
            logger.warning(f"Config file corrupted, using defaults: {e}")
        except OSError as e:
            logger.warning(f"Cannot read config file {self._config_path}: {e}")

    def _deep_merge(self, base, override):
        #深度合并字典
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self):
        #保存到文件，先写临时文件再替换（防止写入过程中崩溃丢数据）
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                tmp = self._config_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                if os.path.isfile(self._config_path):
                    os.replace(tmp, self._config_path)
                else:
                    os.rename(tmp, self._config_path)
                # logger.debug("Config saved to %s", self._config_path)
            except OSError as e:
                logger.error(f"Failed to save config to {self._config_path}: {e}")

    def get(self, key: str, default=None):
        #取配置值，支持点号分隔：get("window.width")
        with self._lock:
            keys = key.split(".")
            node = self._data
            for k in keys:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    return default
            return deepcopy(node)

    def set(self, key: str, value):
        #设配置值并自动保存
        with self._lock:
            keys = key.split(".")
            node = self._data
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value
        self.save()

    def get_all(self) -> dict:
        #返回全部配置的深拷贝
        with self._lock:
            return deepcopy(self._data)

    def reset_defaults(self, key: str = ""):
        #重置为默认值
        with self._lock:
            if key:
                keys = key.split(".")
                src = _DEFAULT_CONFIG
                dst = self._data
                for k in keys[:-1]:
                    src = src.get(k, {})
                    dst = dst.setdefault(k, {})
                if keys[-1] in src:
                    dst[keys[-1]] = deepcopy(src[keys[-1]])
            else:
                self._data = deepcopy(_DEFAULT_CONFIG)
                self._ensure_dirs()
        self.save()

    @property
    def config_path(self) -> str:
        return self._config_path

    @property
    def workspace(self) -> str:
        return self.get("workspace", "")

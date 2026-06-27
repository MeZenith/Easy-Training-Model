import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("EasyTraining")

#内置模型列表
BUILTIN_MODELS = [
    {
        "id": "unsloth/Qwen2.5-Coder-1.5B-Instruct",
        "name": "Qwen2.5-Coder-1.5B-Instruct",
        "params": "1.5B",
        "size_gb": 1.5,
        "min_vram_gb": 4,
        "model_type": "code",
    },
    {
        "id": "unsloth/Qwen2.5-Coder-3B-Instruct",
        "name": "Qwen2.5-Coder-3B-Instruct",
        "params": "3B",
        "size_gb": 3,
        "min_vram_gb": 8,
        "model_type": "code",
    },
    {
        "id": "unsloth/Qwen2.5-Coder-7B-Instruct",
        "name": "Qwen2.5-Coder-7B-Instruct",
        "params": "7B",
        "size_gb": 7,
        "min_vram_gb": 16,
        "model_type": "code",
    },
    {
        "id": "unsloth/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B-Instruct",
        "params": "3B",
        "size_gb": 3,
        "min_vram_gb": 8,
        "model_type": "general",
    },
    {
        "id": "unsloth/Qwen2.5-7B-Instruct",
        "name": "Qwen2.5-7B-Instruct",
        "params": "7B",
        "size_gb": 7,
        "min_vram_gb": 16,
        "model_type": "general",
    },
    {
        "id": "unsloth/Llama-3.2-3B-Instruct",
        "name": "Llama-3.2-3B-Instruct",
        "params": "3B",
        "size_gb": 3,
        "min_vram_gb": 8,
        "model_type": "general",
    },
]

#模型必需文件
REQUIRED_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json"]


class ModelManager:
    #模型管理器 — 下载/校验/列出模型

    def __init__(self, download_dir: str, hf_mirror: str = ""):
        self._download_dir = download_dir
        self._hf_mirror = hf_mirror
        os.makedirs(download_dir, exist_ok=True)

    @property
    def download_dir(self) -> str:
        return self._download_dir

    @download_dir.setter
    def download_dir(self, value: str):
        self._download_dir = value
        os.makedirs(value, exist_ok=True)

    @property
    def hf_mirror(self) -> str:
        return self._hf_mirror

    @hf_mirror.setter
    def hf_mirror(self, value: str):
        self._hf_mirror = value

    def list_builtin_models(self) -> list:
        #返回内置模型列表
        return BUILTIN_MODELS.copy()

    def list_downloaded_models(self) -> list:
        #扫描下载目录，列出已下载的模型
        models = []
        if not os.path.isdir(self._download_dir):
            return models
        for entry in os.listdir(self._download_dir):
            model_path = os.path.join(self._download_dir, entry)
            if not os.path.isdir(model_path):
                continue
            config_path = os.path.join(model_path, "config.json")
            info = {
                "name": entry,
                "path": model_path,
                "params": "-",
                "size_bytes": self._calc_dir_size(model_path),
                "download_time": "",
                "status": "unknown",
            }

            #读config.json估算参数量
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    info["params"] = self._estimate_params(cfg)
                except (json.JSONDecodeError, OSError):
                    pass

            #获取下载时间（用目录最新文件时间）
            mtime = self._get_dir_mtime(model_path)
            if mtime:
                info["download_time"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))

            #校验模型完整性
            valid, missing = self.validate_model(model_path)
            info["status"] = "ok" if valid else f"missing: {', '.join(missing)}"

            models.append(info)
        return models

    @staticmethod
    def _calc_dir_size(path: str) -> int:
        #计算目录总大小
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    @staticmethod
    def _get_dir_mtime(path: str) -> Optional[float]:
        #获取目录里最新的文件修改时间
        latest = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                try:
                    t = os.path.getmtime(os.path.join(root, f))
                    if t > latest:
                        latest = t
                except OSError:
                    pass
        return latest if latest > 0 else None

    @staticmethod
    def _estimate_params(cfg: dict) -> str:
        #从config.json粗略估算参数量
        try:
            hidden = cfg.get("hidden_size", 0)
            layers = cfg.get("num_hidden_layers", 0)
            if hidden and layers:
                rough = (hidden * hidden * 4 * layers) // (10 ** 9)
                if rough > 0:
                    return f"{rough}B"
        except Exception:
            logger.warning("Failed to estimate params from config")
        return "-"

    @staticmethod
    def validate_model(model_path: str) -> tuple:
        #验证模型目录是否完整，返回 (是否完整, 缺少的文件列表)
        if not os.path.isdir(model_path):
            return False, REQUIRED_FILES

        missing = []
        for fname in REQUIRED_FILES:
            if not os.path.isfile(os.path.join(model_path, fname)):
                missing.append(fname)

        #检查有没有权重文件
        has_weights = False
        for f in os.listdir(model_path):
            if f.startswith("model") and (f.endswith(".safetensors") or f.endswith(".bin")):
                has_weights = True
                break
        if not has_weights:
            missing.append("model weights (safetensors/bin)")

        return len(missing) == 0, missing

    def download_model(self, model_id: str, target_dir: str = "",
                       progress_callback=None) -> str:
        #下载模型（同步方法，要放在子线程里调）
        if not target_dir:
            target_dir = self._download_dir
        os.makedirs(target_dir, exist_ok=True)

        #如果已经下载了就跳过
        model_name = model_id.split("/")[-1]
        model_dir = os.path.join(target_dir, model_name)
        if os.path.isdir(model_dir):
            valid, _ = self.validate_model(model_dir)
            if valid:
                logger.info(f"Model already downloaded: {model_id}")
                return model_dir

        try:
            from huggingface_hub import snapshot_download

            kwargs = {
                "repo_id": model_id,
                "local_dir": model_dir,
            }
            if self._hf_mirror:
                kwargs["endpoint"] = self._hf_mirror

            path = snapshot_download(**kwargs)
            logger.info(f"Model downloaded to: {path}")
            return path

        except ImportError:
            raise ImportError("huggingface_hub is required. Run: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def delete_model(self, model_path: str) -> bool:
        #删除模型目录
        import shutil
        try:
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
                logger.info(f"Model deleted: {model_path}")
                return True
        except OSError as e:
            logger.error(f"Delete failed: {e}")
        return False

    @staticmethod
    def get_model_detail(model_path: str) -> dict:
        #获取模型详细信息
        detail = {
            "path": model_path,
            "config": {},
            "files": [],
            "valid": False,
            "missing": [],
        }
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    detail["config"] = json.load(f)
            except Exception:
                logger.warning(f"Failed to read model config {config_path}")

        if os.path.isdir(model_path):
            for f in os.listdir(model_path):
                fpath = os.path.join(model_path, f)
                if os.path.isfile(fpath):
                    detail["files"].append({
                        "name": f,
                        "size": os.path.getsize(fpath),
                    })

        valid, missing = ModelManager.validate_model(model_path)
        detail["valid"] = valid
        detail["missing"] = missing
        return detail

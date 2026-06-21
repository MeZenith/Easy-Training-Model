"""模型下载 Worker — 从 HuggingFace Hub 下载模型文件

继承 BaseWorker，在子线程中执行耗时下载操作，通过 Signal 报告进度。
"""

import logging
import os

from utils.worker import BaseWorker

logger = logging.getLogger("EasyTinking")


class DownloadWorker(BaseWorker):
    """HuggingFace 模型下载 Worker

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2.5-Coder-3B-Instruct")
        download_dir: 本地下载目录
        hf_mirror: HuggingFace 镜像 URL（可选）

    信号:
        progress: (int, str) 进度百分比和描述
        finished: (dict) 完成结果 {"path": str, "model_id": str}
        error: (str, str) 错误码和详情
    """

    def __init__(self, model_id: str, download_dir: str, hf_mirror: str = "",
                 parent=None):
        super().__init__(parent)
        self._model_id = model_id
        self._download_dir = download_dir
        self._hf_mirror = hf_mirror

    def do_work(self) -> dict:
        """执行下载任务

        Returns:
            dict: {"path": downloaded_path, "model_id": model_id}
        """
        from huggingface_hub import snapshot_download

        model_name = self._model_id.split("/")[-1]
        model_dir = os.path.join(self._download_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        self.signals.progress.emit(5, f"Downloading {self._model_id}...")

        kwargs = {
            "repo_id": self._model_id,
            "local_dir": model_dir,
        }
        if self._hf_mirror:
            kwargs["endpoint"] = self._hf_mirror

        try:
            path = snapshot_download(**kwargs)
            self.signals.progress.emit(100, "Download complete")
            return {"path": path, "model_id": self._model_id}
        except Exception as e:
            logger.error(f"Download failed for {self._model_id}: {e}")
            self.signals.error.emit("ERR_NETWORK", str(e))
            return {}

import logging
import os

from utils.worker import BaseWorker

logger = logging.getLogger("EasyTraining")


class DownloadWorker(BaseWorker):
    #HuggingFace模型下载Worker
    #参数: model_id, download_dir, hf_mirror
    #发出信号: progress(进度,描述) finished(结果) error(错误码,详情)

    def __init__(self, model_id: str, download_dir: str, hf_mirror: str = "",
                 parent=None):
        super().__init__(parent)
        self._model_id = model_id
        self._download_dir = download_dir
        self._hf_mirror = hf_mirror

    def do_work(self) -> dict:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            self.signals.error.emit("ERR_DEP_MISSING", f"huggingface_hub未安装: {e}")
            return {}

        model_name = self._model_id.split("/")[-1]
        model_dir = os.path.join(self._download_dir, model_name)

        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            self.signals.error.emit("ERR_PERMISSION", f"无法创建下载目录: {e}")
            return {}

        self.signals.progress.emit(5, f"Downloading {self._model_id}...")

        kwargs = {"repo_id": self._model_id, "local_dir": model_dir}
        if self._hf_mirror:
            kwargs["endpoint"] = self._hf_mirror

        try:
            path = snapshot_download(**kwargs)
            if path is None:
                self.signals.error.emit("ERR_NETWORK", "下载失败：返回了空结果，检查网络或镜像源")
                return {}
            self.signals.progress.emit(100, "Download complete")
            return {"path": path, "model_id": self._model_id}
        except Exception as e:
            logger.error(f"Download failed for {self._model_id}: {e}")
            self.signals.error.emit("ERR_NETWORK", str(e))
            return {}

"""训练逻辑"""

import os
import sys
import json
import time
import logging
import tempfile
from typing import Optional

from PySide6.QtCore import QProcess, QObject, Signal

logger = logging.getLogger("EasyTinking")


class ProcessTrainer(QObject):
    """QProcess-based training manager — isolates CUDA ops in subprocess

    Communicates with train_worker.py via stdout prefix protocol.
    Handles process lifecycle: start, stop, stdout parsing, error handling.

    Args:
        workspace: workspace root directory dictating lora/ subfolder

    Signals:
        progress(int, str): training progress percentage + description
        finished(dict): training complete result {final_loss, elapsed_seconds, ...}
        error(str, str): error code + detail message
        log_message(str): log lines from subprocess
        metric(dict): per-step metric {step, loss, lr}
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log_message = Signal(str)
    metric = Signal(dict)

    def __init__(self, workspace: str):
        super().__init__()
        self._workspace = workspace
        self._lora_dir = os.path.join(workspace, "lora")
        os.makedirs(self._lora_dir, exist_ok=True)
        self._process = None
        self._config_path = ""
        self._pending_output = ""

    def start_training(self, config: dict):
        """启动子进程训练"""
        lora_name = config.get("lora_name", "untitled")
        output_dir = os.path.join(self._lora_dir, self._sanitize(lora_name))
        os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        # 保存配置到临时 JSON 文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            self._config_path = f.name

        # 启动子进程
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_process_finished)
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "train_worker.py"
        )
        self._process.start(
            sys.executable,
            [worker_path, "--config", self._config_path]
        )
        logger.info(f"Training process started (PID: {self._process.processId()})")

    def stop_training(self):
        """停止训练"""
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
            logger.info("Training process killed")

    def _process_line(self, line: str):
        """解析一行子进程输出"""
        line = line.strip()
        if not line:
            return

        if line.startswith("PROGRESS:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                self.progress.emit(int(parts[1]), parts[2])

        elif line.startswith("RESULT:"):
            try:
                result = json.loads(line.split(":", 1)[1])
                self.finished.emit(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse training result: {e}")

        elif line.startswith("ERROR:"):
            parts = line.split(":", 2)
            code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
            detail = parts[2] if len(parts) > 2 else "Unknown error"
            self.error.emit(code, detail)

        elif line.startswith("LOG:"):
            self.log_message.emit(line[4:])

        elif line.startswith("METRIC:"):
            try:
                parts = dict(p.split("=", 1) for p in line[7:].split())
                self.metric.emit({
                    "step": int(parts.get("step", 0)),
                    "loss": float(parts.get("loss", 0)),
                    "lr": float(parts.get("lr", 0)),
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse metric: {e}")

        elif line == "DONE":
            pass

    def _on_stdout(self):
        """读取子进程输出"""
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._pending_output += data
        while "\n" in self._pending_output:
            line, self._pending_output = self._pending_output.split("\n", 1)
            self._process_line(line)

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """子进程退出处理"""
        self._on_stdout()
        while self._pending_output.strip():
            self._process_line(self._pending_output.strip())
            self._pending_output = ""

        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Training process crashed (exit code {exit_code})")

        if self._config_path and os.path.isfile(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temp config {self._config_path}: {e}")

    def list_loras(self) -> list:
        """列出已训练的 LoRA"""
        loras = []
        if not os.path.isdir(self._lora_dir):
            return loras
        for entry in os.listdir(self._lora_dir):
            path = os.path.join(self._lora_dir, entry)
            if os.path.isdir(path):
                meta_path = os.path.join(path, "metadata.json")
                info = {"name": entry, "path": path, "metadata": {}}
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            info["metadata"] = json.load(f)
                        mp = info["metadata"].get("model_path", "")
                        if mp:
                            info["metadata"]["model_path"] = os.path.normpath(mp).replace("\\", "/")
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to read LoRA metadata from {meta_path}: {e}")
                loras.append(info)
        return loras

    @staticmethod
    def _sanitize(name: str) -> str:
        import re
        name = name.strip()
        return re.sub(r'[^\w\-.]', '_', name) or "untitled"


# 向后兼容别名
Trainer = ProcessTrainer

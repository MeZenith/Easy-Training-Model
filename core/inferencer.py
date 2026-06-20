"""推理 QProcess 管理器 — 与训练相同的子进程隔离模式"""

import os
import sys
import json
import logging
import tempfile

from PySide6.QtCore import QProcess, QProcessEnvironment, QObject, Signal

logger = logging.getLogger("EasyTinking")


class Inferencer(QObject):
    """QProcess-based inference manager — isolates CUDA operations in subprocess

    Communicates with infer_worker.py via stdin/stdout JSON lines.
    Prefix-based line routing: LOG:→progress, TOKEN:→token, RESULT:→result, ERROR:→error, LOADED:{}→loaded.

    Signals:
        loaded(): subprocess finished loading model
        progress(str): loading progress messages
        token(str): streaming token output
        result(dict): final generation result {text, tokens, speed, ...}
        error(str, str): error code and detail message
    """

    loaded = Signal()
    progress = Signal(str)
    token = Signal(str)
    result = Signal(dict)
    error = Signal(str, str)

    def __init__(self):
        super().__init__()
        self._process = None
        self._config_path = ""
        self._pending = ""

    def start(self, model_path: str, lora_path: str = ""):
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)

        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(env)

        cfg = {"model_path": model_path, "lora_path": lora_path}
        fd, self._config_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "infer_worker.py"
        )
        self._process.start(
            sys.executable,
            [worker_path, "--config", self._config_path]
        )
        logger.info(f"Inferencer process started (PID: {self._process.processId()})")

    def generate(self, messages: list, params: dict):
        if not self._process or self._process.state() == QProcess.NotRunning:
            self.error.emit("ERR_NO_MODEL", "Model not loaded")
            return
        req = {"action": "generate", "messages": messages, "params": params}
        self._process.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))

    def quit(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            try:
                self._process.write(b'{"action": "quit"}\n')
                self._process.waitForBytesWritten(1000)
                self._process.waitForFinished(3000)
            except Exception:
                pass
            if self._process.state() != QProcess.NotRunning:
                self._process.kill()

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return

        if line.startswith("LOG:"):
            self.progress.emit(line[4:])

        elif line == "LOADED:{}":
            self.loaded.emit()

        elif line.startswith("TOKEN:"):
            self.token.emit(line[6:])

        elif line.startswith("RESULT:"):
            try:
                data = json.loads(line.split(":", 1)[1])
                self.result.emit(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse inference result: {e}")

        elif line.startswith("ERROR:"):
            parts = line.split(":", 2)
            code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
            detail = parts[2] if len(parts) > 2 else ""
            self.error.emit(code, detail)

        elif line == "DONE":
            pass

    def _on_stdout(self):
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._pending += data
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._process_line(line)

    def _on_finished(self, exit_code, exit_status):
        self._on_stdout()
        while self._pending.strip():
            self._process_line(self._pending.strip())
            self._pending = ""

        if self._config_path and os.path.isfile(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temp config {self._config_path}: {e}")

        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Inference process crashed (exit code {exit_code})")

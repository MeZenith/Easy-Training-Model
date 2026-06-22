"""QProcess-based export manager — isolates CUDA/memory ops in subprocess"""

import json
import logging
import os
import sys
import tempfile

from PySide6.QtCore import QObject, QProcess, Signal

logger = logging.getLogger("EasyTinking")


class ProcessExporter(QObject):
    """QProcess-based export manager

    Spawns export_worker.py as a subprocess to prevent OOM crashes
    from taking down the Qt main thread.

    Signals:
        progress(int, str): export progress percentage + description
        finished(dict): export result {"files": [...], "errors": [...]}
        error(str, str): error code + detail
        log_message(str): log lines from subprocess
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._config_path = ""
        self._pending_output = ""

    def start_export(self, config: dict):
        """启动子进程导出"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            self._config_path = f.name

        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_process_finished)
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        env = self._process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)

        worker_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workers", "export_worker.py"
        )
        self._process.start(sys.executable, [worker_path, "--config", self._config_path])

        if not self._process.waitForStarted(5000):
            logger.error("Export process failed to start")
            self.error.emit("ERR_START", "Failed to start export process")
            return

        logger.info(f"Export process started (PID: {self._process.processId()})")

    def stop_export(self):
        """停止导出"""
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
            logger.info("Export process killed")

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return

        if line.startswith("PROGRESS:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                self.progress.emit(int(parts[1]), parts[2])

        elif line.startswith("RESULT:"):
            try:
                data = json.loads(line.split(":", 1)[1])
                self.finished.emit(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse export result: {e}")

        elif line.startswith("ERROR:"):
            parts = line.split(":", 2)
            code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
            detail = parts[2] if len(parts) > 2 else "Unknown error"
            self.error.emit(code, detail)

        elif line.startswith("LOG:"):
            self.log_message.emit(line[4:])

        elif line == "DONE":
            pass

    def _on_stdout(self):
        if not self._process:
            return
        raw = self._process.readAllStandardOutput()
        if not raw:
            return
        data = raw.data().decode("utf-8", errors="replace")
        self._pending_output += data
        while "\n" in self._pending_output:
            line, self._pending_output = self._pending_output.split("\n", 1)
            self._process_line(line)

    def _on_process_finished(self, exit_code, exit_status):
        self._on_stdout()
        while self._pending_output.strip():
            self._process_line(self._pending_output.strip())
            self._pending_output = ""

        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Export process crashed (exit code {exit_code})")

        if self._config_path and os.path.isfile(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError:
                pass

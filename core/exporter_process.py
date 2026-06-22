"""Export process manager — subprocess.Popen + QThread for reliable pipe reading

Replaces QProcess which has unreliable stdout pipe behaviour on Windows
when the child process calls sys.stdout.reconfigure().
"""

import json
import logging
import os
import subprocess
import sys
import tempfile

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger("EasyTinking")


class _ReaderThread(QThread):
    """Reads subprocess stdout line by line, emits parsed signals"""

    progress_line = Signal(int, str)
    finished_line = Signal(dict)
    error_line = Signal(str, str)
    log_line = Signal(str)

    def __init__(self, proc: subprocess.Popen, config_path: str, parent=None):
        super().__init__(parent)
        self._proc = proc
        self._config_path = config_path

    def run(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("PROGRESS:"):
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        self.progress_line.emit(int(parts[1]), parts[2])

                elif line.startswith("RESULT:"):
                    try:
                        data = json.loads(line.split(":", 1)[1])
                        self.finished_line.emit(data)
                    except json.JSONDecodeError:
                        pass

                elif line.startswith("ERROR:"):
                    parts = line.split(":", 2)
                    code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
                    detail = parts[2] if len(parts) > 2 else "Unknown error"
                    self.error_line.emit(code, detail)

                elif line.startswith("LOG:"):
                    self.log_line.emit(line[4:])

        except Exception as e:
            self.error_line.emit("ERR_READ", str(e))

        finally:
            self._proc.stdout.close()
            self._proc.wait()

            if self._config_path and os.path.isfile(self._config_path):
                try:
                    os.unlink(self._config_path)
                except OSError:
                    pass


class ProcessExporter(QObject):
    """Export manager using subprocess for reliable pipe I/O

    Signals:
        progress(int, str)
        finished(dict)
        error(str, str)
        log_message(str)
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._thread = None

    def start_export(self, config: dict):
        """Launch export_worker.py in a subprocess"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            config_path = f.name

        worker_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workers", "export_worker.py"
        )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            [sys.executable, worker_path, "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        logger.info(f"Export process started (PID: {self._proc.pid})")

        self._thread = _ReaderThread(self._proc, config_path, self)
        self._thread.progress_line.connect(self.progress)
        self._thread.finished_line.connect(self.finished)
        self._thread.error_line.connect(self.error)
        self._thread.log_line.connect(self.log_message)
        self._thread.start()

    def stop_export(self):
        """Kill the export subprocess"""
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            logger.info("Export process killed")

        if self._thread and self._thread.isRunning():
            self._thread.wait(3000)

import json
import logging
import os
import subprocess
import sys
import tempfile

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger("EasyTraining")


class _ReaderThread(QThread):
    #子进程stdout读取线程，逐行解析协议发信号

    progress_line = Signal(int, str)
    finished_line = Signal(dict)
    error_line = Signal(str, str)
    log_line = Signal(str)

    def __init__(self, proc: subprocess.Popen, config_path: str, parent=None):
        super().__init__(parent)
        self._proc = proc
        self._cfg = config_path
        self._err = False

    def _dispatch(self, line: str):
        #解析一行协议文本
        line = line.strip()
        if not line:
            return

        if line.startswith("PROGRESS:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    self.progress_line.emit(int(parts[1]), parts[2])
                except (ValueError, IndexError):
                    logger.warning("Invalid PROGRESS line: %s", line)

        elif line.startswith("RESULT:"):
            try:
                data = json.loads(line.split(":", 1)[1])
                self.finished_line.emit(data)
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning("Invalid RESULT line: %s — %s", line, e)

        elif line.startswith("ERROR:"):
            self._err = True
            parts = line.split(":", 2)
            code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
            detail = parts[2] if len(parts) > 2 else "Unknown error"
            self.error_line.emit(code, detail)

        elif line.startswith("LOG:"):
            self.log_line.emit(line[4:])

        else:
            logger.debug("Skipped non-protocol line from subprocess: %s", line)

    def run(self):
        #阻塞读取子进程stdout直到管道关闭
        try:
            for raw_line in self._proc.stdout:
                decoded = raw_line.decode("utf-8", errors="replace")
                self._dispatch(decoded)
        except OSError as e:
            logger.warning("Reader thread OS error: %s", e)
        except Exception as e:
            logger.warning("Reader thread unexpected error: %s", e)
            self.error_line.emit("ERR_READ", str(e))
        finally:
            self._proc.stdout.close()
            self._proc.wait()

            exit_code = self._proc.returncode
            #进程非0退出但没发过ERROR的情况
            if exit_code != 0 and not self._err:
                self.error_line.emit(
                    "ERR_EXIT",
                    f"子进程异常退出，退出码: {exit_code}",
                )

            #清理临时配置
            if self._cfg and os.path.isfile(self._cfg):
                try:
                    os.unlink(self._cfg)
                except OSError as e:
                    logger.warning("Failed to delete temp config %s: %s", self._cfg, e)


class ProcessExporter(QObject):
    #导出子进程管理器
    #用subprocess.Popen替代QProcess（Windows下管道行为更可靠）

    progress = Signal(int, str)     #进度百分比 + 描述
    finished = Signal(dict)         #完成结果
    error = Signal(str, str)        #错误码 + 详情
    log_message = Signal(str)       #子进程日志

    _WORKER_SCRIPT = "workers/export_worker.py"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._thread = None

    @classmethod
    def _resolve_worker_path(cls) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), cls._WORKER_SCRIPT)

    def start_export(self, config: dict):
        #启动导出子进程
        #配置写到临时json文件，通过--config传给子进程
        worker_path = self._resolve_worker_path()
        if not os.path.isfile(worker_path):
            msg = f"找不到导出脚本: {worker_path}"
            logger.error(msg)
            self.error.emit("ERR_WORKER", msg)
            return

        config_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False)
                config_path = f.name
        except OSError as e:
            logger.error("Failed to create config temp file: %s", e)
            self.error.emit("ERR_CONFIG", f"无法创建临时配置文件: {e}")
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        #启动子进程
        try:
            if getattr(sys, "frozen", False):
                self._proc = subprocess.Popen(
                    [sys.executable, "--worker", "export", "--config", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            else:
                self._proc = subprocess.Popen(
                    [sys.executable, worker_path, "--config", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        except OSError as e:
            logger.error("Failed to start export subprocess: %s", e)
            self.error.emit("ERR_START", f"无法启动导出进程: {e}")
            return

        logger.info("Export process started (PID: %s)", self._proc.pid)
        # logger.debug("config: %s", json.dumps(config, ensure_ascii=False))

        #创建读取线程
        self._thread = _ReaderThread(self._proc, config_path, self)
        self._thread.progress_line.connect(self.progress)
        self._thread.finished_line.connect(self.finished)
        self._thread.error_line.connect(self.error)
        self._thread.log_line.connect(self.log_message)
        self._thread.start()

    def stop_export(self):
        #安全终止导出子进程
        if self._proc is None:
            return

        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Terminate timed out, force killing export process")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Export process did not respond to force kill")
            logger.info("Export process stopped")

        if self._thread and self._thread.isRunning():
            self._thread.wait(10000)

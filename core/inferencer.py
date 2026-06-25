import json
import logging
import os
import sys
import tempfile

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from utils.worker import read_process_lines

logger = logging.getLogger("EasyTinking")


class Inferencer(QObject):
    #推理管理器
    #用QProcess子进程隔离CUDA，避免推理时主进程卡死
    #和infer_worker.py通过stdout前缀协议 + stdin JSON通信

    #发出信号
    loaded = Signal()           #模型加载完成
    progress = Signal(str)      #加载进度
    token = Signal(str)         #流式输出
    result = Signal(dict)       #生成结果 {text, tokens, speed}
    error = Signal(str, str)    #错误码 + 详情

    def __init__(self):
        super().__init__()
        self._proc = None
        self._cfg = ""
        self._buf = ""

    def start(self, model_path: str, lora_path: str = ""):
        #启动推理子进程
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(3000)

        #创建子进程
        self._proc = QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.finished.connect(self._on_done)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self._proc.setProcessEnvironment(env)

        #把配置写到临时文件
        cfg = {"model_path": model_path, "lora_path": lora_path}
        fd, self._cfg = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "infer_worker.py"
        )
        self._proc.start(
            sys.executable,
            [worker_path, "--config", self._cfg]
        )
        logger.info(f"Inferencer process started (PID: {self._proc.processId()})")

    def generate(self, messages: list, params: dict):
        #发送生成请求给子进程
        if not self._proc or self._proc.state() == QProcess.NotRunning:
            self.error.emit("ERR_NO_MODEL", "Model not loaded")
            return
        req = {"action": "generate", "messages": messages, "params": params}
        self._proc.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))

    def quit(self):
        #退出子进程
        if self._proc and self._proc.state() != QProcess.NotRunning:
            try:
                self._proc.write(b'{"action": "quit"}\n')
                self._proc.waitForBytesWritten(1000)
                self._proc.waitForFinished(3000)
            except Exception:
                logger.warning("Failed to gracefully quit inference process")
            if self._proc.state() != QProcess.NotRunning:
                self._proc.kill()

    def _process_line(self, line: str):
        #解析子进程输出的一行
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
        #读取子进程全部标准输出
        self._buf, lines = read_process_lines(self._proc, self._buf)
        for line in lines:
            self._process_line(line)

    def _on_done(self, exit_code, exit_status):
        #子进程退出，清空缓冲
        self._buf, lines = read_process_lines(self._proc, self._buf)
        for line in lines:
            self._process_line(line)
        if self._buf.strip():
            self._process_line(self._buf.strip())
            self._buf = ""

        #清理临时配置
        if self._cfg and os.path.isfile(self._cfg):
            try:
                os.unlink(self._cfg)
            except OSError as e:
                logger.warning(f"Failed to clean up temp config {self._cfg}: {e}")

        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Inference process crashed (exit code {exit_code})")

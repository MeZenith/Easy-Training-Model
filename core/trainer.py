import json
import logging
import os
import sys
import tempfile

from PySide6.QtCore import QObject, QProcess, Signal

logger = logging.getLogger("EasyTinking")


class ProcessTrainer(QObject):
    #训练管理器
    #用QProcess子进程隔离CUDA，避免训练崩了拖垮主进程
    #和train_worker.py通过stdout前缀协议通信

    #发出信号
    progress = Signal(int, str)     #进度百分比 + 描述
    finished = Signal(dict)         #训练结果
    error = Signal(str, str)        #错误码 + 详情
    log_message = Signal(str)       #子进程日志行
    metric = Signal(dict)           #每步指标 {step, loss, lr}

    def __init__(self, workspace: str):
        super().__init__()
        self._lora_dir = os.path.join(workspace, "lora")
        os.makedirs(self._lora_dir, exist_ok=True)
        #子进程相关
        self._proc = None
        self._cfg = ""
        self._buf = ""

    def start_training(self, config: dict):
        #启动子进程训练
        lora_name = config.get("lora_name", "untitled")
        output_dir = os.path.join(self._lora_dir, self._sanitize(lora_name))
        os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        #把配置写到临时json文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            self._cfg = f.name

        #创建子进程
        self._proc = QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.finished.connect(self._on_done)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)

        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "train_worker.py"
        )
        self._proc.start(
            sys.executable,
            [worker_path, "--config", self._cfg]
        )
        logger.info(f"Training process started (PID: {self._proc.processId()})")

    def stop_training(self):
        #停止训练 直接kill
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
            self._proc.waitForFinished(3000)
            logger.info("Training process killed")

    def _process_line(self, line: str):
        #解析子进程输出的一行，按前缀协议分发信号
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
        #读取子进程全部标准输出
        if not self._proc:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf += data
        #按行切分处理
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._process_line(line)

    def _on_done(self, exit_code: int, exit_status: QProcess.ExitStatus):
        #子进程退出，把缓冲里的残留数据读完
        self._on_stdout()
        while self._buf.strip():
            self._process_line(self._buf.strip())
            self._buf = ""

        #坑：QProcess崩溃时stdout可能没读完
        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Training process crashed (exit code {exit_code})")

        #清理临时配置
        if self._cfg and os.path.isfile(self._cfg):
            try:
                os.unlink(self._cfg)
            except OSError as e:
                logger.warning(f"Failed to clean up temp config {self._cfg}: {e}")

    def list_loras(self) -> list:
        #列出已训练的lora列表
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
        #清理名称 只保留字母数字下划线和点横线
        import re
        name = name.strip()
        return re.sub(r'[^\w\-.]', '_', name) or "untitled"


def list_loras_for_combo(workspace: str) -> list:
    #获取lora列表 给下拉框用
    #返回 [{display: "lora名 -> 模型名", lora_path, model_path}]
    trainer = ProcessTrainer(workspace)
    items = []
    for lora in trainer.list_loras():
        meta = lora.get("metadata", {})
        model_path = meta.get("model_path", "")
        if model_path:
            display = f"{lora['name']} -> {os.path.basename(model_path)}"
        else:
            display = lora["name"]
        items.append({
            "display": display,
            "lora_path": lora["path"],
            "model_path": model_path,
        })
    return items


# 向后兼容别名
Trainer = ProcessTrainer

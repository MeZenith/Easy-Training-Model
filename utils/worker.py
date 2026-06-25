import re
import logging

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger("EasyTinking")


#清理文件名，去掉不安全字符
def clean_name(name: str) -> str:
    name = name.strip()
    return re.sub(r'[^\w\-.]', '_', name) or "untitled"


#从QProcess读取stdout，处理缓冲拼接，返回 (新buf, 完成的行的列表)
def read_process_lines(proc, buf: str):
    if not proc:
        return buf, []
    data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
    buf += data
    lines = []
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        lines.append(line)
    return buf, lines


class WorkerSignals(QObject):
    #子线程通用信号
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log = Signal(str)


class BaseWorker(QThread):
    #所有耗时操作的子线程基类

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self):
        #请求取消
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        import traceback
        try:
            result = self.do_work()
            if not self._cancelled:
                self.signals.finished.emit(result or {})
        except SystemExit:
            raise
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit("ERR_UNKNOWN", str(e))

    def do_work(self) -> dict:
        #子类重写这个方法实现具体逻辑
        raise NotImplementedError

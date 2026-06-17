from PySide6.QtCore import QThread, QObject, Signal


class WorkerSignals(QObject):
    """子线程通用信号定义"""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log = Signal(str)


class BaseWorker(QThread):
    """所有耗时操作子线程的基类"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self):
        """请求取消任务"""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        """重写 run 以确保线程异常正确传播"""
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
        """子类重写此方法实现具体工作"""
        raise NotImplementedError

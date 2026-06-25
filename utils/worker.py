from PySide6.QtCore import QObject, QThread, Signal


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

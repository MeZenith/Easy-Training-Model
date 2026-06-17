"""Loss 曲线图组件"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class LossChart(QWidget):
    """实时训练 Loss 曲线图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("transparent")
        self._plot_widget.setLabel("left", "Loss")
        self._plot_widget.setLabel("bottom", "Step")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMinimumHeight(200)

        # 禁用鼠标缩放/平移，自动适配数据范围
        vb = self._plot_widget.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self._loss_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#6366f1", width=2)
        )
        self._lr_curve = self._plot_widget.plot(
            pen=pg.mkPen(color="#f59e0b", width=1, style=pg.QtCore.Qt.DashLine)
        )

        self._loss_data = []
        self._lr_data = []
        self._steps = []

        layout.addWidget(self._plot_widget)

    def add_point(self, step: int, loss: float, lr: float = 0):
        """添加一个数据点"""
        self._steps.append(step)
        self._loss_data.append(loss)
        self._lr_data.append(lr)
        self._loss_curve.setData(self._steps, self._loss_data)
        if lr > 0:
            self._lr_curve.setData(self._steps, self._lr_data)
        self._plot_widget.getViewBox().autoRange()

    def clear(self):
        self._steps.clear()
        self._loss_data.clear()
        self._lr_data.clear()
        self._loss_curve.setData([], [])
        self._lr_curve.setData([], [])

    def update_data(self, values: list):
        self._loss_data = list(values)
        self._steps = list(range(1, len(values) + 1))
        self._loss_curve.setData(self._steps, self._loss_data)
        self._plot_widget.getViewBox().autoRange()
        if lr > 0:
            self._lr_curve.setData(self._steps, self._lr_data)

    def clear(self):
        """清空数据"""
        self._steps.clear()
        self._loss_data.clear()
        self._lr_data.clear()
        self._loss_curve.setData([], [])
        self._lr_curve.setData([], [])

    def update_data(self, values: list):
        """用列表整体设置 Loss 数据"""
        self._loss_data = list(values)
        self._steps = list(range(1, len(values) + 1))
        self._loss_curve.setData(self._steps, self._loss_data)

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class LossChart(QWidget):
    #Loss曲线图，支持缩放/拖拽

    def __init__(self, parent=None, i18n=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        #绘图区域
        self._plot = pg.PlotWidget()
        self._plot.setBackground("transparent")
        self._plot.setLabel("left", "Loss" if i18n is None else i18n.t("chart.loss"))
        self._plot.setLabel("bottom", "Step" if i18n is None else i18n.t("chart.step"))
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMinimumHeight(200)

        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        #两条曲线：loss（实线）和lr（虚线）
        self._loss_curve = self._plot.plot(
            pen=pg.mkPen(color="#6366f1", width=2)
        )
        self._lr_curve = self._plot.plot(
            pen=pg.mkPen(color="#f59e0b", width=1, style=pg.QtCore.Qt.DashLine)
        )

        self._loss_data = []
        self._lr_data = []
        self._steps = []

        #按钮行
        btn_row = QHBoxLayout()
        txt_export = "Export PNG" if i18n is None else i18n.t("chart.export_png")
        txt_reset = "Reset" if i18n is None else i18n.t("common.reset")
        self._export_btn = QPushButton(txt_export)
        self._export_btn.setMinimumWidth(80)
        self._export_btn.clicked.connect(self.export_png)
        self._reset_btn = QPushButton(txt_reset)
        self._reset_btn.setMinimumWidth(50)
        self._reset_btn.clicked.connect(self.reset_view)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        btn_row.addWidget(self._reset_btn)

        layout.addWidget(self._plot)
        layout.addLayout(btn_row)

    def add_point(self, step: int, loss: float, lr: float = 0):
        #添加一个数据点
        self._steps.append(step)
        self._loss_data.append(loss)
        self._lr_data.append(lr)
        self._loss_curve.setData(self._steps, self._loss_data)
        if lr > 0:
            self._lr_curve.setData(self._steps, self._lr_data)
        self._plot.getViewBox().autoRange()

    def clear(self):
        #清空数据
        self._steps.clear()
        self._loss_data.clear()
        self._lr_data.clear()
        self._loss_curve.setData([], [])
        self._lr_curve.setData([], [])

    def update_data(self, values: list):
        #用已有数据批量更新loss曲线
        self._loss_data = list(values)
        self._steps = list(range(1, len(values) + 1))
        self._loss_curve.setData(self._steps, self._loss_data)
        self._plot.getViewBox().autoRange()

    def reset_view(self):
        #重置视图到自动范围
        self._plot.getViewBox().autoRange()

    def export_png(self):
        #导出为PNG图片
        import time

        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", f"loss_{time.strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png)"
        )
        if path:
            self._plot.grab().save(path, "PNG")

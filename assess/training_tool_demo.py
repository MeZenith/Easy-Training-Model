"""
AI训练工具 - 专业级UI Demo
展示如何避免"AI味"，打造真正的工程级界面

设计原则：
1. 信息密度高但层次清晰
2. 数据驱动，一切以指标为核心
3. 暗色主题，终端/IDE 风格
4. 等宽字体用于所有数据展示
5. 无装饰性元素，每个像素都有目的
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QListWidget, QGroupBox,
    QFrame, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QFontDatabase
import random


# ============================================================
#  自定义组件 - 这是去AI味的核心
# ============================================================

class MetricCard(QFrame):
    """指标卡片 - 不是简单的QLabel，而是有结构的数据展示"""
    def __init__(self, label: str, value: str, unit: str = "", trend: str = ""):
        super().__init__()
        self.setObjectName("metricCard")
        self.setFixedHeight(80)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        
        # 标签行
        label_row = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        label_row.addWidget(self.label)
        label_row.addStretch()
        if trend:
            self.trend = QLabel(trend)
            self.trend.setStyleSheet("color: #3fb950; font-size: 11px;")
            label_row.addWidget(self.trend)
        layout.addLayout(label_row)
        
        # 数值行
        value_row = QHBoxLayout()
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        value_row.addWidget(self.value_label)
        if unit:
            self.unit_label = QLabel(unit)
            self.unit_label.setStyleSheet("color: #8b949e; font-size: 12px;")
            self.unit_label.setAlignment(Qt.AlignBottom)
            value_row.addWidget(self.unit_label)
        value_row.addStretch()
        layout.addLayout(value_row)
        
        self.setStyleSheet("""
            #metricCard {
                background-color: #161b22;
                border: 1px solid #21262d;
                border-radius: 8px;
            }
        """)


class LossChart(QWidget):
    """自绘 Loss 曲线 - 不用第三方库，纯 QPainter"""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.data = []
        self.setStyleSheet("background-color: #0d1117; border-radius: 8px;")
        
    def set_data(self, data):
        self.data = data
        self.update()
        
    def append_data(self, value):
        self.data.append(value)
        if len(self.data) > 100:
            self.data.pop(0)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        margin = 20
        
        # 背景网格线
        pen = QPen(QColor("#161b22"))
        pen.setWidth(1)
        painter.setPen(pen)
        for i in range(5):
            y = margin + (h - 2 * margin) * i / 4
            painter.drawLine(margin, int(y), w - margin, int(y))
        
        if len(self.data) < 2:
            painter.setPen(QColor("#484f58"))
            painter.setFont(QFont("JetBrains Mono", 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待训练数据...")
            return
        
        # 绘制曲线
        max_val = max(self.data) if self.data else 1
        min_val = min(self.data) if self.data else 0

"""导航栏图标 — QPainter 自绘，跨平台可靠"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

_SIZE = 18
_PAD = 3


def _make_icon(draw_func):
    pix = QPixmap(_SIZE, _SIZE)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    draw_func(p, _SIZE, _PAD)
    p.end()
    return QIcon(pix)


def _pen(painter, color="#888888", width=1.8):
    painter.setPen(QPen(QColor(color), width))
    painter.setBrush(Qt.NoBrush)


def model_icon():
    def draw(p, s, pad):
        _pen(p)
        m = s / 2
        p.drawLine(QPointF(m, pad), QPointF(s - pad, m))
        p.drawLine(QPointF(s - pad, m), QPointF(m, s - pad))
        p.drawLine(QPointF(m, s - pad), QPointF(pad, m))
        p.drawLine(QPointF(pad, m), QPointF(m, pad))
    return _make_icon(draw)


def data_icon():
    def draw(p, s, pad):
        _pen(p)
        gap = (s - 2 * pad) / 3
        for i in range(4):
            y = pad + i * gap
            p.drawLine(QPointF(pad, y), QPointF(s - pad, y))
            x = pad + i * gap
            p.drawLine(QPointF(x, pad), QPointF(x, s - pad))
    return _make_icon(draw)


def train_icon():
    def draw(p, s, pad):
        _pen(p, color="#3fb950", width=2.0)
        pts = [
            QPointF(pad + 1, pad),
            QPointF(s - pad, s / 2),
            QPointF(pad + 1, s - pad),
        ]
        p.drawPolygon(pts)
    return _make_icon(draw)


def export_icon():
    def draw(p, s, pad):
        _pen(p)
        m = s / 2
        p.drawLine(QPointF(m, pad), QPointF(m, s - pad - 2))
        p.drawLine(QPointF(m, s - pad - 2), QPointF(pad + 1, s - pad - 6))
        p.drawLine(QPointF(m, s - pad - 2), QPointF(s - pad - 1, s - pad - 6))
        p.drawLine(QPointF(pad, s - pad), QPointF(s - pad, s - pad))
    return _make_icon(draw)


def test_icon():
    def draw(p, s, pad):
        _pen(p, color="#8b5cf6", width=2.0)
        r = QRectF(pad, pad, s - 2 * pad, s - 2 * pad)
        p.drawRoundedRect(r, 3, 3)
        tail_size = 4
        p.drawLine(
            QPointF(s - pad - tail_size, s - pad),
            QPointF(s - pad, s - pad - tail_size),
        )
    return _make_icon(draw)


def settings_icon():
    def draw(p, s, pad):
        _pen(p)
        m = s / 2
        r = (s - 2 * pad) / 2 - 1
        center = QPointF(m, m)
        p.drawEllipse(center, r * 0.55, r * 0.55)
        import math
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            outer = QPointF(
                m + (r + 1) * math.cos(angle),
                m + (r + 1) * math.sin(angle),
            )
            inner = QPointF(
                m + (r - 2) * math.cos(angle),
                m + (r - 2) * math.sin(angle),
            )
            p.drawLine(inner, outer)
    return _make_icon(draw)


def logs_icon():
    def draw(p, s, pad):
        _pen(p)
        gap = (s - 2 * pad) / 5
        for i in range(3):
            y = pad + i * gap * 2
            end = s - pad if i < 2 else s - pad - 4
            p.drawLine(QPointF(pad, y), QPointF(end, y))
    return _make_icon(draw)


ICON_MAP = {
    "model": model_icon,
    "data": data_icon,
    "train": train_icon,
    "export": export_icon,
    "test": test_icon,
    "settings": settings_icon,
    "logs": logs_icon,
}

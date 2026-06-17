"""数据编辑表格组件"""

from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QMenu
)
from PySide6.QtCore import Qt, Signal


class DataTable(QTableWidget):
    """训练数据编辑表格"""

    data_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["#", "Instruction", "Input", "Output"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def load_data(self, data: list):
        """加载数据到表格"""
        self.setRowCount(len(data))
        for i, item in enumerate(data):
            self.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.setItem(i, 1, QTableWidgetItem(item.get("instruction", "")))
            self.setItem(i, 2, QTableWidgetItem(item.get("input", "")))
            self.setItem(i, 3, QTableWidgetItem(item.get("output", "")))
            # 序号列不可编辑
            if self.item(i, 0):
                self.item(i, 0).setFlags(self.item(i, 0).flags() & ~Qt.ItemIsEditable)

    def get_data(self) -> list:
        """从表格提取数据"""
        data = []
        for row in range(self.rowCount()):
            inst = self.item(row, 1)
            inp = self.item(row, 2)
            out = self.item(row, 3)
            data.append({
                "instruction": inst.text() if inst else "",
                "input": inp.text() if inp else "",
                "output": out.text() if out else "",
            })
        return data

    def add_row(self, instruction: str = "", input_text: str = "", output: str = ""):
        """添加一行"""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.setItem(row, 1, QTableWidgetItem(instruction))
        self.setItem(row, 2, QTableWidgetItem(input_text))
        self.setItem(row, 3, QTableWidgetItem(output))
        if self.item(row, 0):
            self.item(row, 0).setFlags(self.item(row, 0).flags() & ~Qt.ItemIsEditable)

    def delete_selected(self):
        """删除选中行"""
        rows = set()
        for item in self.selectedItems():
            rows.add(item.row())
        for row in sorted(rows, reverse=True):
            self.removeRow(row)
        self.data_changed.emit()

    def _show_context_menu(self, pos):
        """右键菜单"""
        menu = QMenu(self)
        delete_action = menu.addAction("Delete")
        add_action = menu.addAction("Add Row")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.delete_selected()
        elif action == add_action:
            self.add_row()

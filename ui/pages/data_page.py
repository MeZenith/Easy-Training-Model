import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.data_manager import DataManager
from core.error_handler import friendly_error_message
from ui.components.data_table import DataTable


class CreateDatasetDialog(QDialog):
    #创建数据集的弹窗

    def __init__(self, i18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self.setWindowTitle(i18n.t("data.create"))
        self.setMinimumWidth(400)

        layout = QFormLayout(self)
        self._name_edit = QLineEdit()
        layout.addRow(self._i18n.t("data.dataset_name") + ":", self._name_edit)
        self._desc_edit = QTextEdit()
        self._desc_edit.setMaximumHeight(80)
        layout.addRow(self._i18n.t("data.desc") + ":", self._desc_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @property
    def dataset_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def dataset_desc(self) -> str:
        return self._desc_edit.toPlainText().strip()


class IdentityDialog(QDialog):
    #模型身份数据生成弹窗

    def __init__(self, i18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self.setWindowTitle(i18n.t("train.identity"))
        self.setMinimumWidth(500)

        layout = QFormLayout(self)

        self._name_edit = QLineEdit()
        layout.addRow(i18n.t("train.identity_name") + ":", self._name_edit)
        self._creator_edit = QLineEdit()
        layout.addRow(i18n.t("train.identity_creator") + ":", self._creator_edit)
        self._desc_edit = QTextEdit()
        self._desc_edit.setMaximumHeight(60)
        layout.addRow(i18n.t("train.identity_desc") + ":", self._desc_edit)
        self._specialty_edit = QLineEdit()
        layout.addRow(self._i18n.t("data.specialties") + ":", self._specialty_edit)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(150)
        layout.addRow(self._i18n.t("data.preview") + ":", self._preview)

        gen_btn = QPushButton(self._i18n.t("data.gen_preview"))
        gen_btn.clicked.connect(self._generate_preview)
        layout.addRow(gen_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._generated_data = []

    def _generate_preview(self):
        #生成身份数据的预览
        from core.data_manager import DataManager
        self._generated_data = DataManager.generate_identity_data(
            name=self._name_edit.text().strip() or "AI",
            creator=self._creator_edit.text().strip() or "Unknown",
            description=self._desc_edit.toPlainText().strip(),
            specialties=self._specialty_edit.text().strip(),
        )
        preview_lines = []
        for i, item in enumerate(self._generated_data):
            preview_lines.append(f"[{i+1}] Q: {item['instruction']}")
            preview_lines.append(f"    A: {item['output']}")
            preview_lines.append("")
        self._preview.setPlainText("\n".join(preview_lines))

    @property
    def identity_data(self) -> list:
        return self._generated_data


class DataPage(QWidget):
    #数据管理页

    datasets_changed = Signal()

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._manager = None
        self._current_dataset = None
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        #左侧：数据集列表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        list_label = QLabel()
        list_label.setObjectName("label-secondary")
        left_layout.addWidget(list_label)
        self._list_label = list_label

        self._dataset_list = QListWidget()
        self._dataset_list.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(self._dataset_list, 1)

        btn_row = QHBoxLayout()
        self._create_btn = QPushButton()
        self._import_btn = QPushButton()
        self._identity_btn = QPushButton(self._i18n.t("train.identity"))
        self._identity_btn.setMinimumWidth(60)
        self._identity_btn.setToolTip(self._i18n.t("train.identity"))
        self._delete_btn = QPushButton()
        self._delete_btn.setObjectName("dangerBtn")
        btn_row.addWidget(self._create_btn)
        btn_row.addWidget(self._import_btn)
        btn_row.addWidget(self._identity_btn)
        btn_row.addWidget(self._delete_btn)
        left_layout.addLayout(btn_row)

        self._multi_select_hint = QLabel()
        self._multi_select_hint.setObjectName("label-muted")
        self._multi_select_hint.setStyleSheet("font-size: 11px;")
        left_layout.addWidget(self._multi_select_hint)

        splitter.addWidget(left)

        #右侧：数据表格
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        info_row = QHBoxLayout()
        self._ds_name_label = QLabel()
        self._ds_name_label.setStyleSheet("font-weight: bold;")
        info_row.addWidget(self._ds_name_label)

        self._ds_stats_label = QLabel()
        self._ds_stats_label.setObjectName("label-secondary")
        info_row.addWidget(self._ds_stats_label)
        info_row.addStretch()

        self._validate_btn = QPushButton()
        info_row.addWidget(self._validate_btn)
        right_layout.addLayout(info_row)

        #搜索框
        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText(self._i18n.t("common.search"))
        search_row.addWidget(self._search_edit, 1)
        right_layout.addLayout(search_row)

        #数据表格
        self._data_table = DataTable(parent=self, i18n=self._i18n)
        right_layout.addWidget(self._data_table, 1)

        #表格操作按钮
        table_btn_row = QHBoxLayout()
        self._add_row_btn = QPushButton("+")
        self._add_row_btn.setMaximumWidth(40)
        self._del_row_btn = QPushButton("-")
        self._del_row_btn.setObjectName("dangerBtn")
        self._del_row_btn.setMaximumWidth(40)
        self._save_btn = QPushButton()
        table_btn_row.addWidget(self._add_row_btn)
        table_btn_row.addWidget(self._del_row_btn)
        table_btn_row.addStretch()
        table_btn_row.addWidget(self._save_btn)
        right_layout.addLayout(table_btn_row)

        #验证结果
        self._validate_label = QLabel()
        self._validate_label.setObjectName("label-secondary")
        self._validate_label.setWordWrap(True)
        self._validate_label.setStyleSheet("font-size: 12px;")
        right_layout.addWidget(self._validate_label)

        #空状态提示
        self._empty_label = QLabel()
        self._empty_label.setObjectName("label-muted")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("font-size: 16px; padding: 40px;")
        right_layout.addWidget(self._empty_label)

        splitter.addWidget(right)
        splitter.setSizes([250, 500])

        self._refresh_texts()

    def _connect_signals(self):
        self._create_btn.clicked.connect(self._on_create)
        self._import_btn.clicked.connect(self._on_import)
        self._delete_btn.clicked.connect(self._on_delete)
        self._identity_btn.clicked.connect(self._on_identity)
        self._dataset_list.currentRowChanged.connect(self._on_select_dataset)
        self._add_row_btn.clicked.connect(self._on_add_row)
        self._del_row_btn.clicked.connect(self._on_del_row)
        self._save_btn.clicked.connect(self._on_save)
        self._validate_btn.clicked.connect(self._on_validate)
        self._search_edit.textChanged.connect(self._on_search)

    def _init_manager(self):
        if self._manager is None:
            data_dir = os.path.join(self._config.workspace, "data")
            self._manager = DataManager(data_dir)

    def _refresh_dataset_list(self):
        #刷新左侧数据集列表
        self._init_manager()
        self._dataset_list.clear()
        for name in self._manager.list_names():
            ds = self._manager.get(name)
            item = QListWidgetItem(f"{name} ({ds.count})")
            item.setData(Qt.UserRole, name)
            self._dataset_list.addItem(item)

    def _on_select_dataset(self, row: int):
        #选中某个数据集
        item = self._dataset_list.item(row)
        if not item:
            self._current_dataset = None
            self._data_table.setRowCount(0)
            return

        name = item.data(Qt.UserRole)
        self._init_manager()
        self._current_dataset = self._manager.get(name)
        if self._current_dataset:
            self._data_table.load_data(self._current_dataset.data)
            self._ds_name_label.setText(self._current_dataset.name)
            self._ds_stats_label.setText(
                f"{self._i18n.t('data.count')}: {self._current_dataset.count} | "
                f"{self._i18n.t('data.avg_length')}: {self._current_dataset.avg_length()}"
            )
            self._empty_label.setVisible(False)

    def _on_create(self):
        #创建新数据集
        dlg = CreateDatasetDialog(self._i18n, self)
        if dlg.exec() == QDialog.Accepted and dlg.dataset_name:
            self._init_manager()
            try:
                self._manager.create(dlg.dataset_name, dlg.dataset_desc)
                self._refresh_dataset_list()
            except ValueError as e:
                QMessageBox.warning(self, self._i18n.t("common.warning"), str(e))

    def _on_import(self):
        #从文件导入数据
        file_path, _ = QFileDialog.getOpenFileName(
            self, self._i18n.t("data.import"),
            "", "Data Files (*.jsonl *.json *.csv);;All Files (*)"
        )
        if not file_path:
            return

        self._init_manager()

        name, ok = QInputDialog.getText(
            self, self._i18n.t("data.import"),
            self._i18n.t("data.dataset_name") + ":"
        )
        if not ok or not name.strip():
            return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".jsonl":
                result = self._manager.import_jsonl(file_path, name.strip())
            elif ext == ".json":
                result = self._manager.import_json(file_path, name.strip())
            elif ext == ".csv":
                result = self._manager.import_csv(file_path, name.strip())
            else:
                QMessageBox.warning(self, self._i18n.t("common.error"),
                                    self._i18n.t("data.format_error"))
                return

            msg = f"{self._i18n.t('common.success')}: {result['success']}\n"
            msg += f"{self._i18n.t('common.failed')}: {result['failed']}"
            if result['errors']:
                msg += f"\nErrors: {'; '.join(result['errors'][:5])}"
            QMessageBox.information(self, self._i18n.t("data.import"), msg)
            self._refresh_dataset_list()

        except Exception as e:
            msg = friendly_error_message(e, self._i18n.t)
            QMessageBox.critical(self, self._i18n.t("common.error"), msg)

    def _on_delete(self):
        #删除选中的数据集
        items = self._dataset_list.selectedItems()
        if not items:
            return

        reply = QMessageBox.question(
            self, self._i18n.t("common.confirm"),
            f"{self._i18n.t('common.delete')}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._init_manager()
            for item in items:
                name = item.data(Qt.UserRole)
                self._manager.delete(name)
            self._refresh_dataset_list()
            self._data_table.setRowCount(0)
            self._ds_name_label.setText("")
            self._ds_stats_label.setText("")

    def _on_identity(self):
        #生成身份认知数据
        dlg = IdentityDialog(self._i18n, self)
        if dlg.exec() == QDialog.Accepted and dlg.identity_data:
            self._init_manager()
            name, ok = QInputDialog.getText(
                self, self._i18n.t("train.identity"),
                self._i18n.t("data.dataset_name") + ":", text="identity_data"
            )
            if ok and name.strip():
                try:
                    ds = self._manager.create(name.strip())
                    ds.data = dlg.identity_data
                    ds.save()
                    self._refresh_dataset_list()
                except ValueError:
                    ds = self._manager.get(name.strip())
                    if ds:
                        ds.data.extend(dlg.identity_data)
                        ds.save()
                        self._refresh_dataset_list()

    def _on_add_row(self):
        if self._current_dataset:
            self._data_table.add_row()

    def _on_del_row(self):
        self._data_table.delete_selected()

    def _on_save(self):
        #保存到磁盘
        if not self._current_dataset:
            return
        self._current_dataset.data = self._data_table.get_data()
        self._current_dataset.save()
        self._ds_stats_label.setText(
            f"{self._i18n.t('data.count')}: {self._current_dataset.count} | "
            f"{self._i18n.t('data.avg_length')}: {self._current_dataset.avg_length()}"
        )

    def _on_validate(self):
        #验证当前数据集
        if not self._current_dataset:
            return
        self._current_dataset.data = self._data_table.get_data()
        issues = self._current_dataset.validate(
            max_length=self._config.get("default_train_params.max_seq_length", 2048)
        )
        if not issues:
            self._validate_label.setText(
                f"[OK] {self._i18n.t('train.check_data')}"
            )
            self._validate_label.setStyleSheet("color: #22c55e; font-size: 12px;")
        else:
            lines = [f"[{len(issues)} issues]"]
            for issue in issues[:10]:
                lines.append(
                    f"  Row {issue['index']}: {issue['message']}"
                )
            self._validate_label.setText("\n".join(lines))
            self._validate_label.setStyleSheet("color: #ef4444; font-size: 12px;")

    def _on_search(self, text: str):
        #搜索过滤
        if not self._current_dataset:
            return
        keyword = text.strip().lower()
        if not keyword:
            self._data_table.load_data(self._current_dataset.data)
            return
        filtered = [
            d for d in self._current_dataset.data
            if keyword in d.get("instruction", "").lower()
            or keyword in d.get("input", "").lower()
            or keyword in d.get("output", "").lower()
        ]
        self._data_table.load_data(filtered)

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.data"))
        self._list_label.setText(self._i18n.t("data.multi_select"))
        self._create_btn.setText(self._i18n.t("data.create"))
        self._import_btn.setText(self._i18n.t("data.import"))
        self._delete_btn.setText(self._i18n.t("common.delete"))
        self._save_btn.setText(self._i18n.t("common.save"))
        self._validate_btn.setText(self._i18n.t("train.check_before"))
        self._identity_btn.setText(self._i18n.t("train.identity"))
        self._multi_select_hint.setText(self._i18n.t("data.multi_select"))
        self._search_edit.setPlaceholderText(self._i18n.t("common.search"))

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_dataset_list()

    def get_selected_datasets(self) -> list:
        #获取多选的数据集名字（给训练页用）
        self._init_manager()
        names = []
        for item in self._dataset_list.selectedItems():
            names.append(item.data(Qt.UserRole))
        return names

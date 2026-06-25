import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.error_handler import friendly_error_message
from core.inferencer import Inferencer
from core.services.train_service import list_loras_for_combo

PRESET_QUESTION_KEYS = [
    "test.preset_q1",
    "test.preset_q2",
    "test.preset_q3",
    "test.preset_q4",
]


class TestPage(QWidget):
    #推理测试页

    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._inferencer = None
        self._model_loaded = False
        self._stream_tokens = 0
        self._messages = []
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self._title_label)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        #左侧：对话区
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 8, 0)

        #模型选择行
        model_row = QHBoxLayout()
        self._lora_combo = QComboBox()
        model_row.addWidget(QLabel(self._i18n.t("export.lora_adapter") + ":"), 0)
        model_row.addWidget(self._lora_combo, 1)
        self._load_btn = QPushButton()
        self._load_btn.setObjectName("primaryBtn")
        model_row.addWidget(self._load_btn)
        chat_layout.addLayout(model_row)

        #对话显示
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        chat_layout.addWidget(self._chat_display, 1)

        #输入行
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input_edit, 1)

        self._send_btn = QPushButton()
        self._send_btn.setObjectName("primaryBtn")
        self._send_btn.setMinimumWidth(60)
        input_row.addWidget(self._send_btn)
        chat_layout.addLayout(input_row)

        #预设问题按钮
        preset_row = QHBoxLayout()
        self._preset_row = preset_row
        self._preset_btns = []
        for key in PRESET_QUESTION_KEYS:
            q = self._i18n.t(key)
            btn = QPushButton(q)
            btn.setMinimumWidth(85)
            btn.clicked.connect(lambda checked, k=key: self._send_preset(self._i18n.t(k)))
            preset_row.addWidget(btn)
            self._preset_btns.append(btn)
        chat_layout.addLayout(preset_row)

        #操作按钮
        btn_row = QHBoxLayout()
        self._clear_btn = QPushButton()
        self._save_btn = QPushButton()
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        chat_layout.addLayout(btn_row)

        splitter.addWidget(chat_widget)

        #右侧：参数和性能
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        #生成参数
        params_group = QGroupBox()
        params_form = QFormLayout(params_group)

        self._temp_slider = QSlider(Qt.Horizontal)
        self._temp_slider.setRange(0, 200)
        self._temp_slider.setValue(self._config.get("ui_constants.slider_defaults.temperature", 70))
        self._temp_label = QLabel("0.7")
        temp_row = QHBoxLayout()
        temp_row.addWidget(self._temp_slider, 1)
        temp_row.addWidget(self._temp_label)
        params_form.addRow(self._i18n.t("test.temperature") + ":", temp_row)
        self._temp_desc = QLabel()
        self._temp_desc.setStyleSheet("font-size: 10px; color: #8b949e;")
        params_form.addRow("", self._temp_desc)

        self._topp_slider = QSlider(Qt.Horizontal)
        self._topp_slider.setRange(0, 100)
        self._topp_slider.setValue(self._config.get("ui_constants.slider_defaults.top_p", 90))
        self._topp_label = QLabel("0.90")
        topp_row = QHBoxLayout()
        topp_row.addWidget(self._topp_slider, 1)
        topp_row.addWidget(self._topp_label)
        params_form.addRow(self._i18n.t("test.top_p") + ":", topp_row)

        self._max_tokens_edit = QLineEdit(str(self._config.get("ui_constants.slider_defaults.max_tokens", 1024)))
        self._max_tokens_edit.setMaximumWidth(80)
        params_form.addRow(self._i18n.t("test.max_tokens") + ":", self._max_tokens_edit)

        self._topk_slider = QSlider(Qt.Horizontal)
        self._topk_slider.setRange(1, 200)
        self._topk_slider.setValue(self._config.get("ui_constants.slider_defaults.top_k", 50))
        self._topk_label = QLabel("50")
        topk_row = QHBoxLayout()
        topk_row.addWidget(self._topk_slider, 1)
        topk_row.addWidget(self._topk_label)
        params_form.addRow(self._i18n.t("test.top_k") + ":", topk_row)

        self._rep_penalty_slider = QSlider(Qt.Horizontal)
        self._rep_penalty_slider.setRange(100, 200)
        self._rep_penalty_slider.setValue(self._config.get("ui_constants.slider_defaults.repetition_penalty", 110))
        self._rep_penalty_label = QLabel("1.10")
        rep_row = QHBoxLayout()
        rep_row.addWidget(self._rep_penalty_slider, 1)
        rep_row.addWidget(self._rep_penalty_label)
        params_form.addRow(self._i18n.t("test.rep_penalty") + ":", rep_row)

        self._pres_penalty_slider = QSlider(Qt.Horizontal)
        self._pres_penalty_slider.setRange(0, 100)
        self._pres_penalty_slider.setValue(self._config.get("ui_constants.slider_defaults.presence_penalty", 0))
        self._pres_penalty_label = QLabel("0.00")
        pres_row = QHBoxLayout()
        pres_row.addWidget(self._pres_penalty_slider, 1)
        pres_row.addWidget(self._pres_penalty_label)
        params_form.addRow(self._i18n.t("test.pres_penalty") + ":", pres_row)

        right_layout.addWidget(params_group)
        self._params_group = params_group

        #性能面板
        perf_group = QGroupBox()
        perf_layout = QVBoxLayout(perf_group)
        self._perf_text = QLabel()
        self._perf_text.setObjectName("label-secondary")
        self._perf_text.setWordWrap(True)
        self._perf_text.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; padding: 10px;"
        )
        perf_layout.addWidget(self._perf_text)
        right_layout.addWidget(perf_group)
        self._perf_group = perf_group

        self._perf_history = []

        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 250])

        #滑块事件
        self._temp_slider.valueChanged.connect(self._update_temp_display)
        self._update_temp_display(self._temp_slider.value())
        self._topk_slider.valueChanged.connect(
            lambda v: self._topk_label.setText(str(v)))
        self._rep_penalty_slider.valueChanged.connect(
            lambda v: self._rep_penalty_label.setText(f"{v / 100:.2f}"))
        self._pres_penalty_slider.valueChanged.connect(
            lambda v: self._pres_penalty_label.setText(f"{v / 100:.2f}"))

        self._refresh_texts()

    def _connect_signals(self):
        self._send_btn.clicked.connect(self._on_send)
        self._clear_btn.clicked.connect(self._on_clear)
        self._save_btn.clicked.connect(self._on_save)
        self._load_btn.clicked.connect(self._on_load_model)

    def _load_loras(self):
        #加载LoRA适配器列表
        self._lora_combo.clear()
        loras = list_loras_for_combo(self._config.workspace)
        for item in loras:
            self._lora_combo.addItem(
                item["display"],
                userData={"lora_path": item["lora_path"], "model_path": item["model_path"]}
            )

    def _on_load_model(self):
        #通过子进程加载模型
        data = self._lora_combo.currentData() or {}
        model_path = data.get("model_path", "")
        lora_path = data.get("lora_path", "")

        if not model_path:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("error.no_model"))
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setText(self._i18n.t("test.loading"))
        self._model_loaded = False
        self._messages.clear()

        self._inferencer = Inferencer()
        self._inferencer.loaded.connect(self._on_infer_loaded)
        self._inferencer.progress.connect(
            lambda msg: self._chat_display.append(f"<i>{msg}</i>")
        )
        self._inferencer.token.connect(self._on_infer_token)
        self._inferencer.result.connect(self._on_infer_result)
        self._inferencer.error.connect(self._on_infer_error)
        self._inferencer.start(model_path, lora_path)

    def _on_send(self):
        #发送消息
        text = self._input_edit.text().strip()
        if not text or not self._model_loaded:
            return

        self._send_btn.setEnabled(False)
        self._input_edit.setEnabled(False)
        self._stream_tokens = 0

        self._chat_display.append(
            f"<div style='text-align:right; color:#8b5cf6; margin:8px 0;'>"
            f"<b>{self._i18n.t('test.you')}</b> {text}</div>"
        )
        self._messages.append({"role": "user", "content": text})
        self._input_edit.clear()

        self._chat_display.append(f"<b>{self._i18n.t('test.assistant')}</b> ")

        try:
            max_tokens = int(self._max_tokens_edit.text())
        except ValueError:
            max_tokens = 1024

        params = {
            "temperature": self._temp_slider.value() / 100,
            "top_p": self._topp_slider.value() / 100,
            "top_k": self._topk_slider.value(),
            "max_tokens": max_tokens,
            "repetition_penalty": self._rep_penalty_slider.value() / 100,
            "presence_penalty": self._pres_penalty_slider.value() / 100,
        }

        self._inferencer.generate(self._messages.copy(), params)

    def _update_temp_display(self, v: int):
        #滑块调温度时更新显示
        t = v / 100
        self._temp_label.setText(f"{t:.2f}")
        if t <= 0.2:
            desc = self._i18n.t("test.temp_precise")
        elif t <= 0.5:
            desc = self._i18n.t("test.temp_focused")
        elif t <= 0.8:
            desc = self._i18n.t("test.temp_balanced")
        elif t <= 1.2:
            desc = self._i18n.t("test.temp_creative")
        else:
            desc = self._i18n.t("test.temp_random")
        self._temp_desc.setText(desc)

    def _send_preset(self, text: str):
        self._input_edit.setText(text)
        self._on_send()

    def _on_infer_loaded(self):
        self._model_loaded = True
        self._load_btn.setEnabled(True)
        self._load_btn.setText(self._i18n.t("test.load"))
        self._chat_display.append(
            f"<i>{self._i18n.t('test.model_loaded')}</i>"
        )

    def _on_infer_token(self, text: str):
        text = text.replace("\x00", "")
        if not text:
            return
        safe = html.escape(text, quote=False).replace("\n", "<br>")
        self._chat_display.append(safe)
        self._chat_display.ensureCursorVisible()
        self._stream_tokens += 1
        self._perf_text.setText(self._i18n.t("test.streaming_format").format(self._stream_tokens))

    def _on_infer_result(self, result: dict):
        self._send_btn.setEnabled(True)
        self._input_edit.setEnabled(True)

        text = result.get("text", "")
        error = result.get("error", "")

        if error:
            self._chat_display.append(
                f"<i style='color:red;'>Error: {error}</i>"
            )
        elif not text.strip():
            self._chat_display.append(self._i18n.t("test.no_response"))
        else:
            self._messages.append({"role": "assistant", "content": text})
            safe = html.escape(text, quote=False).replace("\n", "<br>")
            self._chat_display.append(safe)

        #显示性能数据
        perf_lines = []
        perf_lines.append(f"--- {self._i18n.t('test.performance')} ---")
        perf_lines.append(f"{self._i18n.t('test.gen_speed_label')}:       {result.get('gen_speed', 0):.1f} tokens/s")
        perf_lines.append(f"{self._i18n.t('test.first_token_label')}:     {result.get('first_token_latency', 0):.3f} s")
        perf_lines.append(f"{self._i18n.t('test.tokens_total')}:    {result.get('total_tokens', 0)}")
        perf_lines.append(f"  {self._i18n.t('test.tokens_prompt')}:        {result.get('prompt_tokens', 0)}")
        perf_lines.append(f"  {self._i18n.t('test.tokens_completion')}:    {result.get('completion_tokens', 0)}")
        perf_lines.append(f"{self._i18n.t('test.gen_time_label')}:        {result.get('gen_time', 0):.2f} s")
        self._perf_text.setText("\n".join(perf_lines))
        self._perf_history.append(result)

    def _on_infer_error(self, code: str, detail: str):
        self._send_btn.setEnabled(True)
        self._input_edit.setEnabled(True)
        self._load_btn.setEnabled(True)
        self._load_btn.setText(self._i18n.t("test.load"))
        self._model_loaded = False
        msg = friendly_error_message(Exception(detail), self._i18n.t)
        self._chat_display.append(f"<i style='color:red;'>{msg}</i>")

    def _on_clear(self):
        self._messages.clear()
        self._chat_display.clear()
        self._perf_text.clear()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self._i18n.t("test.save_chat"),
            "chat.txt", "Text Files (*.txt)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for msg in self._messages:
                        role = msg.get("role", "").upper()
                        content = msg.get("content", "")
                        f.write(f"[{role}]\n{content}\n\n")
            except OSError as e:
                QMessageBox.critical(self, self._i18n.t("common.error"), str(e))

    def _refresh_texts(self):
        self._title_label.setText(self._i18n.t("nav.test"))
        self._load_btn.setText(self._i18n.t("test.load"))
        self._send_btn.setText(self._i18n.t("test.send"))
        self._clear_btn.setText(self._i18n.t("test.clear"))
        self._save_btn.setText(self._i18n.t("test.save_chat"))
        self._params_group.setTitle(self._i18n.t("test.chat_params"))
        self._perf_group.setTitle(self._i18n.t("test.performance"))
        self._input_edit.setPlaceholderText(self._i18n.t("test.input"))
        for i, btn in enumerate(self._preset_btns):
            if i < len(PRESET_QUESTION_KEYS):
                q = self._i18n.t(PRESET_QUESTION_KEYS[i])
                btn.setText(q)

    def showEvent(self, event):
        super().showEvent(event)
        self._load_loras()

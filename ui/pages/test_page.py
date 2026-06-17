"""Test / Chat page - Full implementation"""

import os
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QSplitter, QSlider, QGroupBox,
    QFormLayout, QFrame, QComboBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QTextCursor

from core.error_handler import friendly_error_message
from core.trainer import ProcessTrainer
from utils.worker import BaseWorker


class LoadModelWorker(BaseWorker):
    """Model loading worker - imports done in main thread"""

    def __init__(self, model_path: str, lora_path: str = "", parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._lora_path = lora_path

    def do_work(self) -> dict:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.signals.progress.emit(10, "Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        self.signals.progress.emit(30, "Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path, trust_remote_code=True, device_map="auto",
            torch_dtype="auto",
        )
        if self._lora_path and os.path.isdir(self._lora_path):
            self.signals.progress.emit(70, "Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, self._lora_path)
            model = model.merge_and_unload()
        self.signals.progress.emit(100, "Model loaded")
        return {"model": model, "tokenizer": tokenizer, "path": self._model_path}


class GenerateWorker(BaseWorker):
    """Text generation worker"""

    def __init__(self, model, tokenizer, messages: list, params: dict, parent=None):
        super().__init__(parent)
        self._model = model
        self._tokenizer = tokenizer
        self._messages = messages
        self._params = params

    def do_work(self) -> dict:
        import time as _time
        import traceback
        import torch
        from transformers import TextIteratorStreamer

        start_time = _time.time()

        try:
            messages = self._messages
            query = messages[-1]["content"] if messages else ""
            prompt = f"### Instruction:\n{query}\n### Input:\n\n### Response:\n"
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": self._params.get("max_tokens", 256),
                    "temperature": self._params.get("temperature", 0.7),
                    "top_p": self._params.get("top_p", 0.9),
                    "top_k": self._params.get("top_k", 50),
                    "do_sample": True,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                    "repetition_penalty": self._params.get("repetition_penalty", 1.1),
                    "no_repeat_ngram_size": 4,
                }

                outputs = self._model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generate_kwargs,
                )

            # Strip input tokens, keep generated content only
            if outputs.shape[1] > prompt_len:
                new_ids = outputs[0][prompt_len:]
                generated_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            else:
                generated_text = ""

            # Cut at template markers to avoid model hallucination
            for stop_marker in ["\n###", "### Instruction", "### Input", "Human:", "# Human"]:
                idx = generated_text.find(stop_marker)
                if idx > 0:
                    generated_text = generated_text[:idx].strip()
                    break

            gen_time = _time.time() - start_time
            completion_tokens = len(new_ids) if outputs.shape[1] > prompt_len else 0
            total_tokens = prompt_len + completion_tokens
            speed = completion_tokens / gen_time if gen_time > 0 else 0

            return {
                "text": generated_text,
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "gen_time": round(gen_time, 2),
                "first_token_latency": 0,
                "gen_speed": round(speed, 1),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"text": "", "error": str(e),
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "gen_time": 0, "first_token_latency": 0, "gen_speed": 0}


PRESET_QUESTION_KEYS = [
    "test.preset_q1",
    "test.preset_q2",
    "test.preset_q3",
    "test.preset_q4",
]


class TestPage(QWidget):
    def __init__(self, config, i18n, parent=None):
        super().__init__(parent)
        self._config = config
        self._i18n = i18n
        self._model = None
        self._tokenizer = None
        self._messages = []
        self._load_worker = None
        self._gen_worker = None
        self._setup_ui()
        self._connect_signals()
        self._i18n.language_changed.connect(self._refresh_texts)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._title_label)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ===== Left: Chat area =====
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 8, 0)

        # Model selection
        model_row = QHBoxLayout()
        self._lora_combo = QComboBox()
        model_row.addWidget(QLabel(self._i18n.t("export.lora_adapter") + ":"), 0)
        model_row.addWidget(self._lora_combo, 1)
        self._load_btn = QPushButton()
        self._load_btn.setObjectName("primaryBtn")
        model_row.addWidget(self._load_btn)
        chat_layout.addLayout(model_row)

        # Chat display
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        chat_layout.addWidget(self._chat_display, 1)

        # Input row
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Type a message...")
        self._input_edit.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input_edit, 1)

        self._send_btn = QPushButton()
        self._send_btn.setObjectName("primaryBtn")
        self._send_btn.setMinimumWidth(60)
        input_row.addWidget(self._send_btn)
        chat_layout.addLayout(input_row)

        # Preset question buttons
        preset_row = QHBoxLayout()
        self._preset_row = preset_row
        self._preset_btns = []
        for key in PRESET_QUESTION_KEYS:
            q = self._i18n.t(key)
            btn = QPushButton(q[:12] + ".." if len(q) > 12 else q)
            btn.setMaximumWidth(110)
            btn.clicked.connect(lambda checked, k=key: self._send_preset(self._i18n.t(k)))
            preset_row.addWidget(btn)
            self._preset_btns.append(btn)
        chat_layout.addLayout(preset_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self._clear_btn = QPushButton()
        self._save_btn = QPushButton()
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        chat_layout.addLayout(btn_row)

        splitter.addWidget(chat_widget)

        # ===== Right: Parameters and performance =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        # Generation parameters
        params_group = QGroupBox()
        params_form = QFormLayout(params_group)

        self._temp_slider = QSlider(Qt.Horizontal)
        self._temp_slider.setRange(0, 200)
        self._temp_slider.setValue(70)
        self._temp_label = QLabel("0.7")
        temp_row = QHBoxLayout()
        temp_row.addWidget(self._temp_slider, 1)
        temp_row.addWidget(self._temp_label)
        params_form.addRow("Temperature:", temp_row)

        self._topp_slider = QSlider(Qt.Horizontal)
        self._topp_slider.setRange(0, 100)
        self._topp_slider.setValue(90)
        self._topp_label = QLabel("0.90")
        topp_row = QHBoxLayout()
        topp_row.addWidget(self._topp_slider, 1)
        topp_row.addWidget(self._topp_label)
        params_form.addRow("Top-p:", topp_row)

        self._max_tokens_edit = QLineEdit("256")
        self._max_tokens_edit.setMaximumWidth(80)
        params_form.addRow("Max Tokens:", self._max_tokens_edit)

        self._topk_slider = QSlider(Qt.Horizontal)
        self._topk_slider.setRange(1, 200)
        self._topk_slider.setValue(50)
        self._topk_label = QLabel("50")
        topk_row = QHBoxLayout()
        topk_row.addWidget(self._topk_slider, 1)
        topk_row.addWidget(self._topk_label)
        params_form.addRow(self._i18n.t("test.top_k") + ":", topk_row)

        self._rep_penalty_slider = QSlider(Qt.Horizontal)
        self._rep_penalty_slider.setRange(100, 200)
        self._rep_penalty_slider.setValue(110)
        self._rep_penalty_label = QLabel("1.10")
        rep_row = QHBoxLayout()
        rep_row.addWidget(self._rep_penalty_slider, 1)
        rep_row.addWidget(self._rep_penalty_label)
        params_form.addRow(self._i18n.t("test.rep_penalty") + ":", rep_row)

        self._pres_penalty_slider = QSlider(Qt.Horizontal)
        self._pres_penalty_slider.setRange(0, 100)
        self._pres_penalty_slider.setValue(0)
        self._pres_penalty_label = QLabel("0.00")
        pres_row = QHBoxLayout()
        pres_row.addWidget(self._pres_penalty_slider, 1)
        pres_row.addWidget(self._pres_penalty_label)
        params_form.addRow(self._i18n.t("test.pres_penalty") + ":", pres_row)

        right_layout.addWidget(params_group)
        self._params_group = params_group

        # Performance panel
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

        # Slider connections
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_label.setText(f"{v / 100:.1f}"))
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
        """Load LoRA adapter list"""
        self._lora_combo.clear()
        trainer = ProcessTrainer(self._config.workspace)
        loras = trainer.list_loras()
        for lora in loras:
            meta = lora.get("metadata", {})
            model_path = meta.get("model_path", "")
            self._lora_combo.addItem(
                f"{lora['name']} ({os.path.basename(model_path)})" if model_path else lora["name"],
                userData={"lora_path": lora["path"], "model_path": model_path}
            )

    def _on_load_model(self):
        """Load model"""
        data = self._lora_combo.currentData() or {}
        model_path = data.get("model_path", "")
        lora_path = data.get("lora_path", "")

        if not model_path:
            QMessageBox.warning(self, self._i18n.t("common.warning"),
                                self._i18n.t("error.no_model"))
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setText(self._i18n.t("test.loading"))

        self._load_worker = LoadModelWorker(model_path, lora_path)
        self._load_worker.signals.progress.connect(
            lambda p, d: self._chat_display.append(f"<i>{d}</i>")
        )
        self._load_worker.signals.finished.connect(self._on_model_loaded)
        self._load_worker.signals.error.connect(self._on_model_load_error)
        self._load_worker.start()

    def _on_model_loaded(self, result: dict):
        """Model loading complete"""
        self._model = result.get("model")
        self._tokenizer = result.get("tokenizer")
        self._load_btn.setEnabled(True)
        self._load_btn.setText(self._i18n.t("test.load"))
        self._chat_display.append(
            f"<i>{self._i18n.t('test.model_loaded')}: {result.get('path', '')}</i>"
        )

    def _on_model_load_error(self, code: str, detail: str):
        self._load_btn.setEnabled(True)
        self._load_btn.setText(self._i18n.t("test.load"))
        msg = self._i18n.t("common.error") + f": {detail}"
        self._chat_display.append(f"<i style='color:red;'>{msg}</i>")

    def _on_send(self):
        """Send message / stop generation"""
        if self._gen_worker and self._gen_worker.isRunning():
            self._gen_worker.cancel()
            self._send_btn.setText(self._i18n.t("test.send"))
            self._chat_display.append("<i style='color:#f59e0b;'>Stopped.</i>")
            return

        text = self._input_edit.text().strip()
        if not text or self._model is None:
            return

        self._send_btn.setEnabled(False)
        self._input_edit.setEnabled(False)

        # Display user message
        self._chat_display.append(
            f"<div style='text-align:right; color:#8b5cf6; margin:8px 0;'>"
            f"<b>You:</b> {text}</div>"
        )
        self._messages.append({"role": "user", "content": text})
        self._input_edit.clear()

        # Display thinking placeholder
        self._chat_display.append(
            "<div id='thinking' style='text-align:left; color:#888; margin:8px 0;'>"
            "<b>Assistant:</b> ...</div>"
        )

        # Build generation params
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

        # Start generation
        self._gen_worker = GenerateWorker(self._model, self._tokenizer,
                                          self._messages.copy(), params)
        self._gen_worker.signals.progress.connect(self._on_gen_progress)
        self._gen_worker.signals.finished.connect(self._on_gen_finished)
        self._gen_worker.signals.error.connect(self._on_gen_error)
        self._gen_worker.start()

    def _send_preset(self, text: str):
        """Send preset question"""
        self._input_edit.setText(text)
        self._on_send()

    def _on_gen_progress(self, percent: int, text: str):
        """Streaming output - append directly to chat"""
        pass  # Not streaming per-token; output at _on_gen_finished

    def _on_gen_finished(self, result: dict):
        """Generation complete"""
        self._send_btn.setEnabled(True)
        self._input_edit.setEnabled(True)

        text = result.get("text", "")
        error = result.get("error", "")

        # Remove placeholder "..."
        html = self._chat_display.toHtml()
        if "...</div>" in html:
            html = html.replace(
                "<b>Assistant:</b> ...</div>", ""
            )
            self._chat_display.setHtml(html)
            self._chat_display.moveCursor(QTextCursor.End)

        if error:
            self._chat_display.append(
                f"<i style='color:red;'>Error: {error}</i>"
            )
        elif not text.strip():
            self._chat_display.append(
                "<div style='text-align:left; color:#888; margin:8px 0;'>"
                "<b>Assistant:</b> (no response)</div>"
            )
        else:
            self._messages.append({"role": "assistant", "content": text})
            self._chat_display.append(
                f"<div style='text-align:left; margin:8px 0;'>"
                f"<b>Assistant:</b> {text}</div>"
            )

        # Display performance data
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

    def _on_gen_error(self, code: str, detail: str):
        self._send_btn.setEnabled(True)
        self._input_edit.setEnabled(True)
        self._chat_display.append(f"<i style='color:red;'>Error: {detail}</i>")

    def _on_clear(self):
        """Clear chat history"""
        self._messages.clear()
        self._chat_display.clear()
        self._perf_text.clear()

    def _on_save(self):
        """Save chat to file"""
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
                btn.setText(q[:14] + ".." if len(q) > 14 else q)

    def showEvent(self, event):
        super().showEvent(event)
        self._load_loras()

"""训练逻辑"""

import os
import sys
import json
import time
import logging
import tempfile
from typing import Optional

from PySide6.QtCore import QProcess, QObject, Signal
from utils.worker import BaseWorker

logger = logging.getLogger("EasyTinking")


class ProcessTrainer(QObject):
    """基于 QProcess 的训练管理器 -- 子进程隔离，避免 CUDA/QThread 崩溃"""

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)
    log_message = Signal(str)
    metric = Signal(dict)

    def __init__(self, workspace: str):
        super().__init__()
        self._workspace = workspace
        self._lora_dir = os.path.join(workspace, "lora")
        os.makedirs(self._lora_dir, exist_ok=True)
        self._process = None
        self._config_path = ""
        self._pending_output = ""

    def start_training(self, config: dict):
        """启动子进程训练"""
        lora_name = config.get("lora_name", "untitled")
        output_dir = os.path.join(self._lora_dir, self._sanitize(lora_name))
        os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        # 保存配置到临时 JSON 文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            self._config_path = f.name

        # 启动子进程
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_process_finished)
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "train_worker.py"
        )
        self._process.start(
            sys.executable,
            [worker_path, "--config", self._config_path]
        )
        logger.info(f"Training process started (PID: {self._process.processId()})")

    def stop_training(self):
        """停止训练"""
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
            logger.info("Training process killed")

    def _process_line(self, line: str):
        """解析一行子进程输出"""
        line = line.strip()
        if not line:
            return

        if line.startswith("PROGRESS:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                self.progress.emit(int(parts[1]), parts[2])

        elif line.startswith("RESULT:"):
            try:
                result = json.loads(line.split(":", 1)[1])
                self.finished.emit(result)
            except json.JSONDecodeError:
                pass

        elif line.startswith("ERROR:"):
            parts = line.split(":", 2)
            code = parts[1] if len(parts) > 1 else "ERR_UNKNOWN"
            detail = parts[2] if len(parts) > 2 else "Unknown error"
            self.error.emit(code, detail)

        elif line.startswith("LOG:"):
            self.log_message.emit(line[4:])

        elif line.startswith("METRIC:"):
            try:
                parts = dict(p.split("=", 1) for p in line[7:].split())
                self.metric.emit({
                    "step": int(parts.get("step", 0)),
                    "loss": float(parts.get("loss", 0)),
                    "lr": float(parts.get("lr", 0)),
                })
            except (ValueError, KeyError):
                pass

        elif line == "DONE":
            pass

    def _on_stdout(self):
        """读取子进程输出"""
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._pending_output += data
        while "\n" in self._pending_output:
            line, self._pending_output = self._pending_output.split("\n", 1)
            self._process_line(line)

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """子进程退出处理"""
        self._on_stdout()
        while self._pending_output.strip():
            self._process_line(self._pending_output.strip())
            self._pending_output = ""

        if exit_status == QProcess.CrashExit:
            self.error.emit("ERR_CRASH", f"Training process crashed (exit code {exit_code})")

        if self._config_path and os.path.isfile(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError:
                pass

    def list_loras(self) -> list:
        """列出已训练的 LoRA"""
        loras = []
        if not os.path.isdir(self._lora_dir):
            return loras
        for entry in os.listdir(self._lora_dir):
            path = os.path.join(self._lora_dir, entry)
            if os.path.isdir(path):
                meta_path = os.path.join(path, "metadata.json")
                info = {"name": entry, "path": path, "metadata": {}}
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            info["metadata"] = json.load(f)
                        mp = info["metadata"].get("model_path", "")
                        if mp:
                            info["metadata"]["model_path"] = os.path.normpath(mp).replace("\\", "/")
                    except Exception:
                        pass
                loras.append(info)
        return loras

    @staticmethod
    def _sanitize(name: str) -> str:
        import re
        name = name.strip()
        return re.sub(r'[^\w\-.]', '_', name) or "untitled"


class TrainWorker(BaseWorker):
    """训练子线程"""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

    def do_work(self) -> dict:
        """执行训练流程"""
        import sys
        print("[TRAIN] Worker started", file=sys.stderr, flush=True)
        logger.info("Training worker started")
        try:
            # 尝试 Unsloth 路径
            has_unsloth = True
            try:
                import unsloth
            except ImportError:
                has_unsloth = False

            if has_unsloth:
                try:
                    return self._train_unsloth()
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Unsloth training failed: {e}, falling back...")

            # transformers + peft 备选
            print("[TRAIN] Using transformers path", file=sys.stderr, flush=True)
            return self._train_transformers()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Training failed: {e}", exc_info=True)
            self.signals.error.emit("ERR_UNKNOWN", str(e))
            return {}

    def _train_unsloth(self) -> dict:
        """使用 Unsloth 训练"""
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset as HFDataset

        cfg = self._config
        model_path = cfg.get("model_path", "")
        data = cfg.get("data", [])
        output_dir = cfg.get("output_dir", "")

        if not model_path or not data:
            raise ValueError("model_path and data are required")

        if self.is_cancelled:
            return {}

        logger.info(f"Loading model with Unsloth from: {model_path}")
        self.signals.progress.emit(5, "Loading model (Unsloth)...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=cfg.get("max_seq_length", 2048),
                dtype=None,
                load_in_4bit=False,
            )
        except Exception as e:
            raise RuntimeError(f"Unsloth model loading failed: {e}")

        # 添加 LoRA
        self.signals.progress.emit(10, "Applying LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.get("lora_rank", 16),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0),
            target_modules=cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            use_gradient_checkpointing="unsloth",
        )

        # 准备数据集
        self.signals.progress.emit(15, "Preparing dataset...")
        def formatting_func(examples):
            texts = []
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                prompt = f"### Instruction:\n{inst}\n"
                if inp.strip():
                    prompt += f"### Input:\n{inp}\n"
                prompt += f"### Response:\n{out}"
                texts.append(prompt)
            return {"text": texts}

        hf_data = {
            "instruction": [d.get("instruction", "") for d in data],
            "input": [d.get("input", "") for d in data],
            "output": [d.get("output", "") for d in data],
        }
        dataset = HFDataset.from_dict(hf_data)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("epochs", 3),
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            learning_rate=cfg.get("learning_rate", 2e-4),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
            warmup_steps=cfg.get("warmup_steps", 5),
            weight_decay=cfg.get("weight_decay", 0.01),
            seed=cfg.get("seed", 3407),
            logging_steps=1,
            save_strategy="epoch",
            optim=cfg.get("optimizer", "adamw_8bit"),
            fp16=not cfg.get("bf16", False),
            bf16=cfg.get("bf16", False),
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=training_args,
        )

        # 训练前记录
        start_time = time.time()

        # 开始训练
        self.signals.progress.emit(20, "Training...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.signals.error.emit("ERR_OOM", str(e))
                return {}
            raise

        elapsed = time.time() - start_time

        # 保存 LoRA
        self.signals.progress.emit(90, "Saving LoRA weights...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 收集结果
        log_history = trainer.state.log_history
        initial_loss = log_history[0].get("loss", 0) if log_history else 0
        final_loss = log_history[-1].get("loss", 0) if log_history else 0

        self.signals.progress.emit(100, "Training complete")

        # 保存元数据
        metadata = {
            "model_path": model_path,
            "config": cfg,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "elapsed_seconds": round(elapsed, 1),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop_pct": round((initial_loss - final_loss) / initial_loss * 100, 1) if initial_loss > 0 else 0,
            "datasets_used": cfg.get("dataset_names", []),
            "total_samples": len(data),
        }
        meta_path = os.path.join(output_dir, "metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        return metadata

    def _train_transformers(self) -> dict:
        """使用 transformers + peft 训练（备选方案）"""
        import sys
        print("[TRAIN] _train_transformers starting", file=sys.stderr, flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import Dataset as HFDataset

        cfg = self._config
        model_path = cfg.get("model_path", "")
        data = cfg.get("data", [])
        output_dir = cfg.get("output_dir", "")

        if not model_path or not os.path.isdir(model_path):
            raise ValueError(f"Model path not found: {model_path}")

        if not data:
            raise ValueError("No training data provided")

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # 1. 加载 tokenizer
        self.signals.progress.emit(2, "Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. 加载模型，带降级策略
        self.signals.progress.emit(3, "Loading model...")
        quant = cfg.get("quantization", "none")
        model = self._load_model_with_fallback(model_path, quant)

        # 3. 应用 LoRA
        self.signals.progress.emit(10, "Applying LoRA...")
        lora_config = LoraConfig(
            r=cfg.get("lora_rank", 16),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0),
            target_modules=cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 4. 准备数据集
        self.signals.progress.emit(12, "Preparing dataset...")
        def formatting_func(examples):
            texts = []
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                prompt = f"### Instruction:\n{inst}\n"
                if inp.strip():
                    prompt += f"### Input:\n{inp}\n"
                prompt += f"### Response:\n{out}"
                texts.append(prompt)
            return {"text": texts}

        hf_data = {
            "instruction": [d.get("instruction", "") for d in data],
            "input": [d.get("input", "") for d in data],
            "output": [d.get("output", "") for d in data],
        }
        dataset = HFDataset.from_dict(hf_data)

        # 启用梯度检查点以节省显存
        model.gradient_checkpointing_enable()

        # 5. 训练参数
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("epochs", 3),
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            learning_rate=cfg.get("learning_rate", 2e-4),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
            warmup_steps=cfg.get("warmup_steps", 5),
            weight_decay=cfg.get("weight_decay", 0.01),
            seed=cfg.get("seed", 3407),
            logging_steps=1,
            save_strategy="epoch",
            optim=cfg.get("optimizer", "adamw_8bit"),
            fp16=not use_bf16,
            bf16=use_bf16,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=training_args,
        )

        start_time = time.time()
        self.signals.progress.emit(20, "Training...")
        trainer.train()
        elapsed = time.time() - start_time

        self.signals.progress.emit(90, "Saving LoRA weights...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        log_history = trainer.state.log_history
        initial_loss = log_history[0].get("loss", 0) if log_history else 0
        final_loss = log_history[-1].get("loss", 0) if log_history else 0

        self.signals.progress.emit(100, "Training complete")

        metadata = {
            "model_path": model_path,
            "config": cfg,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "elapsed_seconds": round(elapsed, 1),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop_pct": round((initial_loss - final_loss) / initial_loss * 100, 1) if initial_loss > 0 else 0,
            "datasets_used": cfg.get("dataset_names", []),
            "total_samples": len(data),
        }
        meta_path = os.path.join(output_dir, "metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        return metadata

    def _load_model_with_fallback(self, model_path: str, quant: str):
        """加载模型：fp16 直接加载，bitsandbytes 仅作可选降级"""
        import sys
        from transformers import AutoModelForCausalLM
        import torch

        print(f"[TRAIN] Loading model from: {model_path}", file=sys.stderr, flush=True)
        print(f"[TRAIN] CUDA: {torch.cuda.is_available()}", file=sys.stderr, flush=True)

        # fp16 永远优先（安全，不依赖 bitsandbytes）
        try:
            logger.info("Loading model with fp16 (no bitsandbytes)...")
            print("[TRAIN] Calling from_pretrained (fp16)...", file=sys.stderr, flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            print("[TRAIN] Moving to GPU...", file=sys.stderr, flush=True)
            model = model.to("cuda")
            logger.info("Model loaded successfully with fp16")
            return model
        except Exception as e:
            logger.warning(f"fp16 loading failed: {e}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 显存不够时尝试 8bit（可能闪退，由用户选择）
        raise RuntimeError(
            f"Failed to load model with fp16. Your GPU may not have enough VRAM.\n"
            f"Error: {e}\n"
            f"Try: reduce sequence length, use a smaller model, or free up GPU memory."
        )


# 向后兼容别名
Trainer = ProcessTrainer

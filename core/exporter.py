"""导出逻辑"""

import os
import json
import shutil
import logging
from utils.worker import BaseWorker

logger = logging.getLogger("EasyTinking")


class ExportWorker(BaseWorker):
    """导出子线程"""

    def __init__(self, lora_path: str, model_path: str, export_dir: str,
                 export_name: str, formats: list, parent=None):
        super().__init__(parent)
        self._lora_path = lora_path
        self._model_path = model_path
        self._export_dir = export_dir
        self._export_name = export_name
        self._formats = formats

    def do_work(self) -> dict:
        results = {"files": [], "errors": []}
        out_dir = os.path.join(self._export_dir, self._export_name)
        os.makedirs(out_dir, exist_ok=True)

        total = len(self._formats)
        for i, fmt in enumerate(self._formats):
            if self.is_cancelled:
                break
            pct = int((i / total) * 100) if total > 0 else 0
            self.signals.progress.emit(pct, f"Exporting {fmt}...")

            try:
                if fmt == "16bit":
                    files = self._export_16bit(out_dir)
                elif fmt.startswith("gguf_"):
                    quant = fmt.replace("gguf_", "")
                    files = self._export_gguf(out_dir, quant)
                elif fmt == "lora_only":
                    files = self._export_lora_only(out_dir)
                else:
                    files = []
                    results["errors"].append(f"Unknown format: {fmt}")
                results["files"].extend(files)
            except Exception as e:
                logger.error(f"Export {fmt} failed: {e}")
                results["errors"].append(f"{fmt}: {e}")

        self.signals.progress.emit(100, "Export complete")
        return results

    def _export_16bit(self, out_dir: str) -> list:
        """导出 16 位完整模型（合并 LoRA 适配器）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.signals.log.emit("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path, trust_remote_code=True, device_map="cpu",
            torch_dtype="auto",
        )

        # 合并 LoRA 适配器
        if self._lora_path and os.path.isdir(self._lora_path):
            self.signals.log.emit("Merging LoRA adapter...")
            try:
                model = PeftModel.from_pretrained(model, self._lora_path)
                model = model.merge_and_unload()
                self.signals.log.emit("LoRA merged into base model")
            except Exception as e:
                self.signals.log.emit(f"LoRA merge failed: {e}, exporting base model only")

        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        model_path = os.path.join(out_dir, "model_16bit")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # 复制 LoRA 元数据到导出目录
        meta_path = os.path.join(self._lora_path or "", "metadata.json")
        if os.path.isfile(meta_path):
            try:
                shutil.copy(meta_path, os.path.join(out_dir, "training_metadata.json"))
            except OSError:
                pass

        return self._list_files(model_path)

    def _export_gguf(self, out_dir: str, quantization: str) -> list:
        """导出 GGUF 格式"""
        from peft import PeftModel
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._model_path,
                load_in_4bit=False,
            )
            if self._lora_path and os.path.isdir(self._lora_path):
                self.signals.log.emit("Merging LoRA for GGUF export...")
                model = PeftModel.from_pretrained(model, self._lora_path)
                model = model.merge_and_unload()

            gguf_path = os.path.join(out_dir, f"model-{quantization}.gguf")
            model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=quantization)
            return [{"name": os.path.basename(gguf_path), "path": gguf_path,
                     "size": os.path.getsize(gguf_path) if os.path.isfile(gguf_path) else 0}]
        except ImportError as e:
            raise ImportError(
                "GGUF export needs llama-cpp-python.\n\n"
                "Install: pip install llama-cpp-python\n"
                "Or use Ollama deploy with 16-bit safetensors instead."
            )
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            raise

    def _export_lora_only(self, out_dir: str) -> list:
        """仅导出 LoRA 适配器"""
        lora_out = os.path.join(out_dir, "lora_adapter")
        if os.path.isdir(self._lora_path):
            shutil.copytree(self._lora_path, lora_out, dirs_exist_ok=True)
            return self._list_files(lora_out)
        return []

    @staticmethod
    def _list_files(directory: str) -> list:
        """列出目录中的文件"""
        files = []
        if not os.path.isdir(directory):
            return files
        for f in os.listdir(directory):
            fpath = os.path.join(directory, f)
            if os.path.isfile(fpath):
                files.append({
                    "name": f,
                    "path": fpath,
                    "size": os.path.getsize(fpath),
                })
        return files


class Exporter:
    """导出管理器"""

    def __init__(self, export_dir: str):
        self._export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def list_exports(self) -> list:
        """列出已导出的模型"""
        exports = []
        if not os.path.isdir(self._export_dir):
            return exports
        for entry in os.listdir(self._export_dir):
            path = os.path.join(self._export_dir, entry)
            if os.path.isdir(path):
                total_size = 0
                file_list = []
                for root, dirs, files in os.walk(path):
                    for f in files:
                        fpath = os.path.join(root, f)
                        try:
                            sz = os.path.getsize(fpath)
                            total_size += sz
                            file_list.append({"name": f, "size": sz, "path": fpath})
                        except OSError:
                            pass
                mtime = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        try:
                            t = os.path.getmtime(os.path.join(root, f))
                            if t > mtime:
                                mtime = t
                        except OSError:
                            pass
                exports.append({
                    "name": entry,
                    "path": path,
                    "size": total_size,
                    "file_count": len(file_list),
                    "export_time": mtime,
                })
        return exports

    def start_export(self, lora_path: str, model_path: str, export_name: str,
                     formats: list, on_progress=None, on_finished=None,
                     on_error=None) -> ExportWorker:
        """启动导出"""
        worker = ExportWorker(lora_path, model_path, self._export_dir,
                              export_name, formats)
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if on_error:
            worker.signals.error.connect(on_error)
        worker.start()
        return worker

"""导出逻辑"""

import logging
import os
import shutil

import torch

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
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            except OSError as e:
                logger.warning(f"Failed to copy training metadata: {e}")

        return self._list_files(model_path)

    def _export_gguf(self, out_dir: str, quantization: str) -> list:
        """导出 GGUF 格式 — 使用 gguf 库完整写入 tokenizer 元数据"""
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.signals.log.emit("Loading base model for GGUF...")
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path, trust_remote_code=True,
            device_map="cpu", torch_dtype=torch.float16,
        )
        config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)

        if self._lora_path and os.path.isdir(self._lora_path):
            self.signals.log.emit("Merging LoRA for GGUF export...")
            try:
                model = PeftModel.from_pretrained(model, self._lora_path)
                model = model.merge_and_unload()
            except Exception as e:
                self.signals.log.emit(f"LoRA merge failed: {e}")

        state_dict = model.state_dict()

        # Write to temp F16 file first
        f16_path = os.path.join(out_dir, "model-F16.gguf")
        self._write_gguf_file(f16_path, config, tokenizer, state_dict)

        if quantization == "F16":
            final_path = os.path.join(out_dir, "model-F16.gguf")
            os.replace(f16_path, final_path)
        else:
            final_path = os.path.join(out_dir, f"model-{quantization}.gguf")
            self._quantize_gguf_file(f16_path, final_path, quantization)
            os.remove(f16_path)

        return [{"name": os.path.basename(final_path), "path": final_path,
                 "size": os.path.getsize(final_path) if os.path.isfile(final_path) else 0}]

    @staticmethod
    def _write_gguf_file(path: str, config, tokenizer, state_dict: dict):
        """将 HF 模型写入完整 GGUF 文件（含 tokenizer 元数据）"""

        from gguf import GGUFWriter, TokenType

        arch = getattr(config, "model_type", "qwen2")
        writer = GGUFWriter(path, arch)

        writer.add_name(getattr(config, "_name_or_path", ""))
        writer.add_context_length(getattr(config, "max_position_embeddings", 2048))
        writer.add_embedding_length(config.hidden_size)
        writer.add_block_count(config.num_hidden_layers)
        writer.add_feed_forward_length(config.intermediate_size)
        writer.add_head_count(config.num_attention_heads)
        num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        writer.add_head_count_kv(num_kv)
        writer.add_rope_freq_base(getattr(config, "rope_theta", 1000000.0))
        writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))
        writer.add_file_type(1)

        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1
        writer.add_vocab_size(vocab_size)

        tokens = [""] * vocab_size
        for tok, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = tok
        writer.add_token_list(tokens)

        writer.add_token_scores([0.0] * vocab_size)

        specials = set()
        for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
            tok = getattr(tokenizer, key, None)
            if tok is not None:
                specials.add(tok)
        token_types = []
        for tok in tokens:
            if tok in specials or tok.startswith("<|"):
                token_types.append(TokenType.CONTROL)
            else:
                token_types.append(TokenType.NORMAL)
        writer.add_token_types(token_types)

        mergeable = getattr(tokenizer, "_mergeable_ranks", None)
        if mergeable:
            merges = [" ".join(p) for p, _ in sorted(mergeable.items(), key=lambda x: x[1])]
            writer.add_token_merges(merges)

        writer.add_tokenizer_model("gpt2")

        for attr, method in [
            ("bos_token_id", writer.add_bos_token_id),
            ("eos_token_id", writer.add_eos_token_id),
            ("pad_token_id", writer.add_pad_token_id),
            ("unk_token_id", writer.add_unk_token_id),
        ]:
            val = getattr(tokenizer, attr, None)
            if val is not None:
                method(val)

        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)

        for name, tensor in state_dict.items():
            writer.add_tensor(name, tensor.detach().cpu().float().numpy())

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @staticmethod
    def _quantize_gguf_file(input_path: str, output_path: str, quant_type: str):
        """GGUF 量化"""
        from gguf import GGUFReader, GGUFWriter
        from gguf.quants import quantize

        reader = GGUFReader(input_path)
        arch = "qwen2"
        for field in reader.fields.values():
            if field.name == "general.architecture":
                arch = field.parts[-1].decode() if isinstance(field.parts[-1], bytes) else str(field.parts[-1])
                break

        writer = GGUFWriter(output_path, arch)
        for field in reader.fields.values():
            for i, val in enumerate(field.parts):
                vtype = field.types[i] if i < len(field.types) else field.types[0]
                writer.add_key_value(field.name, val, vtype)

        qtype = getattr(__import__("gguf.quants", fromlist=[quant_type]), quant_type, None)
        for tensor in reader.tensors:
            if tensor.name.endswith("weight") and len(tensor.data.shape) >= 2 and qtype:
                try:
                    q_data = quantize(tensor.data, qtype)
                    writer.add_tensor(tensor.name, q_data, raw_shape=tensor.shape)
                except Exception:
                    writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

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
                        except OSError as e:
                            logger.warning(f"Failed to get file size for {fpath}: {e}")
                mtime = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        try:
                            t = os.path.getmtime(os.path.join(root, f))
                            if t > mtime:
                                mtime = t
                        except OSError as e:
                            logger.warning(f"Failed to get mtime for {fpath}: {e}")
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

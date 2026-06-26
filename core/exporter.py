import logging
import os
import re
import shutil

import torch

from utils.worker import BaseWorker

logger = logging.getLogger("EasyTinking")


def _hf_to_gguf_name(name: str) -> str:
    #把HuggingFace张量名转成GGUF标准名，比如model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"

    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if m:
        layer = m.group(1)
        rest = m.group(2)
        mapping = {
            "input_layernorm.weight": f"blk.{layer}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{layer}.ffn_norm.weight",
            "self_attn.q_proj.weight": f"blk.{layer}.attn_q.weight",
            "self_attn.q_proj.bias": f"blk.{layer}.attn_q.bias",
            "self_attn.k_proj.weight": f"blk.{layer}.attn_k.weight",
            "self_attn.k_proj.bias": f"blk.{layer}.attn_k.bias",
            "self_attn.v_proj.weight": f"blk.{layer}.attn_v.weight",
            "self_attn.v_proj.bias": f"blk.{layer}.attn_v.bias",
            "mlp.gate_proj.weight": f"blk.{layer}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer}.ffn_down.weight",
            "self_attn.o_proj.weight": f"blk.{layer}.attn_output.weight",
        }
        return mapping.get(rest, name)

    return name


class ExportWorker(BaseWorker):
    #导出工作线程

    def __init__(self, lora_path: str, model_path: str, export_dir: str, export_name: str, formats: list, parent=None):
        super().__init__(parent)
        self._lora_path = lora_path
        self._model_path = model_path
        self._export_dir = export_dir
        self._export_name = export_name
        self._formats = formats

    def do_work(self) -> dict:
        #执行导出
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
                elif fmt == "lora_only":
                    files = self._export_lora_only(out_dir)
                elif fmt.startswith("gguf_"):
                    quant = fmt.replace("gguf_", "")
                    files = self._export_gguf(out_dir, quant)
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
        #导出16位完整模型（合并LoRA）
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.signals.log.emit("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
        )

        #有lora就合并进去
        if self._lora_path and os.path.isdir(self._lora_path):
            self.signals.log.emit("Merging LoRA adapter...")
            try:
                model = PeftModel.from_pretrained(model, self._lora_path)
                model = model.merge_and_unload()
                self.signals.log.emit("LoRA merged into base model")
            except Exception as e:
                self.signals.log.emit(f"LoRA merge failed: {e}, exporting base model only")
        #没有lora就直接导出基模型
        else:
            self.signals.log.emit("No LoRA adapter, exporting base model directly")

        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        model_path = os.path.join(out_dir, "model_16bit")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        #复制训练元数据
        meta_path = os.path.join(self._lora_path or "", "metadata.json")
        if os.path.isfile(meta_path):
            try:
                shutil.copy(meta_path, os.path.join(out_dir, "training_metadata.json"))
            except OSError as e:
                logger.warning(f"Failed to copy training metadata: {e}")

        return self._list_files(model_path)

    def _export_gguf(self, out_dir: str, quantization: str) -> list:
        #导出GGUF格式
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.signals.log.emit("Loading base model for GGUF...")
        use_fp32 = quantization in ("FP32",)
        dtype = torch.float32 if use_fp32 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=dtype,
        )
        config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)

        #合并lora
        if self._lora_path and os.path.isdir(self._lora_path):
            self.signals.log.emit("Merging LoRA for GGUF export...")
            try:
                model = PeftModel.from_pretrained(model, self._lora_path)
                model = model.merge_and_unload()
            except Exception as e:
                self.signals.log.emit(f"LoRA merge failed: {e}")

        #先写出F16的gguf
        f16_path = os.path.join(out_dir, "model-F16.gguf")
        self._write_gguf(f16_path, config, tokenizer, model)
        # logger.debug("GGUF F16 written to %s, size=%d", f16_path, os.path.getsize(f16_path))

        #不需要量化就直接返回
        if quantization in ("F16", "FP32"):
            final_path = f16_path
        else:
            final_path = os.path.join(out_dir, f"model-{quantization}.gguf")
            self.signals.log.emit(f"Quantizing to {quantization}...")
            self._quantize_gguf(f16_path, final_path, quantization)
            os.remove(f16_path)

        return [
            {
                "name": os.path.basename(final_path),
                "path": final_path,
                "size": os.path.getsize(final_path) if os.path.isfile(final_path) else 0,
            }
        ]

    @staticmethod
    def _write_gguf(path: str, config, tokenizer, model) -> None:
        #把模型写成GGUF文件，包含tokenizer元数据
        from gguf import GGUFWriter, TokenType

        arch = getattr(config, "model_type", "qwen2")
        writer = GGUFWriter(path, arch)

        #模型基本信息
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

        writer.add_file_type(0)

        #从tokenizer和嵌入层计算词表大小
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1
        for name, param in model.named_parameters():
            if name == "model.embed_tokens.weight":
                if param.shape[0] > vocab_size:
                    vocab_size = param.shape[0]
                break

        writer.add_vocab_size(vocab_size)

        #构建token列表
        tokens = [""] * vocab_size
        for tok, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = tok
        writer.add_token_list(tokens)
        writer.add_token_scores([0.0] * vocab_size)

        #标记特殊token
        specials = set()
        for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
            tok = getattr(tokenizer, key, None)
            if tok is not None:
                specials.add(tok)
        token_types = []
        for tok in tokens:
            is_control = tok in specials or tok.startswith("<|")
            token_types.append(TokenType.CONTROL if is_control else TokenType.NORMAL)
        writer.add_token_types(token_types)

        #BPE merges — 兼容两种格式
        mergeable = getattr(tokenizer, "_mergeable_ranks", None)
        if mergeable:
            merges = [" ".join(p) for p, _ in sorted(mergeable.items(), key=lambda x: x[1])]
            writer.add_token_merges(merges)
        elif hasattr(tokenizer, "_merges"):
            merges = []
            for m in tokenizer._merges:
                if isinstance(m, (tuple, list)):
                    merges.append(" ".join(str(x) for x in m))
                elif isinstance(m, str):
                    merges.append(m)
            if merges:
                writer.add_token_merges(merges)

        writer.add_tokenizer_model("gpt2")

        #写入特殊token ID（不用循环，直接写——循环写法容易漏掉某个ID）
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None:
            writer.add_bos_token_id(bos_id)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            writer.add_eos_token_id(eos_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            writer.add_pad_token_id(pad_id)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None:
            writer.add_unk_token_id(unk_id)

        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)

        #逐张量写入
        with torch.no_grad():
            for name, param in model.named_parameters():
                gguf_name = _hf_to_gguf_name(name)
                tensor = param.detach().cpu().float().numpy()
                writer.add_tensor(gguf_name, tensor)
                del tensor

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    #量化类型映射（gguf 0.19.0 Q4_K还没实现，先用Q4_0代替）
    _QUANT_ENUM_MAP = {
        "Q8_0": "Q8_0",
        "Q4_K_M": "Q4_0",
    }

    @staticmethod
    def _get_quant_enum(quant_name: str):
        from gguf import GGMLQuantizationType

        enum_name = ExportWorker._QUANT_ENUM_MAP.get(quant_name, quant_name)
        return getattr(GGMLQuantizationType, enum_name, None)

    @staticmethod
    def _quantize_gguf(input_path: str, output_path: str, quant_type: str) -> None:
        #GGUF量化：F16 → Q8_0 / Q4_K_M
        from gguf import GGUFReader, GGUFWriter
        from gguf.quants import quantize

        reader = GGUFReader(input_path)

        #从源文件读架构名
        arch = "qwen2"
        for field in reader.fields.values():
            if field.name == "general.architecture":
                parts_val = field.parts[-1]
                arch = parts_val.decode() if isinstance(parts_val, bytes) else str(parts_val)
                break

        writer = GGUFWriter(output_path, arch)

        from gguf import GGUFValueType

        #复制元数据字段
        for field in reader.fields.values():
            types = list(field.types)
            if any(t in (GGUFValueType.ARRAY,) for t in types):
                continue
            vtype = types[-1]
            val = field.parts[-1]

            if vtype == GGUFValueType.STRING:
                if hasattr(val, "tobytes"):
                    val = bytes(val).decode("utf-8", errors="replace")
                elif isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                else:
                    continue
            elif hasattr(val, "item") and hasattr(val, "size") and val.size == 1:
                val = val.item()
            else:
                continue

            try:
                writer.add_key_value(field.name, val, vtype)
            except Exception:
                pass

        qtype = ExportWorker._get_quant_enum(quant_type)
        if qtype is None:
            logger.warning("Unknown quant type: %s, keeping F16", quant_type)
        #量化每个张量（只量化权重矩阵）
        for tensor in reader.tensors:
            if tensor.name.endswith("weight") and len(tensor.data.shape) >= 2 and qtype:
                try:
                    q_data = quantize(tensor.data, qtype)
                    writer.add_tensor(tensor.name, q_data, raw_dtype=qtype)
                except Exception:
                    logger.warning(f"Quant failed for tensor {tensor.name}, keeping F16")
                    writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def _export_lora_only(self, out_dir: str) -> list:
        #只导出LoRA适配器
        lora_out = os.path.join(out_dir, "lora_adapter")
        if os.path.isdir(self._lora_path):
            shutil.copytree(self._lora_path, lora_out, dirs_exist_ok=True)
            return self._list_files(lora_out)
        return []

    @staticmethod
    def _list_files(directory: str) -> list:
        #列出目录里所有文件
        files = []
        if not os.path.isdir(directory):
            return files
        for f in os.listdir(directory):
            fpath = os.path.join(directory, f)
            if os.path.isfile(fpath):
                files.append(
                    {
                        "name": f,
                        "path": fpath,
                        "size": os.path.getsize(fpath),
                    }
                )
        return files


class Exporter:
    #导出管理器，列出/启动导出任务

    def __init__(self, export_dir: str):
        self._export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def list_exports(self) -> list:
        #列出已导出的模型目录
        exports = []
        if not os.path.isdir(self._export_dir):
            return exports
        for entry in os.listdir(self._export_dir):
            path = os.path.join(self._export_dir, entry)
            if os.path.isdir(path):
                #统计目录大小和修改时间
                total_size = 0
                mtime = 0
                gguf_files = []
                for root, dirs, files in os.walk(path):
                    for f in files:
                        fpath = os.path.join(root, f)
                        try:
                            total_size += os.path.getsize(fpath)
                            t = os.path.getmtime(fpath)
                            if t > mtime:
                                mtime = t
                        except OSError:
                            pass
                        if f.endswith(".gguf"):
                            gguf_files.append(fpath)
                exports.append(
                    {
                        "name": entry,
                        "path": path,
                        "size": total_size,
                        "gguf_files": gguf_files,
                        "export_time": mtime,
                    }
                )
        return exports

    def start_export(
        self,
        lora_path: str,
        model_path: str,
        export_name: str,
        formats: list,
        on_progress=None,
        on_finished=None,
        on_error=None,
    ) -> ExportWorker:
        #启动导出
        worker = ExportWorker(lora_path, model_path, self._export_dir, export_name, formats)
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if on_error:
            worker.signals.error.connect(on_error)
        worker.start()
        return worker

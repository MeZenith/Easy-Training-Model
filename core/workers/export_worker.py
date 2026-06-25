import argparse
import gc
import json
import logging
import os
import re
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    elif hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception as e:
    print(f"LOG:stdout reconfigure failed: {e}", flush=True)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("export_worker")


def _msg(prefix: str, *args):
    print(f"{prefix}:{':'.join(str(a) for a in args)}", flush=True)


def progress(pct: int, desc: str):
    _msg("PROGRESS", pct, desc)


def log(msg: str):
    _msg("LOG", msg)


def result(data: dict):
    _msg("RESULT", json.dumps(data, ensure_ascii=False))


def error(code: str, detail: str):
    _msg("ERROR", code, detail)


def _hf_to_gguf_name(name: str) -> str:
    #HF张量名转GGUF名
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
            "self_attn.o_proj.weight": f"blk.{layer}.attn_output.weight",
            "mlp.gate_proj.weight": f"blk.{layer}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer}.ffn_down.weight",
        }
        return mapping.get(rest, name)
    return name


def _write_gguf(path: str, config, tokenizer, model) -> None:
    #写入GGUF F16文件
    import torch
    from gguf import GGUFWriter, TokenType

    arch = getattr(config, "model_type", "qwen2")
    writer = GGUFWriter(path, arch)

    #模型元数据
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

    #词表
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1
    for name, param in model.named_parameters():
        if name == "model.embed_tokens.weight":
            if param.shape[0] > vocab_size:
                vocab_size = param.shape[0]
            break

    writer.add_vocab_size(vocab_size)
    tokens = [""] * vocab_size
    for tok, idx in vocab.items():
        if 0 <= idx < vocab_size:
            tokens[idx] = tok
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab_size)

    #特殊token标记
    specials = set()
    for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
        tok = getattr(tokenizer, key, None)
        if tok is not None:
            specials.add(tok)
    token_types = []
    for tok in tokens:
        token_types.append(TokenType.CONTROL if tok in specials or tok.startswith("<|") else TokenType.NORMAL)
    writer.add_token_types(token_types)

    #BPE merges
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

    #特殊token ID（不用循环，直接写）
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
    total = sum(1 for _ in model.named_parameters())
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            gguf_name = _hf_to_gguf_name(name)
            tensor = param.detach().cpu().float().numpy()
            writer.add_tensor(gguf_name, tensor)
            del tensor
            if (i + 1) % 100 == 0:
                progress(10 + int(80 * (i + 1) / total), f"Writing tensors {i + 1}/{total}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


_QUANT_ENUM_MAP = {
    "Q8_0": "Q8_0",
    "Q4_K_M": "Q4_0",
}


def _get_quant_enum(quant_name: str):
    from gguf import GGMLQuantizationType
    enum_name = _QUANT_ENUM_MAP.get(quant_name, quant_name)
    return getattr(GGMLQuantizationType, enum_name, None)


def _quantize_gguf(input_path: str, output_path: str, quant_type: str) -> None:
    #GGUF量化
    from gguf import GGUFReader, GGUFValueType, GGUFWriter
    from gguf.quants import quantize

    rdr = GGUFReader(input_path)

    arch = "qwen2"
    for f in rdr.fields.values():
        if f.name == "general.architecture":
            v = f.parts[-1]
            if hasattr(v, "tobytes"):
                arch = bytes(v).decode("utf-8", errors="replace")
            elif isinstance(v, bytes):
                arch = v.decode("utf-8", errors="replace")
            else:
                arch = str(v)
            break

    writer = GGUFWriter(output_path, arch)

    #复制元数据
    for field in rdr.fields.values():
        types = list(field.types)
        if not types:
            continue
        if field.name in ("general.architecture",) or field.name.startswith("GGUF."):
            continue

        if GGUFValueType.ARRAY in types:
            indices = list(field.data)
            if not indices:
                continue
            elem_type = types[-1]
            elements = []
            for idx in indices:
                part = field.parts[idx]
                if elem_type == GGUFValueType.STRING:
                    if hasattr(part, "tobytes"):
                        elements.append(bytes(part).decode("utf-8", errors="replace"))
                    elif isinstance(part, bytes):
                        elements.append(part.decode("utf-8", errors="replace"))
                    else:
                        elements.append(str(part))
                elif hasattr(part, "item"):
                    elements.append(part.item())
                else:
                    elements.append(part)
            try:
                writer.add_array(field.name, elements)
            except Exception as e:
                logger.warning("Failed to copy array field %s: %s", field.name, e)
            continue

        if len(types) != 1:
            continue
        vtype = types[0]
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
        elif not isinstance(val, (int, float, bool)):
            continue

        try:
            writer.add_key_value(field.name, val, vtype)
        except Exception as e:
            logger.warning("Failed to copy metadata field %s: %s", field.name, e)

    #量化张量
    total = len(rdr.tensors)
    qtype = _get_quant_enum(quant_type)

    if qtype is None:
        log(f"Unknown quant type: {quant_type}, keeping F16")

    for i, tensor in enumerate(rdr.tensors):
        data = tensor.data
        tensor_qtype = None
        if tensor.name.endswith("weight") and len(data.shape) >= 2 and qtype:
            try:
                data = quantize(data, qtype)
                tensor_qtype = qtype
            except Exception as e:
                log(f"Quant failed for {tensor.name}: {e}, keeping F16")
        if tensor_qtype is not None:
            writer.add_tensor(tensor.name, data, raw_dtype=tensor_qtype)
        else:
            writer.add_tensor(tensor.name, data, raw_shape=tensor.shape)
        if (i + 1) % 100 == 0:
            progress(10 + int(80 * (i + 1) / total), f"Quantizing {i + 1}/{total}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    #读取配置
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        error("ERR_CONFIG", f"Cannot read config: {e}")
        sys.exit(1)

    model_path = cfg.get("model_path", "")
    lora_path = cfg.get("lora_path", "")
    out_dir = cfg.get("out_dir", "")
    formats = cfg.get("formats", [])
    results = {"files": [], "errors": []}

    if not model_path or not out_dir:
        error("ERR_ARGS", "Missing model_path or out_dir")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    try:
        #加载配置
        progress(0, "Loading model config...")
        log(f"Model path: {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        log(
            f"Architecture: {getattr(config, 'model_type', '?')}, "
            f"Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}"
        )

        #加载tokenizer
        progress(1, "Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

        #加载模型
        progress(2, "Loading model weights...")
        log("This may take a minute...")
        use_fp32 = any(f in formats for f in ("gguf_FP32", "16bit"))
        dtype = torch.float32 if use_fp32 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=dtype,
        )
        log("Model loaded")

        #合并lora
        if lora_path and os.path.isdir(lora_path):
            progress(5, "Merging LoRA adapter...")
            log(f"LoRA path: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            progress(7, "Fusing LoRA weights...")
            model = model.merge_and_unload()
            log("LoRA merged successfully")

        for i, fmt in enumerate(formats):
            base_pct = 10 + int(80 * i / len(formats)) if formats else 10

            try:
                if fmt == "16bit":
                    progress(base_pct, "Exporting 16-bit...")
                    out = os.path.join(out_dir, "model_16bit")
                    model.save_pretrained(out)
                    tokenizer.save_pretrained(out)
                    results["files"].append({"name": "model_16bit", "path": out})

                elif fmt.startswith("gguf_"):
                    quant = fmt.replace("gguf_", "")
                    progress(base_pct, "Writing GGUF F16 head...")
                    log("Writing GGUF F16...")

                    f16_path = os.path.join(out_dir, "model-F16.gguf")
                    try:
                        _write_gguf(f16_path, config, tokenizer, model)
                    except Exception as e:
                        raise RuntimeError(f"Write F16 failed: {e}") from e

                    if quant in ("F16", "FP32"):
                        final_path = f16_path
                    else:
                        progress(base_pct + 40, f"Quantizing to {quant}...")
                        log(f"Quantizing to {quant}...")
                        final_path = os.path.join(out_dir, f"model-{quant}.gguf")
                        try:
                            _quantize_gguf(f16_path, final_path, quant)
                        except Exception as e:
                            raise RuntimeError(f"Quantize {quant} failed: {e}") from e
                        os.remove(f16_path)

                    sz = os.path.getsize(final_path) if os.path.isfile(final_path) else 0
                    results["files"].append({"name": os.path.basename(final_path), "path": final_path, "size": sz})

                elif fmt == "lora_only":
                    import shutil
                    lora_out = os.path.join(out_dir, "lora_adapter")
                    if lora_path and os.path.isdir(lora_path):
                        shutil.copytree(lora_path, lora_out, dirs_exist_ok=True)
                    results["files"].append({"name": "lora_adapter", "path": lora_out})

            except Exception as e:
                logger.exception(f"Export format {fmt} failed")
                results["errors"].append(f"{fmt}: {e}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        progress(100, "Done")
        result(results)
        log("DONE")

    except Exception as e:
        logger.exception("Export process failed")
        err_msg = str(e)
        if getattr(e, "args", None):
            sub = "; ".join(str(a) for a in e.args[1:] if a)
            if sub:
                err_msg = f"{e.args[0]}; {sub}"
        error("ERR_EXPORT", err_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""独立导出进程 — 通过 --config 接收参数，执行 GGUF 转换/量化

协议:
    PROGRESS:<pct>:<desc>  → 进度
    LOG:<message>          → 日志
    RESULT:<json>          → 完成 {"files": [...], "errors": [...]}
    ERROR:<code>:<detail>  → 错误
    DONE                   → 结束
"""

import argparse
import gc
import json
import logging
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
    """写入完整 GGUF F16 文件"""
    import torch
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

    specials = set()
    for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
        tok = getattr(tokenizer, key, None)
        if tok is not None:
            specials.add(tok)
    token_types = []
    for tok in tokens:
        token_types.append(
            TokenType.CONTROL if tok in specials or tok.startswith("<|")
            else TokenType.NORMAL
        )
    writer.add_token_types(token_types)

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


def _quantize_gguf(input_path: str, output_path: str, quant_type: str) -> None:
    """量化 GGUF F16 → Q8_0 / Q4_K_M"""
    from gguf import GGUFReader, GGUFWriter
    from gguf.quants import quantize

    reader = GGUFReader(input_path)
    arch = "qwen2"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = field.parts[-1].decode() if isinstance(field.parts[-1], bytes) else str(field.parts[-1])
            break

    writer = GGUFWriter(output_path, arch)
    skip_keys = {"tokenizer.ggml.tokens", "tokenizer.ggml.scores",
                 "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
    for field in reader.fields.values():
        if field.name in skip_keys:
            continue
        if len(field.parts) == 1:
            try:
                writer.add_key_value(field.name, field.parts[0], field.types[0])
            except Exception:
                pass

    qtype = getattr(__import__("gguf.quants", fromlist=[quant_type]), quant_type, None)
    total = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        if tensor.name.endswith("weight") and len(tensor.data.shape) >= 2 and qtype:
            try:
                q_data = quantize(tensor.data, qtype)
                writer.add_tensor(tensor.name, q_data, raw_shape=tensor.shape)
            except Exception:
                writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)
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
        progress(0, "Loading model config...")
        log(f"Model path: {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        log(f"Architecture: {getattr(config, 'model_type', '?')}, "
            f"Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")

        progress(1, "Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

        progress(2, "Loading model weights...")
        log("This may take a minute for large models...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            device_map="cpu", torch_dtype=torch.float16,
        )
        log("Model loaded")

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
                    progress(base_pct, "Writing GGUF F16...")
                    log("Writing GGUF F16 head...")

                    f16_path = os.path.join(out_dir, "model-F16.gguf")
                    _write_gguf(f16_path, config, tokenizer, model)

                    if quant == "F16":
                        final_path = f16_path
                    else:
                        progress(base_pct + 40, f"Quantizing to {quant}...")
                        log(f"Quantizing to {quant}...")
                        final_path = os.path.join(out_dir, f"model-{quant}.gguf")
                        _quantize_gguf(f16_path, final_path, quant)
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
        error("ERR_EXPORT", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

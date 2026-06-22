"""GGUF 导出独立测试脚本

用法:
    python test_gguf.py <model_path> [--lora <lora_path>] [--out <output_dir>] [--quant Q4_K_M]

示例:
    python test_gguf.py "D:/python/模型训练/qwen_model/Qwen/Qwen2.5-Coder-3B-Instruct"
    python test_gguf.py "D:/python/模型训练/qwen_model/Qwen/Qwen2.5-Coder-3B-Instruct" --lora "workspace/lora/KyLin-Code" --quant Q4_K_M

量化选项: F16 (不量化), Q8_0, Q4_K_M (默认)
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import traceback

# ---- 设置 ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_gguf")

# 确保项目根目录在 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def get_memory_usage() -> str:
    """获取当前进程内存使用量"""
    try:
        import psutil
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        return f"{mem_mb:.0f} MB"
    except ImportError:
        return "N/A (install psutil)"


def step_header(step: int, total: int, title: str):
    logger.info("=" * 60)
    logger.info(f"[{step}/{total}] {title}")
    logger.info(f"Memory: {get_memory_usage()}")


def load_model(model_path: str):
    """加载模型和 tokenizer"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"  Architect: {getattr(config, 'model_type', '?')}")
    logger.info(f"  Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Model loaded. Memory: {get_memory_usage()}")
    return model, config, tokenizer


def merge_lora(model, lora_path: str):
    """合并 LoRA 适配器"""
    from peft import PeftModel

    logger.info(f"Merging LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    logger.info(f"LoRA merged. Memory: {get_memory_usage()}")
    return model


def hf_to_gguf_tensor_name(name: str) -> str:
    """将 HuggingFace 张量名映射为 GGUF 标准名称"""
    import re

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

def convert_to_gguf(model, config, tokenizer, output_path: str):
    """将模型写入 GGUF F16 文件 — 逐张量写入，不缓存 state_dict"""
    import torch
    from gguf import GGUFWriter, TokenType

    logger.info(f"Writing GGUF to: {output_path}")

    arch = getattr(config, "model_type", "qwen2")
    writer = GGUFWriter(output_path, arch)

    # ---- 元数据 ----
    writer.add_name(getattr(config, "_name_or_path", os.path.basename(output_path)))
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

    # ---- Tokenizer ----
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1

    # 从模型嵌入表中获取真实大小，优先使用（Qwen2.5 有额外 padding）
    for name, param in model.named_parameters():
        if name == "model.embed_tokens.weight":
            real_vocab_size = param.shape[0]
            if real_vocab_size > vocab_size:
                vocab_size = real_vocab_size
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
        if tok in specials or tok.startswith("<|"):
            token_types.append(TokenType.CONTROL)
        else:
            token_types.append(TokenType.NORMAL)
    writer.add_token_types(token_types)

    # BPE merges — 支持 _mergeable_ranks (dict) 和 _merges (list of tuples)
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

    # ---- 逐张量写入 ----
    logger.info("Writing tensors (streaming, no state_dict copy)...")
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total params: {total_params:,}")

        tensor_count = 0
        for name, param in model.named_parameters():
            gguf_name = hf_to_gguf_tensor_name(name)
            tensor = param.detach().cpu().float().numpy()

            # GGUF 格式自动反转 numpy 维度 (行优先→列优先)
            # HF token_embd (151936, 2048) → GGUF [2048, 151936] 无需手动转置

            writer.add_tensor(gguf_name, tensor)
            del tensor
            tensor_count += 1
            if tensor_count <= 2:
                logger.info(f"  Tensor {tensor_count}: HF={name} -> GGUF={gguf_name} shape={param.shape}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"GGUF F16 written: {output_path} ({size_mb:.0f} MB)")


def quantize_gguf(input_path: str, output_path: str, quant_type: str):
    """量化 GGUF 文件"""
    from gguf import GGUFReader, GGUFWriter
    from gguf.quants import quantize

    logger.info(f"Quantizing {input_path} -> {quant_type}")

    reader = GGUFReader(input_path)

    # 读取架构
    arch = "qwen2"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = field.parts[-1].decode() if isinstance(field.parts[-1], bytes) else str(field.parts[-1])
            break

    writer = GGUFWriter(output_path, arch)

    # 复制元数据 — 跳过数组类型字段，避免 gguf 0.19 的类型广播错误
    array_keys = {
        "tokenizer.ggml.tokens", "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
    }
    for field in reader.fields.values():
        if field.name in array_keys:
            continue
        if len(field.parts) == 1:
            try:
                writer.add_key_value(field.name, field.parts[0], field.types[0])
            except Exception:
                logger.warning(f"Skip field: {field.name}")

    # 量化张量
    qtype = getattr(__import__("gguf.quants", fromlist=[quant_type]), quant_type, None)
    total = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        if tensor.name.endswith("weight") and len(tensor.data.shape) >= 2 and qtype:
            try:
                q_data = quantize(tensor.data, qtype)
                writer.add_tensor(tensor.name, q_data, raw_shape=tensor.shape)
            except Exception:
                logger.warning(f"  Quant failed for {tensor.name}, keeping F16")
                writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{total} tensors processed")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Quantized GGUF written: {output_path} ({size_mb:.0f} MB)")


def test_ollama(model_path: str, model_name: str):
    """注册到 Ollama 并测试运行"""
    logger.info(f"\nTesting with Ollama as '{model_name}'...")

    # 检测 ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        logger.info(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("Ollama not found! Install from https://ollama.com")
        return False
    except Exception as e:
        logger.error(f"Failed to detect ollama: {e}")
        return False

    # 生成 Modelfile（Go 模板语法: {{ }} 双花括号）
    gguf_path = os.path.abspath(model_path).replace("\\", "/")
    modelfile_content = (
        f'FROM "{gguf_path}"\n'
        'TEMPLATE """{{ if .System }}<|im_start|>system\n'
        '{{ .System }}<|im_end|>\n'
        '{{ end }}{{ if .Prompt }}<|im_start|>user\n'
        '{{ .Prompt }}<|im_end|>\n'
        '{{ end }}<|im_start|>assistant\n'
        '"""\n'
        'PARAMETER stop "<|im_end|>"\n'
    )

    modelfile_path = os.path.join(os.path.dirname(model_path), "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    logger.info(f"Modelfile written: {modelfile_path}")

    # 创建 ollama 模型
    logger.info(f"Creating ollama model '{model_name}'...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error(f"Ollama create failed: {result.stderr}")
        return False
    logger.info(f"Ollama model created: {result.stdout.strip()}")

    # 测试推理
    logger.info(f"Testing inference with '{model_name}'...")
    result = subprocess.run(
        ["ollama", "run", model_name],
        input="你好，你是谁？",
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        logger.info(f"Inference result:\n{result.stdout[:500]}")
        return True
    else:
        logger.error(f"Inference failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GGUF 导出测试脚本")
    parser.add_argument("model_path", help="基础模型路径 (HF 格式)")
    parser.add_argument("--lora", default="", help="LoRA 适配器路径 (可选)")
    parser.add_argument("--out", default="workspace/exports/test_gguf", help="输出目录")
    parser.add_argument("--quant", default="Q4_K_M", choices=["F16", "Q8_0", "Q4_K_M"],
                        help="量化类型 (默认: Q4_K_M)")
    parser.add_argument("--name", default="test-model", help="Ollama 模型名")
    parser.add_argument("--no-ollama", action="store_true", help="跳过 Ollama 测试")
    args = parser.parse_args()

    # 验证路径
    if not os.path.isdir(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        sys.exit(1)
    if args.lora and not os.path.isdir(args.lora):
        logger.error(f"LoRA path not found: {args.lora}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("GGUF Export Test Script")
    logger.info(f"  Model:  {args.model_path}")
    logger.info(f"  LoRA:   {args.lora or '(none)'}")
    logger.info(f"  Output: {args.out}")
    logger.info(f"  Quant:  {args.quant}")
    logger.info(f"  Memory: {get_memory_usage()}")

    total_steps = 3 if args.quant == "F16" else 4
    if not args.no_ollama:
        total_steps += 1

    step = 0

    # ---- Step 1: 加载模型 ----
    step += 1
    step_header(step, total_steps, "Loading model + tokenizer")
    try:
        model, config, tokenizer = load_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Step 2: 合并 LoRA (可选) ----
    if args.lora:
        step += 1
        step_header(step, total_steps, "Merging LoRA adapter")
        try:
            model = merge_lora(model, args.lora)
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ---- Step 3: 写入 GGUF F16 ----
    step += 1
    step_header(step, total_steps, "Writing GGUF F16")
    os.makedirs(args.out, exist_ok=True)
    f16_path = os.path.join(args.out, "model-F16.gguf")
    try:
        convert_to_gguf(model, config, tokenizer, f16_path)
    except Exception as e:
        logger.error(f"Failed to write GGUF: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 释放模型内存
    logger.info("Freeing model from memory...")
    del model
    gc.collect()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"After cleanup. Memory: {get_memory_usage()}")

    # ---- Step 4: 量化 ----
    if args.quant != "F16":
        step += 1
        step_header(step, total_steps, f"Quantizing to {args.quant}")
        final_path = os.path.join(args.out, f"model-{args.quant}.gguf")
        try:
            quantize_gguf(f16_path, final_path, args.quant)
            # 删除中间 F16 文件
            os.remove(f16_path)
            logger.info(f"Removed intermediate F16 file: {f16_path}")
        except Exception as e:
            logger.error(f"Failed to quantize: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        final_path = f16_path

    logger.info(f"\nFinal GGUF file: {final_path}")
    logger.info(f"File size: {os.path.getsize(final_path) / (1024**3):.2f} GB")

    # ---- Step 5: Ollama 测试 ----
    if not args.no_ollama:
        step += 1
        step_header(step, total_steps, "Testing with Ollama")
        try:
            success = test_ollama(final_path, args.name)
            if success:
                logger.info("\n✓ GGUF export + Ollama test PASSED!")
            else:
                logger.warning("\n⚠ GGUF file created, but Ollama test failed")
        except Exception as e:
            logger.error(f"Ollama test error: {e}")
            traceback.print_exc()
    else:
        logger.info("\n✓ GGUF export PASSED! (Ollama test skipped)")


if __name__ == "__main__":
    main()

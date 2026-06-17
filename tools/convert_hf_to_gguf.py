"""HF 模型 → GGUF — 使用 gguf 0.19+ API"""

import sys, os, json
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from gguf import GGUFWriter
from gguf.gguf_writer import GGUFWriter as Writer


def convert(model_dir: str, output_path: str):
    print(f"Model: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = config.model_type  # "qwen2"

    print("Loading weights (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map="cpu",
        torch_dtype=torch.float16,
    )
    state_dict = model.state_dict()
    print(f"  Tensors: {len(state_dict)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # GGUF Writer
    writer = GGUFWriter(output_path, arch)

    # 元数据
    writer.add_context_length(getattr(config, 'max_position_embeddings', 2048))
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    num_kv = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    writer.add_head_count_kv(num_kv)
    writer.add_rope_freq_base(getattr(config, 'rope_theta', 1000000.0))
    writer.add_layer_norm_rms_eps(getattr(config, 'rms_norm_eps', 1e-6))
    writer.add_file_type(1)  # FP16

    # Tokenizer
    vocab_size = len(tokenizer)
    writer.add_vocab_size(vocab_size)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    # writer.add_tokenizer_model(2)  # BPE - gguf 0.19 序列化有 bug

    # Vocab tokens
    vocab = tokenizer.get_vocab()
    tokens = [""] * vocab_size
    for tok, idx in vocab.items():
        if idx < vocab_size:
            tokens[idx] = tok
    writer.add_token_list(tokens)

    # 写入 tensors
    print("Writing tensors...")
    total = len(state_dict)
    for i, (name, tensor) in enumerate(state_dict.items()):
        tensor = tensor.detach().cpu().float().numpy()
        writer.add_tensor(name, tensor)
        if i % 100 == 0:
            print(f"  {i}/{total} {name}")

    # 完成写入
    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_hf_to_gguf.py <model_dir> <output.gguf>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

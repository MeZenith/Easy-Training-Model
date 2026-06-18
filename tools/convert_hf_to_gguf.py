"""HF 模型 → GGUF — 完整转换，包含 BPE tokenizer 元数据

用法: python tools/convert_hf_to_gguf.py <model_dir> <output.gguf>
"""

import sys
import os
import json
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from gguf import GGUFWriter, TokenType


def convert(model_dir: str, output_path: str):
    print(f"Model: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = getattr(config, "model_type", "qwen2")

    print("Loading weights (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map="cpu",
        torch_dtype=torch.float16,
    )
    state_dict = model.state_dict()
    print(f"  Tensors: {len(state_dict)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # ---- GGUF Writer ----
    writer = GGUFWriter(output_path, arch)

    # ---- Model metadata ----
    writer.add_name(os.path.basename(model_dir))
    writer.add_context_length(getattr(config, "max_position_embeddings", 2048))
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    writer.add_head_count_kv(num_kv)
    writer.add_rope_freq_base(getattr(config, "rope_theta", 1000000.0))
    writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))
    writer.add_file_type(1)  # FP16

    # ---- Tokenizer ----
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1
    writer.add_vocab_size(vocab_size)

    # Build token list sorted by token id
    tokens = [""] * vocab_size
    for tok, idx in vocab.items():
        if 0 <= idx < vocab_size:
            tokens[idx] = tok
    writer.add_token_list(tokens)

    # Token scores (all 0.0 for BPE)
    scores = [0.0] * vocab_size
    writer.add_token_scores(scores)

    # Token types: mark special tokens as CONTROL
    special_tokens = set()
    special_keys = {
        "pad_token", "bos_token", "eos_token", "unk_token",
        "sep_token", "mask_token", "cls_token",
    }
    for key in special_keys:
        tok = getattr(tokenizer, key, None)
        if tok is not None:
            special_tokens.add(tok)

    token_types = []
    for tok in tokens:
        if tok in special_tokens:
            token_types.append(TokenType.CONTROL)
        elif tok.startswith("<|"):
            token_types.append(TokenType.CONTROL)
        else:
            token_types.append(TokenType.NORMAL)
    writer.add_token_types(token_types)

    # BPE merges
    merges = []
    mergeable_ranks = getattr(tokenizer, "_mergeable_ranks", None)
    if mergeable_ranks:
        sorted_merges = sorted(mergeable_ranks.items(), key=lambda x: x[1])
        merges = [" ".join(pair) for pair, _ in sorted_merges]
    if not merges:
        bpe_ranks = getattr(tokenizer, "bpe_ranks", None)
        if bpe_ranks:
            sorted_merges = sorted(bpe_ranks.items(), key=lambda x: x[1])
            merges = [" ".join(pair) for pair, _ in sorted_merges]
    if merges:
        writer.add_token_merges(merges)

    writer.add_tokenizer_model("gpt2")

    # Special token IDs
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        writer.add_unk_token_id(tokenizer.unk_token_id)

    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    # ---- Tensors ----
    print("Writing tensors...")
    total = len(state_dict)
    for i, (name, tensor) in enumerate(state_dict.items()):
        tensor = tensor.detach().cpu().float().numpy()
        writer.add_tensor(name, tensor)
        if i % 100 == 0:
            print(f"  {i}/{total} {name}")

    # ---- Finalize ----
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

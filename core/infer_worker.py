"""推理子进程 — 加载模型后常驻，通过 stdin JSON 接收生成请求

协议格式:
    stdout 前缀输出：
        LOG:<message>          → Inferencer.progress Signal
        LOADED:{}              → Inferencer.loaded Signal
        TOKEN:<text>           → Inferencer.token Signal (流式，当前未启用)
        RESULT:<json>          → Inferencer.result Signal {text, tokens, speed, ...}
        ERROR:<code>:<detail>  → Inferencer.error Signal
        DONE                   → 忽略

    stdin JSON 请求:
        {"action": "generate", "messages": [...], "params": {...}}
        {"action": "quit"}

CUDA 操作全部隔离在子进程，避免 Qt 主线程崩溃。
"""

import sys
import os
import json
import time
import argparse

# Force UTF-8 encoding for stdout when piped to QProcess on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def log(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        log(f"ERROR:ERR_CONFIG:Cannot read config: {e}")
        sys.exit(1)

    model_path = os.path.normpath(cfg.get("model_path", ""))
    lora_path = cfg.get("lora_path", "")

    if not model_path or not os.path.isdir(model_path):
        log(f"ERROR:ERR_MODEL:Model path not found: {model_path}")
        sys.exit(1)

    log(f"LOG:Model: {os.path.basename(model_path)}")

    import torch
    log(f"LOG:CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"LOG:GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("LOG:Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"LOG:Tokenizer loaded, vocab={len(tokenizer)}")

    log("LOG:Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    log("LOG:Moving model to GPU...")
    model = model.to("cuda")
    used = torch.cuda.memory_allocated() / 1024**3
    log(f"LOG:Model loaded, VRAM={used:.1f}GB")

    if lora_path and os.path.isdir(lora_path):
        log("LOG:Loading LoRA adapter...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            log("LOG:LoRA merged into base model")
        except Exception as e:
            log(f"LOG:LoRA load skipped: {e}")

    model.eval()
    log("LOADED:{}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log("ERROR:ERR_FORMAT:Invalid JSON")
            continue

        action = req.get("action", "")
        if action == "quit":
            break
        elif action == "generate":
            try:
                messages = req.get("messages", [])
                params = req.get("params", {})

                prompt_parts = []
                for msg in messages:
                    if msg["role"] == "user":
                        prompt_parts.append(
                            f"### Instruction:\n{msg['content']}\n"
                            f"### Input:\n\n### Response:\n"
                        )
                    elif msg["role"] == "assistant":
                        prompt_parts.append(f"{msg['content']}\n")
                prompt = "".join(prompt_parts)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                prompt_len = inputs["input_ids"].shape[1]

                start_time = time.time()

                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=params.get("max_tokens", 256),
                        temperature=params.get("temperature", 0.7),
                        top_p=params.get("top_p", 0.9),
                        top_k=params.get("top_k", 50),
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=params.get("repetition_penalty", 1.1),
                        no_repeat_ngram_size=4,
                    )

                gen_time = time.time() - start_time

                generated_text = ""
                if outputs.shape[1] > prompt_len:
                    new_ids = outputs[0][prompt_len:].cpu().tolist()
                    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    if not generated_text.strip():
                        generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                log(f"LOG:gen={outputs.shape[1] - prompt_len} tok, text_len={len(generated_text)}")

                for stop_marker in ["\n###", "### Instruction", "### Input", "Human:", "# Human"]:
                    idx = generated_text.find(stop_marker)
                    if idx > 0:
                        generated_text = generated_text[:idx].strip()
                        break

                completion_tokens = len(tokenizer.encode(
                    generated_text, add_special_tokens=False
                ))
                total_tokens = prompt_len + completion_tokens
                speed = completion_tokens / gen_time if gen_time > 0 else 0

                result = {
                    "text": generated_text,
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "gen_time": round(gen_time, 2),
                    "gen_speed": round(speed, 1),
                }
                log(f"RESULT:{json.dumps(result, ensure_ascii=False)}")

            except Exception as e:
                import traceback
                log(f"ERROR:ERR_GEN:{e}")
                traceback.print_exc()

    log("DONE")


if __name__ == "__main__":
    main()

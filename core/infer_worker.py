"""推理子进程 — 加载模型 + 常驻等待生成请求，CUDA 操作全部隔离在子进程"""

import sys
import os
import json
import time
import argparse


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
            messages = req.get("messages", [])
            params = req.get("params", {})

            query = messages[-1]["content"] if messages else ""
            prompt = f"### Instruction:\n{query}\n### Input:\n\n### Response:\n"
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

            if outputs.shape[1] > prompt_len:
                new_ids = outputs[0][prompt_len:]
                generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            else:
                generated_text = ""
                new_ids = torch.tensor([], dtype=torch.long)

            for stop_marker in ["\n###", "### Instruction", "### Input", "Human:", "# Human"]:
                idx = generated_text.find(stop_marker)
                if idx > 0:
                    generated_text = generated_text[:idx].strip()
                    break

            completion_tokens = len(new_ids) if new_ids.numel() > 0 else 0
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

    log("DONE")


if __name__ == "__main__":
    main()

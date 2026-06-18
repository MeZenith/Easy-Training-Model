"""独立训练进程 -- 纯 PyTorch 训练循环，不依赖 Trainer/trl/accelerate/datasets"""

import sys
import os
import json
import time
import math
import argparse


def log(msg: str):
    print(msg, flush=True)


def progress(pct: int, desc: str):
    log(f"PROGRESS:{pct}:{desc}")


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

    model_path = cfg.get("model_path", "")
    if model_path:
        model_path = os.path.normpath(model_path).replace("\\", "/")
    data = cfg.get("data", [])
    output_dir = cfg.get("output_dir", "")
    lora_rank = cfg.get("lora_rank", 16)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0)
    target_modules = cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    epochs = cfg.get("epochs", 3)
    batch_size = cfg.get("batch_size", 1)
    grad_accum = cfg.get("grad_accum", 4)
    learning_rate = cfg.get("learning_rate", 2e-4)
    lr_scheduler = cfg.get("lr_scheduler", "cosine")
    warmup_steps = cfg.get("warmup_steps", 5)
    max_seq_length = cfg.get("max_seq_length", 1024)
    weight_decay = cfg.get("weight_decay", 0.01)
    seed = cfg.get("seed", 3407)

    if not model_path or not os.path.isdir(model_path):
        log(f"ERROR:ERR_MODEL:Model path not found: {model_path}")
        sys.exit(1)
    if not data:
        log(f"ERROR:ERR_DATA:No training data")
        sys.exit(1)

    log(f"LOG:Model: {os.path.basename(model_path)}")
    log(f"LOG:Samples: {len(data)}, Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

    # ---- torch ----
    try:
        import torch
        log(f"LOG:CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            torch.cuda.init()
            log(f"LOG:GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        log(f"ERROR:ERR_CUDA:torch init: {e}")
        sys.exit(1)

    # ---- 仅导入 AutoModel + AutoTokenizer（已验证安全）----
    try:
        log("LOG:Importing transformers (core)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        log(f"ERROR:ERR_DEP:transformers core: {e}")
        sys.exit(1)

    # ---- tokenizer ----
    progress(2, "Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"LOG:Tokenizer loaded, vocab={len(tokenizer)}")
    except Exception as e:
        log(f"ERROR:ERR_TOKENIZER:{e}")
        sys.exit(1)

    # ---- model ----
    progress(3, "Loading model (fp16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        log("LOG:Moving model to GPU...")
        model = model.to("cuda")
        used = torch.cuda.memory_allocated() / 1024**3
        log(f"LOG:Model loaded, VRAM={used:.1f}GB")
    except Exception as e:
        log(f"ERROR:ERR_LOAD:{e}")
        sys.exit(1)

    # ---- peft ----
    progress(6, "Importing peft...")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        log(f"ERROR:ERR_PEFT:{e}")
        sys.exit(1)

    progress(8, "Applying LoRA...")
    try:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"LOG:LoRA applied, trainable={trainable:,}, VRAM={torch.cuda.memory_allocated()/1024**3:.1f}GB")
    except Exception as e:
        log(f"ERROR:ERR_LORA:{e}")
        sys.exit(1)

    # ---- 分词数据 ----
    progress(10, "Tokenizing dataset...")
    TASK_TEMPLATE = "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"

    tokenized_items = []
    for item in data:
        text = TASK_TEMPLATE.format(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            output=item.get("output", ""),
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized_items.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    log(f"LOG:Tokenized {len(tokenized_items)} samples")

    # ---- PyTorch Dataset ----
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            item = self._items[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["input_ids"], dtype=torch.long),
            }

    train_dataset = SimpleDataset(tokenized_items)

    # ---- DataLoader ----
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )

    # ---- Optimizer & Scheduler ----
    log("LOG:Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = (len(train_dataset) // (batch_size * grad_accum)) * epochs
    actual_steps = max(total_steps, 1)
    warmup = min(warmup_steps, actual_steps // 2)

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_steps
        )
    elif lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=actual_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # ---- 混合精度 ----
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---- 训练循环 ----
    torch.manual_seed(seed)
    model.train()
    device = next(model.parameters()).device
    log(f"LOG:Device: {device}, dtype: {next(model.parameters()).dtype}")

    start_time = time.time()
    global_step = 0
    initial_loss = None
    all_losses = []

    progress(20, "Training...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda") if use_amp else torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward() if scaler else loss.backward()

            epoch_loss += loss.item()
            epoch_steps += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                if global_step < warmup:
                    lr_scale = (global_step + 1) / max(warmup, 1)
                    for pg in optimizer.param_groups:
                        pg["lr"] = learning_rate * lr_scale
                else:
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0 or global_step == 1:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    all_losses.append(avg_loss)
                    if initial_loss is None:
                        initial_loss = avg_loss
                    pct = min(20 + int((global_step / max(actual_steps, 1)) * 70), 90)
                    progress(pct, f"Epoch {epoch+1}/{epochs} Step {global_step} loss={avg_loss:.4f}")
                    log(f"LOG:Step {global_step} loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
                    log(f"METRIC:loss={avg_loss:.4f} step={global_step} lr={optimizer.param_groups[0]['lr']:.2e}")

        epoch_avg_loss = epoch_loss / max(epoch_steps, 1)
        log(f"LOG:Epoch {epoch+1} done, avg_loss={epoch_avg_loss:.4f}")

    elapsed = time.time() - start_time
    final_loss = epoch_avg_loss
    loss_drop = round((initial_loss - final_loss) / initial_loss * 100, 1) if initial_loss and initial_loss > 0 else 0
    log(f"LOG:Training done in {elapsed:.1f}s, final_loss={final_loss:.4f}")

    # ---- 保存 ----
    progress(90, "Saving LoRA weights...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---- 元数据 ----
    metadata = {
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "elapsed_seconds": round(elapsed, 1),
        "initial_loss": round(initial_loss, 4) if initial_loss else 0,
        "final_loss": round(final_loss, 4),
        "loss_drop_pct": loss_drop,
        "total_samples": len(data),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "output_dir": output_dir,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

    result = {
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "elapsed_seconds": round(elapsed, 1),
        "initial_loss": round(initial_loss, 4) if initial_loss else 0,
        "final_loss": round(final_loss, 4),
        "loss_drop_pct": loss_drop,
        "total_samples": len(data),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "output_dir": output_dir,
    }
    log(f"RESULT:{json.dumps(result, ensure_ascii=False)}")
    progress(100, "Training complete")
    log("DONE")


def _collate_fn(batch):
    """动态 padding：将不等长序列 pad 到 batch 内最长"""
    import torch

    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = 0  # tokenizer pad_token_id, default 0

    padded_input_ids = []
    padded_attention = []
    padded_labels = []

    for item in batch:
        ids = item["input_ids"]
        att = item["attention_mask"]
        lbl = item["labels"]

        pad_size = max_len - ids.size(0)
        if pad_size > 0:
            ids = torch.cat([ids, torch.full((pad_size,), pad_id, dtype=torch.long)])
            att = torch.cat([att, torch.zeros(pad_size, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_size,), pad_id, dtype=torch.long)])

        padded_input_ids.append(ids)
        padded_attention.append(att)
        padded_labels.append(lbl)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention),
        "labels": torch.stack(padded_labels),
    }


if __name__ == "__main__":
    main()

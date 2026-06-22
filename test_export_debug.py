"""Debug: test export_worker.py directly"""
import json
import os
import subprocess
import sys

cfg = {
    "model_path": r"D:/python/模型训练/qwen_model/Qwen/Qwen2.5-Coder-3B-Instruct",
    "lora_path": "workspace/lora/KyLin-Code",
    "out_dir": "workspace/exports/test_debug",
    "formats": ["gguf_Q8_0"],
}

with open("workspace/test_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False)

os.environ["PYTHONUNBUFFERED"] = "1"
result = subprocess.run(
    [sys.executable, "core/workers/export_worker.py", "--config", "workspace/test_config.json"],
    capture_output=True, text=True, encoding="utf-8", timeout=600,
)
print("=== STDOUT ===")
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
print(f"=== Return code: {result.returncode} ===")

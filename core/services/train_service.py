"""训练服务 — LoRA 列表查询、配置组装等纯业务逻辑"""

import os

from core.trainer import ProcessTrainer


def list_loras_for_combo(workspace: str) -> list:
    """列出已训练 LoRA 适配器，返回适合 QComboBox 填充的列表

    Returns:
        [{"display": "lora_name -> base_model", "lora_path": ..., "model_path": ...}]
    """
    trainer = ProcessTrainer(workspace)
    items = []
    for lora in trainer.list_loras():
        meta = lora.get("metadata", {})
        model_path = meta.get("model_path", "")
        if model_path:
            display = f"{lora['name']} -> {os.path.basename(model_path)}"
        else:
            display = lora["name"]
        items.append({
            "display": display,
            "lora_path": lora["path"],
            "model_path": model_path,
        })
    return items

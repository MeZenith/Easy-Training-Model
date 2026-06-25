import os

from core.trainer import ProcessTrainer


def list_loras_for_combo(workspace: str) -> list:
    #列出已训练LoRA，给下拉框用
    #返回: [{display: "lora名 -> 模型名", lora_path, model_path}]
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

import os


def find_gguf(directory: str) -> str:
    #递归找GGUF文件，返回第一个找到的路径
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".gguf"):
                return os.path.join(root, f)
    return ""


def find_safetensors_dir(directory: str) -> str:
    #找包含config.json和safetensors的HF格式目录
    if os.path.isfile(os.path.join(directory, "config.json")):
        for f in os.listdir(directory):
            if f.endswith(".safetensors"):
                return directory
    for entry in os.listdir(directory):
        sub = os.path.join(directory, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
            for f in os.listdir(sub):
                if f.endswith(".safetensors"):
                    return sub
    return ""


def detect_format(path: str) -> str:
    #检测导出目录的模型格式
    if os.path.isfile(os.path.join(path, "model-F16.gguf")):
        return "GGUF F16"
    if os.path.isfile(os.path.join(path, "model-Q4_K_M.gguf")):
        return "GGUF Q4"
    if os.path.isfile(os.path.join(path, "model-Q8_0.gguf")):
        return "GGUF Q8"
    if os.path.isdir(os.path.join(path, "model_16bit")):
        return "16-bit"
    if os.path.isdir(os.path.join(path, "lora_adapter")):
        return "LoRA"
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".gguf"):
                return "GGUF"
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".safetensors"):
                return "SafeTensors"
    return "-"

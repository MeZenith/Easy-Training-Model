import logging
import os
import subprocess

logger = logging.getLogger("EasyTinking")


def get_gpu_info() -> list:
    #用nvidia-smi获取GPU信息，返回 [{index, name, vram_total_mb, vram_used_mb, vram_free_mb, temperature_c, driver_version}]
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        if result.returncode != 0:
            return gpus
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "vram_free_mb": int(float(parts[4])),
                    "temperature_c": int(float(parts[5])),
                    "driver_version": parts[6],
                })
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        logger.warning("Failed to query GPU info via nvidia-smi")
        pass
    return gpus


def get_cuda_version() -> str:
    #查CUDA版本
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.warning("Failed to query CUDA version via nvidia-smi")
        pass
    return ""

import logging
import os
import platform

logger = logging.getLogger("EasyTraining")


def get_system_info() -> dict:
    #获取系统信息
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["memory_available_gb"] = round(mem.available / (1024 ** 3), 1)
        info["memory_percent"] = mem.percent
    except ImportError:
        pass
    try:
        disk = os.statvfs if hasattr(os, "statvfs") else None
        if disk:
            stat = disk(os.getcwd())
            info["disk_total_gb"] = round(stat.f_blocks * stat.f_frsize / (1024 ** 3), 1)
            info["disk_free_gb"] = round(stat.f_bavail * stat.f_frsize / (1024 ** 3), 1)
        elif os.name == "nt":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                os.getcwd(), None, None, ctypes.pointer(free_bytes)
            )
            info["disk_free_gb"] = round(free_bytes.value / (1024 ** 3), 1)
    except Exception:
        logger.warning("Failed to get disk free space")
        pass
    return info


def get_dependency_versions() -> dict:
    #获取关键依赖的版本
    deps = {}
    for name in ["torch", "transformers", "unsloth", "trl", "peft",
                 "huggingface_hub", "accelerate", "bitsandbytes",
                 "safetensors", "llama_cpp"]:
        try:
            mod = __import__(name)
            deps[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[name] = None
    return deps

"""统一错误分类与格式化模块 — 纯逻辑，不依赖 UI"""

import logging
from functools import wraps

logger = logging.getLogger("EasyTinking")

# 错误码到 i18n key 的映射
_ERROR_MAP = {
    "ERR_OOM": "error.oom",
    "ERR_MODEL_NOT_FOUND": "error.no_model",
    "ERR_DATA_FORMAT": "error.format_error",
    "ERR_NETWORK": "error.network",
    "ERR_DISK_FULL": "error.disk_full",
    "ERR_DEP_MISSING": "error.dep_missing",
    "ERR_PERMISSION": "error.permission",
    "ERR_NO_CUDA": "error.cuda",
    "ERR_GGUF_CONVERT": "error.gguf_convert",
    "ERR_OLLAMA": "error.unknown",
    "ERR_NO_DATA": "error.no_data",
    "ERR_UNKNOWN": "error.unknown",
}

# 异常类型到错误码的映射
_EXCEPTION_MAP = {
    "RuntimeError": [
        ("CUDA out of memory", "ERR_OOM"),
    ],
    "FileNotFoundError": [
        ("model", "ERR_MODEL_NOT_FOUND"),
    ],
    "json.JSONDecodeError": [
        ("", "ERR_DATA_FORMAT"),
    ],
    "ConnectionError": [
        ("", "ERR_NETWORK"),
    ],
    "OSError": [
        ("No space left", "ERR_DISK_FULL"),
    ],
    "ImportError": [
        ("", "ERR_DEP_MISSING"),
    ],
    "PermissionError": [
        ("", "ERR_PERMISSION"),
    ],
}


def classify_error(exc: Exception) -> tuple:
    """将异常分类为 (error_code, i18n_key)

    返回: (error_code, i18n_key, detail_str)
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    for mapped_type, patterns in _EXCEPTION_MAP.items():
        if exc_type == mapped_type or exc_type == mapped_type.split(".")[-1]:
            for substring, code in patterns:
                if not substring or substring.lower() in exc_msg.lower():
                    i18n_key = _ERROR_MAP.get(code, "error.unknown")
                    return code, i18n_key, exc_msg

    return "ERR_UNKNOWN", "error.unknown", exc_msg


def friendly_error_message(exc: Exception, i18n_func=None) -> str:
    """生成友好错误提示

    Args:
        exc: 异常对象
        i18n_func: i18n.t 函数，如果提供则使用国际化文本
    """
    code, i18n_key, detail = classify_error(exc)

    if i18n_func:
        msg = i18n_func(i18n_key)
    else:
        msg = i18n_key

    logger.error(f"[{code}] {detail}")
    return msg


def safe_call(func, i18n_func=None, error_title="Error"):
    """安全调用装饰器 — 异常时记录日志并返回 None，由 UI 层决定如何展示

    Args:
        func: 要包装的函数
        i18n_func: 可选 i18n.t 函数，用于生成多语言错误消息
        error_title: 错误标题

    Returns:
        装饰后的函数，异常时返回 None
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = friendly_error_message(e, i18n_func)
            logger.exception(f"Unhandled error in {func.__name__}: {msg}")
            return None
    return wrapper

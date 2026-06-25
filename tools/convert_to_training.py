#对话数据转训练数据的工具
import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from core.data_manager import REQUIRED_FIELDS
except ImportError:
    REQUIRED_FIELDS = ["instruction", "output"]

SKIP_SYSTEM_CONTENT = [
    "你是一个专业的",
    "你是一个",
    "You are a",
    "system",
]


def _is_system_like(content: str) -> bool:
    c = content.strip().lower()
    return any(c.startswith(p.lower()) for p in SKIP_SYSTEM_CONTENT)


def _detect_format(data) -> str:
    #自动检测输入格式
    if not isinstance(data, list) or not data:
        return "unknown"
    first = data[0]
    if isinstance(first, dict):
        if "messages" in first:
            return "messages"
        if "choices" in first and isinstance(first.get("choices"), list):
            return "api_response"
        if "instruction" in first and "output" in first:
            return "instruction_output"
        if "conversations" in first:
            return "messages"
    return "unknown"


def _convert_messages(item: dict, keep_reasoning: bool = False) -> list[dict]:
    #messages格式 → instruction/output
    messages = item.get("messages", item.get("conversations", []))
    if not messages:
        return []

    results = []
    pending_user = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        if not isinstance(content, str) or not content.strip():
            continue

        if role == "user":
            pending_user = content.strip()
        elif role == "assistant":
            if pending_user:
                final_content = content.strip()
                if keep_reasoning and reasoning and isinstance(reasoning, str) and reasoning.strip():
                    final_content = f"【思考过程】\n{reasoning.strip()}\n\n【回答】\n{final_content}"
                results.append(
                    {
                        "instruction": pending_user,
                        "input": "",
                        "output": final_content,
                    }
                )
                pending_user = None

    return results


def _convert_api_response(item: dict, keep_reasoning: bool = False) -> list[dict]:
    #API响应格式 → instruction/output
    results = []
    choices = item.get("choices", [])
    if not choices:
        return results

    message = choices[0].get("message", {})
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")

    if not isinstance(content, str) or not content.strip():
        return results

    final_content = content.strip()
    if keep_reasoning and reasoning and isinstance(reasoning, str) and reasoning.strip():
        final_content = f"【思考过程】\n{reasoning.strip()}\n\n【回答】\n{final_content}"

    #从消息上下文恢复用户问题
    instruction = ""
    orig_messages = item.get("messages", [])
    for msg in reversed(orig_messages):
        if msg.get("role") == "user":
            instruction = msg.get("content", "").strip()
            break

    if not instruction:
        instruction = item.get("prompt", item.get("question", item.get("instruction", "")))

    results.append(
        {
            "instruction": str(instruction) if instruction else "(未记录的用户问题)",
            "input": "",
            "output": final_content,
        }
    )
    return results


def _convert_instruction_output(item: dict) -> list[dict]:
    #已转换格式直接透传
    item = dict(item)
    item.setdefault("input", "")
    return [item]


def convert_data(input_data: list, fmt: str = "auto", keep_reasoning: bool = False) -> tuple[list, int]:
    if fmt == "auto":
        fmt = _detect_format(input_data)

    converters = {
        "messages": _convert_messages,
        "api_response": _convert_api_response,
        "instruction_output": _convert_instruction_output,
    }
    converter = converters.get(fmt)
    if converter is None:
        raise ValueError(
            f"无法识别的格式 '{fmt}'。检测到的格式: {_detect_format(input_data)}\n"
            "支持: messages, api_response, instruction_output"
        )

    results = []
    skipped = 0
    for item in input_data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        if fmt == "api_response":
            converted = converter(item, keep_reasoning)
        elif fmt == "messages":
            converted = converter(item, keep_reasoning)
        else:
            converted = converter(item)
        if not converted:
            skipped += 1
        results.extend(converted)

    return results, skipped


def _resolve_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        return output_path

    in_dir = os.path.dirname(os.path.abspath(input_path))
    base = os.path.splitext(os.path.basename(input_path))[0]

    idx = 1
    while True:
        name = f"{base}_new_{idx}.json"
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            return path
        idx += 1


def _validate_items(items: list[dict]) -> tuple[list[dict], int]:
    valid = []
    bad = 0
    for item in items:
        if all(k in item and item[k] for k in REQUIRED_FIELDS):
            item.setdefault("input", "")
            valid.append(item)
        else:
            bad += 1
    return valid, bad


def main():
    parser = argparse.ArgumentParser(
        description="对话数据 → Easy Training 训练数据 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/convert_to_training.py chat_data.json
  python tools/convert_to_training.py api_responses.json --format api_response --output my_data.json
  python tools/convert_to_training.py raw.json --keep-reasoning --output train.jsonl

输出格式: JSON 数组 [{instruction,input,output}, ...]，可直接导入 Easy Training。
        """,
    )
    parser.add_argument("input", help="输入文件路径 (JSON)")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径")
    parser.add_argument(
        "--format", "-f",
        choices=["auto", "messages", "api_response", "instruction_output"],
        default="auto", help="输入数据格式 (默认: auto 自动检测)",
    )
    parser.add_argument("--keep-reasoning", action="store_true", help="保留 reasoning_content")
    parser.add_argument("--jsonl", action="store_true", help="输出JSONL格式")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误: 找不到文件 {args.input}")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8-sig") as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}")
        sys.exit(1)

    if not isinstance(input_data, list):
        input_data = [input_data]

    detected = _detect_format(input_data)
    if args.format == "auto":
        print(f"检测到格式: {detected}")
    else:
        print(f"指定格式: {args.format} (检测到: {detected})")

    fmt = detected if args.format == "auto" else args.format

    try:
        converted, skipped = convert_data(input_data, fmt, args.keep_reasoning)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    if not converted:
        print("错误: 未能从输入中提取任何有效训练数据")
        if skipped:
            print(f"  跳过了 {skipped} 条无效数据")
        sys.exit(1)

    valid, bad = _validate_items(converted)

    output_path = _resolve_output_path(args.input, args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if args.jsonl:
                for item in valid:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(valid, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"错误: 写入文件失败 - {e}")
        sys.exit(1)

    print("\n转换完成!")
    print(f"  输入记录: {len(input_data)}")
    print(f"  生成样本: {len(valid)}")
    if bad:
        print(f"  校验失败: {bad}")
    if skipped:
        print(f"  跳过无效: {skipped}")
    print(f"  输出文件: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path):,} 字节")


if __name__ == "__main__":
    main()

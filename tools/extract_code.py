#从API返回中提取代码块的工具
import json
import os
import re
import sys

if len(sys.argv) < 2:
    print("用法: python tools/extract_code.py <api_response.json> [--output <path>]")
    print("  从 DeepSeek/GLM API 返回的 JSON 中提取代码块，保存为文件")
    print("  支持: --lang html|python|js 指定要提取的代码语言（默认 html）")
    print("        --output 指定输出路径（默认自动生成）")
    sys.exit(1)

input_path = sys.argv[1]
lang = "html"
output_path = ""

i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
        lang = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
        output_path = sys.argv[i + 1]
        i += 2
    else:
        i += 1

#读取数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    items = data
else:
    items = [data]

for idx, item in enumerate(items):
    #从 choices[0].message.content 提取
    content = ""
    if "choices" in item:
        choices = item["choices"]
        if choices:
            content = choices[0].get("message", {}).get("content", "")
    elif "messages" in item:
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

    if not content:
        print(f"[{idx}] 无内容，跳过")
        continue

    #提取代码块
    pattern = rf"```{lang}\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        #不指定语言再试试
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches and lang == "html" and ("<html" in matches[0].lower() or "<!doctype" in matches[0].lower()):
            pass
        else:
            print(f"[{idx}] 未找到 {lang} 代码块")
            continue

    code = matches[0].strip()

    #确定输出路径
    if output_path:
        out = output_path
    else:
        in_dir = os.path.dirname(os.path.abspath(input_path))
        ext_map = {"html": "html", "python": "py", "javascript": "js", "js": "js"}
        ext = ext_map.get(lang, "txt")
        model = item.get("model", "unknown").replace("/", "_")
        idx_str = f"_{idx}" if len(items) > 1 else ""
        out = os.path.join(in_dir, f"extracted_{model}{idx_str}.{ext}")
        count = 1
        while os.path.exists(out):
            out = os.path.join(in_dir, f"extracted_{model}{idx_str}_{count}.{ext}")
            count += 1

    with open(out, "w", encoding="utf-8") as f:
        f.write(code)

    size = os.path.getsize(out)
    print(f"[{idx}] 已提取 {len(code):,} 字符 → {out} ({size:,} 字节)")

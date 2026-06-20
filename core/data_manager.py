"""数据集管理/验证/导入导出"""

import os
import json
import csv
import copy
import time
import logging
from typing import Optional

logger = logging.getLogger("EasyTinking")

# 单条数据的必需字段
REQUIRED_FIELDS = ["instruction", "output"]


class Dataset:
    """单个训练数据集 — 内存中的指令-响应数据集合

    Attributes:
        name: 数据集名称
        path: 磁盘 JSONL 文件路径
        data: list[dict] 每个元素含 instruction/input/output 字段
        description: 数据集描述文本
        count: 数据条数 (由 data 长度计算)
    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.data = []
        self.description = ""
        self.created_at = ""
        self._load()

    def _load(self):
        """从磁盘加载数据"""
        meta_path = os.path.join(self.path, "meta.json")
        data_path = os.path.join(self.path, "data.jsonl")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.description = meta.get("description", "")
                self.created_at = meta.get("created_at", "")
            except Exception:
                pass

        self.data = []
        if os.path.isfile(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.data.append(json.loads(line))
            except Exception:
                logger.error(f"Failed to load dataset: {self.name}")

    def save(self):
        """保存数据集到磁盘"""
        os.makedirs(self.path, exist_ok=True)
        meta_path = os.path.join(self.path, "meta.json")
        data_path = os.path.join(self.path, "data.jsonl")

        meta = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M"),
            "count": len(self.data),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Save meta failed: {e}")

        try:
            with open(data_path, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Save data failed: {e}")

    @property
    def count(self) -> int:
        return len(self.data)

    def avg_length(self) -> int:
        """计算平均字符长度"""
        if not self.data:
            return 0
        total = sum(len(str(d.get("instruction", "")) + str(d.get("input", "")) + str(d.get("output", "")))
                     for d in self.data)
        return total // len(self.data) if self.data else 0

    def validate(self, max_length: int = 0) -> list:
        """验证数据集，返回问题列表

        每个问题为: {index, field, message}
        """
        issues = []
        seen = set()
        for i, item in enumerate(self.data):
            for field in REQUIRED_FIELDS:
                if not item.get(field, "").strip():
                    issues.append({
                        "index": i,
                        "field": field,
                        "message": f"Missing required field: {field}"
                    })

            text_len = len(str(item.get("instruction", "")) + str(item.get("input", "")) + str(item.get("output", "")))
            if max_length > 0 and text_len > max_length:
                issues.append({
                    "index": i,
                    "field": "length",
                    "message": f"Length {text_len} exceeds max {max_length}"
                })

            content_key = (item.get("instruction", ""), item.get("input", ""), item.get("output", ""))
            if content_key in seen:
                issues.append({
                    "index": i,
                    "field": "duplicate",
                    "message": "Duplicate entry"
                })
            else:
                seen.add(content_key)

        return issues


class DataManager:
    """数据集管理器 — CRUD 操作 + JSONL 持久化

    Args:
        data_dir: 数据集存储目录

    Public API:
        list_names() → list[str]
        get(name) → Dataset | None
        create(name, description) → Dataset
        delete(name) → bool
        import_jsonl(path, name) → Dataset
        generate_identity_data(name, creator, description) → list[dict]
    """

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._datasets = {}
        os.makedirs(data_dir, exist_ok=True)
        self._scan()

    def _scan(self):
        """扫描数据目录加载数据集"""
        self._datasets.clear()
        if not os.path.isdir(self._data_dir):
            return
        for entry in os.listdir(self._data_dir):
            path = os.path.join(self._data_dir, entry)
            if os.path.isdir(path):
                ds = Dataset(entry, path)
                self._datasets[entry] = ds

    @property
    def datasets(self) -> dict:
        return self._datasets

    def list_names(self) -> list:
        """返回所有数据集名称"""
        return sorted(self._datasets.keys())

    def get(self, name: str) -> Optional[Dataset]:
        return self._datasets.get(name)

    def create(self, name: str, description: str = "") -> Dataset:
        """创建新数据集"""
        name = self._sanitize_name(name)
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' already exists")
        path = os.path.join(self._data_dir, name)
        ds = Dataset(name, path)
        ds.description = description
        ds.save()
        self._datasets[name] = ds
        return ds

    def delete(self, name: str) -> bool:
        """删除数据集"""
        import shutil
        ds = self._datasets.get(name)
        if not ds:
            return False
        try:
            shutil.rmtree(ds.path)
            del self._datasets[name]
            return True
        except OSError as e:
            logger.error(f"Delete dataset failed: {e}")
            return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """重命名数据集"""
        import shutil
        new_name = self._sanitize_name(new_name)
        ds = self._datasets.get(old_name)
        if not ds or new_name in self._datasets:
            return False
        new_path = os.path.join(self._data_dir, new_name)
        try:
            shutil.move(ds.path, new_path)
            del self._datasets[old_name]
            ds.name = new_name
            ds.path = new_path
            self._datasets[new_name] = ds
            ds.save()
            return True
        except OSError as e:
            logger.error(f"Rename dataset failed: {e}")
            return False

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """清理数据集名称，只保留安全字符"""
        import re
        name = name.strip()
        name = re.sub(r'[^\w\-.]', '_', name)
        return name or "untitled"

    def import_jsonl(self, file_path: str, dataset_name: str,
                     field_map: dict = None) -> dict:
        """导入 JSONL 文件

        Args:
            file_path: JSONL 文件路径
            dataset_name: 目标数据集名
            field_map: 字段映射 {源字段: 目标字段}，默认自动检测

        Returns:
            {success: int, failed: int, errors: list}
        """
        result = {"success": 0, "failed": 0, "errors": []}
        ds = self._datasets.get(dataset_name)
        if not ds:
            try:
                ds = self.create(dataset_name)
            except ValueError:
                result["errors"].append("Dataset already exists with different content")
                return result

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        mapped = self._map_fields(item, field_map)
                        if self._validate_item(mapped):
                            ds.data.append(mapped)
                            result["success"] += 1
                        else:
                            result["failed"] += 1
                            result["errors"].append(f"Line {line_num}: validation failed")
                    except json.JSONDecodeError as e:
                        result["failed"] += 1
                        result["errors"].append(f"Line {line_num}: {e}")
            ds.save()
        except OSError as e:
            result["errors"].append(str(e))
        return result

    def import_json(self, file_path: str, dataset_name: str,
                    field_map: dict = None) -> dict:
        """导入 JSON 文件（数组格式）"""
        result = {"success": 0, "failed": 0, "errors": []}
        ds = self._datasets.get(dataset_name)
        if not ds:
            try:
                ds = self.create(dataset_name)
            except ValueError:
                result["errors"].append("Dataset already exists")
                return result

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if not isinstance(data, list):
                result["errors"].append("JSON must be an array")
                return result
            for i, item in enumerate(data):
                mapped = self._map_fields(item, field_map)
                if self._validate_item(mapped):
                    ds.data.append(mapped)
                    result["success"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(f"Item {i}: validation failed")
            ds.save()
        except (json.JSONDecodeError, OSError) as e:
            result["errors"].append(str(e))
        return result

    def import_csv(self, file_path: str, dataset_name: str,
                   field_map: dict = None) -> dict:
        """导入 CSV 文件"""
        result = {"success": 0, "failed": 0, "errors": []}
        ds = self._datasets.get(dataset_name)
        if not ds:
            try:
                ds = self.create(dataset_name)
            except ValueError:
                result["errors"].append("Dataset already exists")
                return result

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        mapped = self._map_fields(dict(row), field_map)
                        if self._validate_item(mapped):
                            ds.data.append(mapped)
                            result["success"] += 1
                        else:
                            result["failed"] += 1
                            result["errors"].append(f"Row {i}: validation failed")
                    except Exception as e:
                        result["failed"] += 1
                        result["errors"].append(f"Row {i}: {e}")
            ds.save()
        except (OSError, csv.Error) as e:
            result["errors"].append(str(e))
        return result

    def export_dataset(self, name: str, output_path: str, fmt: str = "jsonl") -> bool:
        """导出数据集到文件"""
        ds = self._datasets.get(name)
        if not ds:
            return False
        try:
            if fmt == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in ds.data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif fmt == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(ds.data, f, ensure_ascii=False, indent=2)
            return True
        except OSError as e:
            logger.error(f"Export failed: {e}")
            return False

    @staticmethod
    def _map_fields(item: dict, field_map: dict = None) -> dict:
        """字段映射

        默认自动检测 instruction/input/output，
        如果提供了 field_map 则使用映射
        """
        if field_map:
            mapped = {}
            for src, dst in field_map.items():
                if src in item:
                    mapped[dst] = item[src]
            for key in ["instruction", "input", "output"]:
                if key not in mapped:
                    mapped[key] = item.get(key, "")
            return mapped

        mapped = {"instruction": "", "input": "", "output": ""}
        for key in ["instruction", "input", "output"]:
            if key in item:
                mapped[key] = str(item[key])
            elif key == "instruction":
                for alt in ["prompt", "question", "Q"]:
                    if alt in item:
                        mapped[key] = str(item[alt])
                        break
            elif key == "output":
                for alt in ["response", "answer", "A"]:
                    if alt in item:
                        mapped[key] = str(item[alt])
                        break
        return mapped

    @staticmethod
    def _validate_item(item: dict) -> bool:
        """验证单条数据"""
        for field in REQUIRED_FIELDS:
            if not item.get(field, "").strip():
                return False
        return True

    @staticmethod
    def generate_identity_data(name: str, creator: str, description: str,
                               specialties: str = "") -> list:
        """生成模型身份认知数据"""
        data = []
        qa_pairs = [
            ("你是谁？", f"我是{name}，由{creator}创建。{description}"),
            ("你的名字是什么？", f"我叫{name}。"),
            ("谁创建了你？", f"我由{creator}创建。"),
            ("你擅长什么？", f"我擅长{specialties}。" if specialties else f"我的能力包括：{description}"),
            ("请介绍一下你自己", f"我是{name}，一个AI助手。{description}我由{creator}开发。"),
        ]
        if specialties:
            qa_pairs.append(
                ("你能做什么？", f"我是{name}，擅长{specialties}。我可以帮助你解决相关问题。")
            )
        for q, a in qa_pairs:
            data.append({"instruction": q, "input": "", "output": a})
        return data

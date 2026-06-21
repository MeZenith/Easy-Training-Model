"""Ollama 部署"""

import logging
import os
import subprocess

logger = logging.getLogger("EasyTinking")


class OllamaDeployer:
    """Ollama 模型部署管理"""

    def __init__(self):
        self._ollama_path = self._find_ollama()

    @staticmethod
    def _find_ollama() -> str:
        """查找 ollama 可执行文件"""
        for candidate in ["ollama", "ollama.exe"]:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True, text=True, encoding="utf-8", timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return ""

    def is_installed(self) -> bool:
        """检测 Ollama 是否已安装"""
        return bool(self._ollama_path)

    def get_version(self) -> str:
        """获取 Ollama 版本"""
        if not self._ollama_path:
            return ""
        try:
            result = subprocess.run(
                [self._ollama_path, "--version"],
                capture_output=True, text=True, encoding="utf-8", timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def generate_modelfile(self, model_path: str, model_name: str,
                           system_prompt: str = "", is_directory: bool = False) -> str:
        """生成 Ollama Modelfile 内容"""
        if is_directory:
            # HuggingFace 目录 —— 需要 FROM + 转换参数
            modelfile = f'FROM {model_path}\n'
            if system_prompt:
                modelfile += f'SYSTEM """{system_prompt}"""\n'
            modelfile += '''TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""\n'''
            modelfile += 'PARAMETER stop "<|im_start|>"\n'
            modelfile += 'PARAMETER stop "<|im_end|>"\n'
            modelfile += 'PARAMETER temperature 0.7\n'
            modelfile += 'PARAMETER num_ctx 2048\n'
        else:
            # GGUF 文件
            modelfile = f'FROM {model_path}\n'
            if system_prompt:
                modelfile += f'SYSTEM """{system_prompt}"""\n'
            modelfile += '''TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""\n'''
            modelfile += 'PARAMETER stop "<|im_start|>"\n'
            modelfile += 'PARAMETER stop "<|im_end|>"\n'
            modelfile += 'PARAMETER temperature 0.7\n'
            modelfile += 'PARAMETER num_ctx 2048\n'
        return modelfile

    def create_model(self, modelfile_content: str, model_name: str,
                     working_dir: str = "") -> tuple:
        """创建 Ollama 模型

        Returns:
            (success: bool, output: str)
        """
        if not self._ollama_path:
            return False, "Ollama not installed"

        if not working_dir:
            working_dir = os.path.dirname(os.path.abspath("."))

        modelfile_path = os.path.join(working_dir, "Modelfile")
        try:
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)
        except OSError as e:
            return False, str(e)

        try:
            result = subprocess.run(
                [self._ollama_path, "create", model_name, "-f", modelfile_path],
                capture_output=True, text=True, encoding="utf-8", timeout=300,
                cwd=working_dir,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr or result.stdout
        except subprocess.TimeoutExpired:
            return False, "Ollama create timed out"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.remove(modelfile_path)
            except OSError:
                pass

    def run_model(self, model_name: str) -> tuple:
        """运行 Ollama 模型

        Returns:
            (success: bool, message: str)
        """
        if not self._ollama_path:
            return False, "Ollama not installed"
        try:
            subprocess.Popen(
                [self._ollama_path, "run", model_name],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            )
            return True, "Ollama model started"
        except Exception as e:
            return False, str(e)

    def list_models(self) -> list:
        """列出已安装的 Ollama 模型"""
        if not self._ollama_path:
            return []
        try:
            result = subprocess.run(
                [self._ollama_path, "list"],
                capture_output=True, text=True, encoding="utf-8", timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                models = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        models.append({
                            "name": parts[0],
                            "id": parts[1] if len(parts) > 1 else "",
                            "size": parts[2] if len(parts) > 2 else "",
                        })
                return models
        except Exception:
            pass
        return []

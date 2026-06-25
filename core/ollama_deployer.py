import logging
import os
import subprocess

logger = logging.getLogger("EasyTinking")


class OllamaDeployer:
    #Ollama模型部署管理

    def __init__(self):
        self._ollama_path = self._find_ollama()

    @staticmethod
    def _find_ollama() -> str:
        #找ollama可执行文件
        for candidate in ["ollama", "ollama.exe"]:
            try:
                #Win下要隐藏命令行窗口
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=5,
                    creationflags=flags,
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return ""

    def is_installed(self) -> bool:
        #检测有没有装ollama
        return bool(self._ollama_path)

    def get_version(self) -> str:
        #获取版本号
        if not self._ollama_path:
            return ""
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                [self._ollama_path, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                creationflags=flags,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Failed to detect ollama version: %s", e)
            return ""

    def generate_modelfile(
        self, model_path: str, model_name: str, system_prompt: str = "", is_directory: bool = False
    ) -> str:
        #生成Ollama Modelfile内容
        safe_path = model_path.replace("\\", "/")
        modelfile = f'FROM "{safe_path}"\n'
        if system_prompt:
            modelfile += f'SYSTEM """{system_prompt}"""\n'
        #Qwen2格式的模板
        modelfile += '''TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""\n'''
        modelfile += 'PARAMETER stop "<|im_start|>"\n'
        modelfile += 'PARAMETER stop "<|im_end|>"\n'
        modelfile += "PARAMETER temperature 0.7\n"
        modelfile += "PARAMETER num_ctx 2048\n"
        # logger.debug("Generated modelfile for %s", model_name)
        return modelfile

    def create_model(self, modelfile_content: str, model_name: str, working_dir: str = "") -> tuple:
        #创建ollama模型，返回 (成功?, 输出)
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

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            result = subprocess.run(
                [self._ollama_path, "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
                cwd=working_dir,
                creationflags=flags,
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
            except OSError as e:
                logger.warning("Failed to remove modelfile %s: %s", modelfile_path, e)

    def run_model(self, model_name: str) -> tuple:
        #运行ollama模型
        if not self._ollama_path:
            return False, "Ollama not installed"
        try:
            flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            subprocess.Popen(
                [self._ollama_path, "run", model_name],
                creationflags=flags,
            )
            return True, "Ollama model started"
        except Exception as e:
            return False, str(e)

    def delete_model(self, model_name: str) -> tuple:
        #删除ollama模型
        if not self._ollama_path:
            return False, "Ollama not installed"
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                [self._ollama_path, "rm", model_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                creationflags=flags,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip() or result.stdout.strip()
        except Exception as e:
            return False, str(e)

    def list_models(self) -> list:
        #列出已安装的ollama模型
        if not self._ollama_path:
            return []
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                [self._ollama_path, "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                creationflags=flags,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                models = []
                #第一行是表头，跳过
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        size = " ".join(parts[2:4]) if len(parts) >= 4 else parts[2] if len(parts) > 2 else ""
                        modified = " ".join(parts[4:]) if len(parts) > 4 else ""
                        models.append(
                            {
                                "name": parts[0],
                                "id": parts[1],
                                "size": size,
                                "modified": modified,
                            }
                        )
                return models
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Failed to list ollama models: %s", e)
        return []

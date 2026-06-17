# Easy Tinking

**零代码 AI 大模型微调桌面工具** —— 下载模型、准备数据、训练、导出、部署到 Ollama，全部在图形界面完成，无需写一行代码。

基于 PySide6 + PyTorch + PEFT (LoRA) 构建。

## 适用人群

- 想拥有专属 AI 编程助手但不想学深度学习的**全栈开发者**
- 需要本地部署微调模型、保护代码隐私的**企业开发团队**
- 正在学习大模型微调、需要一个可视化工具的**AI 初学者**
- 想快速验证训练数据和参数效果的**算法工程师**

不需要懂 Python、不需要写训练脚本、不需要配环境——打开软件就能训。

## 快速开始

```bash
# 1. 克隆仓库
git clone git@github.com:MeZenith/Easy-Training-Model.git
cd Easy-Training-Model

# 2. 安装依赖
pip install PySide6 pyqtgraph torch transformers peft huggingface_hub safetensors accelerate bitsandbytes

# 3. 启动
python main.py
```

**硬件要求**：NVIDIA GPU, CUDA 12.4+, VRAM >= 8GB（3B 模型 fp16 训练约需 6GB）

## 功能

| 模块 | 说明 |
|------|------|
| 模型管理 | 内置 6 个预选模型，支持 HuggingFace 下载、本地模型导入、完整性校验、删除 |
| 数据管理 | 创建数据集、导入 JSONL/JSON/CSV、在线编辑、多选、身份数据自动生成、数据校验 |
| 训练中心 | LoRA 微调，参数预设（快速/标准/精细/自定义），双页布局（配置 + 实时监控），Loss 曲线图 |
| 导出部署 | 16 位完整模型导出（自动合并 LoRA），LoRA 适配器导出，HF→GGUF 转换工具，Ollama 部署 |
| 对话测试 | 加载训练好的模型对话，Temperature/Top-P/Top-K/重复惩罚/存在惩罚 可调，性能统计 |
| 系统设置 | 中英文切换、深色/浅色主题、HF 镜像、代理设置、系统/GPU 信息 |
| 运行日志 | 实时日志查看、级别过滤、关键词搜索、自动滚动 |

### 使用流程

```
1. 设置 → 配置工作目录和镜像源
2. 模型管理 → 下载底座模型（推荐 Qwen2.5-Coder-3B）
3. 数据管理 → 导入/创建训练数据（Alpaca 格式 JSONL）
4. 训练中心 → 选模型 + 勾数据集 + 调参数 → 开始训练
5. 导出部署 → 16位导出（自动合并 LoRA）→ Ollama 部署
6. 对话测试 → 加载训练好的模型聊天
```

## 技术栈

| 类别 | 技术 |
|------|------|
| UI | PySide6 (Qt 6), pyqtgraph |
| 深度学习 | PyTorch 2.5, Transformers 5.x, PEFT (LoRA) |
| 训练 | 子进程隔离, 纯 PyTorch 循环, AMP 混合精度, 梯度检查点 |
| 导出 | HuggingFace safetensors, GGUF (llama.cpp) |
| 部署 | Ollama API |
| 国际化 | 自研 Signal 驱动的 i18n 系统 |
| 打包 | PyInstaller |

### 训练架构

```
main.py (Qt GUI)
  └── ProcessTrainer (QProcess 子进程管理)
        └── train_worker.py (独立 Python 进程)
              ├── 加载底座模型 + LoRA 适配器
              ├── Alpaca 格式分词
              ├── AdamW + 梯度累积 + AMP + 梯度检查点
              ├── Cosine/Linear/Constant 调度器 + 预热
              └── 保存权重 + 元数据
```

> **为什么用子进程？** 在 Windows + RTX 4060 上，CUDA DLL 在 QThread 中首次加载会触发 0xC0000005 崩溃。子进程完全隔离，崩溃只影响子进程。

## 项目结构

```
Easy Tinking/
├── main.py               # 程序入口
├── setup_icon.py          # 图标设置（任务栏/标题栏/Alt+Tab）
├── EasyTinking.spec       # PyInstaller 打包配置
├── core/                  # 核心业务逻辑
│   ├── config.py          # 配置（深合并、自动保存、原子写入）
│   ├── model_manager.py   # 模型管理（下载/校验/删除）
│   ├── data_manager.py    # 数据集管理（CRUD/导入/导出/校验）
│   ├── train_worker.py    # 独立训练进程（PyTorch 原生循环）
│   ├── trainer.py         # ProcessTrainer（QProcess 管理）
│   ├── exporter.py        # 导出（16位/LoRA/GGUF，合并 LoRA）
│   ├── ollama_deployer.py # Ollama 部署（检测/创建/运行）
│   └── error_handler.py   # 错误分类/友好提示
├── ui/                    # 用户界面
│   ├── app.py             # 主窗口、标题栏、导航、全局异常处理
│   ├── theme.py           # 主题管理（深色黑+青绿/浅色白+蓝）
│   ├── pages/             # 7 个功能页面
│   └── components/        # 可复用组件（Loss 曲线、GPU 监控等）
├── utils/                 # 工具（i18n、日志、GPU 信息、Worker 基类）
├── locale/                # 翻译文件（zh.json / en.json）
├── tools/                 # 转换工具（HF → GGUF）
├── res/                   # 应用图标
├── assess/                # QSS 主题样式
└── workspace/             # 用户工作目录（不上传 Git）
```

## 已知问题

### 已修复
| 问题 | 根因 |
|------|------|
| 训练闪退(0xC0000005) | QThread 内 CUDA DLL 初始化崩溃 → 子进程隔离 |
| 对话输出模板垃圾 | Alpaca 格式不匹配 → 遇到 ### 停止 |
| 数据导入 BOM 报错 | utf-8 vs BOM → utf-8-sig |
| Loss 图表空白 | 没喂数据 → 解析子进程 LOG |
| 中/英文切换失效 | load_language 缓存 → force 参数 |
| 预设问题硬编码 | 直接字符串 → i18n 动态切换 |
| 删除按钮看不见 | Unicode 无字形 → X + 红框 QSS |

### 功能限制
| 限制 | 原因 |
|------|------|
| GGUF 导出不可用 | unsloth 不兼容 PyTorch 2.5.1 |
| Ollama 直接导入 safetensors 乱码 | Ollama 只完全支持 GGUF |
| 任务栏图标开发期不显示 | python.exe 图标，打包后正常 |
| 对话多轮不支持 | 当前 Alpaca 单轮格式 |
| 单 GPU | 训练循环无分布式 |

## License

Apache 2.0 — BlueCorner Studio

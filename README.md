# Easy Tinking

**一键式 AI 大模型微调工具** — 让每个人都能训练自己的专属 AI 模型，无需编写任何代码。

PySide6 桌面应用，支持模型下载、数据管理、LoRA 微调训练、模型导出、Ollama 部署和对话测试。

## 功能

| 模块 | 功能 |
|------|------|
| 模型管理 | 内置 6 个预选模型，支持 HuggingFace 下载、本地模型导入、完整性校验、删除 |
| 数据管理 | 创建数据集、导入 JSONL/JSON/CSV、在线编辑、多选、身份数据自动生成、数据校验 |
| 训练中心 | LoRA 微调，参数预设（快速/标准/精细/自定义），双页布局（配置 + 实时监控），Loss 曲线图 |
| 导出部署 | 16 位完整模型导出（自动合并 LoRA）、LoRA 适配器导出、Ollama 一键部署 |
| 对话测试 | 加载训练好的模型进行对话，支持 Temperature/Top-p/Max-tokens 调节，性能统计面板 |
| 系统设置 | 中英文界面切换、暗色/浅色主题、工作目录配置、HuggingFace 镜像、代理设置、系统/GPU 信息 |
| 运行日志 | 实时日志查看、级别过滤、关键词搜索、自动滚动、导出 |

### 训练功能详解

- 完整 LoRA 微调流程：选择底座模型 → 准备训练数据 → 设置超参数 → 开始训练 → 导出模型
- 支持 Alpaca 格式指令微调（`### Instruction: ... ### Response: ...`）
- 训练在独立子进程中执行，与主程序完全隔离，避免 CUDA 崩溃影响 UI
- 纯 PyTorch 训练循环，不依赖 trl/datasets/accelerate
- 实时 Loss 曲线（pyqtgraph），训练结果面板展示完整指标
- 支持梯度累积、混合精度训练(AMP)、梯度检查点、预热调度

## 技术栈

| 类别 | 技术 |
|------|------|
| UI 框架 | PySide6 (Qt 6) |
| 深度学习 | PyTorch 2.5+、Transformers 5.x、PEFT (LoRA) |
| 图表 | pyqtgraph |
| 国际化 | 自研 I18n 单例，支持运行时语言切换 |
| 打包 | PyInstaller |
| 部署 | Ollama API |

### 依赖

```
PySide6 pyqtgraph torch transformers peft
huggingface_hub bitsandbytes accelerate safetensors
```

可选：`llama-cpp-python`（GGUF 导出）、`ollama`（本地部署）

## 项目结构

```
Easy Training/
├── main.py                     # 程序入口
├── core/                       # 核心业务逻辑
│   ├── config.py               # 应用配置（深合并、自动保存、原子写入）
│   ├── model_manager.py        # 模型管理（下载/校验/删除/内置列表）
│   ├── data_manager.py         # 数据集管理（CRUD/导入/导出/校验/身份生成）
│   ├── train_worker.py         # 独立训练进程（PyTorch 原生训练循环）
│   ├── trainer.py              # ProcessTrainer（QProcess 子进程管理）
│   ├── exporter.py             # 模型导出（16位/LoRA/GGUF，合并 LoRA）
│   ├── ollama_deployer.py      # Ollama 部署（检测/创建/运行/列表）
│   └── error_handler.py        # 错误分类、友好提示、safe_call 装饰器
├── ui/                         # 用户界面
│   ├── app.py                  # 主窗口、侧边栏导航、全局异常处理
│   ├── theme.py                # 主题管理（暗色/浅色 QSS）
│   ├── pages/                  # 7 个功能页面
│   │   ├── model_page.py       # 模型管理页
│   │   ├── data_page.py        # 数据管理页
│   │   ├── train_page.py       # 训练中心（配置页 + 监控页双页布局）
│   │   ├── export_page.py      # 导出部署页
│   │   ├── test_page.py        # 对话测试页
│   │   ├── settings_page.py    # 系统设置页
│   │   └── logs_page.py        # 运行日志页
│   └── components/             # 可复用组件
│       ├── loss_chart.py       # Loss 曲线图（pyqtgraph）
│       ├── gpu_monitor.py      # GPU 显存/温度监控
│       ├── model_card.py       # 模型卡片
│       ├── data_table.py       # 数据表格
│       └── progress_bar.py     # 进度条
├── utils/                      # 工具模块
│   ├── i18n.py                 # 国际化管理器（Signal 驱动 UI 刷新）
│   ├── worker.py               # BaseWorker 基类（QThread + WorkerSignals）
│   ├── logger.py               # 日志系统（TimedRotatingFileHandler）
│   ├── gpu_info.py             # GPU 信息获取（nvidia-smi）
│   └── system_info.py          # 系统信息获取
├── locale/                     # 翻译文件
│   ├── zh.json                 # 中文
│   └── en.json                 # 英文
├── tools/                      # 辅助工具
│   ├── convert_hf_to_gguf.py   # HuggingFace → GGUF 转换
│   └── quantize_gguf.py        # GGUF 量化（WIP）
└── workspace/                  # 工作目录
    ├── models/                 # 下载的底座模型
    ├── data/                   # 训练数据集
    ├── lora/                   # 训练好的 LoRA 适配器
    ├── exports/                # 导出的模型
    ├── logs/                   # 运行日志
    └── config.json             # 运行时配置
```

## 实现原理

### 训练架构

训练使用**子进程隔离**方案。主程序通过 QProcess 启动独立的 Python 进程执行训练：

```
main.py (Qt 事件循环)
    │
    └── ProcessTrainer (QProcess)
            │
            └── python train_worker.py --config /tmp/config.json
                    │
                    ├── 1. 加载底座模型 + LoRA 适配器
                    ├── 2. 分词训练数据（Alpaca 格式）
                    ├── 3. 纯 PyTorch 训练循环
                    │       ├── AdamW 优化器 + 梯度累积
                    │       ├── 混合精度 AMP（GradScaler）
                    │       ├── 梯度检查点（节省显存）
                    │       ├── Cosine/Linear/Constant 调度器
                    │       └── 预热（Warmup）
                    ├── 4. 保存 LoRA 权重 + 元数据
                    └── 5. stdout 输出进度/结果
                            ↓
              主进程解析并更新 UI
```

**为什么用子进程而不是 QThread？** 在 Windows + RTX 4060 上，`transformers`/`torch` 的 CUDA DLL 在 QThread 中首次加载会导致 `0xC0000005` 内存访问违规（进程级崩溃）。子进程完全隔离，崩溃只影响子进程，主程序正常弹窗报错。

### 对话测试

```
1. 主线程加载模型（避免 QThread CUDA 崩溃）
     ├── LoadModelWorker: 加载底座 + PEFT 适配器 → merge_and_unload()
     └── 返回合并后的模型对象

2. 用户输入 → GenerateWorker
     ├── 构建 Alpaca 格式 prompt（与训练格式一致）
     ├── model.generate() 生成回复
     ├── 截断 input tokens
     └── 遇到 ### / Human: 模板标记自动停止
```

### 导出流程

```
16 位导出：
  Load base model + LoRA adapter → merge_and_unload() → save_pretrained()

LoRA 仅导出：
  复制 adapter 文件到导出目录

Ollama 部署：
  查找 GGUF 或 safetensors 目录 → 生成 Modelfile → ollama create → ollama run
```

## 已知问题与待修复

### 待修复

| 问题 | 详情 | 优先级 |
|------|------|--------|
| GGUF 量化 | `tools/quantize_gguf.py` 因 `gguf` 库 API 变更暂不可用，需适配 | 高 |
| GGUF 导出（App 内） | Unsloth 不兼容 PyTorch 2.5.1+cu124，需等上游更新 | 中 |
| 对话测试流式显示 | 当前生成完后一次性显示，无逐字输出效果 | 低 |
| 对话测试多轮 | 只取最后一轮消息，不支持带历史的连续对话 | 低 |
| 预设参数切换 | 预设切换时部分参数不更新（如 grad_accum） | 低 |

### 平台限制

- 训练需要 NVIDIA GPU（CUDA 12.4+），V RAM >= 8GB（3B 模型 fp16）
- Windows 专属：`os.startfile()`、`subprocess.CREATE_NO_WINDOW`
- Ollama 部署依赖已安装的 Ollama 客户端

## 快速开始

```bash
# 安装依赖
pip install PySide6 pyqtgraph torch transformers peft huggingface_hub accelerate safetensors bitsandbytes

# 启动
python main.py
```

### 使用流程

1. 设置：配置工作目录和 HuggingFace 镜像
2. 模型：下载底座模型（Qwen2.5-Coder-3B 推荐）
3. 数据：创建数据集，导入 JSONL 训练数据
4. 训练：选择模型 + 数据集 + 参数 → 开始训练
5. 导出：16 位导出（含 LoRA 合并）
6. 测试：加载训练好的模型进行对话

## License

Apache 2.0

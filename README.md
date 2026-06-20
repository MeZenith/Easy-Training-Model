# Easy Tinking

**一站式 AI 模型微调工作站** —— 从下载模型到训练、导出、部署、对话测试，全流程图形化操作，无需编写任何代码。

基于 PySide6 + PyTorch + PEFT (LoRA) · 子进程隔离 CUDA · 中英双语 · 薄荷主题

[中文] | [English](README.en.md)

## 界面预览

| 训练中心 | 数据管理 | 模型管理 |
|:---:|:---:|:---:|
| ![](image/训练中心.png) | ![](image/数据管理.png) | ![](image/模型管理.png) |

| 对话测试 | 导出部署 | 运行日志 |
|:---:|:---:|:---:|
| ![](image/对话测试.png) | ![](image/导出部署.png) | ![](image/运行日志.png) |

| 系统设置 | 浅色主题 | 英文界面 |
|:---:|:---:|:---:|
| ![](image/系统设置.png) | ![](image/浅色主题.png) | ![](image/切换英文界面.png) |

## 适用人群

- 想拥有个人 AI 助手但不想学习深度学习的**全栈开发者**
- 需要本地私有部署微调模型、保护数据隐私的**企业团队**
- 正在学习大模型微调、需要可视化实验环境的**AI 初学者**
- 想快速迭代训练数据和参数效果的**算法工程师**

## 快速开始

```bash
git clone git@github.com:MeZenith/Easy-Training-Model.git
cd Easy-Training-Model
pip install -r requirements.txt
python main.py
```

**硬件要求**：NVIDIA GPU，CUDA 12.4+，VRAM ≥ 8GB（Qwen2.5-3B 模型 fp16 训练约需 6GB）

## 功能模块

| 模块 | 功能 |
|------|------|
| **模型管理** | 内置 7 个预选模型，支持 HuggingFace 自定义下载、本地验证、删除；可管理已训练的 LoRA 适配器 |
| **数据管理** | 创建/导入/编辑数据集，支持 JSONL/JSON/CSV，在线表格编辑，身份数据自动生成，数据校验 |
| **训练中心** | LoRA 微调，4 档预设（快速/标准/精细/自定义），双页布局（配置 + 实时监控），Loss 曲线+GPU 监控 |
| **导出部署** | 16 位完整模型（自动合并 LoRA），GGUF Q4/Q8/F16 量化，LoRA 适配器单独导出，Ollama 一键部署 |
| **对话测试** | 加载已训练模型对话，Temperature / Top-P / Top-K / 重复惩罚 / 存在惩罚 可调，实时性能统计 |
| **系统设置** | 中英文切换、深色/浅色薄荷主题、工作目录/HF 镜像/代理配置、系统/GPU 信息 |

### 使用流程

```
1. 系统设置 → 配置工作目录和 HuggingFace 镜像源
2. 模型管理 → 下载底座模型（推荐 Qwen2.5-Coder-3B-Instruct）
3. 数据管理 → 导入或创建训练数据（Alpaca 格式 JSONL）
4. 训练中心 → 选择模型 + 勾选数据集 + 设置输出名称 + 调整参数 → 开始训练
5. 导出部署 → 选择 LoRA → 勾选导出格式 → 保存模型 → Ollama 部署
6. 对话测试 → 选择 LoRA → 加载模型 → 开始对话
```

## 技术架构

### 技术栈

| 类别 | 技术 |
|------|------|
| UI 框架 | PySide6 (Qt 6), pyqtgraph |
| 深度学习 | PyTorch 2.5+, Transformers 4.45+, PEFT (LoRA) |
| 训练引擎 | 子进程隔离, 纯 PyTorch 训练循环, AMP 混合精度, 梯度检查点 |
| 模型导出 | HuggingFace safetensors, GGUF (llama.cpp) |
| 模型部署 | Ollama CLI |
| 国际化 | Signal 驱动的 i18n 系统 (zh / en) |
| 测试 | pytest, py_compile |

### 子进程架构

```
main.py → MainWindow (Qt GUI)
  ├── ProcessTrainer (QProcess)
  │     └── train_worker.py (独立进程, CUDA 隔离)
  └── Inferencer (QProcess)
        └── infer_worker.py (独立进程, stdin/stdout JSON 通信)
```

> **为什么用子进程？** 在 Windows + RTX 4060 上，CUDA DLL 在 QThread 中加载会触发 `0xC0000005` 崩溃。子进程完全隔离 CUDA 运行时，崩溃不影响主界面。

### 项目结构

```
Easy Tinking/
├── main.py                    # 程序入口
├── pyproject.toml              # 项目元数据
├── requirements.txt            # 宽松依赖
├── requirements-lock.txt       # 锁定依赖 (可重现构建)
├── README.md / README.en.md    # 文档 (中/英)
├── setup_icon.py               # 图标设置 (任务栏/窗口)
├── .github/workflows/          # CI (ruff + pytest)
├── core/                       # 业务逻辑层
│   ├── config.py               # 配置管理 (JSON 持久化, 线程安全)
│   ├── model_manager.py        # 模型下载/验证/删除
│   ├── data_manager.py         # 数据集 CRUD / JSONL 持久化
│   ├── trainer.py              # ProcessTrainer (QProcess 训练管理)
│   ├── train_worker.py         # 独立训练子进程 (PyTorch 原生循环)
│   ├── inferencer.py           # Inferencer (QProcess 推理管理)
│   ├── infer_worker.py         # 独立推理子进程 (stdin JSON 通信)
│   ├── exporter.py             # 模型导出 (16bit / GGUF / LoRA)
│   ├── ollama_deployer.py      # Ollama 部署 (检测/创建/运行)
│   ├── error_handler.py        # 错误分类与格式化
│   └── workers/
│       └── download_worker.py  # HuggingFace 下载 Worker
├── ui/                         # UI 层
│   ├── app.py                  # 主窗口, 标题栏, 侧边栏, 快捷键
│   ├── theme.py                # 主题管理器 (深色/浅色单例)
│   ├── error_dialog.py         # 错误弹窗 (UI 层)
│   ├── pages/                  # 7 个功能页面
│   │   ├── model_page.py       # 模型管理 + LoRA 管理
│   │   ├── data_page.py        # 数据管理
│   │   ├── train/              # 训练中心 (拆分为3文件)
│   │   │   ├── train_page.py   # 训练编排
│   │   │   ├── config_panel.py # 配置面板
│   │   │   └── monitor_panel.py# 监控面板
│   │   ├── export_page.py      # 导出部署
│   │   ├── test_page.py        # 对话测试
│   │   ├── settings_page.py    # 系统设置
│   │   └── logs_page.py        # 运行日志
│   └── components/             # 可复用组件
│       ├── model_card.py       # 模型信息卡片
│       ├── loss_chart.py       # 实时 Loss 曲线 (pyqtgraph)
│       ├── gpu_monitor.py      # GPU 显存监控
│       ├── data_table.py       # 可编辑数据表格
│       └── progress_bar.py     # 自定义进度条
├── utils/                      # 工具层
│   ├── i18n.py                 # Signal 驱动的国际化管理器
│   ├── logger.py               # 日志系统
│   ├── worker.py               # BaseWorker (QThread 基类)
│   ├── gpu_info.py             # nvidia-smi GPU 信息查询
│   └── system_info.py          # 系统环境信息
├── locale/                     # 翻译文件
│   ├── zh.json                 # 简体中文
│   └── en.json                 # English
├── tests/                      # 单元测试
│   ├── test_config.py          # AppConfig 配置测试
│   └── test_i18n.py            # I18n 国际化测试
├── assess/                     # QSS 主题样式
│   ├── professional_theme.qss  # 深色薄荷主题
│   ├── light_theme.qss         # 浅色薄荷主题
│   └── nav_icons.py            # 侧边栏矢量图标
├── tools/                      # CLI 工具
│   ├── convert_hf_to_gguf.py   # HF → GGUF 转换
│   └── quantize_gguf.py        # GGUF 量化
├── res/                        # 应用图标 (ico)
└── image/                      # 截图 (文档用)
```

## 已完成的质量改进

| 类别 | 内容 |
|------|------|
| **架构** | error_handler 分层 (core 纯逻辑 / ui 弹窗)，TrainWorker 废弃类删除，train_page 728行拆分为3文件 |
| **代码整洁** | 17 处静默异常 → logger 记录，6 个硬编码阈值 → config.get()，8 个硬编码文本 → i18n |
| **UI** | 深色/浅色薄荷主题，毛玻璃侧边栏，统一 8px 圆角，40+ 处硬编码 → i18n |
| **基础设施** | pyproject.toml，requirements-lock.txt，CI pytest 集成，13 个单元测试，7 个核心类 docstring |
| **功能** | LoRA 命名输入框，模型页 LoRA 删除管理，ModelCard Load 按钮，GPU 状态栏 5 秒自动刷新 |

## 许可证

本软件采用 **CC BY-NC 4.0** 许可证：

- **署名** — 使用、修改、再分发时须注明来源 [MeZenith/Easy-Training-Model](https://github.com/MeZenith/Easy-Training-Model)
- **非商业性** — 不得将本软件用于商业目的
- **可自由** — 复制、再分发、修改、二次创作

完整条款见 [LICENSE](LICENSE)

---

BlueCorner Studio · 2026

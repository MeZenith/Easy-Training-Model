# Easy Training

**一站式 AI 模型微调工作站** —— 从下载模型到训练、导出、部署、对话测试，全流程图形化操作，无需编写任何代码。

基于 PySide6 + PyTorch + PEFT (LoRA) · 子进程隔离 CUDA · 中英双语 · 深色/浅色主题

[中文] | [English](README.en.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green)](https://pypi.org/project/PySide6/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

## 界面预览

| 训练中心 | 数据管理 | 模型管理 |
|:---:|:---:|:---:|
| ![](image/%E8%AE%AD%E7%BB%83%E4%B8%AD%E5%BF%83%E7%95%8C%E9%9D%A2.png) | ![](image/%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86%E7%95%8C%E9%9D%A2.png) | ![](image/%E6%A8%A1%E5%9E%8B%E7%AE%A1%E7%90%86%E7%95%8C%E9%9D%A2.png) |

| 对话测试 | 导出部署 | 运行日志 |
|:---:|:---:|:---:|
| ![](image/%E5%AF%B9%E8%AF%9D%E6%B5%8B%E8%AF%95%E7%95%8C%E9%9D%A2.png) | ![](image/%E5%AF%BC%E5%87%BA%E9%83%A8%E7%BD%B2%E7%95%8C%E9%9D%A2.png) | ![](image/%E8%BF%90%E8%A1%8C%E6%97%A5%E5%BF%97%E7%95%8C%E9%9D%A2.png) |

| 系统设置 |
|:---:|
| ![](image/%E8%AE%BE%E7%BD%AE%E7%95%8C%E9%9D%A2.png) |

## 为什么需要这个工具

大语言模型（LLM）已经广泛应用于各个领域，但要让通用模型在特定领域表现得更好，微调是必不可少的步骤。

然而传统微调流程对普通用户极其不友好：配置 Linux CUDA 环境、编写 Python 训练脚本、手动管理 LoRA 适配器、用命令行导出合并模型——每一步都是门槛。很多有能力应用 AI 的人，因为不会写训练代码而无法拥有属于自己的专属模型。

Easy Training 就是为了解决这个问题而设计的。它把整个微调流程做成了桌面软件：**像用 Word 一样用 PyTorch**。你只需要点几下鼠标，就能完成从下载原始模型到部署最终产品的全过程。

## 设计理念

- **零代码**：不需要写一行代码，不需要懂 Python，不需要会 Linux。能操作电脑就能训练 AI
- **子进程隔离**：训练/推理/导出全部放在独立子进程中运行，和主界面完全隔离。这么做既解决了 Windows 下 CUDA 与 Qt GUI 线程不兼容导致的崩溃问题，也保证了即使训练出错，主程序也不会跟着挂
- **轻量化依赖**：训练循环完全用原生 PyTorch 实现，不依赖 trl、datasets、accelerate 等第三方训练框架。代码更少、兼容性更好、打包体积更可控
- **本地优先**：所有数据和模型都保存在你自己的电脑上，不需要上传到任何云端服务。数据隐私和安全由你掌控
- **真实场景驱动**：每个功能的加入都来源于实际问题——显存估算器的动态计算公式来自反复调参踩坑，LoRA metadata 记录基座模型路径是为了让导出流程不用手动匹配

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

**硬件要求**：NVIDIA GPU，CUDA 12.4+，VRAM ≥ 8GB（1.5B 模型 fp16 训练约需 4GB，3B 模型约需 6GB）

## 功能模块

| 模块 | 功能 |
|------|------|
| **模型管理** | 内置 6 个预选模型（涵盖 Qwen2.5 系列和 Llama 系列，从 1.5B 到 7B），一键下载到本地；自动校验模型文件完整性，缺失文件列表一目了然；支持切换 HuggingFace 镜像源解决国内网络问题；可管理已训练的 LoRA 适配器——查看、删除、查看关联的基座模型 |
| **数据管理** | 在线创建和编辑数据集，支持 JSONL、JSON、CSV 三种格式导入；表格直接双击即可修改内容，改完点保存持久化到磁盘；内置身份数据生成器——填写模型名称、创建者和描述，自动生成"你是谁""你叫什么"等训练样本；训练前可运行数据校验，自动检测缺失字段、重复数据、长度超限等问题 |
| **训练中心** | LoRA 微调的核心工作台。预设了快速、标准、精细、自定义四种训练方案，每种方案对应对不同的设备和时间预算；支持手动调整 lora_rank、epochs、batch_size、learning_rate、max_seq_length、梯度累积、优化器、学习率调度器等全部关键参数；训练前自动预检查——校验模型文件完整性、数据格式正确性、GPU 显存余量、磁盘剩余空间；训练中实时显示 Loss 折线图和 GPU 显存使用，每 5 步汇报一次进度；显存估算器根据当前参数动态计算预期消耗，帮助用户在训练前就判断是否可以流畅运行 |
| **导出部署** | 训练完成后的模型只是 LoRA 适配器，需要合并并导出成可独立运行的文件。支持导出 16 位完整模型（自动合并 LoRA 权重到基座模型）、GGUF 格式 F16/Q8/Q4 三种量化级别、仅 LoRA 适配器三种方式；导出采用子进程防止阻塞界面，实时日志可以看进度；内置 Ollama 部署功能——自动检测 Ollama 是否安装、生成 Modelfile、创建模型，部署完即可通过 `ollama run` 命令运行 |
| **对话测试** | 加载训练好的模型进行对话测试，不用等部署完成就能直观感受训练效果。右侧面板提供 Temperature（温度）、Top-P、Top-K、重复惩罚、存在惩罚五个可调参数；预设常用问题按钮，点击即可发送；第一轮对话完成后自动显示性能统计——生成速度（tokens/s）、总 token 数、Prompt 和 Completion 各自用量、总耗时；对话记录可保存为 txt 文件 |
| **系统设置** | 支持中英文界面切换——所有文字、导航、按钮跟随语言即时刷新；深色/浅色两套薄荷主题——深色护眼适合长时间使用，浅色清爽适合白天；可自定义工作目录，所有模型、数据、导出文件统一管理；支持配置 HuggingFace 镜像地址（默认国内镜像站）和 HTTP/SOCKS5 代理；下方显示当前系统硬件信息——GPU 型号、显存、温度、驱动版本、CUDA 版本 |

### 使用流程

```
1. 系统设置 → 配置工作目录和 HuggingFace 镜像源
2. 模型管理 → 下载底座模型（推荐 Qwen2.5-Coder-1.5B-Instruct）
3. 数据管理 → 导入或创建训练数据（Alpaca 格式 JSONL）
4. 训练中心 → 选择模型 + 勾选数据集 + 设置 LoRA 名称 + 调整参数 → 开始训练
5. 导出部署 → 选择 LoRA → 勾选导出格式 → 保存模型 → Ollama 部署
6. 对话测试 → 选择 LoRA → 加载模型 → 开始对话
```

## 技术架构

### 核心设计

**子进程隔离 CUDA**：在 Windows 上，CUDA DLL 在 QThread 中加载会触发 `0xC0000005` 内存访问错误，直接导致整个程序崩溃。Easy Training 将训练、推理、导出三类耗时操作全部放在独立的子进程中执行，通过标准输入输出与主进程通信。子进程崩溃不会影响主界面，用户感知到的只是"这次训练失败了"而不是"程序闪退了"。

**纯 PyTorch 训练循环**：没有使用 trl、datasets、accelerate 等任何第三方训练框架。训练循环完全基于原生 PyTorch 实现，包含梯度累积、AMP 混合精度、梯度检查点、余弦/线性学习率调度、Warmup、梯度裁剪等功能。这样做的好处是——代码完全可控、依赖冲突概率大幅降低、打包体积更小、理解和修改成本更低。

**QProcess 前缀协议**：主进程和子进程之间通过 stdout 输出的前缀行通信。`PROGRESS:` 表示进度、`LOG:` 表示日志、`METRIC:` 每步指标、`RESULT:` 最终结果、`ERROR:` 错误信息。推理子进程额外通过 stdin 接收 JSON 请求，支持 `generate` 和 `quit` 两种操作。

**Signal 驱动的国际化**：所有界面文字通过 `I18n` 单例管理，切换语言时发射 `language_changed` Signal，各个页面监听信号后调用 `refresh_texts()` 统一更新。翻译数据存储在独立的 `locale/zh.json` 和 `locale/en.json` 中，编辑 JSON 文件即可增删翻译条目，不需要改动任何代码。

### 技术栈

| 类别 | 技术 |
|------|------|
| UI 框架 | PySide6 (Qt 6), pyqtgraph |
| 深度学习 | PyTorch 2.5+, Transformers 4.45+, PEFT (LoRA) |
| 训练引擎 | 子进程隔离, 纯 PyTorch 训练循环, AMP 混合精度, 梯度检查点, 梯度累积 |
| 模型导出 | HuggingFace safetensors, GGUF (llama.cpp) |
| 模型部署 | Ollama CLI |
| 国际化 | Signal 驱动的 i18n 系统 (zh / en) |
| 测试 | pytest, ruff |

### 子进程架构

```
main.py → MainWindow (Qt GUI)
  ├── ProcessTrainer (QProcess)
  │     └── train_worker.py (独立进程, CUDA 隔离)
  ├── Inferencer (QProcess)
  │     └── infer_worker.py (独立进程, stdin/stdout JSON 通信)
  └── ProcessExporter (subprocess.Popen)
        └── export_worker.py (独立进程, 逐行协议)
```

> **为什么用子进程？** 在 Windows + RTX 4060 上，CUDA DLL 在 QThread 中加载会触发 `0xC0000005` 崩溃。子进程完全隔离 CUDA 运行时，即使训练崩溃，主界面也不受影响。

### 项目结构

```
Easy Training/
├── main.py                    # 程序入口（含 --worker 子进程路由）
├── pyproject.toml              # 项目元数据 & ruff 配置
├── requirements.txt            # 依赖
├── EasyTraining.spec           # PyInstaller 打包配置
├── EasyTraining.iss            # Inno Setup 安装器脚本
├── README.md / README.en.md    # 文档（中/英）
├── setup_icon.py               # 图标设置（任务栏/窗口）
├── pack.bat                    # 一键压缩脚本
├── .github/workflows/          # CI (ruff + py_compile + pytest)
├── core/                       # 业务逻辑层
│   ├── config.py               # 配置管理（JSON 持久化，点号路径，线程安全）
│   ├── model_manager.py        # 模型下载/验证/删除
│   ├── data_manager.py         # 数据集 CRUD / JSONL 持久化
│   ├── trainer.py              # ProcessTrainer（QProcess 训练管理）
│   ├── train_worker.py         # 独立训练子进程（PyTorch 原生循环，不依赖 trl / datasets）
│   ├── inferencer.py           # Inferencer（QProcess 推理管理）
│   ├── infer_worker.py         # 独立推理子进程（stdin JSON 通信）
│   ├── exporter.py             # 模型导出（16bit / GGUF / LoRA）
│   ├── exporter_process.py     # 导出子进程管理器（subprocess.Popen）
│   ├── ollama_deployer.py      # Ollama 部署（检测/创建/运行）
│   ├── error_handler.py        # 错误分类与格式化
│   ├── services/               # 服务工具
│   │   ├── export_service.py   # 导出路径检测
│   │   └── train_service.py    # LoRA 列表查询
│   └── workers/
│       ├── download_worker.py  # HuggingFace 下载 Worker
│       └── export_worker.py    # 独立导出子进程
├── ui/                         # UI 层
│   ├── app.py                  # 主窗口、标题栏、侧边栏、快捷键
│   ├── theme.py                # 主题管理器（深色/浅色单例）
│   ├── error_dialog.py         # 错误弹窗（UI 层）
│   ├── pages/                  # 7 个功能页面
│   │   ├── model_page.py       # 模型管理 + LoRA 管理
│   │   ├── data_page.py        # 数据管理
│   │   ├── train/              # 训练中心（拆分为 3 文件）
│   │   │   ├── train_page.py   # 训练编排
│   │   │   ├── config_panel.py # 配置面板
│   │   │   └── monitor_panel.py# 监控面板
│   │   ├── export_page.py      # 导出部署
│   │   ├── test_page.py        # 对话测试
│   │   ├── settings_page.py    # 系统设置
│   │   └── logs_page.py        # 运行日志
│   └── components/             # 可复用组件
│       ├── model_card.py       # 模型信息卡片
│       ├── loss_chart.py       # 实时 Loss 曲线（pyqtgraph）
│       ├── gpu_monitor.py      # GPU 显存监控
│       ├── data_table.py       # 可编辑数据表格
│       └── progress_bar.py     # 自定义进度条
├── utils/                      # 工具层
│   ├── i18n.py                 # Signal 驱动的国际化管理器（zh / en）
│   ├── logger.py               # 日志系统（按天切分）
│   ├── worker.py               # BaseWorker（QThread 基类 + 公共函数）
│   ├── gpu_info.py             # nvidia-smi GPU 信息查询
│   └── system_info.py          # 系统环境信息
├── locale/                     # 翻译文件
│   ├── zh.json                 # 简体中文
│   └── en.json                 # English
├── tests/                      # 单元测试
│   ├── test_config.py          # AppConfig 配置测试
│   └── test_i18n.py            # I18n 国际化测试
├── assess/                     # 导航图标 + QSS 主题
│   ├── professional_theme.qss  # 深色主题
│   ├── light_theme.qss         # 浅色主题
│   └── nav_icons.py            # 侧边栏矢量图标
├── tools/                      # CLI 工具
│   ├── convert_hf_to_gguf.py   # HF → GGUF 转换
│   ├── quantize_gguf.py        # GGUF 量化
│   ├── convert_to_training.py  # 对话数据 → 训练数据转换
│   └── extract_code.py         # API 返回提取代码块
├── res/                        # 资源
│   ├── icon.ico                # 应用图标（Windows）
│   ├── icon.png                # 应用图标
│   └── splash.png              # 启动图
└── image/                      # 截图（文档用）
```

## 下载 & 打包

### 下载编译好的安装包

前往 [GitHub Releases](https://github.com/MeZenith/Easy-Training-Model/releases) 下载 `EasyTraining_Setup.exe`，双击安装即可。

### 自行打包

```bash
# 编译 PyInstaller
pyinstaller EasyTraining.spec

# 打成安装包（需要 Inno Setup）
"C:\InnoSetup\ISCC.exe" EasyTraining.iss

# 产物在 dist\EasyTraining_Setup.exe
```

## 参与贡献

欢迎你一起来完善这个项目！无论你是修复 bug、优化 UI、改进文档还是添加新功能，都非常欢迎。

**如何参与**：
1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/你的功能`)
3. 提交你的改动 (`git commit -m '添加了某个功能'`)
4. 推送到你的分支 (`git push origin feature/你的功能`)
5. 发起 Pull Request

**开发环境**：参考上面的"快速开始"章节。运行 `ruff check .` 确保代码风格一致。

**功能建议**：如果你有一个好的想法但不会写代码，直接提 [Issue](https://github.com/MeZenith/Easy-Training-Model/issues) 讨论也可以。

## 许可证

本软件采用 **CC BY-NC 4.0** 许可证：

- **署名** — 使用、修改、再分发时须注明来源 [MeZenith/Easy-Training-Model](https://github.com/MeZenith/Easy-Training-Model)
- **非商业性** — 不得将本软件用于商业目的
- **可自由** — 复制、再分发、修改、二次创作

完整条款见 [LICENSE](LICENSE)

---

蓝隅工作室 (BlueCorner Studio) · 2026

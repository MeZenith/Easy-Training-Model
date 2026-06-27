# Easy Training

**All-in-One AI Model Fine-tuning Workstation** — Download models, prepare data, train, export, deploy, and chat test — all through a graphical interface, no coding required.

Built with PySide6 + PyTorch + PEFT (LoRA) · Subprocess-isolated CUDA · Chinese/English bilingual · Dark/Light theme

[中文](README.md) | [English]

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green)](https://pypi.org/project/PySide6/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

## Screenshots

| Training Center | Data Management | Model Management |
|:---:|:---:|:---:|
| ![](image/%E8%AE%AD%E7%BB%83%E4%B8%AD%E5%BF%83%E7%95%8C%E9%9D%A2.png) | ![](image/%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86%E7%95%8C%E9%9D%A2.png) | ![](image/%E6%A8%A1%E5%9E%8B%E7%AE%A1%E7%90%86%E7%95%8C%E9%9D%A2.png) |

| Chat Test | Export & Deploy | Runtime Logs |
|:---:|:---:|:---:|
| ![](image/%E5%AF%B9%E8%AF%9D%E6%B5%8B%E8%AF%95%E7%95%8C%E9%9D%A2.png) | ![](image/%E5%AF%BC%E5%87%BA%E9%83%A8%E7%BD%B2%E7%95%8C%E9%9D%A2.png) | ![](image/%E8%BF%90%E8%A1%8C%E6%97%A5%E5%BF%97%E7%95%8C%E9%9D%A2.png) |

| Settings |
|:---:|
| ![](image/%E8%AE%BE%E7%BD%AE%E7%95%8C%E9%9D%A2.png) |

## Who is this for

- **Full-stack developers** who want a personal AI assistant without learning deep learning
- **Enterprise teams** needing local private model fine-tuning and data privacy
- **AI beginners** learning model fine-tuning who need a visual experimentation environment
- **Algorithm engineers** who want to rapidly iterate training data and parameter effects

## Quick Start

```bash
git clone git@github.com:MeZenith/Easy-Training-Model.git
cd Easy-Training-Model
pip install -r requirements.txt
python main.py
```

**Hardware**: NVIDIA GPU, CUDA 12.4+, VRAM ≥ 8GB (1.5B model fp16 training ~4GB, 3B model ~6GB)

## Features

| Module | Description |
|--------|-------------|
| **Model Management** | 6 built-in models, HuggingFace download, local validation, deletion; manage trained LoRA adapters |
| **Data Management** | Create/import/edit datasets, JSONL/JSON/CSV support, inline table editing, auto-generate identity data, data validation |
| **Training Center** | LoRA fine-tuning, 4 presets (Quick/Standard/Fine/Custom), dual-page layout (config + real-time monitoring), Loss chart + GPU VRAM monitor |
| **Export & Deploy** | 16-bit full model (auto-merge LoRA), GGUF Q4/Q8/F16 quantization, LoRA adapter export, one-click Ollama deployment |
| **Chat Test** | Load trained models for conversation, adjustable Temperature/Top-P/Top-K/repetition penalty/presence penalty, performance statistics |
| **Settings** | Chinese/English UI, dark/light theme, workspace/HF mirror/proxy config, system/GPU info display |

### Workflow

```
1. Settings → Configure workspace and HuggingFace mirror
2. Model Management → Download base model (recommended: Qwen2.5-Coder-1.5B-Instruct)
3. Data Management → Import or create training data (Alpaca format JSONL)
4. Training Center → Select model + check datasets + set LoRA name + adjust parameters → Start training
5. Export & Deploy → Select LoRA → choose export format → save model → Ollama deploy
6. Chat Test → Select LoRA → load model → start chatting
```

## Architecture

### Tech Stack

| Category | Technology |
|----------|------------|
| UI Framework | PySide6 (Qt 6), pyqtgraph |
| Deep Learning | PyTorch 2.5+, Transformers 4.45+, PEFT (LoRA) |
| Training Engine | Subprocess isolation, pure PyTorch training loop, AMP mixed precision, gradient checkpointing, gradient accumulation |
| Model Export | HuggingFace safetensors, GGUF (llama.cpp) |
| Model Deployment | Ollama CLI |
| Internationalization | Signal-driven i18n system (zh / en) |
| Testing | pytest, ruff |

### Subprocess Architecture

```
main.py → MainWindow (Qt GUI)
  ├── ProcessTrainer (QProcess)
  │     └── train_worker.py (independent process, CUDA isolated)
  ├── Inferencer (QProcess)
  │     └── infer_worker.py (independent process, stdin/stdout JSON)
  └── ProcessExporter (subprocess.Popen)
        └── export_worker.py (independent process, line protocol)
```

> **Why subprocesses?** On Windows with RTX 4060, loading CUDA DLLs in a QThread triggers an `0xC0000005` crash. Subprocesses completely isolate the CUDA runtime — even if training crashes, the main UI stays intact.

### Project Structure

```
Easy Training/
├── main.py                    # Entry point (includes --worker subprocess routing)
├── pyproject.toml              # Project metadata & ruff config
├── requirements.txt            # Dependencies
├── EasyTraining.spec           # PyInstaller packaging config
├── EasyTraining.iss            # Inno Setup installer script
├── README.md / README.en.md    # Documentation (CN / EN)
├── setup_icon.py               # Icon setup (taskbar/window)
├── pack.bat                    # One-click compression script
├── .github/workflows/          # CI (ruff + py_compile + pytest)
├── core/                       # Business logic layer
│   ├── config.py               # Config management (JSON persistence, dot-path, thread-safe)
│   ├── model_manager.py        # Model download/validation/deletion
│   ├── data_manager.py         # Dataset CRUD / JSONL persistence
│   ├── trainer.py              # ProcessTrainer (QProcess training manager)
│   ├── train_worker.py         # Standalone training subprocess (pure PyTorch, no trl/datasets)
│   ├── inferencer.py           # Inferencer (QProcess inference manager)
│   ├── infer_worker.py         # Standalone inference subprocess (stdin JSON)
│   ├── exporter.py             # Model export (16bit / GGUF / LoRA)
│   ├── exporter_process.py     # Export subprocess manager (subprocess.Popen)
│   ├── ollama_deployer.py      # Ollama deployment (detect/create/run)
│   ├── error_handler.py        # Error classification & formatting
│   ├── services/               # Service utilities
│   │   ├── export_service.py   # Export path detection
│   │   └── train_service.py    # LoRA list query
│   └── workers/
│       ├── download_worker.py  # HuggingFace download worker
│       └── export_worker.py    # Standalone export subprocess
├── ui/                         # UI layer
│   ├── app.py                  # Main window, title bar, sidebar, shortcuts
│   ├── theme.py                # Theme manager (dark/light singleton)
│   ├── error_dialog.py         # Error dialog (UI layer)
│   ├── pages/                  # 7 functional pages
│   │   ├── model_page.py       # Model management + LoRA management
│   │   ├── data_page.py        # Data management
│   │   ├── train/              # Training center (3 files)
│   │   │   ├── train_page.py   # Training orchestration
│   │   │   ├── config_panel.py # Config panel
│   │   │   └── monitor_panel.py# Monitor panel
│   │   ├── export_page.py      # Export & deploy
│   │   ├── test_page.py        # Chat testing
│   │   ├── settings_page.py    # System settings
│   │   └── logs_page.py        # Runtime logs
│   └── components/             # Reusable components
│       ├── model_card.py       # Model info card
│       ├── loss_chart.py       # Real-time Loss chart (pyqtgraph)
│       ├── gpu_monitor.py      # GPU VRAM monitor
│       ├── data_table.py       # Editable data table
│       └── progress_bar.py     # Custom progress bar
├── utils/                      # Utility layer
│   ├── i18n.py                 # Signal-driven i18n manager (zh / en)
│   ├── logger.py               # Logging system (daily rotation)
│   ├── worker.py               # BaseWorker (QThread base + shared utilities)
│   ├── gpu_info.py             # nvidia-smi GPU info query
│   └── system_info.py          # System environment info
├── locale/                     # Translation files
│   ├── zh.json                 # Simplified Chinese
│   └── en.json                 # English
├── tests/                      # Unit tests
│   ├── test_config.py          # AppConfig tests
│   └── test_i18n.py            # I18n tests
├── assess/                     # Nav icons + QSS themes
│   ├── professional_theme.qss  # Dark theme
│   ├── light_theme.qss         # Light theme
│   └── nav_icons.py            # Sidebar vector icons
├── tools/                      # CLI tools
│   ├── convert_hf_to_gguf.py   # HF → GGUF converter
│   ├── quantize_gguf.py        # GGUF quantizer
│   ├── convert_to_training.py  # Chat data → training data converter
│   └── extract_code.py         # Code block extractor from API responses
├── res/                        # Resources
│   ├── icon.ico                # App icon (Windows)
│   ├── icon.png                # App icon
│   └── splash.png              # Splash screen
└── image/                      # Screenshots (for documentation)
```

## Download & Build

### Download Installer

Go to [GitHub Releases](https://github.com/MeZenith/Easy-Training-Model/releases) and download `EasyTraining_Setup.exe`. Double-click to install.

### Build from Source

```bash
# PyInstaller build
pyinstaller EasyTraining.spec

# Create installer (requires Inno Setup)
"C:\InnoSetup\ISCC.exe" EasyTraining.iss

# Output: dist\EasyTraining_Setup.exe
```

## License

This software is licensed under **CC BY-NC 4.0**:

- **Attribution** — Credit the source [MeZenith/Easy-Training-Model](https://github.com/MeZenith/Easy-Training-Model) when using, modifying, or redistributing
- **Non-Commercial** — May not be used for commercial purposes
- **Free** — Copy, redistribute, modify, and create derivative works

Full terms in [LICENSE](LICENSE)

---

BlueCorner Studio · 2026

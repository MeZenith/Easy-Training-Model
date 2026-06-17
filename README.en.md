# Easy Tinking

A zero-code desktop tool for fine-tuning large language models. Train your own AI model in minutes — download, prepare data, train, export, and deploy to Ollama.

Built with PySide6 + PyTorch + PEFT (LoRA).

## Features

| Module | Description |
|--------|-------------|
| Model Manager | Download models from HuggingFace, validate, delete. 6 built-in presets |
| Data Manager | Create datasets, import JSONL/JSON/CSV, generate identity data, validate |
| Training | LoRA fine-tuning with isolated subprocess. Loss curve, GPU monitor, presets |
| Export | 16-bit safetensors (LoRA merged), Ollama deployment via GGUF |
| Chat Test | Load trained model, Alpaca-format prompting, generation params |
| Settings | CN/EN i18n, dark/light themes, HF mirror, proxy, system info |

## Quick Start

```bash
pip install PySide6 pyqtgraph torch transformers peft huggingface_hub safetensors accelerate

python main.py
```

### Workflow

1. Download a base model (Qwen2.5-Coder-3B recommended)
2. Create or import training data (Alpaca format)
3. Select model + dataset → Start training
4. Export 16-bit model → Deploy to Ollama
5. Chat with your model in the Test page

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4+ (8GB+ VRAM)
- Windows 10/11

## Architecture

```
main.py (Qt event loop)
  └── ProcessTrainer (QProcess)
        └── train_worker.py (isolated subprocess)
              ├── Load model + LoRA
              ├── Pure PyTorch training loop
              ├── AMP + gradient checkpointing
              └── Save weights + metadata
```

Training runs in a subprocess to isolate CUDA initialization from the Qt event loop — avoiding `0xC0000005` crashes on Windows.

## Project Structure

```
core/          Business logic (config, model, data, training, export, ollama)
ui/            UI components and pages
utils/         Utilities (i18n, logging, GPU info, worker base class)
locale/        Translation files (zh.json, en.json)
tools/         Conversion utilities (HF → GGUF)
res/           Application icons
workspace/     User data (models, datasets, training outputs)
```

## Known Issues

### Fixed
| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Training crash (0xC0000005) | CUDA DLL init in QThread | Subprocess isolation |
| Model outputs template garbage | Alpaca format mismatch | Auto-stop on `###` markers |
| Data import BOM error | utf-8 vs utf-8-sig on Windows | utf-8-sig encoding |
| Model doesn't respond in chat | `_on_gen_progress` didn't write to display | Direct output + truncation |
| Loss chart empty | No data fed to chart | Parse subprocess LOG for loss |
| Dataset checkboxes missing | Train page didn't load data list | QListWidget with checkboxes |
| Preset switching broken | Combo index out of sync | blockSignals + setCurrentIndex |
| Delete button invisible | Unicode glyph not available | X + red bordered QSS |
| Light theme log text white | Inline stylesheet overrides QSS | Removed inline style |
| Preset questions in English | Hardcoded strings | Changed to i18n keys |
| Sidebar collapse misaligned | 48px too narrow | 56px + icon padding |

### Limitations
| Limitation | Reason |
|-----------|--------|
| GGUF export unavailable | unsloth incompatible with PyTorch 2.5.1 |
| Ollama safetensors import garbled | Ollama only fully supports GGUF |
| Taskbar icon not shown | python.exe icon in dev, needs PyInstaller |
| Window rounded corners incomplete | Frameless window needs DWM API |
| Multi-turn chat not supported | Single-turn Alpaca format only |
| Single GPU only | No distributed training |

### To Improve
- Training page advanced params could use collapsible groups
- Training result panel could show more fields (learning rate, etc.)
- Model page path labels could be clearer

## License

Apache 2.0 — BlueCorner Studio

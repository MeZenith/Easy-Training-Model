# -*- mode: python ; coding: utf-8 -*-

import os
BASE = r'D:\python\Easy Training'

import huggingface_hub
HF_HUB_DIR = os.path.dirname(huggingface_hub.__file__)

a = Analysis(
    ['main.py'],
    pathex=[BASE],
    binaries=[],
    datas=[
        (f'{BASE}/locale/zh.json', 'locale'),
        (f'{BASE}/locale/en.json', 'locale'),
        (f'{BASE}/res/icon.ico', 'res'),
        (f'{BASE}/res/icon.png', 'res'),
        (f'{BASE}/res/splash.png', 'res'),
        (f'{BASE}/core/train_worker.py', 'core'),
        (f'{BASE}/core/infer_worker.py', 'core'),
        (f'{BASE}/core/workers/export_worker.py', 'core/workers'),
        (f'{BASE}/assess/professional_theme.qss', 'assess'),
        (f'{BASE}/assess/light_theme.qss', 'assess'),
        (f'{HF_HUB_DIR}/templates', 'huggingface_hub/templates'),
    ],
    hiddenimports=[
        'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        'pyqtgraph',
        'torch', 'transformers', 'peft', 'huggingface_hub',
        'safetensors', 'accelerate', 'bitsandbytes',
        'json', 'logging', 'argparse', 're',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter','matplotlib','notebook','datasets','trl','diffusers','scipy','sklearn','pandas','pytest'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='EasyTraining',
    debug=False,
    strip=True,
    upx=True,
    console=False,
    icon=f'{BASE}/res/icon.ico',
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=True,
    upx=True,
    name='EasyTraining',
)

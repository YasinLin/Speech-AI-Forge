# -*- mode: python ; coding: utf-8 -*-
import whisper,funasr,wandb,lightning_fabric,sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all,copy_metadata

hiddenimports = []
REPO_DIR = lambda name: os.path.abspath(os.path.join('./modules/repos_static', name))

paths = [
    REPO_DIR("cosyvoice"),
    REPO_DIR("openvoice"),
    REPO_DIR("fish_speech"),
    REPO_DIR("FireRedTTS"),
    REPO_DIR("F5TTS"),
]

def setup_repos_paths():
    for pth in paths:
        if pth not in sys.path:
            sys.path.insert(0, pth)

setup_repos_paths()
alltorch = collect_all('torch')
allx_transformers = collect_all('x_transformers')
allopenvoice = collect_all('openvoice')
alltensorrt = collect_all('tensorrt')
alltensorrt_libs = collect_all('tensorrt_libs')
datas = collect_data_files('wandb', include_py_files=True, includes=['**/vendor/**/*.py'])
binaries=[
         # CUDA 运行时库（根据实际 CUDA 版本调整路径）
      ('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\*.dll', '.'),
      (os.path.join(os.path.dirname(funasr.__file__), 'version.txt'), 'funasr'),
      (os.path.join(os.path.dirname(lightning_fabric.__file__), 'version.info'), 'lightning_fabric'),
      (os.path.join(os.path.dirname(whisper.__file__), 'assets', '*'), 'whisper/assets'),
      ('./modules', 'modules')
    ]

datas+=[("./modules/repos_static/ChatTTS/ChatTTS/res/homophones_map.json", 'modules/repos_static/ChatTTS/ChatTTS/res')]
#binaries+=[("./modules/repos_static", '.')]
#datas+=[("./modules", 'modules')]
#datas+=[('ffmpeg/*', 'ffmpeg')]
#datas+=[("./models", 'models')]
#datas+=[("./data", 'data')]
#datas+=[("./playground", 'playground')]
datas+=alltorch[0]
binaries+=alltorch[1]
hiddenimports+=alltorch[2]
datas+=allx_transformers[0]
binaries+=allx_transformers[1]
hiddenimports+=allx_transformers[2]
datas+=allopenvoice[0]
binaries+=allopenvoice[1]
hiddenimports+=allopenvoice[2]
datas+=alltensorrt[0]
binaries+=alltensorrt[1]
hiddenimports+=alltensorrt[2]
datas+=alltensorrt_libs[0]
binaries+=alltensorrt_libs[1]
hiddenimports+=alltensorrt_libs[2]

hiddenimports += ['ChatTTS', 'fish_speech', 'cosyvoice', 'cosyvoice.llm', 'cosyvoice.flow', 'cosyvoice.hifigan', 'cosyvoice.tokenizer', 'fireredtts', 'f5_tts', 'tools', 'rich', 'natsort', 'eng_to_ipa', 'unidecode', 'launch', 'matcha', 'modules.api.worker']
hiddenimports += collect_submodules('ChatTTS')
hiddenimports += collect_submodules('fish_speech')
hiddenimports += collect_submodules('tools')
hiddenimports += collect_submodules('cosyvoice')
hiddenimports += collect_submodules('cosyvoice.llm')
hiddenimports += collect_submodules('cosyvoice.flow')
hiddenimports += collect_submodules('cosyvoice.hifigan')
hiddenimports += collect_submodules('cosyvoice.tokenizer')
hiddenimports += collect_submodules('fireredtts')
hiddenimports += collect_submodules('f5_tts')
hiddenimports += collect_submodules('rich')
hiddenimports += collect_submodules('matcha')
hiddenimports += collect_submodules('modules')

block_cipher = None

a = Analysis(
    ['launch.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={
        'inflect': 'pyz+py',
    },
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='launch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='launch',
)

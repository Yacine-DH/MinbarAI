# PyInstaller spec — builds dist/MinbarAI/MinbarAI.exe (cloud edition client)
# Build:  python -m PyInstaller MinbarAI.spec --noconfirm
#
# The exe expects a `.env` next to it (ELEVENLABS_API_KEY,
# TRANSLATE_SERVER_URL). First run downloads the Silero VAD (~2 MB) into
# the user cache; internet is required anyway (cloud edition).

from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["src/ui.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=[
        "sounddevice",
        "torchaudio",  # silero-vad hub code touches it
    ]
    + collect_submodules("elevenlabs"),
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # server-side / dev-only heavyweights that must not ride along
        "transformers",
        "sentence_transformers",
        "sacrebleu",
        "sklearn",
        "scipy",
        "modal",
        "PyInstaller",
        "pytest",
        "matplotlib",
        "pandas",
        "IPython",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="MinbarAI",
    debug=False,
    console=False,   # windowed app — logs are not shown
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MinbarAI",
)

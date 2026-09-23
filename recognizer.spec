# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置 (完全可移植: 所有路径基于本 spec 文件所在目录 SPECPATH)
# 用法: 先按 README 装好依赖, 然后运行: build.bat 或 pyinstaller --clean --noconfirm recognizer.spec
import os

def has_model(folder):
    """模型目录里是否有可用权重 (safetensors 优先, 兼容旧的 .bin)"""
    return (os.path.exists(os.path.join(folder, "model.safetensors"))
            or os.path.exists(os.path.join(folder, "pytorch_model.bin")))

# 模型目录: 优先用 fp16 (体积小), 没有就用原版
CLIP_SRC = os.path.join(SPECPATH, "clip_model_fp16")
if not has_model(CLIP_SRC):
    CLIP_SRC = os.path.join(SPECPATH, "clip_model")
if not has_model(CLIP_SRC):
    raise SystemExit("找不到 CLIP 模型: 请先运行 python download_clip.py "
                     "(可选再运行 python convert_fp16.py 以减半体积)")

datas = [
    (os.path.join(SPECPATH, "static"), "static"),
    (CLIP_SRC, "clip_model"),
]

hiddenimports = [
    "ultralytics", "ultralytics.nn", "ultralytics.nn.modules", "ultralytics.engine",
    "ultralytics.engine.model", "ultralytics.utils", "ultralytics.data",
    "transformers", "transformers.models.clip", "transformers.models.clip.modeling_clip",
    "cv2", "PIL", "PIL.Image", "yaml", "pandas", "numpy", "flask", "jinja2", "werkzeug",
    "torch", "torchvision", "matplotlib.backends.backend_agg",
]

a = Analysis(
    [os.path.join(SPECPATH, "app_entry.py")],
    pathex=[SPECPATH],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PySide2", "PySide6", "IPython"],
    noarchive=False,
)

# 剔除第三方素材: ultralytics 自带的示例图片 (bus.jpg / zidane.jpg 等)
# 这些是 ultralytics 仓库的演示用图 (含真实人物照片), 程序运行时完全用不到。
def _keep(entry):
    dest = str(entry[0]).replace("\\", "/").lower()
    return not (dest.startswith("ultralytics/assets")
                or "/assets/" in dest and dest.endswith((".jpg", ".jpeg", ".png")))

a.datas = [d for d in a.datas if _keep(d)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="CharacterRecognizer",
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CharacterRecognizer",
)

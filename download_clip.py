# -*- coding: utf-8 -*-
"""
下载 CLIP 模型 (openai/clip-vit-base-patch32) 到本目录的 clip_model 文件夹。
国内网络请把 BASE 换成镜像 (如 https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/)。
用法: python download_clip.py

安全说明 (重要):
1. 下载后会校验大文件的 SHA-256, 与官方值比对, 不一致会给出警告。
2. 校验通过后会**转换成 safetensors 格式并删除原始 .bin 文件**。
   原因: .bin 是 Python pickle 格式, 加载它等于执行文件里的代码;
         safetensors 只是纯张量数据, 无法携带代码, 从根本上避免"模型文件被替换 = 被植入后门"。
         (transformers 会优先加载 model.safetensors。)
3. 请只从官方 HuggingFace 站点或可信镜像下载模型文件。
"""
import urllib.request, os, hashlib

BASE = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_model")
SMALL_FILES = ["config.json", "preprocessor_config.json"]
BIN_FILE = "pytorch_model.bin"

# 官方 openai/clip-vit-base-patch32 pytorch_model.bin 的 SHA-256
EXPECTED_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def download(name, dest):
    print("下载:", name)
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if name.endswith(".bin") and total % (64 << 20) < (1 << 20):
                print("  %.0f MB" % (total / 1e6))
    print("  完成: %.1f MB" % (os.path.getsize(dest) / 1e6))

def convert_to_safetensors(folder):
    """把 pytorch_model.bin 转成 model.safetensors 并删除 .bin"""
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("  (未安装 torch/safetensors, 跳过格式转换)")
        return
    src = os.path.join(folder, BIN_FILE)
    dst = os.path.join(folder, "model.safetensors")
    if not os.path.exists(src):
        return
    print("转换为 safetensors 格式 (消除 pickle 加载风险)...")
    try:
        sd = torch.load(src, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(src, map_location="cpu")
    sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(sd, dst, metadata={"format": "pt", "source": "converted-from-pytorch_model.bin"})
    os.remove(src)
    print("  已生成 model.safetensors 并删除 pytorch_model.bin")

def main():
    os.makedirs(DEST, exist_ok=True)
    if os.path.exists(os.path.join(DEST, "model.safetensors")):
        print("已存在 model.safetensors, 跳过下载。模型位置:", DEST)
        return
    for f in SMALL_FILES:
        d = os.path.join(DEST, f)
        if os.path.exists(d) and os.path.getsize(d) > 100:
            print("已存在, 跳过:", f)
        else:
            download(f, d)

    bin_path = os.path.join(DEST, BIN_FILE)
    if not os.path.exists(bin_path):
        download(BIN_FILE, bin_path)
        print("校验文件完整性 (SHA-256)...")
        got = sha256_of(bin_path)
        if got == EXPECTED_SHA256:
            print("  ✅ 校验通过, 与官方一致")
            convert_to_safetensors(DEST)
        else:
            print("  ⚠️ 校验不一致! 已保留文件但未转换。")
            print("     期望:", EXPECTED_SHA256)
            print("     实际:", got)
            print("     建议: 删除 clip_model 目录后重新运行本脚本。")
    else:
        convert_to_safetensors(DEST)
    print("完成。模型位置:", DEST)

if __name__ == "__main__":
    main()

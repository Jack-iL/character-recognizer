# -*- coding: utf-8 -*-
"""
把 clip_model 转成 fp16 副本 (clip_model_fp16), 打包/分发时体积减半 (605MB -> 303MB)。
用法: python convert_fp16.py

安全说明:
- 以 weights_only=True 读取权重, 只加载张量数据, 不会执行文件里的代码。
- 输出为 safetensors 格式 (纯张量容器, 无法携带可执行代码), 而不是 pickle 格式的 .bin。
"""
import os, shutil
import torch

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clip_model")
DST = os.path.join(HERE, "clip_model_fp16")

def load_state_dict(folder):
    """优先读取 safetensors; 否则读取 .bin (限量加载)"""
    st = os.path.join(folder, "model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        print("  读取 model.safetensors (安全格式)")
        return load_file(st)
    binf = os.path.join(folder, "pytorch_model.bin")
    print("  读取 pytorch_model.bin")
    try:
        return torch.load(binf, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(binf, map_location="cpu")

def main():
    os.makedirs(DST, exist_ok=True)
    print("加载模型...")
    sd = load_state_dict(SRC)
    print("转 fp16...")
    sd16 = {k: v.half().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    if save_file is None:
        raise SystemExit("需要 safetensors 库: pip install safetensors")
    out = os.path.join(DST, "model.safetensors")
    save_file(sd16, out, metadata={"format": "pt", "dtype": "fp16"})
    for f in ("config.json", "preprocessor_config.json"):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
    print("已生成: %s (%.0f MB, safetensors 格式)" % (out, os.path.getsize(out) / 1e6))

if __name__ == "__main__":
    main()

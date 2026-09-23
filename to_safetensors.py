# -*- coding: utf-8 -*-
"""
把已下载的 CLIP 模型转换/加固为 safetensors 格式。

为什么: pytorch_model.bin 是 Python pickle 格式, 加载它等同于执行文件里的代码,
        一旦文件被替换/篡改, 就可能在你的电脑上执行任意代码。
        safetensors 只是纯粹的张量数据容器, 无法携带代码, 从根本上消除这个风险。
        transformers 在同时存在两种格式时, 会优先加载 model.safetensors。

用法: python to_safetensors.py            # 处理 clip_model 与 clip_model_fp16
"""
import os, sys
import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("需要 safetensors 库: pip install safetensors")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))

def convert(folder, delete_bin=True):
    src = os.path.join(folder, "pytorch_model.bin")
    dst = os.path.join(folder, "model.safetensors")
    if not os.path.exists(src):
        if os.path.exists(dst):
            print("  %s: 已是 safetensors 格式, 跳过" % os.path.basename(folder))
        else:
            print("  %s: 没有 pytorch_model.bin, 跳过" % os.path.basename(folder))
        return
    print("  %s: 读取 pytorch_model.bin ..." % os.path.basename(folder))
    try:
        sd = torch.load(src, map_location="cpu", weights_only=True)
    except TypeError:
        print("    (当前 torch 不支持 weights_only, 使用默认方式)")
        sd = torch.load(src, map_location="cpu")
    # safetensors 只接受连续张量
    sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    print("    %d 个张量, 写出 model.safetensors ..." % len(sd))
    save_file(sd, dst, metadata={"format": "pt", "source": "converted-from-pytorch_model.bin"})
    print("    完成: %.1f MB" % (os.path.getsize(dst) / 1e6))
    if delete_bin:
        os.remove(src)
        print("    已删除 pytorch_model.bin (消除 pickle 加载风险)")

def main():
    for name in ("clip_model", "clip_model_fp16"):
        folder = os.path.join(HERE, name)
        if os.path.isdir(folder):
            convert(folder)
    print("全部完成。")

if __name__ == "__main__":
    main()

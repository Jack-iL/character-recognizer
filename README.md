# Character Recognizer

本地运行的目标识别程序：**YOLO 找框 + CLIP 认类**，网页界面，中英双语，数据不出本机。

Made by NEU~Jack Li · 东北大学-李仲洲制作

## 功能

- 任意 YOLO 模型 (用户自己训练/提供) + 任意 CLIP 参考集 (每个子文件夹 = 一个类)
- 排除参考集：匹配的框自动剔除
- 网页界面：拖拽上传图片、下拉改类、删框、一键保存 (标注图 + CSV)
- 中英双语，进入时选择语言
- 关闭网页约 3 秒后程序自动退出
- 完全离线运行 (不联网、不上传任何数据)
- 可打包成单文件夹 exe 分享给他人 (无需 Python)

## 安装

```bash
# 1. Python 3.10 环境 (建议虚拟环境)
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. 安装依赖 (CPU 版 torch)
pip install -r requirements.txt
# 国内网络: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 下载 CLIP 模型 (约 600MB, 自动校验 SHA-256 并转换为 safetensors 格式)
python download_clip.py
# 国内网络: 把 download_clip.py 里的 BASE 改成 https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/
# 已有旧版 .bin 模型? 运行 python to_safetensors.py 转换 (消除 pickle 加载风险)
```

## 使用

1. 准备三样东西：
   - **YOLO 模型**：训练好的 `.pt` 文件
   - **图片**：要识别的图片文件夹
   - **参考集**：一个文件夹，每个子文件夹 = 一个类（类名用英文），里面放该类例图 10~20 张
     （仓库 `reference/` 已给出目录骨架，把例图放进去即可）
   - *可选* **排除集**：仓库 `exclude/` 用于放"不要的东西"（直接放图片即可），
     匹配上的框会被自动排除、不进最终结果
2. 启动：双击 `start.bat`（无窗口版：`start_hidden.vbs`；或直接 `python webapp.py`）
3. 浏览器自动打开 `http://127.0.0.1:8000`，按页面提示操作
4. 如果路径填错，页面会用中文/英文详细告诉你**错在哪、怎么改**

## 打包成 exe

```bash
pip install pyinstaller
# 可选: 模型减半 (605MB -> 303MB)
python convert_fp16.py
build.bat                       # 或: pyinstaller --clean --noconfirm recognizer.spec
build_release.bat               # 发布级: 从中性路径构建, exe 内不含本机用户名/路径
```

产物在 `dist\CharacterRecognizer\`，整个文件夹打包发给别人即可（对方无需 Python）。
**注意：打包用 CPU 版 torch，收件人无需显卡。**

分享前请删除产物目录里的运行期文件：`web_config.json`、`debug_log.txt`、`hf_cache/`。

## 隐私

- 图片、模型、结果全部在本地处理，无任何联网上传
- 程序只监听 `127.0.0.1`，局域网/互联网无法访问
- 已加防护：
  - 只接受本机 Host 请求头（防 DNS 重绑定：恶意网站借域名偷读本机服务）
  - 所有写操作要求自定义请求头（防跨站请求伪造）
  - `/file` 只能读取**你自己选定的图片/输出文件夹**里的图片文件（不能读磁盘上其它位置）
- 模型以 **safetensors** 格式加载（纯张量容器，无法携带可执行代码）
- 已禁用 ultralytics 的更新检查/遥测 (`YOLO_OFFLINE`)

## 安全提示

- **只加载你自己信任的 `.pt` 模型。** PyTorch 权重文件本质上可以携带可执行代码，
  加载来路不明的模型文件等同于运行来路不明的程序。
- 请只从官方站点或可信镜像下载模型文件；`download_clip.py` 会校验大文件的 SHA-256 并给出警告。
- `mini_pip.py` 是网络受限时的应急安装器，**不校验哈希、不处理依赖**，正式环境请用 pip。

## 关于仓库内容

本仓库**只包含程序源码与说明文档**，不含任何图片、模型或数据。`.gitignore` 已默认忽略：

- 模型文件（`*.pt` / `*.pth` / `*.onnx` …）
- 所有图片（`*.png` / `*.jpg` …）——避免误传个人素材或涉及第三方版权的图片
- 输出结果（`*.csv` / `*.xlsx`）、运行日志、缓存与打包产物

如果你确实需要把某张图片纳入版本管理，请显式强制添加：`git add -f 图片路径`。

## 许可证

代码: MIT (见 LICENSE)

第三方组件许可:
- [ultralytics](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- [transformers](https://github.com/huggingface/transformers) — Apache-2.0
- [PyTorch](https://pytorch.org) — BSD-3-Clause
- CLIP 权重 [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) — MIT
- OpenCV — Apache-2.0

注意：`reference/` 中的角色形象可能受版权保护，本仓库不包含任何参考图；
请自行准备并遵守相关法律法规。

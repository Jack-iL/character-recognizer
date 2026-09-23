# -*- coding: utf-8 -*-
"""
识别程序 - 网页版后端
启动后浏览器打开 http://127.0.0.1:8000 即可使用
"""
import sys, os, glob, csv, json, time
from collections import defaultdict

_DEV_BASE = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, "frozen", False):
    # 开发模式才加载本地库; 打包版不需要 (路径用相对方式, 不泄露机器信息)
    sys.path.insert(0, os.path.join(_DEV_BASE, "pylibs310"))
os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("WEBAPP_BASE", _DEV_BASE), "hf_cache"))
# 完全离线运行: 禁止 ultralytics 的更新检查/遥测/字体下载等一切联网行为
os.environ.setdefault("YOLO_OFFLINE", "true")

# pythonw (无窗口) 模式下没有 stdout/stderr: 写进日志文件方便排查
if sys.stdout is None or sys.stderr is None:
    _log = open(os.path.join(os.environ.get("WEBAPP_BASE", _DEV_BASE), "debug_log.txt"),
                "a", encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPVisionModelWithProjection
from ultralytics import YOLO
from flask import Flask, request, jsonify, Response

# (YOLO_OFFLINE 环境变量已在上方设置, 足够彻底离线; 不再写用户配置目录)

# 打包(frozen)时: BASE = exe 所在目录(可写), RES = 资源目录(_MEIPASS)
if getattr(sys, "frozen", False):
    BASE = os.environ.get("WEBAPP_BASE") or os.path.dirname(sys.executable)
    RES  = sys._MEIPASS
else:
    BASE = _DEV_BASE
    RES  = _DEV_BASE

CLIP_DIR    = os.path.join(RES, "clip_model")
INPUT_DIR   = os.path.join(BASE, "input_images")
OUT_DIR     = os.path.join(BASE, "output_results")
CONFIG_PATH = os.path.join(BASE, "config.txt")
CONF        = 0.25
EXCL_MARGIN = 0.02   # 排除集相似度要高出这个余量才排除
IMG_EXTS    = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
COLOR_PALETTE = [
    (0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255),
    (128, 0, 255), (0, 255, 128), (255, 128, 0), (0, 128, 128),
]
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".bmp": "image/bmp", ".webp": "image/webp"}

app = Flask(__name__, static_folder=os.path.join(RES, "static"))
WEB_CFG = os.path.join(BASE, "web_config.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
_YOLO = {"path": None, "model": None}
_CLIP = {"model": None, "prep": None}

# ==================== 安全加固 ====================
# 1) 防 DNS 重绑定: 只接受本机 Host 头 (恶意网站借域名重绑定偷读本机服务的经典手法)
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

# 3) 文件读取白名单: /file 只允许读取"用户自己选定的文件夹"里的图片,
#    避免本机其它程序/扩展借该接口探测磁盘上的私人图片
ALLOWED_FILE_DIRS = set()

def _path_allowed(path):
    try:
        p = os.path.abspath(path)
    except Exception:
        return False
    for d in list(ALLOWED_FILE_DIRS):
        try:
            if os.path.commonpath([p, os.path.abspath(d)]) == os.path.abspath(d):
                return True
        except ValueError:
            continue
    return False

@app.before_request
def _guard_host():
    host = (request.host or "").split(":")[0].lower()
    if host not in ALLOWED_HOSTS:
        return Response("forbidden", status=403)

# 2) 防跨站请求伪造: 所有修改类请求必须带自定义头
#    (跨站表单/脚本无法伪造自定义头, 会被浏览器预检拦截)
def _csrf_ok():
    return request.headers.get("X-Requested-With") == "CharacterRecognizer"

@app.before_request
def _guard_write():
    if request.method in ("POST", "PUT", "DELETE") and not _csrf_ok():
        return Response("forbidden", status=403)
# ===================================================

def load_webcfg():
    if os.path.exists(WEB_CFG):
        try:
            return json.load(open(WEB_CFG, encoding="utf-8"))
        except Exception:
            pass
    return {"defaults": {}, "last": {}}

def save_webcfg(cfg):
    json.dump(cfg, open(WEB_CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def load_config():
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        raw = open(CONFIG_PATH, "rb").read()
        text = ""
        for enc in ("utf-8", "gbk"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        for line in text.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write("%s=%s\n" % (k, v))

def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_u(p, img):
    cv2.imencode(".png", img)[1].tofile(p)

def load_classes(ref_dir):
    classes = []
    if not os.path.isdir(ref_dir):
        return classes
    for name in sorted(os.listdir(ref_dir)):
        d = os.path.join(ref_dir, name)
        if os.path.isdir(d):
            if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(d)):
                classes.append(name)
    return classes

def get_yolo(path):
    if _YOLO["path"] != path:
        _YOLO["model"] = YOLO(path)
        _YOLO["path"] = path
    return _YOLO["model"]

def get_clip():
    if _CLIP["model"] is None:
        _CLIP["model"] = CLIPVisionModelWithProjection.from_pretrained(CLIP_DIR).to(device).eval()
        _CLIP["prep"] = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    return _CLIP["model"], _CLIP["prep"]

def build_proto(ref_dir, classes):
    clip, prep = get_clip()
    ref_paths, ref_y = [], []
    for ci, c in enumerate(classes):
        for ext in IMG_EXTS:
            for f in glob.glob(os.path.join(ref_dir, c, "*" + ext)):
                ref_paths.append(f); ref_y.append(ci)
    if not ref_paths:
        return None
    ref_y = np.array(ref_y)

    @torch.no_grad()
    def embed(imgs, batch=64):
        feats = []
        for i in range(0, len(imgs), batch):
            x = torch.stack([prep(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in imgs[i:i+batch]]).to(device)
            f = clip(x).image_embeds
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats)

    rf = embed([imread_u(p) for p in ref_paths])
    proto = torch.stack([rf[ref_y == k].mean(0) for k in range(len(classes))])
    proto = proto / proto.norm(dim=-1, keepdim=True)
    return proto

def render_boxes(rows, out_dir, input_dir, classes, with_number, lang="zh"):
    os.makedirs(out_dir, exist_ok=True)
    colors = {c: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, c in enumerate(classes)}
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    out_paths = []
    excl_word = "excluded" if lang == "en" else "排除"
    for name, rs in by_img.items():
        stem = os.path.splitext(name)[0]
        src = None
        for ext in IMG_EXTS:
            p = os.path.join(input_dir, stem + ext)
            if os.path.exists(p):
                src = p
                break
        img = imread_u(src) if src else None
        if img is None:
            continue
        for r in rs:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            if r.get("class") == "excluded":
                col = (128, 128, 128)
                label = ("#%d " % r["num"] if with_number else "") + "%s %.0f%%" % (excl_word, float(r["conf"]) * 100)
            else:
                col = colors.get(r["class"], (255, 255, 255))
                label = ("#%d " % r["num"] if with_number else "") + "%s %.0f%%" % (r["class"], float(r["conf"]) * 100)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        out = os.path.join(out_dir, stem + "_annotated.png")
        imwrite_u(out, img)
        out_paths.append(out)
    return out_paths

# ---------------- 页面 ----------------

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/file")
def file_serve():
    p = request.args.get("path", "")
    ext = os.path.splitext(p)[1].lower()
    if ext not in IMG_EXTS:
        return "not allowed", 403
    if not p or not os.path.isfile(p):
        return "not found", 404
    if not _path_allowed(p):
        return "forbidden", 403
    return Response(open(p, "rb").read(), mimetype=MIME.get(ext, "application/octet-stream"))

@app.route("/api/defaults")
def api_defaults():
    cfg = load_webcfg()
    return jsonify({
        "defaults": cfg.get("defaults", {}),
        "last": cfg.get("last", {}),
        "device": device,
        "app_images": INPUT_DIR,
        "app_output": OUT_DIR,
    })

@app.route("/api/settings", methods=["POST"])
def api_settings():
    d = request.json or {}
    cfg = load_webcfg()
    if "defaults" in d:
        cfg["defaults"] = d["defaults"]
    if "last" in d:
        cfg["last"] = d["last"]
    save_webcfg(cfg)
    return jsonify({"ok": True})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    folder = request.form.get("folder", INPUT_DIR)
    os.makedirs(folder, exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        if f and f.filename:
            name = os.path.basename(f.filename)
            if name.lower().endswith(IMG_EXTS):
                f.save(os.path.join(folder, name))
                saved.append(name)
    return jsonify({"ok": True, "saved": saved})

@app.route("/api/close", methods=["POST"])
def api_close():
    """页面正在关闭时由前端发送; 3 秒内没有心跳确认就退出程序
       (刷新页面会在 3 秒内重新心跳, 从而自动取消退出)"""
    global CLOSE_AT
    CLOSE_AT = time.time() + 3.0
    return jsonify({"ok": True})

@app.route("/api/ready")
def api_ready():
    """仅用于启动时探测"服务是否就绪"; 刻意不碰看门狗状态"""
    return jsonify({"ready": True})

@app.route("/api/ping")
def api_ping():
    try:
        global PING_LAST, CLOSE_AT
        PING_LAST = time.time()
        CLOSE_AT = 0.0          # 页面还活着 -> 取消待退出
        PING_STARTED.set()
        return jsonify({"ok": True})
    except Exception:
        import traceback
        traceback.print_exc()
        raise

def collect_images(folder):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, "**", "*" + ext), recursive=True)
    return sorted(files)


def check_inputs(yolo_path, images_dir, ref_dir, out_dir):
    """逐项体检输入; 返回 (错误字典 或 None, 提醒列表)
    错误字典: {"error_code":..., "params":{...}} —— 前端据此显示"问题/原因/怎么改"
    """
    # ---- YOLO 模型 ----
    if not yolo_path:
        return {"error_code": "EMPTY_YOLO"}, []
    if not os.path.exists(yolo_path):
        return {"error_code": "YOLO_NOT_FOUND", "params": {"path": yolo_path}}, []
    if os.path.isdir(yolo_path):
        return {"error_code": "YOLO_IS_DIR", "params": {"path": yolo_path}}, []
    if not yolo_path.lower().endswith((".pt", ".onnx")):
        return {"error_code": "YOLO_BAD_EXT", "params": {"path": yolo_path}}, []

    # ---- 图片文件夹 ----
    if not os.path.isdir(images_dir):
        return {"error_code": "IMG_DIR_NOT_FOUND", "params": {"path": images_dir}}, []
    imgs = collect_images(images_dir)
    if not imgs:
        try:
            items = os.listdir(images_dir)[:5]
        except Exception:
            items = []
        return {"error_code": "NO_IMAGES",
                "params": {"path": images_dir, "items": "、".join(items) if items else "（空文件夹）"}}, []

    # ---- 参考集 ----
    if not ref_dir:
        return {"error_code": "EMPTY_REFSET"}, []
    if not os.path.isdir(ref_dir):
        return {"error_code": "REF_DIR_NOT_FOUND", "params": {"path": ref_dir}}, []
    classes, empty_classes = [], []
    class_counts = {}
    files_at_root = []
    for name in sorted(os.listdir(ref_dir)):
        p = os.path.join(ref_dir, name)
        if os.path.isdir(p):
            n = len(collect_images(p))
            if n:
                classes.append(name)
                class_counts[name] = n
            else:
                empty_classes.append(name)
        else:
            files_at_root.append(name)
    if not classes:
        if empty_classes:
            return {"error_code": "REF_ALL_EMPTY",
                    "params": {"path": ref_dir, "classes": "、".join(empty_classes)}}, []
        if files_at_root:
            return {"error_code": "REF_FILES_NOT_FOLDERS",
                    "params": {"path": ref_dir, "items": "、".join(files_at_root[:5])}}, []
        return {"error_code": "REF_NO_CLASSES", "params": {"path": ref_dir}}, []

    warnings = []
    if empty_classes:
        warnings.append({"code": "WARN_EMPTY_CLASS", "params": {"classes": "、".join(empty_classes)}})
    few = [c for c in classes if class_counts[c] < 5]
    if few:
        warnings.append({"code": "WARN_FEW_IMAGES",
                         "params": {"classes": "、".join("%s(%d张)" % (c, class_counts[c]) for c in few)}})

    # ---- 输出文件夹 ----
    try:
        os.makedirs(out_dir, exist_ok=True)
        probe = os.path.join(out_dir, "_write_test.tmp")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
    except Exception as e:
        return {"error_code": "OUT_DIR_FAILED",
                "params": {"path": out_dir, "reason": str(e)[:120]}}, []

    return None, warnings


@app.route("/api/run", methods=["POST"])
def api_run():
    d = request.json or {}
    yolo_path = (d.get("yolo_model") or "").strip().strip('"').strip("'")
    images_dir = (d.get("images_folder") or "").strip().strip('"').strip("'") or INPUT_DIR
    ref_dir = (d.get("refset") or "").strip().strip('"').strip("'")
    out_dir = (d.get("output_folder") or "").strip().strip('"').strip("'") or OUT_DIR
    excl_dir = (d.get("exclset") or "").strip().strip('"').strip("'")

    err, warnings = check_inputs(yolo_path, images_dir, ref_dir, out_dir)
    if err:
        return jsonify(err)

    # 允许 /file 读取这两个目录 (页面只显示输出目录里的标注图)
    ALLOWED_FILE_DIRS.clear()
    ALLOWED_FILE_DIRS.update({os.path.abspath(out_dir), os.path.abspath(images_dir)})

    classes = load_classes(ref_dir)

    # 记住这次输入, 供"一键填入上次输入"使用
    try:
        cfg = load_webcfg()
        cfg["last"] = {"yolo_model": yolo_path, "images_folder": images_dir,
                       "refset": ref_dir, "output_folder": out_dir, "exclset": excl_dir}
        save_webcfg(cfg)
    except Exception:
        pass

    try:
        try:
            yolo = get_yolo(yolo_path)
        except Exception as e:
            return jsonify({"error_code": "YOLO_LOAD_FAILED",
                            "params": {"path": yolo_path, "reason": str(e)[:200]}})
        proto = build_proto(ref_dir, classes)
        if proto is None:
            return jsonify({"error_code": "NO_REF_IMAGES", "params": {"path": ref_dir}})

        imgs_files = collect_images(images_dir)

        results = yolo.predict(source=imgs_files, conf=CONF, verbose=False)
        clip, prep = get_clip()

        @torch.no_grad()
        def embed(imgs, batch=64):
            feats = []
            for i in range(0, len(imgs), batch):
                x = torch.stack([prep(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in imgs[i:i+batch]]).to(device)
                f = clip(x).image_embeds
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
            return torch.cat(feats)

        rows = []
        # 排除集: 收集所有图片(含子文件夹), 逐个比
        excl_feats = None
        excl_count = 0
        if excl_dir and os.path.isdir(excl_dir):
            excl_files = []
            for ext in IMG_EXTS:
                excl_files += glob.glob(os.path.join(excl_dir, "**", "*" + ext), recursive=True)
            if excl_files:
                excl_feats = embed([imread_u(p) for p in excl_files])
                excl_count = len(excl_files)

        for img_file, res in zip(imgs_files, results):
            img = imread_u(img_file)
            if img is None:
                continue
            n = len(res.boxes)
            if n == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            dets = [img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] for x1, y1, x2, y2 in xyxy]
            qf = embed(dets)
            probs = torch.softmax(qf @ proto.T * 100.0, dim=1).numpy()
            sim_class = (qf @ proto.T).max(1).values.numpy()
            sim_excl = (qf @ excl_feats.T).max(1).values.numpy() if excl_feats is not None else None
            for bi, ((x1, y1, x2, y2), p) in enumerate(zip(xyxy, probs)):
                k = int(p.argmax())
                scores = {classes[ci]: round(float(p[ci]), 3) for ci in range(len(classes))}
                is_excl = False
                if sim_excl is not None and sim_excl[bi] > sim_class[bi] + EXCL_MARGIN:
                    is_excl = True
                scores["excluded"] = round(float(sim_excl[bi]), 3) if sim_excl is not None else 0.0
                rows.append({
                    "image": os.path.basename(img_file),
                    "num": len(rows),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "class": "excluded" if is_excl else classes[k],
                    "conf": round(float(sim_excl[bi] if is_excl else p.max()), 3),
                    "scores": scores,
                })
        os.makedirs(out_dir, exist_ok=True)
        lang = "en" if (d.get("lang") == "en") else "zh"
        render_boxes(rows, out_dir, images_dir, classes, with_number=True, lang=lang)
        return jsonify({"classes": classes, "out_dir": out_dir, "rows": rows,
                        "has_excl": excl_count > 0, "excl_count": excl_count,
                        "warnings": warnings})
    except Exception as e:
        return jsonify({"error_code": "INTERNAL",
                        "params": {"reason": "%s: %s" % (type(e).__name__, str(e)[:300])}})

@app.route("/api/save", methods=["POST"])
def api_save():
    d = request.json or {}
    rows = d.get("rows", [])
    images_dir = d.get("images_folder", "")
    out_dir = d.get("output_folder", "")
    classes = d.get("classes", [])
    if not rows or not images_dir or not out_dir:
        return jsonify({"error_code": "MISSING_DATA"})
    os.makedirs(out_dir, exist_ok=True)
    kept = [r for r in rows if r.get("class") != "excluded"]
    n_excl = len(rows) - len(kept)
    lang = "en" if (d.get("lang") == "en") else "zh"
    render_boxes(kept, out_dir, images_dir, classes, with_number=False, lang=lang)
    csv_path = os.path.join(out_dir, "result_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["image", "num", "x1", "y1", "x2", "y2", "class", "conf"] + classes
        w.writerow(header)
        for r in kept:
            line = [r["image"], r["num"], r["x1"], r["y1"], r["x2"], r["y2"], r["class"], r["conf"]]
            for c in classes:
                line.append(r.get("scores", {}).get(c, ""))
            w.writerow(line)
    return jsonify({"ok": True, "csv": csv_path, "excluded_count": n_excl})

def start_server():
    import threading, webbrowser, time as _time
    global PING_LAST, PING_STARTED, CLOSE_AT
    PING_LAST = _time.time()
    CLOSE_AT = 0.0
    PING_STARTED = threading.Event()

    def watchdog():
        """页面关闭信号 -> 3 秒内退出; 心跳长时间中断(卡死) -> 45 秒兜底退出"""
        PING_STARTED.wait(timeout=600)
        while True:
            _time.sleep(0.5)
            if CLOSE_AT and _time.time() > CLOSE_AT:
                print("页面已关闭, 程序自动退出", flush=True)
                os._exit(0)
            if _time.time() - PING_LAST > 45:
                print("页面长时间无响应, 程序自动退出", flush=True)
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    print("识别程序网页版启动中...")
    print("浏览器将打开: http://127.0.0.1:8000")
    print("(关闭网页后, 本程序会在约 3 秒内自动退出)")

    # 等服务器真正就绪后再打开浏览器 (避免提前弹出"无法访问"页面)
    def _open_when_ready():
        import urllib.request
        for _ in range(120):
            _time.sleep(0.5)
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/api/ready", timeout=1)
                if os.environ.get("CR_NO_BROWSER") != "1":   # 测试时可关掉自动开浏览器
                    webbrowser.open("http://127.0.0.1:8000")
                return
            except Exception:
                pass

    threading.Thread(target=_open_when_ready, daemon=True).start()
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
if __name__ == "__main__":
    start_server()

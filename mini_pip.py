# -*- coding: utf-8 -*-
"""
mini_pip —— 从国内镜像下载并解压 wheel 的简易安装器 (用于网络受限环境)。

用法:
    python mini_pip.py flask transformers==4.44.2 "torch==2.4.1::+cpu"
    (支持 包名 / 包名==版本 / 包名::wheel文件名关键字)

⚠️ 安全提示:
    本脚本为了能在受限环境中工作, 直接下载并解压 wheel, 不校验哈希、不验证签名,
    也不处理依赖关系。它只是 pip 不可用时的应急手段。
    正式环境请优先使用官方 pip:
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    另外, 安装第三方包本身就有供应链风险, 请只在你信任的镜像/来源上使用。
"""
import urllib.request, urllib.parse, zipfile, os, re, sys

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TARGET = os.environ.get("MINIPIP_TARGET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

def get(url, timeout=600):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pip/24"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            print("   (下载中断, 重试 %d/3: %s)" % (attempt + 1, str(e)[:60]), flush=True)
            import time as _t
            _t.sleep(3)
    raise last

def py_tag():
    v = sys.version_info
    return "cp%d%d" % (v.major, v.minor)

def parse_ver(fname):
    parts = fname.split("-")
    if len(parts) < 2:
        return (0,)
    m = re.match(r"^([0-9]+[0-9.]*)", parts[1])
    if not m:
        return (0,)
    nums = [int(x) for x in m.group(1).split(".") if x]
    return tuple(nums) if nums else (0,)

def tag_rank(h):
    f = os.path.basename(h.split("#")[0])
    tag = py_tag()
    if tag + "-" + tag + "-win_amd64" in f: return 0
    if tag + "-none-win_amd64" in f: return 1
    m = re.search(r"cp(\d)(\d+)-abi3-win_amd64", f)
    if m:
        code = int(m.group(1)) * 100 + int(m.group(2))
        cur = sys.version_info.major * 100 + sys.version_info.minor
        if code <= cur:
            return 2
    if "py3-none-win_amd64" in f: return 3
    if "py3-none-any" in f: return 4
    if "py2.py3-none-any" in f: return 5
    return 9

def find_wheel(pkg, exact=None, max_ver=None, contain=None):
    html = get("%s/%s/" % (MIRROR, pkg)).decode("utf-8", "ignore")
    hrefs = re.findall(r'href="([^"]+\.whl(?:#[^"]*)?)"', html)
    cands = []
    for h in hrefs:
        f = os.path.basename(h.split("#")[0])
        if tag_rank(h) >= 5:      # 只要兼容 cp310/py3 的 wheel
            continue
        if contain and contain not in f:
            continue
        v = parse_ver(f)
        if exact and not f.startswith(pkg.replace("-", "_") + "-" + exact + "-"):
            if exact not in f:
                continue
        if max_ver and v > max_ver:
            continue
        cands.append((v, tag_rank(h), urllib.parse.urljoin(MIRROR + "/", h.split("#")[0]), f))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])                # 先按 tag 优先度 (稳定排序)
    cands.sort(key=lambda x: x[0], reverse=True)  # 再按版本从新到旧
    v, r, url, fname = cands[0]
    return url, fname

def install(pkg, exact=None, max_ver=None, contain=None):
    os.makedirs(TARGET, exist_ok=True)
    r = find_wheel(pkg, exact, max_ver, contain)
    if not r:
        print("!! %-22s 没找到" % pkg)
        return False
    url, fname = r
    dest = os.path.join(TARGET, fname)
    data = get(url)
    with open(dest, "wb") as f:
        f.write(data)
    print("OK %-22s %-46s (%.1f MB)" % (pkg, fname, len(data)/1e6))
    with zipfile.ZipFile(dest) as z:
        z.extractall(TARGET)
    return True

if __name__ == "__main__":
    specs = sys.argv[1:]
    for s in specs:
        pkg, exact, max_ver, contain = s, None, None, None
        if "::" in s:
            s, contain = s.split("::", 1)
            pkg = s
        if "==" in s:
            pkg, exact = s.split("==", 1)
        if "<" in pkg:
            pkg, mv = pkg.split("<", 1)
            max_ver = tuple(int(x) for x in mv.split("."))
        try:
            install(pkg, exact, max_ver, contain)
        except Exception as e:
            print("!! %s: %s" % (pkg, str(e)[:90]))

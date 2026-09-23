# -*- coding: utf-8 -*-
"""打包用入口: PyInstaller 从这里启动整个程序"""
import os
import sys

if getattr(sys, "frozen", False):
    # 打包后: 可写目录 = exe 旁边, 资源目录 = 解包目录
    base = os.path.dirname(sys.executable)
    os.environ["WEBAPP_BASE"] = base
    # 无窗口模式: 把所有输出和错误写进日志文件, 方便排查
    log = open(os.path.join(base, "debug_log.txt"), "w", encoding="utf-8", errors="replace")
    sys.stdout = log
    sys.stderr = log

    def excepthook(t, v, tb):
        import traceback
        traceback.print_exception(t, v, tb, file=log)
        log.flush()
        os._exit(1)
    sys.excepthook = excepthook
else:
    base = os.path.dirname(os.path.abspath(__file__))
    os.environ["WEBAPP_BASE"] = base

import webapp

webapp.start_server()

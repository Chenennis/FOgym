#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline包的入口点
允许以python -m形式运行包
"""

import sys
import os

# 处理导入方式
try:
    # 尝试作为包的一部分导入
    from .model_based_pipeline import run_pipeline
except (ImportError, SystemError):
    # 直接运行脚本时的导入方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from model_based_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
    sys.exit(0) 
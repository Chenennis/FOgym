#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline entry point
Allow running the package as python -m
"""

import sys
import os

# handle import method
try:
    # try to import as part of the package
    from .model_based_pipeline import run_pipeline
except (ImportError, SystemError):
    # import method when running script directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from model_based_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
    sys.exit(0) 
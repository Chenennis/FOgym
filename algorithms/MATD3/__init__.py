"""
Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) algorithms

This package contains implementations of MATD3 and related multi-agent 
reinforcement learning algorithms for FlexOffer systems.

Modules:
- fomatd3: FlexOffer-specific MATD3 implementation
- Original MATD3 implementation files (agent.py, matd3.py, etc.)

Classes:
- FOMATD3: FlexOffer-specific MATD3 algorithm
- FOMATd3Policy: FlexOffer-specific MATD3 policy networks
"""

# 导入主要的算法类
try:
    from .fomatd3.fomatd3 import FOMATD3
    from .fomatd3.fomatd3_policy import FOMATd3Policy
    __all__ = ['FOMATD3', 'FOMATd3Policy']
except ImportError:
    # 如果导入失败，至少让模块可以被识别
    __all__ = [] 
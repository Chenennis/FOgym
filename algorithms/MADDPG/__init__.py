"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithms

This package contains implementations of MADDPG and related multi-agent 
reinforcement learning algorithms for FlexOffer systems.

Modules:
- fomaddpg: FlexOffer-specific MADDPG implementation
- maddpg: Original MADDPG implementation  
- experiments: Training and evaluation scripts
"""

# 导入主要的算法类
try:
    from .fomaddpg.fomaddpg import FOMADDPG
    from .fomaddpg.fomaddpg_policy import FOMaddpgPolicy
    __all__ = ['FOMADDPG', 'FOMaddpgPolicy']
except ImportError:
    # 如果导入失败，至少让模块可以被识别
    __all__ = [] 
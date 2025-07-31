"""
Multi-Agent Proximal Policy Optimization (MAPPO) algorithms

This package contains implementations of MAPPO and related multi-agent 
reinforcement learning algorithms for FlexOffer systems.

Modules:
- fomappo: FlexOffer-specific MAPPO implementation
- onpolicy: Original MAPPO implementation framework
"""

# 导入主要的算法类
try:
    from .fomappo.fomappo import FOMAPPO
    from .fomappo.fomappo_policy import FOMAPPOPolicy
    __all__ = ['FOMAPPO', 'FOMAPPOPolicy']
except ImportError:
    # 如果导入失败，至少让模块可以被识别
    __all__ = [] 
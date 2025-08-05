"""
FlexOffer Multi-Agent Deep Deterministic Policy Gradient (FOMADDPG)

Key features:
- Device-level state transition modeling
- Inter-Manager collaboration mechanism
- FlexOffer constraint-aware reward design
- Distributed training and centralized execution
"""

from .fomaddpg import FOMADDPG
from .fomaddpg_policy import FOMaddpgPolicy
from .fomaddpg_adapter import FOMAddpgAdapter

__all__ = ['FOMADDPG', 'FOMaddpgPolicy', 'FOMAddpgAdapter'] 
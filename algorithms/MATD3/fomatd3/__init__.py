"""
FlexOffer Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (FOMATD3)

This module contains the FlexOffer-specific implementation of MATD3 algorithm
for multi-agent reinforcement learning in FlexOffer systems.

Classes:
- FOMATD3: Main FOMATD3 algorithm class
- FOMATd3Policy: FlexOffer-specific MATD3 policy network
- FOMATD3Adapter: FO Pipeline integration adapter for FOMATD3
"""

from .fomatd3 import FOMATD3
from .fomatd3_policy import FOMATd3Policy
from .fomatd3_adapter import FOMATD3Adapter

__all__ = ['FOMATD3', 'FOMATd3Policy', 'FOMATD3Adapter'] 
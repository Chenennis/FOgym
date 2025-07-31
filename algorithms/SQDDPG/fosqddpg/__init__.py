"""
FlexOffer Shapley Q-value Deep Deterministic Policy Gradient (FOSQDDPG) Algorithm

This module implements SQDDPG algorithm adapted for FlexOffer framework.
SQDDPG uses Shapley value-based credit assignment in multi-agent settings.

Key Features:
- Shapley value-based credit assignment for fair multi-agent cooperation
- Actor-Critic architecture with deterministic policy gradient
- FlexOffer constraint-aware training
- Multi-agent coordination through global observation
"""

from .fosqddpg import FOSQDDPG
from .fosqddpg_policy import FOSQDDPGPolicy
from .fosqddpg_adapter import FOSQDDPGAdapter

__all__ = ['FOSQDDPG', 'FOSQDDPGPolicy', 'FOSQDDPGAdapter'] 
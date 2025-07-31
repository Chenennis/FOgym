"""
SQDDPG (Shapley Q-value Deep Deterministic Policy Gradient) Algorithm Package

This package contains the SQDDPG algorithm implementation and its FlexOffer adaptation.
"""

# Import core SQDDPG components if needed
try:
    from .fosqddpg import FOSQDDPG, FOSQDDPGPolicy
    __all__ = ['FOSQDDPG', 'FOSQDDPGPolicy']
except ImportError:
    # FOSQDDPG module not available
    __all__ = [] 
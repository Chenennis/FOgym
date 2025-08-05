"""
FlexOffer Multi-Agent PPO (FOMAPPO) Algorithm

A multi-agent reinforcement learning algorithm specifically designed for the FlexOffer system based on MAPPO algorithm.
Optimized for multi-agent collaboration at the Manager level.
"""

import os
import sys

# Add MAPPO onpolicy module path to ensure the onpolicy module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/
onpolicy_path = os.path.join(mappo_dir, "onpolicy")

if onpolicy_path not in sys.path:
    sys.path.insert(0, onpolicy_path)

# Import complete FOMAPPO algorithm components
try:
    from .fomappo import FOMAPPO
    from .fomappo_policy import FOMAPPOPolicy
    
    # Import complete Dec-POMDP components
    from .dec_pomdp_adapter import DecPOMDPObservationAdapter
    from .dec_pomdp_policy import DecPOMDPFOMAPPOPolicy
    from .dec_pomdp_loss import DecPOMDPLossComputer
    
    # Import standard FOMAPPO adapter (shared policy architecture)
    from .fomappo_adapter import FOMAPPOAdapter
    
    print("[OK] Complete FOMAPPO algorithm module imported successfully (with onpolicy support)")
    
except ImportError as e:
    print(f"[WARN] Some FOMAPPO modules failed to import: {e}")

__all__ = [
    'FOMAPPO', 
    'FOMAPPOPolicy', 
    'FOMAPPOAdapter',
    'DecPOMDPObservationAdapter', 
    'DecPOMDPFOMAPPOPolicy', 
    'DecPOMDPLossComputer'
] 
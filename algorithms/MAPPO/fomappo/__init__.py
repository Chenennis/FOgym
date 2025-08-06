import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/
onpolicy_path = os.path.join(mappo_dir, "onpolicy")

if onpolicy_path not in sys.path:
    sys.path.insert(0, onpolicy_path)

# import complete FOMAPPO algorithm components
try:
    from .fomappo import FOMAPPO
    from .fomappo_policy import FOMAPPOPolicy
    
    # import complete Dec-POMDP components
    from .dec_pomdp_adapter import DecPOMDPObservationAdapter
    from .dec_pomdp_policy import DecPOMDPFOMAPPOPolicy
    from .dec_pomdp_loss import DecPOMDPLossComputer
    
    # import standard FOMAPPO adapter (shared policy architecture)
    from .fomappo_adapter import FOMAPPOAdapter
    
    print("[OK] complete FOMAPPO algorithm module imported successfully (including onpolicy support)")
    
except ImportError as e:
    print(f"[WARN] FOMAPPO module import failed: {e}")

__all__ = [
    'FOMAPPO', 
    'FOMAPPOPolicy', 
    'FOMAPPOAdapter',
    'DecPOMDPObservationAdapter', 
    'DecPOMDPFOMAPPOPolicy', 
    'DecPOMDPLossComputer'
] 
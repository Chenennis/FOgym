"""
FlexOffer Multi-Agent PPO (FOMAPPO) Algorithm

基于MAPPO算法专门为FlexOffer系统设计的多智能体强化学习算法。
针对Manager级别的多智能体协作进行了优化。
"""

import os
import sys

# 添加MAPPO onpolicy模块路径，确保能找到onpolicy模块
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/
onpolicy_path = os.path.join(mappo_dir, "onpolicy")

if onpolicy_path not in sys.path:
    sys.path.insert(0, onpolicy_path)

# 导入完整的FOMAPPO算法组件
try:
    from .fomappo import FOMAPPO
    from .fomappo_policy import FOMAPPOPolicy
    
    # 导入完整的Dec-POMDP组件
    from .dec_pomdp_adapter import DecPOMDPObservationAdapter
    from .dec_pomdp_policy import DecPOMDPFOMAPPOPolicy
    from .dec_pomdp_loss import DecPOMDPLossComputer
    
    # 导入标准FOMAPPO适配器（共享策略架构）
    from .fomappo_adapter import FOMAPPOAdapter
    
    print("[OK] 完整FOMAPPO算法模块导入成功（包含onpolicy支持）")
    
except ImportError as e:
    print(f"[WARN] FOMAPPO部分模块导入失败: {e}")

__all__ = [
    'FOMAPPO', 
    'FOMAPPOPolicy', 
    'FOMAPPOAdapter',
    'DecPOMDPObservationAdapter', 
    'DecPOMDPFOMAPPOPolicy', 
    'DecPOMDPLossComputer'
] 
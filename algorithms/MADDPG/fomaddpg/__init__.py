"""
FlexOffer Multi-Agent Deep Deterministic Policy Gradient (FOMADDPG)

专门为FlexOffer系统设计的多智能体DDPG算法。
支持Manager级别的协作学习和设备级别的精确控制。

主要特点：
- 设备级状态转移建模
- Manager间协作机制  
- FlexOffer约束感知的奖励设计
- 分布式训练和集中式执行
"""

from .fomaddpg import FOMADDPG
from .fomaddpg_policy import FOMaddpgPolicy
from .fomaddpg_adapter import FOMAddpgAdapter

__all__ = ['FOMADDPG', 'FOMaddpgPolicy', 'FOMAddpgAdapter'] 
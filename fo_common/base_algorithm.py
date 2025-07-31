"""
FlexOffer多智能体强化学习算法基础类

本模块提供所有多智能体RL算法的通用基础类和接口，
减少代码重复，提升代码可维护性。
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BaseMARL(ABC):
    """多智能体强化学习算法基础类"""
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 device: str = "cpu"):
        """
        初始化基础MARL算法
        
        Args:
            n_agents: 智能体数量
            state_dim: 状态维度
            action_dim: 动作维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            hidden_dim: 隐藏层维度
            max_action: 最大动作值
            gamma: 折扣因子
            tau: 软更新系数
            batch_size: 批次大小
            buffer_capacity: 缓冲区容量
            device: 计算设备
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.device = torch.device(device)
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # FlexOffer特定参数
        self.fo_generation_mode = True
        self.manager_coordination_weight = 0.1
        
        # 初始化组件
        self._setup_networks()
        self._setup_optimizers()
        self._setup_replay_buffer()
    
    @abstractmethod
    def _setup_networks(self):
        """设置网络结构 - 由子类实现"""
        pass
    
    @abstractmethod
    def _setup_optimizers(self):
        """设置优化器 - 由子类实现"""
        pass
    
    @abstractmethod
    def _setup_replay_buffer(self):
        """设置经验回放缓冲区 - 由子类实现"""
        pass
    
    @abstractmethod
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作 - 由子类实现"""
        pass
    
    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """更新算法 - 由子类实现"""
        pass
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, 
                        rewards: np.ndarray, next_states: np.ndarray, 
                        dones: np.ndarray, **kwargs):
        """存储经验 - 通用实现"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones, **kwargs)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float = None):
        """软更新目标网络 - 通用实现"""
        if tau is None:
            tau = self.tau
            
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def save_models(self, path: str):
        """保存模型 - 通用框架"""
        save_dict = {
            'training_step': self.training_step,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }
        self._add_algorithm_specific_save_data(save_dict)
        torch.save(save_dict, path)
    
    def load_models(self, path: str):
        """加载模型 - 通用框架"""
        checkpoint = torch.load(path, map_location=self.device)
        self.training_step = checkpoint.get('training_step', 0)
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self._load_algorithm_specific_data(checkpoint)
    
    @abstractmethod
    def _add_algorithm_specific_save_data(self, save_dict: Dict):
        """添加算法特定的保存数据"""
        pass
    
    @abstractmethod
    def _load_algorithm_specific_data(self, checkpoint: Dict):
        """加载算法特定的数据"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息 - 通用实现"""
        return {
            'training_step': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'buffer_size': len(self.replay_buffer) if hasattr(self, 'replay_buffer') else 0
        }


class BaseReplayBuffer(ABC):
    """经验回放缓冲区基础类"""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.size = 0
    
    @abstractmethod
    def push(self, *args, **kwargs):
        """存储经验"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int):
        """采样经验"""
        pass
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.size = 0


class BaseActorNetwork(nn.Module, ABC):
    """Actor网络基础类"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass


class BaseCriticNetwork(nn.Module, ABC):
    """Critic网络基础类"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass


class FlexOfferMixin:
    """FlexOffer特定功能混入类"""
    
    def apply_fo_constraints(self, actions: torch.Tensor, 
                           fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用FlexOffer约束"""
        if fo_constraints is None:
            return actions
        
        # 简化的约束应用逻辑
        constrained_actions = torch.clamp(actions, -1.0, 1.0)
        return constrained_actions
    
    def compute_fo_constraint_loss(self, actions: torch.Tensor, 
                                  fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算FlexOffer约束损失"""
        if fo_constraints is None:
            return torch.tensor(0.0, device=actions.device)
        
        # 约束违反损失
        constraint_violations = torch.relu(torch.abs(actions) - 1.0)
        return constraint_violations.mean()
    
    def compute_device_coordination_loss(self, actions: torch.Tensor, 
                                       device_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算设备协调损失"""
        if device_states is None:
            return torch.tensor(0.0, device=actions.device)
        
        # 鼓励适度的动作方差（协调但不完全相同）
        action_variance = torch.var(actions, dim=-1).mean()
        target_variance = 0.5  # 目标方差
        coordination_loss = torch.relu(action_variance - target_variance)
        return coordination_loss


class AlgorithmRegistry:
    """算法注册器"""
    
    _algorithms = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: type):
        """注册算法"""
        cls._algorithms[name] = algorithm_class
        logger.info(f"算法注册成功: {name}")
    
    @classmethod
    def get(cls, name: str):
        """获取算法类"""
        if name not in cls._algorithms:
            raise ValueError(f"未注册的算法: {name}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """列出所有注册的算法"""
        return list(cls._algorithms.keys())


# 算法工厂函数
def create_algorithm(algorithm_name: str, config: Dict[str, Any]) -> BaseMARL:
    """
    创建算法实例
    
    Args:
        algorithm_name: 算法名称
        config: 配置参数
        
    Returns:
        算法实例
    """
    algorithm_class = AlgorithmRegistry.get(algorithm_name)
    return algorithm_class(**config) 
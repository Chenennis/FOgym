"""
FOMADDPG Adapter - FlexOffer multi-agent algorithm adapter based on MADDPG

Provides the same interface as FOMAPPO, but internally uses MADDPG's off-policy learning mechanism.
Supports seamless integration with the FO Pipeline.

Algorithm: FOMADDPG (FlexOffer Multi-Agent Deep Deterministic Policy Gradient)
Base: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
Key Features:
- Off-policy learning with replay buffer
- Continuous action spaces
- Actor-Critic architecture
- Multi-agent coordination
- FlexOffer constraint awareness

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import os

from .fomaddpg import FOMADDPG

logger = logging.getLogger(__name__)

class FOMAddpgArgs:
    """FOMADDPG parameter configuration class - inherits MADDPG parameters and adds FlexOffer specific parameters"""
    
    def __init__(self, **kwargs):
        # ========== Core MADDPG parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.buffer_capacity = kwargs.get('buffer_capacity', 100000)
        self.batch_size = kwargs.get('batch_size', 64)
        
        # Learning rate parameters - 🔧 Use stable learning rates
        self.lr = kwargs.get('lr_actor', 1e-4)
        self.lr_actor = kwargs.get('lr_actor', 1e-4)
        self.critic_lr = kwargs.get('lr_critic', 1e-3)
        self.tau = kwargs.get('tau', 0.005)  # Soft update parameter
        
        # DDPG specific parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.noise_scale = kwargs.get('noise_scale', 0.1)
        self.max_action = kwargs.get('max_action', 1.0)
        
        # Network parameters
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.layer_N = kwargs.get('layer_N', 2)
        self.use_orthogonal = kwargs.get('use_orthogonal', True)
        self.gain = kwargs.get('gain', 0.01)
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)
        self.activation_id = kwargs.get('activation_id', 1)
        self.use_ReLU = kwargs.get('use_ReLU', False)
        
        # Training options
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        
        # Algorithm name
        self.algorithm_name = kwargs.get('algorithm_name', 'fomaddpg')
        
        # ========== FOMADDPG specific parameters ==========
        self.use_device_coordination = kwargs.get('use_device_coordination', True)
        self.device_coordination_weight = kwargs.get('device_coordination_weight', 0.1)
        self.fo_constraint_weight = kwargs.get('fo_constraint_weight', 0.2)
        self.use_manager_coordination = kwargs.get('use_manager_coordination', True)
        self.manager_coordination_weight = kwargs.get('manager_coordination_weight', 0.05)
        
        # Network architecture specific parameters
        self.num_managers = kwargs.get('num_managers', 4)
        self.devices_per_manager = kwargs.get('devices_per_manager', 10)

class FOMAddpgAdapter:
    """
    FOMADDPG Adapter - Multi-agent reinforcement learning based on MADDPG (FlexOffer Multi-Agent DDPG)
    
    Core design principles:
    1. Off-policy algorithm architecture based on MADDPG
    2. Uses experience replay buffer for training
    3. Supports continuous action spaces
    4. Retains FOMADDPG's special FlexOffer features
    5. Seamless integration with FO Pipeline
    
    Advantages:
    - Off-policy learning: Higher sample efficiency
    - Continuous actions: Suitable for continuous parameter adjustment in FlexOffer
    - Experience replay: More stable training process
    - Actor-Critic: Separate optimization of policy and value function
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 device: str = "cpu",
                 **kwargs):
        """
        初始化FOMADDPG适配器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            num_agents: 智能体数量（Manager数量）
            episode_length: Episode长度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            device: 计算设备
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        
        logger.info(f"🔧 初始化FOMADDPG适配器（基于MADDPG架构）")
        logger.info(f"   参数: {num_agents}个Manager, 状态{state_dim}维, 动作{action_dim}维")
        
        # 创建参数对象
        self.args = FOMAddpgArgs(
            episode_length=episode_length,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            num_managers=num_agents,
            **kwargs
        )
        
        # 初始化FOMADDPG算法
        self.fomaddpg = FOMADDPG(
            n_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=kwargs.get('hidden_dim', 256),
            max_action=kwargs.get('max_action', 1.0),
            gamma=kwargs.get('gamma', 0.99),
            tau=kwargs.get('tau', 0.005),
            noise_scale=kwargs.get('noise_scale', 0.1),
            buffer_capacity=kwargs.get('buffer_capacity', 100000),
            batch_size=kwargs.get('batch_size', 64),
            device=device
        )
        
        # 训练统计
        self.training_iterations = 0
        self.total_episodes = 0
        
        # Manager统计
        self.manager_stats = {}
        for i in range(num_agents):
            manager_id = f"manager_{i + 1}"
            self.manager_stats[manager_id] = {
                'total_reward': 0.0,
                'episode_count': 0,
                'avg_reward': 0.0,
                'best_reward': float('-inf'),
                'training_updates': 0
            }
        
        logger.info("✅ FOMADDPG适配器初始化完成")
        logger.info(f"   架构: Off-policy MADDPG，经验回放缓冲区，连续动作空间")
    
    def reset_buffers(self):
        """重置缓冲区 - MADDPG使用经验回放，不需要episode级重置"""
        # MADDPG使用经验回放缓冲区，不需要像PPO那样的episode重置
        # 这里保持接口兼容性，但实际上MADDPG的buffer是持续累积的
        logger.debug("FOMADDPG使用经验回放缓冲区，无需episode级重置")
        pass
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        为所有Manager选择FlexOffer参数生成动作（MADDPG连续动作）
    
        - 动作现在对应FlexOffer参数：[start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight] × 设备数量
        - MADDPG适合连续动作空间，非常适合FlexOffer参数的连续调节
        - 使用经验回放和off-policy学习，样本效率更高
        
        Args:
            obs: 观测字典 {manager_id: observation}
            deterministic: 是否确定性动作
            
        Returns:
            actions: FlexOffer参数动作字典 {manager_id: fo_params_action}
            action_log_probs: 动作对数概率字典（MADDPG中为空，保持接口兼容）
            values: 价值函数预测字典（MADDPG中为空，保持接口兼容）
        """
        actions = {}
        action_log_probs = {}  # MADDPG不使用，但保持接口兼容
        values = {}  # MADDPG不使用，但保持接口兼容
        
        manager_ids = list(obs.keys())
        
        # 准备states数组格式，MADDPG期望numpy数组
        states = []
        for manager_id in manager_ids:
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                states.append(current_obs)
            else:
                states.append(np.array(current_obs))
        
        states = np.array(states)  # Shape: (num_agents, state_dim)
        
        # 使用FOMADDPG选择动作
        try:
            # 调用FOMADDPG的select_actions方法
            agent_actions = self.fomaddpg.select_actions(states, add_noise=not deterministic)
            
            # 转换为字典格式并映射到FlexOffer参数范围
            for i, manager_id in enumerate(manager_ids):
                raw_action = agent_actions[i]
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.zeros_like(fo_action)  # 占位符
                values[manager_id] = np.array([0.0])  # 占位符
                
                logger.debug(f"Manager {manager_id} MADDPG FlexOffer动作: {fo_action.shape} 维, "
                           f"前5个参数: {fo_action[:5]}")
                
        except Exception as e:
            logger.error(f"FOMADDPG动作选择失败: {e}")
            # 提供备用随机动作
            for manager_id in manager_ids:
                actions[manager_id] = np.random.uniform(-1, 1, self.action_dim)
                action_log_probs[manager_id] = np.zeros(self.action_dim)
                values[manager_id] = np.array([0.0])
        
        return actions, action_log_probs, values
    
    def collect_step(self, 
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """
        收集一步的经验数据到经验回放缓冲区
        
        Args:
            obs: 当前观测
            actions: 执行的动作
            rewards: 获得的奖励
            dones: 是否结束
            infos: 额外信息
            action_log_probs: 动作对数概率（MADDPG不使用）
            values: 价值函数预测（MADDPG不使用）
        """
        # 🔧 数值稳定性：检查和修复奖励
        manager_ids = list(obs.keys())
        
        # 准备MADDPG格式的数据
        states = []
        agent_actions = []
        agent_rewards = []
        agent_dones = []
        
        for manager_id in manager_ids:
            # 观测
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                states.append(current_obs)
            else:
                states.append(np.array(current_obs))
            
            # 动作
            action = actions[manager_id]
            if isinstance(action, np.ndarray):
                agent_actions.append(action)
            else:
                agent_actions.append(np.array(action))
            
            # 奖励 - 🔧 数值稳定性修复和奖励缩放优化
            raw_reward = rewards[manager_id]
            if np.isnan(raw_reward) or np.isinf(raw_reward):
                logger.warning(f"Manager {manager_id} 奖励无效({raw_reward})，设置为0")
                raw_reward = 0.0
            
            # 🔧 优化的奖励缩放 - 不过度压缩奖励信号
            # 原来的0.1倍缩放太激进，改为轻微缩放并保留更多信息
            normalized_reward = np.clip(raw_reward, -50.0, 50.0) * 0.5  # 从0.1改为0.5，范围从±10改为±50
            agent_rewards.append(normalized_reward)
            
            # 完成标志
            agent_dones.append(dones[manager_id])
            
            # 更新Manager统计
            self.manager_stats[manager_id]['total_reward'] += normalized_reward
            if normalized_reward > self.manager_stats[manager_id]['best_reward']:
                self.manager_stats[manager_id]['best_reward'] = normalized_reward
        
        # 转换为numpy数组
        states = np.array(states)
        agent_actions = np.array(agent_actions)
        agent_rewards = np.array(agent_rewards)
        agent_dones = np.array(agent_dones)
        
        # 存储到FOMADDPG的经验回放缓冲区
        # 注意：我们需要next_states，但在这个接口中没有提供
        # 我们将在下一次调用时提供next_states
        if hasattr(self, '_prev_states'):
            # 如果有之前的状态，将其作为next_states存储
            try:
                self.fomaddpg.store_experience(
                    states=self._prev_states,
                    actions=self._prev_actions,
                    rewards=self._prev_rewards,
                    next_states=states,
                    dones=self._prev_dones
                )
            except Exception as e:
                logger.warning(f"FOMADDPG经验存储失败: {e}")
        
        # 保存当前状态用于下次存储
        self._prev_states = states.copy()
        self._prev_actions = agent_actions.copy()
        self._prev_rewards = agent_rewards.copy()
        self._prev_dones = agent_dones.copy()
    
    def compute_returns(self):
        """计算returns - MADDPG不需要像PPO那样计算returns"""
        # MADDPG是off-policy算法，不需要像PPO那样计算episode-level的returns
        # 保持接口兼容性
        pass
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        执行一次MADDPG训练更新
        
        Returns:
            训练信息字典
        """
        try:
            # 检查是否有足够的经验进行训练
            if len(self.fomaddpg.replay_buffer) < self.fomaddpg.batch_size:
                logger.debug(f"经验缓冲区不足({len(self.fomaddpg.replay_buffer)}/{self.fomaddpg.batch_size})，跳过训练")
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
            
            # 执行MADDPG更新
            update_info = self.fomaddpg.update()
            
            if update_info:
                self.training_iterations += 1
                
                # 更新Manager统计
                for manager_id in self.manager_stats:
                    self.manager_stats[manager_id]['training_updates'] += 1
                
                return {
                    'actor_loss': update_info.get('actor_loss', 0.0),
                    'critic_loss': update_info.get('critic_loss', 0.0),
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
            else:
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
                
        except Exception as e:
            logger.error(f"FOMADDPG训练更新失败: {e}")
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'training_iterations': self.training_iterations,
                'buffer_size': len(self.fomaddpg.replay_buffer) if hasattr(self.fomaddpg, 'replay_buffer') else 0,
                'error': str(e)
            }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.fomaddpg.replay_buffer) if hasattr(self.fomaddpg, 'replay_buffer') else 0,
            'algorithm': 'FOMADDPG'
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """获取Manager奖励总结"""
        summary = {}
        for manager_id, stats in self.manager_stats.items():
            if stats['episode_count'] > 0:
                stats['avg_reward'] = stats['total_reward'] / stats['episode_count']
            summary[manager_id] = stats.copy()
        return summary
    
    def save_models(self, save_path: str):
        """保存模型"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 使用FOMADDPG的保存方法
            self.fomaddpg.save_models(save_path)
            logger.info(f"FOMADDPG模型已保存至 {save_path}")
        except Exception as e:
            logger.error(f"保存FOMADDPG模型失败: {e}")
    
    def load_models(self, load_path: str):
        """加载模型"""
        try:
            # 使用FOMADDPG的加载方法
            self.fomaddpg.load_models(load_path)
            logger.info(f"FOMADDPG模型已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载FOMADDPG模型失败: {e}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        """
        将原始动作映射到FlexOffer参数范围
        
        FlexOffer参数范围：
        - start_flex: [-1.0, 1.0] → 时间灵活性
        - end_flex: [-1.0, 1.0] → 时间灵活性  
        - energy_min_factor: [0.1, 1.0] → 最小能量因子
        - energy_max_factor: [1.0, 2.0] → 最大能量因子
        - priority_weight: [0.1, 2.0] → 优先级权重
        
        Args:
            raw_action: 原始动作 [-1, 1]范围
            
        Returns:
            fo_action: 映射到FlexOffer参数范围的动作
        """
        fo_action = np.zeros_like(raw_action)
        
        # 假设动作是5的倍数（每个设备5个参数）
        num_devices = len(raw_action) // 5 if len(raw_action) >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < len(raw_action):
                # start_flex: [-1, 1] → [-1, 1] (保持不变)
                fo_action[base_idx] = np.clip(raw_action[base_idx], -1.0, 1.0)
                
                # end_flex: [-1, 1] → [-1, 1] (保持不变)
                fo_action[base_idx + 1] = np.clip(raw_action[base_idx + 1], -1.0, 1.0)
                
                # energy_min_factor: [-1, 1] → [0.1, 1.0]
                fo_action[base_idx + 2] = 0.1 + 0.45 * (raw_action[base_idx + 2] + 1.0)
                
                # energy_max_factor: [-1, 1] → [1.0, 2.0]  
                fo_action[base_idx + 3] = 1.0 + 0.5 * (raw_action[base_idx + 3] + 1.0)
                
                # priority_weight: [-1, 1] → [0.1, 2.0]
                fo_action[base_idx + 4] = 0.1 + 0.95 * (raw_action[base_idx + 4] + 1.0)
        
        return fo_action
    
    def _generate_default_fo_action(self) -> np.ndarray:
        """生成默认的FlexOffer参数动作"""
        # 生成合理的默认FlexOffer参数
        default_action = np.zeros(self.action_dim)
        num_devices = self.action_dim // 5 if self.action_dim >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < self.action_dim:
                default_action[base_idx] = 0.0      # start_flex = 0
                default_action[base_idx + 1] = 0.0  # end_flex = 0  
                default_action[base_idx + 2] = 0.55 # energy_min_factor = 0.55
                default_action[base_idx + 3] = 1.5  # energy_max_factor = 1.5
                default_action[base_idx + 4] = 1.0  # priority_weight = 1.0
        
        return default_action 
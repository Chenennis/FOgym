#!/usr/bin/env python3
"""
FOSQDDPG适配器 - FlexOffer Shapley Q-value Deep Deterministic Policy Gradient Adapter

为FOSQDDPG算法提供与FO Pipeline的完整集成，支持：
1. Shapley值公平信用分配
2. FlexOffer约束集成
3. 多智能体协作训练
4. 与FO Framework的标准化接口

基于FOMADDPG和FOMATD3的成功架构模式设计。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import os
from collections import defaultdict

# FO Framework imports
from .fosqddpg import FOSQDDPG

logger = logging.getLogger(__name__)


class FOSQDDPGAdapter:
    """
    FOSQDDPG算法适配器
    
    集成FOSQDDPG算法到FO Pipeline，提供标准化的多智能体强化学习接口。
    特色功能：Shapley值公平信用分配 + FlexOffer约束优化
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 num_agents: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_scale: float = 0.1,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 sample_size: int = 5,  # Shapley采样大小
                 policy_delay: int = 1,  # FOSQDDPG通常不需要延迟更新
                 device: str = "cpu"):
        
        # 核心参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.policy_delay = policy_delay
        
        # FOSQDDPG特有参数
        self.max_action = max_action
        self.buffer_capacity = buffer_capacity
        
        # 智能体ID管理
        self.agent_ids = [f"manager_{i+1}" for i in range(num_agents)]
        
        # 训练状态
        self.training_iterations = 0
        self.total_iterations = 0
        self.current_episode = 0
        
        # 初始化FOSQDDPG算法
        self.fosqddpg = FOSQDDPG(
            n_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            max_action=max_action,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            sample_size=sample_size,
            device=device
        )
        
        # 缓存机制（兼容FO Pipeline接口）
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
        
        # 管理器奖励统计
        self._manager_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # 奖励标准化器
        self.reward_normalizer = self._create_reward_normalizer()
        
        # 兼容性参数对象
        class Args:
            def __init__(self, policy_delay, noise_scale, sample_size):
                self.policy_delay = policy_delay
                self.noise_scale = noise_scale
                self.sample_size = sample_size
        
        self.args = Args(policy_delay, noise_scale, sample_size)
        
        logger.info(f"FOSQDDPG适配器初始化完成: {num_agents}个智能体, "
                   f"状态维度={state_dim}, 动作维度={action_dim}, "
                   f"Shapley采样={sample_size}")
    
    def _create_reward_normalizer(self):
        """创建奖励标准化器"""
        return {
            'running_min': float('inf'),
            'running_max': float('-inf'),
            'running_mean': 0.0,
            'running_var': 1.0,
            'count': 0,
            'epsilon': 1e-8
        }
    
    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """标准化奖励值 - 保持相对关系的缩放方法"""
        
        # 固定除数缩放（推荐）- 假设奖励在500-600范围
        # 将奖励从~575缩放到~0.575
        fixed_scale_normalized = rewards / 1000.0
        
        # 检查方差以决定使用哪种方法
        reward_variance = np.var(rewards)
        
        if reward_variance < 1e-3:  # 方差很小，使用固定缩放
            return fixed_scale_normalized
        else:
            # 动态范围标准化（保持相对关系）
            reward_min = np.min(rewards)
            reward_max = np.max(rewards)
            if reward_max - reward_min > 1e-8:
                dynamic_normalized = (rewards - reward_min) / (reward_max - reward_min) * 0.1
                return dynamic_normalized
            else:
                return fixed_scale_normalized
    
    def select_actions(self, 
                      observations: Dict[str, np.ndarray], 
                      deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Optional[Dict], Optional[Dict]]:
        """
        选择FlexOffer参数生成动作（Shapley值公平信用分配）
        
        🔧 重构后的环境适配：
        - 动作现在对应FlexOffer参数：[start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight] × 设备数量
        - FOSQDDPG的Shapley值机制特别适合公平的FlexOffer参数分配
        - 确保不同Manager之间FlexOffer生成的公平性
        
        Returns:
            actions: FlexOffer参数动作字典 {agent_id: fo_params_action}
            action_log_probs: None (FOSQDDPG是确定性策略)
            values: None (此接口不返回值函数)
        """
        # 转换为numpy数组
        obs_array = np.array([observations[agent_id] for agent_id in self.agent_ids])
        
        # 使用FOSQDDPG选择原始动作
        raw_actions_array = self.fosqddpg.select_actions(obs_array, add_noise=not deterministic)
        
        # 转换回字典格式并映射到FlexOffer参数范围
        actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            raw_action = raw_actions_array[i]
            fo_action = self._map_action_to_fo_params(raw_action)
            actions[agent_id] = fo_action
            
            logger.debug(f"Manager {agent_id} FOSQDDPG FlexOffer动作: {fo_action.shape} 维, "
                       f"前5个参数: {fo_action[:5]}, Shapley公平性权重应用")
        
        return actions, None, None
    
    def collect_step(self, 
                    obs: Dict[str, np.ndarray],
                    actions: Dict[str, np.ndarray], 
                    rewards: Dict[str, float],
                    dones: Dict[str, bool],
                    infos: Dict[str, Any],
                    timestep: int) -> Dict[str, Any]:
        """
        收集单步经验（标准FO Pipeline接口）
        """
        # 转换格式
        states = np.array([obs[agent_id] for agent_id in self.agent_ids])
        actions_array = np.array([actions[agent_id] for agent_id in self.agent_ids])
        rewards_array = np.array([rewards[agent_id] for agent_id in self.agent_ids])
        dones_array = np.array([dones[agent_id] for agent_id in self.agent_ids])
        
        # 标准化奖励
        normalized_rewards = self._normalize_rewards(rewards_array)
        
        # 更新管理器奖励统计
        for i, agent_id in enumerate(self.agent_ids):
            self._manager_rewards[agent_id].append(rewards_array[i])
        
        # 使用缓存机制存储经验（兼容FO Pipeline）
        if hasattr(self, '_prev_states') and self._prev_states is not None:
            # 存储上一步的经验
            self.fosqddpg.store_experience(
                self._prev_states,
                self._prev_actions, 
                self._prev_rewards,
                states,  # 当前状态作为next_states
                self._prev_dones
            )
        
        # 保存当前状态为下次使用
        self._prev_states = states.copy()
        self._prev_actions = actions_array.copy()
        self._prev_rewards = normalized_rewards.copy()
        self._prev_dones = dones_array.copy()
        
        return {
            "states": states,
            "actions": actions_array,
            "rewards": normalized_rewards,
            "dones": dones_array,
            "normalized_rewards": normalized_rewards
        }
    
    def train_on_batch(self) -> Optional[Dict[str, float]]:
        """
        执行一次批量训练
        """
        if len(self.fosqddpg.replay_buffer) < self.batch_size:
            return None
        
        try:
            # 执行FOSQDDPG训练
            training_info = self.fosqddpg.update()
            
            if training_info:
                self.training_iterations += 1
                self.total_iterations += 1
                
                # 添加FOSQDDPG特有的统计信息
                training_stats = training_info.copy()
                additional_stats = {
                    'total_iterations': self.total_iterations,
                    'buffer_size': len(self.fosqddpg.replay_buffer),
                    'current_noise_scale': self.noise_scale,
                    'sample_size': self.sample_size,
                    'reward_stats': {
                        'mean': self.reward_normalizer['running_mean'],
                        'std': np.sqrt(self.reward_normalizer['running_var']),
                        'count': self.reward_normalizer['count']
                    }
                }
                
                # 合并统计信息
                for key, value in additional_stats.items():
                    training_stats[key] = value
                
                return training_stats
            
        except Exception as e:
            logger.error(f"FOSQDDPG训练出错: {e}")
            return None
        
        return None
    
    def reset_episode(self):
        """重置episode状态"""
        self.current_episode += 1
        
        # 清空缓存
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
    
    def reset_buffers(self):
        """重置经验缓冲区"""
        self.fosqddpg.replay_buffer.buffer.clear()
        logger.debug("FOSQDDPG经验缓冲区已重置")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'training_iterations': self.training_iterations,
            'total_iterations': self.total_iterations, 
            'current_episode': self.current_episode,
            'buffer_size': len(self.fosqddpg.replay_buffer) if hasattr(self.fosqddpg, 'replay_buffer') else 0,
            'noise_scale': self.noise_scale,
            'sample_size': self.sample_size
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Dict[str, float]]:
        """获取管理器奖励统计摘要"""
        summary = {}
        for agent_id in self.agent_ids:
            if agent_id in self._manager_rewards and self._manager_rewards[agent_id]:
                rewards = self._manager_rewards[agent_id]
                summary[agent_id] = {
                    "mean_reward": np.mean(rewards),
                    "total_reward": np.sum(rewards),
                    "episodes": len(rewards),
                    "max_reward": np.max(rewards),
                    "min_reward": np.min(rewards)
                }
            else:
                summary[agent_id] = {
                    "mean_reward": 0.0,
                    "total_reward": 0.0,
                    "episodes": 0,
                    "max_reward": 0.0,
                    "min_reward": 0.0
                }
        return summary
    
    def save_models(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        for i, policy in enumerate(self.fosqddpg.policies):
            agent_id = self.agent_ids[i]
            
            # 保存Actor和Critic网络
            actor_path = os.path.join(save_path, f"{agent_id}_actor.pt")
            critic_path = os.path.join(save_path, f"{agent_id}_critic.pt")
            
            torch.save(policy.actor.state_dict(), actor_path)
            torch.save(policy.critic.state_dict(), critic_path)
            
            logger.info(f"保存{agent_id}的模型到{save_path}")
    
    def load_models(self, load_path: str):
        """加载模型"""
        for i, policy in enumerate(self.fosqddpg.policies):
            agent_id = self.agent_ids[i]
            
            # 加载Actor和Critic网络
            actor_path = os.path.join(load_path, f"{agent_id}_actor.pt")
            critic_path = os.path.join(load_path, f"{agent_id}_critic.pt")
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                policy.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                policy.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                logger.info(f"加载{agent_id}的模型从{load_path}")
            else:
                logger.warning(f"未找到{agent_id}的模型文件")
    
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
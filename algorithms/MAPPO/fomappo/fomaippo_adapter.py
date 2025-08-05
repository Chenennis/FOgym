#!/usr/bin/env python3
"""
FOMAIPPO Adapter - 基于分离策略的独立Agent架构 (FlexOffer Multi-Agent Independent PPO)

架构设计：
- 参考原始MAPPO的separated/base_runner.py架构
- 为每个Manager创建独立的Policy、Trainer、Buffer
- 保留FOMAPPO的特殊功能（设备协调、FlexOffer约束等）
- 与现有FO Framework集成
- 解决策略冲突问题，实现独立学习

关键特性：
1. 独立学习：每个Manager有独立的策略网络，避免策略冲突
2. FOMAPPO特性：保留设备协调和FlexOffer约束感知
3. FO集成：与现有FO Pipeline无缝集成
4. 通用配置：多智能体设定在FO Framework中统一配置

Algorithm: FOMAIPPO (FlexOffer Multi-Agent Independent PPO)
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
import sys

# 添加MAPPO路径
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

# 导入原始MAPPO组件（separated架构）
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

# 导入FOMAPPO特定组件
from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAIPPOArgs:
    """FOMAIPPO参数配置类 - 继承MAPPO参数并添加FlexOffer特定参数"""
    
    def __init__(self, **kwargs):
        # ========== 核心PPO参数 ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 1)
        self.ppo_epoch = kwargs.get('ppo_epoch', 4)
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # 学习率参数 - 🔧 降低学习率提高数值稳定性
        self.lr = kwargs.get('lr_actor', 1e-4)  # 降低actor学习率
        self.lr_actor = kwargs.get('lr_actor', 1e-4)  # 降低actor学习率
        self.critic_lr = kwargs.get('lr_critic', 5e-4)  # 降低critic学习率
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # PPO特定参数 - 🔧 增强数值稳定性
        self.clip_param = kwargs.get('clip_param', 0.1)  # 降低clip范围提高稳定性
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)  # 降低value loss权重
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.2)  # 更强的梯度裁剪
        self.huber_delta = kwargs.get('huber_delta', 1.0)  # 降低huber delta
        
        # GAE参数
        self.use_gae = kwargs.get('use_gae', True)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        
        # 🔧 重要修复：添加缺失的use_proper_time_limits属性
        # 这个属性用于控制在计算回报时是否考虑时间限制
        # 在FlexOffer系统中，每个episode有明确的时间限制（24小时），所以设置为True
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', True)
        
        # 网络参数
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.layer_N = kwargs.get('layer_N', 2)
        self.use_orthogonal = kwargs.get('use_orthogonal', True)
        self.gain = kwargs.get('gain', 0.01)
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)
        self.activation_id = kwargs.get('activation_id', 1)
        self.use_ReLU = kwargs.get('use_ReLU', False)  # 🔧 修复：使用Tanh激活函数（False）或ReLU（True）
        self.stacked_frames = kwargs.get('stacked_frames', 1)  # 堆叠帧数
        self.use_stacked_frames = kwargs.get('use_stacked_frames', False)  # 是否使用堆叠帧
        
        # RNN参数
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # 训练选项
        self.use_centralized_V = kwargs.get('use_centralized_V', True)
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        self.use_popart = kwargs.get('use_popart', False)
        self.use_valuenorm = kwargs.get('use_valuenorm', True)
        self.use_value_active_masks = kwargs.get('use_value_active_masks', True)
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', True)
        
        # 算法名称（用于兼容性）
        self.algorithm_name = kwargs.get('algorithm_name', 'fomaippo')
        
        # 策略共享选项
        self.share_policy = kwargs.get('share_policy', False)  # 🔧 FOMAIPPO使用独立策略，设置为False
        
        # ========== FOMAIPPO特定参数 ==========
        self.use_device_coordination = kwargs.get('use_device_coordination', True)
        self.device_coordination_weight = kwargs.get('device_coordination_weight', 0.1)
        self.fo_constraint_weight = kwargs.get('fo_constraint_weight', 0.2)
        self.use_manager_coordination = kwargs.get('use_manager_coordination', True)
        self.manager_coordination_weight = kwargs.get('manager_coordination_weight', 0.05)
        
        # 网络架构特定参数
        self.num_managers = kwargs.get('num_managers', 4)
        self.devices_per_manager = kwargs.get('devices_per_manager', 10)
        
        # 从kwargs获取观测和动作空间
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')

class FOMAIPPOAdapter:
    """
    FOMAIPPO适配器 - 基于分离策略的多智能体强化学习 (FlexOffer Multi-Agent Independent PPO)
    
    核心设计原则：
    1. 参考原始MAPPO的separated/base_runner.py架构
    2. 每个Manager有独立的Policy、Trainer、Buffer
    3. 保留FOMAPPO的所有特殊功能
    4. 与FO Framework无缝集成
    
    解决的问题：
    - 策略冲突：共享策略导致的学习信号混合
    - 奖励干扰：不同Manager的奖励信号相互干扰
    - 学习效率：独立学习提高收敛速度和稳定性
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 5e-4,
                 device: str = "cpu",
                 **kwargs):
        """
        初始化FOMAIPPO适配器
        
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
        
        logger.info(f"🔧 初始化FOMAIPPO适配器（分离策略架构）")
        logger.info(f"   参数: {num_agents}个Manager, 状态{state_dim}维, 动作{action_dim}维")
        
        # 创建观测和动作空间（与原始MAPPO格式兼容）
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # 创建参数配置
        self.args = FOMAIPPOArgs(
            episode_length=episode_length,
            n_rollout_threads=1,
            num_mini_batch=1,
            ppo_epoch=4,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            num_agents=num_agents,
            obs_space=obs_space,
            share_obs_space=share_obs_space,
            act_space=act_space,
            **kwargs
        )
        
        # ========== 创建独立的Policy、Trainer、Buffer ==========
        # 参考原始MAPPO separated架构
        
        # 1. 为每个Manager创建独立的Policy
        self.policies = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # 使用FOMAPPO策略（保留特殊功能）
            policy = FOMAPPOPolicy(
                args=self.args,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space,
                device=self.device
            )
            
            self.policies.append(policy)
            logger.info(f"   ✅ 创建 {manager_id} 独立FOMAIPPO策略")
        
        # 2. 为每个Manager创建独立的Trainer
        self.trainers = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # 使用FOMAPPO训练器（保留特殊功能）
            trainer = FOMAPPO(
                args=self.args,
                policy=self.policies[agent_id],
                device=self.device
            )
            
            self.trainers.append(trainer)
            logger.info(f"   ✅ 创建 {manager_id} 独立FOMAIPPO训练器")
        
        # 3. 为每个Manager创建独立的Buffer
        self.buffers = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # 使用分离式Buffer（原始MAPPO separated架构）
            buffer = SeparatedReplayBuffer(
                args=self.args,
                obs_space=obs_space,
                share_obs_space=share_obs_space,
                act_space=act_space
            )
            
            self.buffers.append(buffer)
            logger.info(f"   ✅ 创建 {manager_id} 独立SeparatedReplayBuffer")
        
        # 训练统计
        self.total_episodes = 0
        self.training_iterations = 0
        
        # 为每个Manager跟踪训练统计
        self.manager_stats = {}
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            self.manager_stats[manager_id] = {
                'episodes': 0,
                'total_reward': 0.0,
                'best_reward': float('-inf'),
                'avg_loss': 0.0,
                'training_updates': 0
            }
        
        logger.info("✅ FOMAIPPO适配器初始化完成")
        logger.info(f"   架构: {num_agents}个独立Manager, 每个有独立的Policy+Trainer+Buffer")
        logger.info(f"   特性: 保留FOMAPPO设备协调和FlexOffer约束感知")
    
    def reset_buffers(self):
        """重置所有Manager的buffer"""
        for agent_id in range(self.num_agents):
            self.buffers[agent_id].step = 0
        logger.debug("所有Manager的buffer已重置")
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        为所有Manager选择FlexOffer参数生成动作（独立策略）
        
        🔧 重构后的环境适配：
        - 动作现在对应FlexOffer参数：[start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight] × 设备数量
        - 每个Manager使用独立的策略网络，避免策略冲突
        - 观测包含设备状态、环境状态、其他Manager信息、市场状态
        
        Args:
            obs: 观测字典 {manager_id: observation}
            deterministic: 是否确定性动作
            
        Returns:
            actions: FlexOffer参数动作字典 {manager_id: fo_params_action}
            action_log_probs: 动作对数概率字典
            values: 价值函数预测字典
        """
        actions = {}
        action_log_probs = {}
        values = {}
        
        manager_ids = list(obs.keys())
        
        for i, manager_id in enumerate(manager_ids):
            if i >= len(self.policies):
                logger.warning(f"Manager {manager_id} 超出策略数量，跳过")
                continue
                
            policy = self.policies[i]
            current_obs = obs[manager_id]
            
            # 确保观测格式正确
            if isinstance(current_obs, np.ndarray):
                if len(current_obs.shape) == 1:
                    obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor(current_obs).to(self.device)
            else:
                obs_tensor = torch.FloatTensor([current_obs]).to(self.device)
            
            share_obs_tensor = obs_tensor  # 在分离策略中，假设共享观测相同
            
            # 创建RNN状态和掩码
            batch_size = obs_tensor.shape[0]
            rnn_states_actor = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            rnn_states_critic = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            masks = torch.ones(batch_size, 1, device=self.device)
            
            # 使用策略选择动作
            try:
                value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = policy.get_actions(
                    share_obs_tensor,
                    obs_tensor,
                    rnn_states_actor,
                    rnn_states_critic,
                    masks,
                    available_actions=None,
                    deterministic=deterministic
                )
                
                # 转换为numpy格式并映射到FlexOffer参数范围
                raw_action = action.detach().cpu().numpy().squeeze()
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = action_log_prob.detach().cpu().numpy().squeeze()
                values[manager_id] = value.detach().cpu().numpy().squeeze()
                
                logger.debug(f"Manager {manager_id} FlexOffer独立动作: {fo_action.shape} 维, "
                           f"前5个参数: {fo_action[:5]}")
                
            except Exception as e:
                logger.error(f"Manager {manager_id} FlexOffer动作选择失败: {e}")
                # 提供备用FlexOffer参数动作
                fo_action = self._generate_default_fo_action()
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.log(0.5)
                values[manager_id] = 0.0
        
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
        收集一步的经验数据到各自的buffer
        
        Args:
            obs: 当前观测
            actions: 执行的动作
            rewards: 获得的奖励
            dones: 是否结束
            infos: 额外信息
            action_log_probs: 动作对数概率（可选）
            values: 价值函数预测（可选）
        """
        manager_ids = list(obs.keys())
        
        for i, manager_id in enumerate(manager_ids):
            if i >= len(self.buffers):
                continue
                
            buffer = self.buffers[i]
            
            # 准备数据格式（适配SeparatedReplayBuffer）
            current_obs = obs[manager_id].reshape(1, -1)  # (1, obs_dim)
            share_obs = current_obs  # 分离策略中共享观测相同
            
            action = actions[manager_id].reshape(1, -1)  # (1, action_dim)
            
            # 🔧 数值稳定性修复：奖励裁剪和归一化
            raw_reward = rewards[manager_id]
            
            # 检查奖励是否有效
            if np.isnan(raw_reward) or np.isinf(raw_reward):
                logger.warning(f"Manager {manager_id} 奖励无效({raw_reward})，设置为0")
                raw_reward = 0.0
            
            # 裁剪奖励防止极值
            clipped_reward = np.clip(raw_reward, -10.0, 10.0)
            
            # 轻微的奖励缩放
            normalized_reward = clipped_reward * 0.1  # 缩放到较小范围
            
            reward = np.array([[normalized_reward]], dtype=np.float32)  # (1, 1)
            
            # RNN状态（全零，因为不使用RNN）
            rnn_states_actor = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            
            # 掩码
            mask = np.array([[1.0]], dtype=np.float32)  # 未结束为1
            bad_mask = np.array([[1.0]], dtype=np.float32)
            active_mask = np.array([[1.0]], dtype=np.float32)
            
            # 动作对数概率和价值预测
            if action_log_probs is not None and manager_id in action_log_probs:
                action_log_prob = action_log_probs[manager_id].reshape(1, -1)
            else:
                action_log_prob = np.array([[np.log(0.5)]], dtype=np.float32)
                
            if values is not None and manager_id in values:
                value_pred = np.array([[values[manager_id]]], dtype=np.float32)
            else:
                value_pred = np.array([[0.0]], dtype=np.float32)
            
            # 插入到buffer
            try:
                buffer.insert(
                    share_obs=share_obs,
                    obs=current_obs,
                    rnn_states=rnn_states_actor,
                    rnn_states_critic=rnn_states_critic,
                    actions=action,
                    action_log_probs=action_log_prob,
                    value_preds=value_pred,
                    rewards=reward,
                    masks=mask,
                    bad_masks=bad_mask,
                    active_masks=active_mask
                )
            except Exception as e:
                logger.error(f"Manager {manager_id} buffer插入失败: {e}")
    
    def compute_returns(self):
        """计算所有Manager的returns和advantages"""
        for agent_id in range(self.num_agents):
            buffer = self.buffers[agent_id]
            
            try:
                # 🔧 数值稳定性修复：提供安全的next_value而不是None
                # 使用最后一个value_pred作为next_value，如果无效则使用0
                if hasattr(buffer, 'value_preds') and len(buffer.value_preds) > 0:
                    last_value = buffer.value_preds[-1]
                    
                    # 检查last_value是否有效
                    if isinstance(last_value, np.ndarray):
                        if np.isnan(last_value).any() or np.isinf(last_value).any():
                            next_value = np.zeros_like(last_value)
                        else:
                            next_value = last_value
                    else:
                        next_value = np.zeros((1, 1), dtype=np.float32)
                else:
                    next_value = np.zeros((1, 1), dtype=np.float32)
                
                # 计算GAE
                buffer.compute_returns(
                    next_value=next_value,
                    value_normalizer=self.trainers[agent_id].value_normalizer
                )
                
                # 🔧 验证计算结果
                if hasattr(buffer, 'returns'):
                    returns_has_nan = np.isnan(buffer.returns).any() if isinstance(buffer.returns, np.ndarray) else torch.isnan(buffer.returns).any()
                    if returns_has_nan:
                        logger.warning(f"Manager {agent_id} GAE计算后returns包含NaN，使用安全值替代")
                        if isinstance(buffer.returns, np.ndarray):
                            buffer.returns = np.nan_to_num(buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                        else:
                            buffer.returns = torch.nan_to_num(buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
            except Exception as e:
                logger.error(f"Manager {agent_id} GAE计算失败: {e}")
                # 🔧 提供backup：如果GAE计算完全失败，创建安全的returns
                try:
                    if hasattr(buffer, 'rewards'):
                        # 使用简单的累积奖励作为returns
                        buffer.returns = np.cumsum(buffer.rewards, axis=0)
                        logger.warning(f"Manager {agent_id} 使用简单累积奖励作为returns")
                except Exception as backup_error:
                    logger.error(f"Manager {agent_id} backup returns创建也失败: {backup_error}")
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        对所有Manager进行一次训练更新
        
        Returns:
            训练信息字典
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_ratio = 0.0
        update_count = 0
        
        for agent_id in range(self.num_agents):
            trainer = self.trainers[agent_id]
            buffer = self.buffers[agent_id]
            
            try:
                # 🔧 数值稳定性修复：先检查buffer数据质量
                returns = buffer.returns[:-1]
                value_preds = buffer.value_preds[:-1]
                
                # 检查returns和value_preds的数据质量
                returns_has_nan = np.isnan(returns).any() if isinstance(returns, np.ndarray) else torch.isnan(returns).any()
                value_preds_has_nan = np.isnan(value_preds).any() if isinstance(value_preds, np.ndarray) else torch.isnan(value_preds).any()
                
                if returns_has_nan or value_preds_has_nan:
                    logger.warning(f"Manager {agent_id} buffer数据质量问题:")
                    logger.warning(f"  returns有NaN: {returns_has_nan}")
                    logger.warning(f"  value_preds有NaN: {value_preds_has_nan}")
                    
                    # 🔧 数据修复：用安全值替代NaN/Inf
                    if isinstance(returns, np.ndarray):
                        returns = np.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if isinstance(value_preds, np.ndarray):
                        value_preds = np.nan_to_num(value_preds, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        value_preds = torch.nan_to_num(value_preds, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    logger.warning(f"  已修复Manager {agent_id}的NaN/Inf数据")
                
                # 计算advantages
                advantages = returns - value_preds
                
                # 🔧 最终安全检查：如果advantages仍有问题，提供backup
                if isinstance(advantages, np.ndarray):
                    if np.isnan(advantages).any() or np.isinf(advantages).any():
                        logger.warning(f"Manager {agent_id} advantages仍包含NaN/Inf，使用零advantages")
                        advantages = np.zeros_like(advantages)
                    # 转换为torch张量
                    advantages = torch.from_numpy(advantages).float()
                elif torch.is_tensor(advantages):
                    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                        logger.warning(f"Manager {agent_id} advantages仍包含NaN/Inf，使用零advantages")
                        advantages = torch.zeros_like(advantages)
                else:
                    logger.warning(f"Manager {agent_id} advantages数据类型未知({type(advantages)})，使用零advantages")
                    advantages = torch.zeros((len(buffer.returns)-1, 1), dtype=torch.float32)
                
                # 🔧 安全的标准化advantages（确保是torch张量）
                if not torch.is_tensor(advantages):
                    advantages = torch.from_numpy(advantages).float()
                
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                
                if adv_std > 1e-8 and not torch.isnan(adv_std) and not torch.isinf(adv_std):
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    # 如果标准差太小或无效，跳过标准化
                    logger.warning(f"Manager {agent_id} advantages标准差无效({adv_std})，跳过标准化")
                    advantages = advantages - adv_mean
                
                # 🔧 裁剪advantages防止极值
                advantages = torch.clamp(advantages, -10.0, 10.0)
                
                # 执行PPO更新
                train_info = trainer.train(buffer)
                
                # 累积训练统计
                if isinstance(train_info, dict):
                    total_policy_loss += train_info.get('policy_loss', 0.0)
                    total_value_loss += train_info.get('value_loss', 0.0)
                    total_entropy += train_info.get('dist_entropy', 0.0)
                    total_ratio += train_info.get('ratio', 1.0)
                    update_count += 1
                    
                    # 更新Manager统计
                    manager_id = f"manager_{agent_id + 1}"
                    self.manager_stats[manager_id]['training_updates'] += 1
                    self.manager_stats[manager_id]['avg_loss'] = train_info.get('policy_loss', 0.0)
                
                # Buffer重置
                buffer.after_update()
                
            except Exception as e:
                logger.error(f"Manager {agent_id} 训练失败: {e}")
                continue
        
        # 计算平均训练统计
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy = total_entropy / update_count
            avg_ratio = total_ratio / update_count
        else:
            avg_policy_loss = avg_value_loss = avg_entropy = avg_ratio = 0.0
        
        self.training_iterations += 1
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'ratio': avg_ratio,
            'training_iterations': self.training_iterations,
            'updated_managers': update_count
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'num_managers': self.num_agents,
            'algorithm': 'FOMAIPPO',
            'architecture': 'separated_policy'
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """获取Manager奖励总结"""
        return self.manager_stats.copy()
    
    def save_models(self, save_path: str):
        """保存所有Manager的模型"""
        try:
            for agent_id in range(self.num_agents):
                manager_id = f"manager_{agent_id + 1}"
                model_path = f"{save_path}_{manager_id}.pt"
                
                torch.save({
                    'actor_state_dict': self.policies[agent_id].actor.state_dict(),
                    'critic_state_dict': self.policies[agent_id].critic.state_dict(),
                    'actor_optimizer': self.policies[agent_id].actor_optimizer.state_dict(),
                    'critic_optimizer': self.policies[agent_id].critic_optimizer.state_dict(),
                    'training_iterations': self.training_iterations,
                    'manager_stats': self.manager_stats[manager_id]
                }, model_path)
                
            logger.info(f"FOMAIPPO模型已保存至 {save_path}_manager_*.pt")
        except Exception as e:
            logger.error(f"保存FOMAIPPO模型失败: {e}")
    
    def load_models(self, load_path: str):
        """加载所有Manager的模型"""
        try:
            for agent_id in range(self.num_agents):
                manager_id = f"manager_{agent_id + 1}"
                model_path = f"{load_path}_{manager_id}.pt"
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.policies[agent_id].actor.load_state_dict(checkpoint['actor_state_dict'])
                    self.policies[agent_id].critic.load_state_dict(checkpoint['critic_state_dict'])
                    self.policies[agent_id].actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.policies[agent_id].critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                    
                    if 'manager_stats' in checkpoint:
                        self.manager_stats[manager_id] = checkpoint['manager_stats']
                    
                    logger.info(f"已加载 {manager_id} 模型")
                else:
                    logger.warning(f"模型文件 {model_path} 不存在")
                    
            logger.info(f"FOMAIPPO模型已从 {load_path}_manager_*.pt 加载")
        except Exception as e:
            logger.error(f"加载FOMAIPPO模型失败: {e}")
    
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
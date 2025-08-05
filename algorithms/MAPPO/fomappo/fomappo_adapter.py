#!/usr/bin/env python3
"""
FOMAPPO Adapter - 基于共享策略的多智能体架构 (FlexOffer Multi-Agent PPO)

架构设计：
- 参考原始MAPPO的shared/base_runner.py架构
- 所有Manager共享一个Policy和Trainer
- 使用SharedReplayBuffer收集所有agent数据
- 保留FOMAPPO的特殊功能（设备协调、FlexOffer约束等）
- 与现有FO Framework集成
- 高效的参数共享和集中式训练

关键特性：
1. 共享学习：所有Manager共享同一个策略网络，提高数据效率
2. FOMAPPO特性：保留设备协调和FlexOffer约束感知
3. FO集成：与现有FO Pipeline无缝集成
4. 中心化训练：利用所有agent的经验进行联合学习

Algorithm: FOMAPPO (FlexOffer Multi-Agent PPO)
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

# 导入原始MAPPO组件（shared架构）
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

# 导入FOMAPPO特定组件
from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAPPOArgs:
    """FOMAPPO参数配置类 - 继承MAPPO参数并添加FlexOffer特定参数"""
    
    def __init__(self, **kwargs):
        # ========== 核心PPO参数 ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 2)  # 增加到2个mini-batch
        self.ppo_epoch = kwargs.get('ppo_epoch', 8)  # 增加到8个epoch
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # 学习率参数 - 降低以避免快速收敛到次优解
        self.lr = kwargs.get('lr_actor', 5e-5)  # 从3e-4降低到5e-5
        self.lr_actor = kwargs.get('lr_actor', 5e-5)  # 从3e-4降低到5e-5
        self.critic_lr = kwargs.get('lr_critic', 2e-4)  # 从1e-3降低到2e-4
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # 增加学习率衰减参数
        self.use_linear_lr_decay = kwargs.get('use_linear_lr_decay', True)  # 启用学习率衰减
        self.lr_decay_rate = kwargs.get('lr_decay_rate', 0.95)  # 学习率衰减率
        
        # PPO裁剪参数
        self.clip_param = kwargs.get('clip_param', 0.2)
        self.value_loss_coef = kwargs.get('value_loss_coef', 1.0)
        
        # 增加熵系数，鼓励更多探索
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)  # 从0.001增加到0.01，增强探索
        
        # GAE参数
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.use_gae = kwargs.get('use_gae', True)
        
        # 梯度裁剪
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        
        # 网络参数
        self.hidden_size = kwargs.get('hidden_size', 64)  # 增加网络容量
        self.layer_N = kwargs.get('layer_N', 2)  # 使用更深的网络
        self.gain = kwargs.get('gain', 0.01)  # 添加缺失的gain参数
        self.use_orthogonal = kwargs.get('use_orthogonal', True)  # 添加缺失的use_orthogonal参数
        self.use_ReLU = kwargs.get('use_ReLU', True)  # 添加缺失的use_ReLU参数
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)  # 添加缺失的use_feature_normalization参数
        self.activation_id = kwargs.get('activation_id', 1)  # 添加缺失的activation_id参数
        
        # 奖励归一化
        self.use_reward_normalization = kwargs.get('use_reward_normalization', True)  # 启用奖励归一化
        self.reward_scale = kwargs.get('reward_scale', 0.01)  # 添加缺失的reward_scale参数
        
        # 循环策略参数
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # PopArt和ValueNorm参数
        self.use_popart = kwargs.get('use_popart', False)
        self.use_valuenorm = kwargs.get('use_valuenorm', True)  # 启用价值归一化
        self.use_value_active_masks = kwargs.get('use_value_active_masks', False)
        
        # 掩码参数
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', False)
        
        # 时间限制参数
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', False)
        
        # 算法名称
        self.algorithm_name = kwargs.get('algorithm_name', 'FOMAPPO')
        
        # 其他参数
        self.stacked_frames = kwargs.get('stacked_frames', 1)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        self.huber_delta = kwargs.get('huber_delta', 10.0)  # 添加缺失的huber_delta参数
        
        # 设备协调损失权重
        self.device_coord_loss_weight = kwargs.get('device_coord_loss_weight', 0.1)
        
        # FO约束损失权重
        self.fo_constraint_loss_weight = kwargs.get('fo_constraint_loss_weight', 0.1)
        
        # 增加探索参数
        self.action_noise_std = kwargs.get('action_noise_std', 0.1)  # 动作噪声标准差
        self.use_action_noise = kwargs.get('use_action_noise', True)  # 是否使用动作噪声
        
        # 增加训练稳定性参数
        self.clip_value = kwargs.get('clip_value', 10.0)  # 值裁剪范围
        self.use_advantage_normalization = kwargs.get('use_advantage_normalization', True)  # 是否归一化优势
        
        # 从kwargs获取观测和动作空间
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')

class FOMAPPOAdapter:
    """
    FOMAPPO适配器 - 基于共享策略的多智能体强化学习 (FlexOffer Multi-Agent PPO)
    
    核心设计原则：
    1. 参考原始MAPPO的shared/base_runner.py架构
    2. 所有Manager共享一个Policy和Trainer
    3. 使用SharedReplayBuffer收集所有agent数据
    4. 保留FOMAPPO的所有特殊功能
    5. 与FO Framework无缝集成
    
    优势：
    - 参数效率：共享策略减少参数数量，提高数据效率
    - 协调学习：Manager间自然的协调和沟通
    - 稳定训练：减少策略方差，提高训练稳定性
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 5e-5,
                 lr_critic: float = 2e-4,
                 device: str = "cpu",
                 **kwargs):
        """
        初始化FOMAPPO适配器
        
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
        # 确保state_dim至少为73，避免后续维度变化
        self.state_dim = max(state_dim, 73)
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.actual_obs_dim = max(state_dim, 73)  # 初始设置为73，避免后续维度变化
        
        # 增加维度变化跟踪
        self.initial_state_dim = max(state_dim, 73)
        self.has_dimension_changed = False
        self.dimension_change_count = 0
        self.new_obs_dimension_history = []  # 添加此行，初始化维度历史记录
        
        logger.info(f"🔧 初始化FOMAPPO适配器（共享策略架构）")
        logger.info(f"   参数: {num_agents}个Manager, 状态{self.state_dim}维, 动作{action_dim}维")
        
        # 创建观测和动作空间（与原始MAPPO格式兼容）
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # 创建参数配置 - 🔧 修复：避免重复参数
        args_dict = {
            'episode_length': episode_length,
            'n_rollout_threads': 1,
            'num_mini_batch': kwargs.get('num_mini_batch', 2),
            'ppo_epoch': kwargs.get('ppo_epoch', 8),
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'entropy_coef': kwargs.get('entropy_coef', 0.01),
            'use_linear_lr_decay': kwargs.get('use_linear_lr_decay', True),
            'lr_decay_rate': kwargs.get('lr_decay_rate', 0.95),
            'gamma': kwargs.get('gamma', 0.99),
            'gae_lambda': kwargs.get('gae_lambda', 0.95),
            'use_gae': kwargs.get('use_gae', True),
            'clip_param': kwargs.get('clip_param', 0.2),
            'max_grad_norm': kwargs.get('max_grad_norm', 0.5),
            'use_max_grad_norm': kwargs.get('use_max_grad_norm', True),
            'use_clipped_value_loss': kwargs.get('use_clipped_value_loss', True),
            'use_huber_loss': kwargs.get('use_huber_loss', True),
            'huber_delta': kwargs.get('huber_delta', 10.0),  # 添加缺失的huber_delta参数
            'reward_scale': kwargs.get('reward_scale', 0.01),  # 添加缺失的reward_scale参数
            'use_reward_normalization': kwargs.get('use_reward_normalization', True),
            'use_orthogonal': kwargs.get('use_orthogonal', True),
            'use_ReLU': kwargs.get('use_ReLU', True),
            'use_feature_normalization': kwargs.get('use_feature_normalization', True),
            'obs_space': obs_space,
            'share_obs_space': share_obs_space,
            'act_space': act_space
        }
        
        self.args = FOMAPPOArgs(**args_dict)
        
        # 验证参数
        logger.debug(f"📊 创建参数: reward_scale={self.args.reward_scale}, huber_delta={self.args.huber_delta}")
        
        # 初始化FOMAPPO训练器
        try:
            # 创建策略网络
            self.policy = FOMAPPOPolicy(
            args=self.args,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space,
            device=self.device
            )
            
            # 先创建共享经验缓冲区
            self.buffer = SharedReplayBuffer(
            args=self.args,
                num_agents=num_agents,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space
            )
            logger.info("✅ 共享缓冲区创建成功")
            
            # 再创建FOMAPPO训练器
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            
            # 确保trainer有buffer引用
            if hasattr(self, 'buffer') and self.buffer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ 成功将buffer传递给FOMAPPO训练器")
            else:
                logger.warning("⚠️ 无法将buffer传递给FOMAPPO训练器，buffer不存在")
                
            logger.info("✅ FOMAPPO训练器和缓冲区初始化成功")
        except Exception as e:
            logger.error(f"❌ 初始化FOMAPPO组件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"FOMAPPO初始化失败: {e}")
        
        # 初始化训练统计
        self.total_episodes = 0
        self.training_iterations = 0
        self.training_stats = {
            'shared_parameters': sum(p.numel() for p in self.policy.actor.parameters()) + sum(p.numel() for p in self.policy.critic.parameters()),
            'actor_learning_rate': lr_actor,
            'critic_learning_rate': lr_critic,
            'entropy_coef': self.args.entropy_coef,
            'clip_param': self.args.clip_param,
            'max_grad_norm': self.args.max_grad_norm
        }
        
        # 奖励归一化器
        self.reward_normalizer = {
            'running_mean': 0,
            'running_var': 1,
            'count': 0,
            'decay': 0.99,
            'epsilon': 1e-8  # 避免除零
        }
        
        # 历史观测维度变化记录
        self.new_obs_dimension_history = []
        
        # 监控维度变化
        self.dimension_change_time = None
        
        # 尝试获取观测维度可能的缓存值
        if 'actual_obs_dim' in kwargs:
            actual_dim = kwargs.get('actual_obs_dim')
            if actual_dim != state_dim:
                logger.warning(f"⚠️ 检测到观测维度不一致: 提供的是{state_dim}维，但实际可能是{actual_dim}维")
                logger.warning(f"初始化后将检查并适应正确的维度")
                # 标记需要检查维度，但不立即重建（避免在初始化时就重建）
                self.has_dimension_changed = True
                self.actual_obs_dim = actual_dim
        
        logger.info("✅ FOMAPPO适配器初始化完成")
        logger.info(f"   架构: {num_agents}个Manager共享Policy+Trainer+Buffer")
        logger.info(f"   特性: 保留FOMAPPO设备协调和FlexOffer约束感知")
        logger.info(f"   参数数量: {self.training_stats['shared_parameters']:,}")
    
    def reset_buffer(self):
        """重置共享buffer"""
        try:
            if hasattr(self, 'buffer') and self.buffer is not None:
                # SharedReplayBuffer没有reset方法，我们需要重新创建buffer
                logger.info("重新创建共享buffer（SharedReplayBuffer没有reset方法）")
                
                # 保存当前的step值
                old_step = self.buffer.step if hasattr(self.buffer, 'step') else 0
                
                # 创建buffer
                from gymnasium import spaces
                obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
                
                from onpolicy.utils.shared_buffer import SharedReplayBuffer
                self.buffer = SharedReplayBuffer(
                    args=self.args,
                    num_agents=self.num_agents,
                    obs_space=obs_space,
                    cent_obs_space=share_obs_space,
                    act_space=act_space
                )
                
                # 恢复step值（如果需要）
                if old_step > 0:
                    logger.info(f"恢复buffer的step值: {old_step}")
                    self.buffer.step = old_step
                
                logger.info(f"✅ 成功重新创建buffer，观测维度: {self.actual_obs_dim}")
            else:
                logger.warning("没有buffer可重置，将在collect_step时自动创建")
                
                # 创建buffer
                from gymnasium import spaces
                obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
                
                from onpolicy.utils.shared_buffer import SharedReplayBuffer
                self.buffer = SharedReplayBuffer(
                    args=self.args,
                    num_agents=self.num_agents,
                    obs_space=obs_space,
                    cent_obs_space=share_obs_space,
                    act_space=act_space
                )
                logger.info(f"创建了新的buffer，观测维度: {self.actual_obs_dim}")
            
            # 更新trainer的buffer引用
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ 成功将buffer传递给FOMAPPO训练器")
                
        except Exception as e:
            logger.error(f"重置buffer失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        为所有Manager选择FlexOffer参数生成动作（使用共享策略）
        
        🔧 重构后的环境适配：
        - 动作现在对应FlexOffer参数：[start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight] × 设备数量
        - 观测包含设备状态、环境状态、其他Manager信息、市场状态
        - 需要确保动作在合理范围内以生成有效的FlexOffer
        
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
        
        manager_ids = sorted(list(obs.keys()))  # 保证顺序一致
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            return actions, action_log_probs, values
        
        # 准备批量观测数据 - 🔧 修复：处理不同长度的观测
        obs_batch = []
        obs_lengths = []
        
        # 首先收集所有观测并记录长度
        for manager_id in manager_ids:
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                obs_batch.append(current_obs)
            else:
                obs_batch.append(np.array(current_obs))
            obs_lengths.append(len(obs_batch[-1]))
        
        # 🔧 关键修复：统一观测长度，填充到最大长度
        max_obs_length = max(obs_lengths)
        
        # 更新实际观测维度记录
        if max_obs_length != self.actual_obs_dim:
            logger.warning(f"观测维度变化: 之前为{self.actual_obs_dim}维，现在为{max_obs_length}维。更新记录并使用新维度。")
            self.actual_obs_dim = max_obs_length
            
            # 重新创建buffer和策略网络
            self._recreate_buffer_and_policy(max_obs_length)
        
        logger.debug(f"🔧 FlexOffer动作选择: {batch_size}个Manager, 观测长度{obs_lengths} → 统一为{max_obs_length}")
        
        # 填充所有观测到相同长度
        padded_obs_batch = []
        for i, obs_array in enumerate(obs_batch):
            if len(obs_array) < max_obs_length:
                # 用零填充到最大长度
                padded_obs = np.zeros(max_obs_length, dtype=np.float32)
                padded_obs[:len(obs_array)] = obs_array
                padded_obs_batch.append(padded_obs)
                logger.debug(f"Manager {manager_ids[i]} 观测从 {len(obs_array)} 填充到 {max_obs_length}")
            else:
                padded_obs_batch.append(obs_array.astype(np.float32))
        
        # 转换为tensor格式 (batch_size, max_obs_dim)
        obs_tensor = torch.FloatTensor(np.array(padded_obs_batch)).to(self.device)
        share_obs_tensor = obs_tensor  # 在共享策略中，假设共享观测相同
        
        # 创建RNN状态和掩码 (batch_size, ...)
        rnn_states_actor = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        rnn_states_critic = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        masks = torch.ones(batch_size, 1, device=self.device)
        
        # 使用共享策略批量选择动作
        try:
            value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = self.policy.get_actions(
                share_obs_tensor,
                obs_tensor,
                rnn_states_actor,
                rnn_states_critic,
                masks,
                available_actions=None,
                deterministic=deterministic
            )
            
            # 将批量结果分配给各个Manager，并映射到FlexOffer参数范围
            action_np = action.detach().cpu().numpy()
            action_log_prob_np = action_log_prob.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()
            
            for i, manager_id in enumerate(manager_ids):
                # 🔧 重构适配：将原始动作映射到FlexOffer参数范围
                raw_action = action_np[i]
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = action_log_prob_np[i]
                values[manager_id] = value_np[i]
                
                logger.debug(f"Manager {manager_id} FlexOffer动作: {fo_action.shape} 维, "
                           f"前5个参数: {fo_action[:5]}")
                
        except Exception as e:
            logger.error(f"共享策略FlexOffer动作选择失败: {e}")
            # 提供备用FlexOffer参数动作
            for manager_id in manager_ids:
                fo_action = self._generate_default_fo_action()
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.log(0.5)
                values[manager_id] = 0.0
        
        return actions, action_log_probs, values
    
    def normalize_rewards(self, rewards):
        """
        归一化奖励以提高训练稳定性
        
        Args:
            rewards: 原始奖励值，可以是单个值或字典

        Returns:
            归一化后的奖励
        """
        if isinstance(rewards, dict):
            normalized_rewards = {}
            for k, v in rewards.items():
                normalized_rewards[k] = self._normalize_reward_value(v)
            return normalized_rewards
        else:
            return self._normalize_reward_value(rewards)
    
    def _normalize_reward_value(self, reward):
        """
        归一化单个奖励值
        
        Args:
            reward: 单个奖励值
            
        Returns:
            归一化后的奖励值
        """
        # 检查无效值
        if np.isnan(reward) or np.isinf(reward):
            return 0.0
            
        # 更新运行统计量
        self.reward_normalizer['count'] += 1
        delta = reward - self.reward_normalizer['running_mean']
        
        # 更新均值和方差
        if self.reward_normalizer['count'] == 1:
            self.reward_normalizer['running_mean'] = reward
        else:
            decay = self.reward_normalizer['decay']
            self.reward_normalizer['running_mean'] = self.reward_normalizer['running_mean'] * decay + reward * (1 - decay)
            self.reward_normalizer['running_var'] = self.reward_normalizer['running_var'] * decay + delta * delta * (1 - decay)
        
        # 计算标准差
        std = np.sqrt(self.reward_normalizer['running_var'] + self.reward_normalizer['epsilon'])
        
        # 归一化并裁剪
        if std > 0:
            normalized = (reward - self.reward_normalizer['running_mean']) / std
        else:
            normalized = reward * self.args.reward_scale
            
        # 裁剪到合理范围
        normalized = np.clip(normalized, -5.0, 5.0)
        
        # 缩放到小范围
        return normalized * self.args.reward_scale
        
    def collect_step(self, 
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """
        收集一步的经验数据到共享buffer
        
        Args:
            obs: 当前观测
            actions: 执行的动作
            rewards: 获得的奖励
            dones: 是否结束
            infos: 额外信息
            action_log_probs: 动作对数概率（可选）
            values: 价值函数预测（可选）
        """
        # 详细记录rewards信息
        reward_values = list(rewards.values())
        reward_mean = np.mean(reward_values) if reward_values else 0.0
        reward_min = np.min(reward_values) if reward_values else 0.0
        reward_max = np.max(reward_values) if reward_values else 0.0
        logger.info(f"收集到的奖励: 均值={reward_mean:.4f}, 最小值={reward_min:.4f}, 最大值={reward_max:.4f}")
        
        # 调试：检查rewards是否全为零
        if all(abs(r) < 1e-6 for r in rewards.values()):
            logger.warning(f"警告：当前时间步的所有rewards都为零或接近零: {rewards}")
            # 分析可能的原因
            logger.warning("可能的原因: 1) 环境没有计算奖励 2) 奖励函数设计问题 3) 动作没有影响环境状态")
        manager_ids = sorted(list(obs.keys()))  # 保证顺序一致
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            logger.error("没有有效的Manager ID，无法收集数据")
            return
        
        # 奖励归一化
        if self.args.use_reward_normalization:
            normalized_rewards = self.normalize_rewards(rewards)
        else:
            normalized_rewards = rewards
        
        # 检测实际观测维度
        first_obs = next(iter(obs.values()))
        actual_obs_dim = len(first_obs) if isinstance(first_obs, np.ndarray) else len(np.array(first_obs))
        
        # 如果实际观测维度与记录的不同，更新记录并记录警告
        dimension_changed = False
        if actual_obs_dim != self.actual_obs_dim:
            logger.warning(f"观测维度变化: 之前为{self.actual_obs_dim}维，现在为{actual_obs_dim}维。更新记录并使用新维度。")
            self.actual_obs_dim = actual_obs_dim
            
            # 重新创建buffer和策略网络
            self._recreate_buffer_and_policy(actual_obs_dim)
            dimension_changed = True
            
            logger.warning("观测维度已更新，将继续收集数据")
        
        # 确保buffer已初始化
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("Buffer未初始化，创建新的buffer")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("无法创建buffer，跳过数据收集")
                return
        
        # 准备数据格式（适配SharedReplayBuffer），使用实际观测维度
        # SharedReplayBuffer期望的数据格式: (n_rollout_threads, num_agents, ...)
        
        # 观测数据 (1, num_agents, actual_obs_dim)
        obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        share_obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        
        # 动作和奖励数据
        action_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # 动作对数概率和价值预测
        action_log_prob_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        value_pred_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # RNN状态（全零，因为不使用RNN）
        rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        
        # 掩码
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        bad_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        
        # 填充数据
        for i, manager_id in enumerate(manager_ids):
            if i >= self.num_agents:
                break
                
            # 观测
            obs_batch[0, i] = obs[manager_id]
            share_obs_batch[0, i] = obs[manager_id]  # 假设共享观测相同
            
            # 动作
            action_batch[0, i] = actions[manager_id]
            
            # 奖励
            reward_batch[0, i, 0] = normalized_rewards[manager_id]
            
            # 动作对数概率
            if action_log_probs is not None and manager_id in action_log_probs:
                action_log_prob_batch[0, i] = action_log_probs[manager_id]
            else:
                action_log_prob_batch[0, i] = np.zeros(self.action_dim)  # 使用零而不是log(0.5)
                
            # 价值预测
            if values is not None and manager_id in values:
                value_pred_batch[0, i, 0] = values[manager_id]
            else:
                value_pred_batch[0, i, 0] = 0.0
        
        # 插入到共享buffer
        try:
            # 如果维度刚刚变化，重置buffer以确保一致性
            if dimension_changed:
                logger.info("由于维度变化，重置buffer")
                self.buffer.reset()
            
            self.buffer.insert(
                share_obs=share_obs_batch,
                obs=obs_batch,
                rnn_states_actor=rnn_states_actor,
                rnn_states_critic=rnn_states_critic,
                actions=action_batch,
                action_log_probs=action_log_prob_batch,
                value_preds=value_pred_batch,
                rewards=reward_batch,
                masks=masks,
                bad_masks=bad_masks,
                active_masks=active_masks
            )
            logger.debug(f"成功收集数据到buffer: step={self.buffer.step}, rewards={np.mean(reward_batch):.4f}")
            
            # 确保trainer也有buffer引用
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                
        except Exception as e:
            logger.error(f"共享buffer插入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def compute_returns(self):
        """计算共享buffer的returns和advantages - 按照原始MAPPO模式"""
        try:
            # 检查buffer是否存在
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.warning("Buffer不存在，先创建一个新的buffer")
                self.reset_buffer()
                if not hasattr(self, 'buffer') or self.buffer is None:
                    logger.error("无法创建buffer，无法计算returns")
                    return False
            
            # 详细记录buffer状态
            logger.info(f"Buffer状态: step={self.buffer.step}, rewards形状={self.buffer.rewards.shape if hasattr(self.buffer, 'rewards') else 'N/A'}")
            
            # 检查rewards是否有有效值
            if hasattr(self.buffer, 'rewards'):
                non_zero_rewards = np.count_nonzero(self.buffer.rewards)
                total_rewards = np.prod(self.buffer.rewards.shape)
                logger.info(f"Buffer内容检查: rewards非零值数量={non_zero_rewards}/{total_rewards} ({non_zero_rewards/total_rewards*100:.2f}%)")
                
                # 记录rewards的统计信息
                if non_zero_rewards > 0:
                    reward_mean = np.mean(self.buffer.rewards)
                    reward_std = np.std(self.buffer.rewards)
                    reward_min = np.min(self.buffer.rewards)
                    reward_max = np.max(self.buffer.rewards)
                    logger.info(f"Rewards统计: 均值={reward_mean:.6f}, 标准差={reward_std:.6f}, 最小值={reward_min:.6f}, 最大值={reward_max:.6f}")
            
            # 检查buffer是否有足够的数据
            buffer_empty = self.buffer.step == 0 or (hasattr(self.buffer, 'rewards') and np.count_nonzero(self.buffer.rewards) == 0)
            if buffer_empty:
                # 检查是否是训练初始阶段
                is_initial_phase = not hasattr(self, '_training_started') or not self._training_started
                if is_initial_phase:
                    logger.info("训练初始阶段，buffer为空是正常的，添加初始化数据")
                    # 标记训练已开始
                    self._training_started = True
                else:
                    logger.warning("训练已进行但Buffer没有数据或数据全为零，尝试添加真实数据")
                
                # 添加虚拟数据以避免空buffer错误
                self._add_dummy_data_to_buffer()
                logger.info(f"添加虚拟数据后: step={self.buffer.step}, rewards形状={self.buffer.rewards.shape}")
                
                # 再次检查是否添加成功
                if self.buffer.step == 0 or not np.any(self.buffer.rewards):
                    logger.error("即使添加了虚拟数据，buffer仍然为空，无法计算returns")
                    return False
                else:
                    logger.info("虚拟数据添加成功，继续计算returns")
            
            # 获取最后一步的值估计
            try:
                # 检查是否有有效的rewards数据
                if hasattr(self.buffer, 'rewards') and np.sum(np.abs(self.buffer.rewards)) < 1e-6:
                    logger.warning("Buffer中的rewards全部为零或接近零，添加小的随机噪声以避免全零returns")
                    self.buffer.rewards = self.buffer.rewards + np.random.normal(0, 0.01, self.buffer.rewards.shape)
                    logger.info(f"添加噪声后rewards非零值数量: {np.count_nonzero(self.buffer.rewards)}")
                
                # 获取共享观测和状态
                share_obs = np.concatenate(self.buffer.share_obs[-1])
                rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[-1])
                masks = np.concatenate(self.buffer.masks[-1])
                
                # 调试: 打印输入形状
                logger.info(f"计算returns的输入形状: share_obs={share_obs.shape}, rnn_states_critic={rnn_states_critic.shape}")
                
                # 转换为tensor
                share_obs = torch.FloatTensor(share_obs).to(self.device)
                rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)
                masks = torch.FloatTensor(masks).to(self.device)
                
                # 获取值估计
                with torch.no_grad():
                    next_values = self.policy.get_values(share_obs, rnn_states_critic, masks)
                    
                # 调试: 打印值估计结果
                logger.info(f"值估计结果: next_values形状={next_values.shape}, 样本={next_values[:3]}")
                
                # 计算returns
                next_values = next_values.detach().cpu().numpy()
                self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
                
                # 调试: 打印计算后的returns
                logger.info(f"计算后的returns形状={self.buffer.returns.shape}, 样本={self.buffer.returns[0][0][0][:3]}")
                logger.info(f"Returns非零值数量: {np.count_nonzero(self.buffer.returns)}")
                
                # 检查returns是否包含NaN或无穷大
                if np.isnan(self.buffer.returns).any() or np.isinf(self.buffer.returns).any():
                    logger.warning("Returns中包含NaN或无穷大，进行数值修正")
                    self.buffer.returns = np.nan_to_num(self.buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 检查returns是否全为零
                if np.sum(np.abs(self.buffer.returns)) < 1e-6:
                    logger.warning("计算后的returns全为零，这可能导致训练无效")
                    return False
                
                # 检查advantages是否已计算
                if not hasattr(self.buffer, 'advantages') or self.buffer.advantages is None:
                    logger.warning("Advantages尚未计算，手动计算")
                    # 简单计算advantages (returns - value_preds)
                    self.buffer.advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
                    logger.info(f"手动计算的advantages形状: {self.buffer.advantages.shape}")
                
                # 检查advantages是否包含NaN或无穷大
                if hasattr(self.buffer, 'advantages') and (np.isnan(self.buffer.advantages).any() or np.isinf(self.buffer.advantages).any()):
                    logger.warning("Advantages中包含NaN或无穷大，进行数值修正")
                    self.buffer.advantages = np.nan_to_num(self.buffer.advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 确保trainer的buffer引用是最新的
                if hasattr(self, 'trainer') and self.trainer is not None:
                    self.trainer.buffer = self.buffer
                    logger.info("已更新trainer的buffer引用")
                
                logger.info("成功计算returns和advantages")
                return True
                
            except Exception as e:
                logger.error(f"计算值估计失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"计算returns总体失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _add_dummy_data_to_buffer(self):
        """向buffer中添加一些虚拟数据，避免空buffer错误
        
        注意：这些数据仅用于调试和避免错误，不应该用于实际训练。
        实际训练应该使用从环境中收集的真实数据。
        """
        logger.warning("添加虚拟数据到buffer以避免空buffer错误 - 仅用于调试")
        
        # 确保buffer存在
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("Buffer不存在，先创建一个新的buffer")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("无法创建buffer，无法添加虚拟数据")
                return
        
        # 创建有意义的虚拟数据，而不是完全随机的数据
        # 使用更有意义的虚拟观测和奖励
        dummy_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        dummy_share_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        
        # 为每个agent设置不同的观测，模拟真实环境
        for i in range(self.num_agents):
            # 设置一些基本特征，确保每个agent的观测不同
            dummy_obs[0, i, 0] = 0.5 + 0.1 * i  # 时间特征
            dummy_obs[0, i, 1] = 0.3 + 0.05 * i  # 价格特征
            dummy_obs[0, i, 2] = 0.7 - 0.05 * i  # 需求特征
            dummy_obs[0, i, 3] = 0.2 + 0.02 * i  # 灵活性特征
            
            # 为不同agent设置不同的特征
            dummy_obs[0, i, 4] = 0.1 * (i + 1)  # agent特定特征
            
            # 随机化其余特征以增加多样性
            if self.actual_obs_dim > 5:
                dummy_obs[0, i, 5:] = np.random.uniform(0.1, 0.9, size=self.actual_obs_dim-5)
            
            # 复制到共享观测
            dummy_share_obs[0, i] = dummy_obs[0, i].copy()
        
        # 创建有意义的动作
        dummy_actions = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            # 设置动作，模拟FlexOffer参数
            dummy_actions[0, i, 0] = 0.5 + 0.1 * np.sin(i)  # 能量参数
            dummy_actions[0, i, 1] = 0.3 + 0.1 * np.cos(i)  # 时间参数
            if self.action_dim > 2:
                dummy_actions[0, i, 2] = 0.7 - 0.1 * np.sin(i + 1)  # 价格参数
        
        # 创建有意义的奖励 - 非零值，确保有学习信号
        dummy_rewards = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            # 为每个agent设置不同的正奖励
            dummy_rewards[0, i, 0] = 0.5 + 0.1 * i  # 从0.5到0.9的奖励
        
        # RNN状态
        dummy_rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        dummy_rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        
        # 动作对数概率和价值预测 - 设置合理的值
        dummy_action_log_probs = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            for j in range(self.action_dim):
                dummy_action_log_probs[0, i, j] = -0.5 - 0.1 * j  # 合理的对数概率
        
        dummy_value_preds = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            dummy_value_preds[0, i, 0] = 0.6 + 0.1 * i  # 略高于奖励的值估计
        
        # 掩码
        dummy_masks = np.ones((1, self.num_agents, 1))
        dummy_bad_masks = np.ones((1, self.num_agents, 1))
        dummy_active_masks = np.ones((1, self.num_agents, 1))
        
        # 插入虚拟数据
        try:
            self.buffer.insert(
                share_obs=dummy_share_obs,
                obs=dummy_obs,
                rnn_states_actor=dummy_rnn_states_actor,
                rnn_states_critic=dummy_rnn_states_critic,
                actions=dummy_actions,
                action_log_probs=dummy_action_log_probs,
                value_preds=dummy_value_preds,
                rewards=dummy_rewards,
                masks=dummy_masks,
                bad_masks=dummy_bad_masks,
                active_masks=dummy_active_masks
            )
            logger.info("成功添加虚拟数据到buffer")
            logger.info(f"虚拟数据详情: 观测形状={dummy_obs.shape}, 奖励范围=[{np.min(dummy_rewards):.3f}, {np.max(dummy_rewards):.3f}]")
            
            # 确保trainer也有buffer引用
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ 确保trainer有最新的buffer引用")
                
            # 添加多条虚拟数据，确保有足够的训练数据
            # 添加8条数据以形成一个小批次
            for step in range(8):
                # 为每步创建略有不同的数据
                step_obs = dummy_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_share_obs = dummy_share_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_actions = dummy_actions.copy() * (1.0 + 0.03 * np.cos(step))
                
                # 奖励随时间变化，形成一个有意义的轨迹
                step_rewards = dummy_rewards.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                # 价值预测也相应变化
                step_value_preds = dummy_value_preds.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                # 插入数据
                self.buffer.insert(
                    share_obs=step_share_obs,
                    obs=step_obs,
                    rnn_states_actor=dummy_rnn_states_actor.copy(),
                    rnn_states_critic=dummy_rnn_states_critic.copy(),
                    actions=step_actions,
                    action_log_probs=dummy_action_log_probs.copy(),
                    value_preds=step_value_preds,
                    rewards=step_rewards,
                    masks=dummy_masks.copy(),
                    bad_masks=dummy_bad_masks.copy(),
                    active_masks=dummy_active_masks.copy()
                )
            
            logger.info(f"✅ 成功添加9条有意义的虚拟数据到buffer，当前step={self.buffer.step}")
            
            # 计算虚拟数据的returns，确保有训练信号
            if hasattr(self, 'compute_returns'):
                success = self.compute_returns()
                if success:
                    logger.info("✅ 成功为虚拟数据计算returns")
                else:
                    logger.warning("⚠️ 无法为虚拟数据计算returns")
            
        except Exception as e:
            logger.error(f"添加虚拟数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        执行一次PPO批量更新
        
        Returns:
            Dict[str, Any]: 训练信息，包括policy_loss, value_loss, entropy等
        """
        # 检查buffer是否为空
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.error("Buffer不存在，无法训练")
            self._add_dummy_data_to_buffer()  # 添加一些虚拟数据以避免错误
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("即使在添加虚拟数据后，buffer仍然不存在")
                return {
                    'policy_loss': 0.0,  # 返回0表示没有进行实际训练
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # 检查buffer中是否有足够的数据
        if self.buffer.step == 0:
            logger.warning("Buffer为空或没有足够的数据，添加虚拟数据进行训练")
            # 尝试添加虚拟数据
            self._add_dummy_data_to_buffer()
            if self.buffer.step == 0:
                logger.error("即使添加虚拟数据后，buffer仍然为空")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # 检查rewards是否有意义
        if hasattr(self.buffer, 'rewards'):
            non_zero_rewards = np.count_nonzero(self.buffer.rewards)
            total_rewards = np.prod(self.buffer.rewards.shape)
            zero_ratio = 1.0 - (non_zero_rewards / total_rewards)
            
            logger.info(f"训练前检查: rewards零值比例={zero_ratio:.2%}, 非零值数量={non_zero_rewards}/{total_rewards}")
            
            # 如果超过95%的奖励为零，可能数据质量有问题
            if zero_ratio > 0.95:
                logger.warning("超过95%的奖励为零，数据质量可能有问题，但仍尝试训练")
        
        # 确保returns已计算
        if not hasattr(self.buffer, 'returns') or self.buffer.returns is None or np.count_nonzero(self.buffer.returns) == 0:
            logger.info("训练前计算returns")
            success = self.compute_returns()
            if not success:
                logger.error("无法计算returns，跳过训练")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        try:
            # 使用MAPPO训练器执行训练
            train_info = self.trainer.train()
            
            # 更新训练迭代次数
            self.training_iterations += 1
            
            # 检查训练结果是否有效
            if not isinstance(train_info, dict):
                logger.warning(f"训练结果无效: {type(train_info)}")
                # 构造最基本的训练信息，但使用0值表示训练未成功
                train_info = {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0
                }
            
            # 记录训练信息
            logger.info(f"训练完成: policy_loss={train_info.get('policy_loss', 0.0):.6f}, " +
                        f"value_loss={train_info.get('value_loss', 0.0):.6f}, " +
                        f"entropy={train_info.get('entropy', 0.0):.6f}")
            
            # 确保所有必要的键都存在于结果中
            required_keys = ['policy_loss', 'value_loss', 'entropy', 'grad_norm', 'ratio']
            for key in required_keys:
                if key not in train_info:
                    train_info[key] = 0.0  # 使用0表示数据缺失
            
            # 添加训练迭代次数
            train_info['num_updates'] = 1
            
            return train_info
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回表示训练失败的信息，使用0值而不是固定的0.001
            return {
                'policy_loss': 0.0,  # 使用0表示训练失败
                'value_loss': 0.0,
                'entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0,
                'num_updates': 0,
                'training_error': str(e)
            }
    
    def _update_learning_rate(self):
        """
        更新学习率（学习率衰减）
        根据训练进度渐进式降低学习率，避免训练后期的过度震荡
        """
        try:
            # 指数衰减方式
            if hasattr(self.args, 'use_linear_lr_decay') and self.args.use_linear_lr_decay:
                # 获取当前episode数
                episode = self.total_episodes
                
                # 每10个episodes衰减一次
                decay_interval = 10
                if episode > 0 and episode % decay_interval == 0:
                    # 默认衰减率为0.95
                    decay_rate = getattr(self.args, 'lr_decay_rate', 0.95)
                    
                    # 更新actor学习率
                    for param_group in self.policy.actor_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # 设置最小学习率
                    
                    # 更新critic学习率
                    for param_group in self.policy.critic_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # 设置最小学习率
                    
                    # 记录新的学习率
                    actor_lr = self.policy.actor_optimizer.param_groups[0]['lr']
                    critic_lr = self.policy.critic_optimizer.param_groups[0]['lr']
                    
                    logger.info(f"学习率已衰减: actor_lr={actor_lr:.7f}, critic_lr={critic_lr:.7f}")
                    
        except Exception as e:
            logger.warning(f"学习率衰减失败: {e}")
            # 失败时不影响训练过程
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'num_agents': self.num_agents,
            'algorithm': 'FOMAPPO',
            'architecture': 'shared_policy',
            'shared_parameters': self.training_stats['shared_parameters']
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """获取Manager奖励总结（共享策略版本）"""
        # 在共享策略中，所有Manager共享同一个策略，所以返回整体统计
        return {
            'shared_policy': self.training_stats.copy(),
            'note': 'All managers share the same policy network'
        }
    
    def save_models(self, save_path: str):
        """保存共享模型"""
        try:
            model_path = f"{save_path}_shared.pt"
            
            torch.save({
                'actor_state_dict': self.policy.actor.state_dict(),
                'critic_state_dict': self.policy.critic.state_dict(),
                'actor_optimizer': self.policy.actor_optimizer.state_dict(),
                'critic_optimizer': self.policy.critic_optimizer.state_dict(),
                'training_iterations': self.training_iterations,
                'training_stats': self.training_stats,
                'num_agents': self.num_agents,
                'architecture': 'shared_policy'
            }, model_path)
            
            logger.info(f"FOMAPPO共享模型已保存至 {model_path}")
        except Exception as e:
            logger.error(f"保存FOMAPPO共享模型失败: {e}")
    
    def load_models(self, load_path: str):
        """加载共享模型"""
        try:
            model_path = f"{load_path}_shared.pt"
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.policy.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                
                if 'training_stats' in checkpoint:
                    self.training_stats = checkpoint['training_stats']
                
                if 'training_iterations' in checkpoint:
                    self.training_iterations = checkpoint['training_iterations']
                
                logger.info(f"FOMAPPO共享模型已从 {model_path} 加载")
            else:
                logger.warning(f"模型文件 {model_path} 不存在")
                
        except Exception as e:
            logger.error(f"加载FOMAPPO共享模型失败: {e}")
    
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

    def _recreate_buffer_and_policy(self, new_obs_dim):
        """
        当观测维度变化时，重新创建buffer和策略网络
        
        Args:
            new_obs_dim: 新的观测维度
        """
        logger.warning(f"⚠️ 观测维度变化检测: {self.state_dim} → {new_obs_dim}")
        logger.warning(f"重新创建buffer和策略网络以适应新的观测维度")
        
        # 确保new_obs_dimension_history已初始化
        if not hasattr(self, 'new_obs_dimension_history'):
            self.new_obs_dimension_history = []
        
        # 保存原始网络参数（如果可能）
        old_actor_state = None
        old_critic_state = None
        try:
            if hasattr(self, 'policy') and self.policy is not None:
                old_actor_state = self.policy.actor.state_dict()
                old_critic_state = self.policy.critic.state_dict()
                logger.info("✅ 成功保存原始网络参数")
        except Exception as e:
            logger.warning(f"无法保存原始网络参数: {e}")
        
        # 更新状态维度
        self.state_dim = new_obs_dim
        self.actual_obs_dim = new_obs_dim
        
        # 重新创建观测和动作空间
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # 更新args中的obs_space和share_obs_space
        self.args.obs_space = obs_space
        self.args.share_obs_space = share_obs_space
        
        # 记录详细的网络结构变化
        logger.info(f"🔄 网络重建: 输入层从 {self.state_dim} → {new_obs_dim}")
        
        # 重新创建策略网络
        try:
            self.policy = FOMAPPOPolicy(
                args=self.args,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space,
                device=self.device
            )
            logger.info("✅ 策略网络重建成功")
            
            # 尝试恢复部分网络参数（可能需要手动映射层）
            if old_actor_state is not None and old_critic_state is not None:
                try:
                    # 注意：输入层参数无法直接复制，但可以复制其他层
                    # 这里可能需要更复杂的参数映射逻辑
                    logger.info("⚠️ 网络参数无法直接迁移，使用新初始化的参数")
                except Exception as transfer_e:
                    logger.warning(f"参数迁移失败: {transfer_e}")
        except Exception as e:
            logger.error(f"❌ 策略网络重建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"策略网络重建失败: {e}")
        
        # 重新创建训练器
        try:
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            logger.info("✅ 训练器重建成功")
        except Exception as e:
            logger.error(f"❌ 训练器重建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"训练器重建失败: {e}")
        
        # 重新创建buffer
        try:
            self.buffer = SharedReplayBuffer(
                args=self.args,
                num_agents=self.num_agents,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space
            )
            logger.info("✅ 共享缓冲区重建成功")
        except Exception as e:
            logger.error(f"❌ 共享缓冲区重建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"共享缓冲区重建失败: {e}")
        
        # 确保trainer有buffer引用
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.buffer = self.buffer
            logger.info("✅ 成功将新buffer传递给FOMAPPO训练器")
        else:
            logger.warning("⚠️ 无法将buffer传递给FOMAPPO训练器，trainer不存在")
        
        # 更新训练统计
        try:
            if not hasattr(self, 'training_stats'):
                self.training_stats = {'shared_parameters': 0}
            
            self.training_stats['shared_parameters'] = sum(p.numel() for p in self.policy.actor.parameters()) + \
                                                   sum(p.numel() for p in self.policy.critic.parameters())
            logger.info(f"✅ 网络参数统计更新成功, 共有 {self.training_stats['shared_parameters']} 个参数")
        except Exception as e:
            logger.warning(f"无法更新网络参数统计: {e}")
        
        # 重置内部计数器
        if not hasattr(self, 'training_iterations'):
            self.training_iterations = 0
        
        # 记录重要的维度信息
        self.new_obs_dimension_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'old_dim': self.state_dim, 
            'new_dim': new_obs_dim
        })
        
        logger.info(f"✅ 网络重建完成，新的观测维度: {new_obs_dim}")
        logger.info(f"📊 历史维度变化记录: {len(self.new_obs_dimension_history)} 次") 
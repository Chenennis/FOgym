import torch
import numpy as np
import sys
import os

# 添加onpolicy模块路径（修正版）
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/

# 🔧 关键修复：添加包含onpolicy的父目录，而不是onpolicy目录本身
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

# 现在可以安全导入onpolicy模块
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class FOMAPPOPolicy:
    """
    FlexOffer Multi-Agent PPO Policy类
    
    专门为FlexOffer系统设计的策略网络，支持：
    - Manager级别的观测和动作
    - 设备级别的状态感知
    - FlexOffer约束的集成
    - 多智能体协作机制
    
    Args:
        args: 参数配置
        obs_space: 观测空间
        cent_obs_space: 集中式观测空间（用于critic）
        action_space: 动作空间
        device: 计算设备
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        # FlexOffer特定参数
        self.num_managers = getattr(args, 'num_managers', 4)
        self.devices_per_manager = getattr(args, 'devices_per_manager', 10)
        self.use_device_attention = getattr(args, 'use_device_attention', True)
        self.use_manager_coordination = getattr(args, 'use_manager_coordination', True)

        # 创建actor和critic网络
        self.actor = FOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = FOCritic(args, self.share_obs_space, self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """学习率衰减"""
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, device_states=None, fo_constraints=None):
        """
        计算动作和价值函数预测
        
        Args:
            cent_obs: 集中式观测（用于critic）
            obs: 局部观测（用于actor）
            rnn_states_actor: actor的RNN状态
            rnn_states_critic: critic的RNN状态
            masks: RNN状态重置掩码
            available_actions: 可用动作掩码
            deterministic: 是否确定性动作
            device_states: 设备状态信息
            fo_constraints: FlexOffer约束信息
            
        Returns:
            values: 价值函数预测
            actions: 选择的动作
            action_log_probs: 动作对数概率
            rnn_states_actor: 更新后的actor RNN状态
            rnn_states_critic: 更新后的critic RNN状态
        """
        # 处理FlexOffer特定信息
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        
        # Actor前向传播
        actions, action_log_probs, rnn_states_actor = self.actor(enhanced_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        # Critic前向传播
        values, rnn_states_critic = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        
        # 应用FlexOffer约束
        if fo_constraints is not None:
            actions = self._apply_fo_constraints(actions, fo_constraints)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, device_states=None, fo_constraints=None):
        """获取价值函数预测"""
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        values, _ = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, device_states=None, fo_constraints=None):
        """
        评估动作的对数概率、熵和价值函数
        
        用于actor更新时计算梯度
        """
        # 增强观测
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        
        # Actor评估
        action_log_probs, dist_entropy = self.actor.evaluate_actions(enhanced_obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        # Critic评估
        values, _ = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, 
            device_states=None, fo_constraints=None):
        """仅计算动作（用于推理）"""
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        actions, _, rnn_states_actor = self.actor(enhanced_obs, rnn_states_actor, masks, 
                                                  available_actions, deterministic)
        
        # 应用FlexOffer约束
        if fo_constraints is not None:
            actions = self._apply_fo_constraints(actions, fo_constraints)
            
        return actions, rnn_states_actor

    def _enhance_observation(self, obs, device_states=None, fo_constraints=None):
        """增强观测信息，集成设备状态和FlexOffer约束"""
        if device_states is None and fo_constraints is None:
            return obs
        
        enhanced_obs = obs
        
        # 添加设备状态信息
        if device_states is not None:
            if isinstance(device_states, np.ndarray):
                device_features = torch.FloatTensor(device_states).to(self.device)
            else:
                device_features = device_states
            enhanced_obs = torch.cat([enhanced_obs, device_features], dim=-1)
        
        # 添加FlexOffer约束信息
        if fo_constraints is not None:
            if isinstance(fo_constraints, np.ndarray):
                constraint_features = torch.FloatTensor(fo_constraints).to(self.device)
            else:
                constraint_features = fo_constraints
            enhanced_obs = torch.cat([enhanced_obs, constraint_features], dim=-1)
        
        return enhanced_obs

    def _enhance_centralized_observation(self, cent_obs, device_states=None, fo_constraints=None):
        """增强集中式观测信息"""
        if device_states is None and fo_constraints is None:
            return cent_obs
        
        enhanced_cent_obs = cent_obs
        
        # 添加全局设备状态信息
        if device_states is not None:
            if isinstance(device_states, np.ndarray):
                global_device_features = torch.FloatTensor(device_states).to(self.device)
            else:
                global_device_features = device_states
            
            # 对于集中式观测，可能需要聚合所有Manager的设备状态
            if len(global_device_features.shape) > 2:
                global_device_features = global_device_features.mean(dim=1)  # 聚合Manager维度
            
            enhanced_cent_obs = torch.cat([enhanced_cent_obs, global_device_features], dim=-1)
        
        # 添加全局FlexOffer约束信息
        if fo_constraints is not None:
            if isinstance(fo_constraints, np.ndarray):
                global_constraint_features = torch.FloatTensor(fo_constraints).to(self.device)
            else:
                global_constraint_features = fo_constraints
            
            if len(global_constraint_features.shape) > 2:
                global_constraint_features = global_constraint_features.mean(dim=1)
            
            enhanced_cent_obs = torch.cat([enhanced_cent_obs, global_constraint_features], dim=-1)
        
        return enhanced_cent_obs

    def _apply_fo_constraints(self, actions, fo_constraints):
        """应用FlexOffer约束到动作"""
        if fo_constraints is None:
            return actions
        
        # 简化实现：将动作限制在约束范围内
        # 在实际应用中，这里应该实现更复杂的约束处理逻辑
        constrained_actions = torch.clamp(actions, 0.0, 1.0)
        
        return constrained_actions


class FOActor(R_Actor):
    """FlexOffer专用Actor网络"""
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(FOActor, self).__init__(args, obs_space, action_space, device)
        
        # FlexOffer特定的网络层
        self.device_attention_dim = getattr(args, 'device_attention_dim', 64)
        self.use_device_attention = getattr(args, 'use_device_attention', True)
        
        if self.use_device_attention:
            # 设备注意力机制
            self.device_attention = torch.nn.MultiheadAttention(
                embed_dim=self.device_attention_dim,
                num_heads=4,
                batch_first=True
            ).to(device)


class FOCritic(R_Critic):
    """FlexOffer专用Critic网络"""
    
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(FOCritic, self).__init__(args, cent_obs_space, device)
        
        # FlexOffer特定的网络层
        self.manager_coordination_dim = getattr(args, 'manager_coordination_dim', 128)
        self.use_manager_coordination = getattr(args, 'use_manager_coordination', True)
        
        if self.use_manager_coordination:
            # Manager协作机制
            self.manager_coordination = torch.nn.MultiheadAttention(
                embed_dim=self.manager_coordination_dim,
                num_heads=4,
                batch_first=True
            ).to(device) 
#!/usr/bin/env python3
"""
Dec-POMDP感知的FOMAPPO策略网络
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPFOMAPPOPolicy:
    """Dec-POMDP感知的FOMAPPO策略类"""
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.device = device
        self.dec_pomdp_config = dec_pomdp_config
        
        # 基础参数
        self.lr = getattr(args, 'lr', 0.0003)
        self.critic_lr = getattr(args, 'critic_lr', 0.0003)
        self.opti_eps = getattr(args, 'opti_eps', 1e-5)
        self.weight_decay = getattr(args, 'weight_decay', 0)
        
        # 观测空间维度（基于Dec-POMDP架构）
        # 从传入的obs_space获取总维度
        self.total_obs_dim = obs_space.shape[0]
        
        # 根据总维度计算各部分维度，保持比例
        total_parts = 72  # 原始总和
        self.private_dim = int(self.total_obs_dim * (39/total_parts))  # 私有信息层维度
        self.public_dim = int(self.total_obs_dim * (18/total_parts))   # 公共信息层维度
        self.others_dim = self.total_obs_dim - self.private_dim - self.public_dim  # 有限他者信息层维度
        
        # 确保维度总和正确
        assert self.private_dim + self.public_dim + self.others_dim == self.total_obs_dim, f"维度不匹配: {self.private_dim} + {self.public_dim} + {self.others_dim} != {self.total_obs_dim}"
        
        # 动作空间
        self.act_space = act_space
        self.action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 10
        
        # 创建Dec-POMDP感知的网络
        self.actor = DecPOMDPActor(args, self.private_dim, self.public_dim, self.others_dim, 
                                   self.action_dim, dec_pomdp_config, device)
        self.critic = DecPOMDPCritic(args, self.private_dim, self.public_dim, self.others_dim, 
                                     dec_pomdp_config, device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    
    def parse_observation(self, obs):
        """解析Dec-POMDP观测空间，安全处理不同维度的观测"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # 处理批次维度
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # 获取实际观测维度
        actual_dim = obs.shape[1]
        
        # 检查维度是否匹配预期
        if actual_dim != self.total_obs_dim:
            logger.warning(f"观测维度不匹配: 预期{self.total_obs_dim}维，实际{actual_dim}维。尝试安全处理。")
            
            # 计算安全的切片索引
            safe_private_end = min(self.private_dim, actual_dim)
            safe_public_end = min(self.private_dim + self.public_dim, actual_dim)
            
            # 安全分离三层观测信息
            private_obs = obs[:, :safe_private_end]
            public_obs = obs[:, min(self.private_dim, actual_dim):safe_public_end]
            others_obs = obs[:, min(safe_public_end, actual_dim):]
            
            # 如果维度不足，用零填充
            if safe_private_end < self.private_dim:
                padding = torch.zeros(obs.shape[0], self.private_dim - safe_private_end, device=self.device)
                private_obs = torch.cat([private_obs, padding], dim=1)
                
            if safe_public_end - self.private_dim < self.public_dim:
                padding = torch.zeros(obs.shape[0], self.public_dim - (safe_public_end - self.private_dim), device=self.device)
                public_obs = torch.cat([public_obs, padding], dim=1)
                
            if actual_dim - safe_public_end < self.others_dim:
                padding = torch.zeros(obs.shape[0], self.others_dim - (actual_dim - safe_public_end), device=self.device)
                others_obs = torch.cat([others_obs, padding], dim=1)
        else:
            # 正常分离三层观测信息
            private_obs = obs[:, :self.private_dim]
            public_obs = obs[:, self.private_dim:self.private_dim + self.public_dim]
            others_obs = obs[:, self.private_dim + self.public_dim:]
        
        return private_obs, public_obs, others_obs
    
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False):
        """获取动作和价值预测"""
        # 解析观测
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        # Actor前向传播
        actions, action_log_probs, rnn_states_actor = self.actor(
            private_obs, public_obs, others_obs, rnn_states_actor, masks, 
            available_actions, deterministic
        )
        
        # Critic前向传播
        values, rnn_states_critic = self.critic(
            private_obs, public_obs, others_obs, rnn_states_critic, masks
        )
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """获取价值函数预测"""
        private_obs, public_obs, others_obs = self.parse_observation(cent_obs)
        values, _ = self.critic(private_obs, public_obs, others_obs, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, 
                         masks, available_actions=None, active_masks=None):
        """评估动作的对数概率、熵和价值函数"""
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        # Actor评估
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            private_obs, public_obs, others_obs, rnn_states_actor, action, 
            masks, available_actions, active_masks
        )
        
        # Critic评估
        values, _ = self.critic(private_obs, public_obs, others_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """仅计算动作（用于推理）"""
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        actions, _, rnn_states_actor = self.actor(
            private_obs, public_obs, others_obs, rnn_states_actor, masks,
            available_actions, deterministic
        )
        
        return actions, rnn_states_actor


class DecPOMDPActor(nn.Module):
    """Dec-POMDP感知的Actor网络"""
    
    def __init__(self, args, private_dim, public_dim, others_dim, action_dim, 
                 dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        super(DecPOMDPActor, self).__init__()
        
        self.device = device
        self.config = dec_pomdp_config
        self.hidden_size = getattr(args, 'hidden_size', 256)
        self.action_dim = action_dim
        
        # 记录输入维度
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        logger.info(f"Actor网络输入维度: 私有={private_dim}, 公共={public_dim}, 他者={others_dim}")
        
        # 私有信息处理网络（无噪声，高权重）
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU()
        )
        
        # 公共信息处理网络（标准处理）
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # 他者信息处理网络（带不确定性处理）
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # 信息融合网络
        fusion_input_dim = (self.hidden_size // 2) + (self.hidden_size // 4) + (self.hidden_size // 4)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # 动作输出网络
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.action_dim)
        )
        
        # 动作分布参数
        self.action_std = nn.Parameter(torch.ones(self.action_dim) * 0.1)
        
        self.to(device)
    
    def forward(self, private_obs, public_obs, others_obs, rnn_states, masks, 
                available_actions=None, deterministic=False):
        """前向传播"""
        
        # 处理私有信息（最可靠）
        private_features = self.private_encoder(private_obs)
        
        # 处理公共信息（标准可靠性）
        public_features = self.public_encoder(public_obs)
        
        # 处理他者信息（考虑不确定性）
        others_features = self.others_encoder(others_obs)
        
        # 处理他者信息的可用性
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        # 融合所有信息
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        # 生成动作分布
        action_mean = self.action_head(fused_output)
        action_std = self.action_std.expand_as(action_mean)
        
        # 创建动作分布
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = action_dist.sample()
        
        action_log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # RNN状态处理（简化实现）
        new_rnn_states = rnn_states
        
        return actions, action_log_probs, new_rnn_states
    
    def evaluate_actions(self, private_obs, public_obs, others_obs, rnn_states, action, 
                         masks, available_actions=None, active_masks=None):
        """评估给定动作的对数概率和熵"""
        
        # 重新计算动作分布
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        others_features = self.others_encoder(others_obs)
        
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        action_mean = self.action_head(fused_output)
        action_std = self.action_std.expand_as(action_mean)
        
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        action_log_probs = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_probs, dist_entropy


class DecPOMDPCritic(nn.Module):
    """Dec-POMDP感知的Critic网络"""
    
    def __init__(self, args, private_dim, public_dim, others_dim, 
                 dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        super(DecPOMDPCritic, self).__init__()
        
        self.device = device
        self.config = dec_pomdp_config
        self.hidden_size = getattr(args, 'hidden_size', 256)
        
        # 记录输入维度
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        logger.info(f"Critic网络输入维度: 私有={private_dim}, 公共={public_dim}, 他者={others_dim}")
        
        # 与Actor相似的架构
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU()
        )
        
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # 信息融合网络
        fusion_input_dim = (self.hidden_size // 2) + (self.hidden_size // 4) + (self.hidden_size // 4)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # 价值函数输出
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.to(device)
    
    def forward(self, private_obs, public_obs, others_obs, rnn_states, masks):
        """前向传播"""
        
        # 处理各层观测信息
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        others_features = self.others_encoder(others_obs)
        
        # 处理他者信息的可用性
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        # 融合信息
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        # 计算价值函数
        values = self.value_head(fused_output)
        
        # RNN状态处理（简化实现）
        new_rnn_states = rnn_states
        
        return values, new_rnn_states 
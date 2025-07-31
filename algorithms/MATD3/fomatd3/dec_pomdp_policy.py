#!/usr/bin/env python3
"""
FOMATD3 Dec-POMDP策略网络

为FOMATD3算法提供Dec-POMDP感知的Actor-Critic网络架构。
支持分层观测处理、Twin Delayed DDPG和FlexOffer约束集成。

核心功能：
1. Dec-POMDP分层观测编码
2. Twin Critic网络架构
3. 目标策略平滑化
4. 延迟更新机制
5. FlexOffer约束感知
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from algorithms.MATD3.fomatd3.dec_pomdp_adapter import FOMAtd3DecPOMDPAdapter


class DecPOMDPTD3Actor(nn.Module):
    """
    Dec-POMDP感知的TD3 Actor网络
    
    处理3层观测信息：
    - Private: 40维（39基础 + 1趋势）
    - Public: 18维
    - Others: 15维
    """
    
    def __init__(self, 
                 private_dim: int = 40, 
                 public_dim: int = 18, 
                 others_dim: int = 15,
                 action_dim: int = 4,
                 hidden_dim: int = 256,
                 lr: float = 1e-4,
                 device: torch.device = torch.device("cpu"),
                 config: Optional[DecPOMDPConfig] = None):
        super(DecPOMDPTD3Actor, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.config = config or DecPOMDPConfig()
        
        # 分层信息编码器
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU()
        )
        
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # TD3特定特征融合
        fusion_dim = 64 + 24 + (16 if self.config.enable_other_manager_info else 0)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # FlexOffer约束感知输出层
        self.fo_constraint_layer = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.action_output = nn.Linear(hidden_dim // 4, action_dim)
        
        # 权重初始化
        self._init_weights()
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # 移动到设备
        self.to(device)
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: 包含private, public, others观测的字典
            
        Returns:
            确定性动作输出 [batch_size, action_dim]
        """
        # 确保观测张量具有正确的批次维度
        private_obs = observations['private']
        public_obs = observations['public']
        others_obs = observations['others']
        
        # 如果是1维张量，增加批次维度
        if private_obs.dim() == 1:
            private_obs = private_obs.unsqueeze(0)
        if public_obs.dim() == 1:
            public_obs = public_obs.unsqueeze(0)
        if others_obs.dim() == 1:
            others_obs = others_obs.unsqueeze(0)
        
        # 分层编码
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        # 处理他者信息
        if self.config.enable_other_manager_info:
            others_features = self.others_encoder(others_obs)
            fused_features = torch.cat([private_features, public_features, others_features], dim=1)
        else:
            fused_features = torch.cat([private_features, public_features], dim=1)
        
        # 特征融合
        batch_size = fused_features.size(0)
        if batch_size > 1:
            fused_output = self.fusion_network(fused_features)
        else:
            # 处理单样本情况（跳过BatchNorm）
            x = F.relu(self.fusion_network[0](fused_features))
            x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(self.fusion_network[4](x))
            fused_output = x
        
        # FlexOffer约束处理
        fo_features = F.relu(self.fo_constraint_layer(fused_output))
        
        # 确定性动作输出（TD3特性）
        actions = torch.tanh(self.action_output(fo_features))
        
        return actions
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """获取特征维度信息"""
        return {
            'private_dim': self.private_dim,
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'action_dim': self.action_dim,
            'fusion_dim': 64 + 24 + (16 if self.config.enable_other_manager_info else 0)
        }


class DecPOMDPTD3TwinCritic(nn.Module):
    """
    Dec-POMDP感知的Twin Critic网络
    
    实现TD3的双Critic架构，处理Dec-POMDP观测空间
    """
    
    def __init__(self,
                 private_dim: int = 40,
                 public_dim: int = 18,
                 others_dim: int = 15,
                 action_dim: int = 4,
                 n_agents: int = 4,
                 hidden_dim: int = 256,
                 lr: float = 1e-3,
                 device: torch.device = torch.device("cpu"),
                 config: Optional[DecPOMDPConfig] = None):
        super(DecPOMDPTD3TwinCritic, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = device
        self.config = config or DecPOMDPConfig()
        
        # 输入维度：全局状态+全局动作
        global_state_dim = (private_dim + public_dim + others_dim) * n_agents
        global_action_dim = action_dim * n_agents
        input_dim = global_state_dim + global_action_dim
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2网络（Twin Critic）
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 权重初始化
        self._init_weights()
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # 移动到设备
        self.to(device)
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, global_states: torch.Tensor, global_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回双Critic的Q值
        
        Args:
            global_states: 全局状态 [batch_size, global_state_dim]
            global_actions: 全局动作 [batch_size, global_action_dim]
            
        Returns:
            (Q1值, Q2值)
        """
        # 拼接状态和动作
        x = torch.cat([global_states, global_actions], dim=1)
        batch_size = x.size(0)
        
        # Q1值计算
        if batch_size > 1:
            q1_value = self.q1_network(x)
        else:
            # 处理单样本情况（跳过BatchNorm）
            q1_x = x
            for i, layer in enumerate(self.q1_network):
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                q1_x = layer(q1_x)
            q1_value = q1_x
        
        # Q2值计算
        if batch_size > 1:
            q2_value = self.q2_network(x)
        else:
            # 处理单样本情况（跳过BatchNorm）
            q2_x = x
            for i, layer in enumerate(self.q2_network):
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                q2_x = layer(q2_x)
            q2_value = q2_x
        
        return q1_value.squeeze(-1), q2_value.squeeze(-1)
    
    def Q1(self, global_states: torch.Tensor, global_actions: torch.Tensor) -> torch.Tensor:
        """仅返回Q1值（用于策略更新）"""
        x = torch.cat([global_states, global_actions], dim=1)
        batch_size = x.size(0)
        
        if batch_size > 1:
            q1_value = self.q1_network(x)
        else:
            q1_x = x
            for layer in self.q1_network:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                q1_x = layer(q1_x)
            q1_value = q1_x
        
        return q1_value.squeeze(-1)
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        return {
            'input_dim': (self.private_dim + self.public_dim + self.others_dim) * self.n_agents + self.action_dim * self.n_agents,
            'hidden_dim': self.hidden_dim,
            'n_agents': self.n_agents,
            'twin_critic': True
        }


class DecPOMDPFOMAtd3Policy:
    """
    Dec-POMDP感知的FOMATD3策略
    
    集成Dec-POMDP观测适配器、Actor和Twin Critic网络
    """
    
    def __init__(self,
                 agent_id: int,
                 private_dim: int = 40,
                 public_dim: int = 18,
                 others_dim: int = 15,
                 action_dim: int = 4,
                 n_agents: int = 4,
                 hidden_dim: int = 256,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 device: torch.device = torch.device("cpu"),
                 dec_pomdp_config: Optional[DecPOMDPConfig] = None):
        
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.config = dec_pomdp_config or DecPOMDPConfig()
        
        # 初始化Dec-POMDP观测适配器
        self.observation_adapter = FOMAtd3DecPOMDPAdapter(self.config, device)
        
        # 初始化网络
        self.actor = DecPOMDPTD3Actor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr_actor,
            device=device,
            config=self.config
        )
        
        self.critic = DecPOMDPTD3TwinCritic(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            lr=lr_critic,
            device=device,
            config=self.config
        )
        
        # 目标网络
        self.target_actor = DecPOMDPTD3Actor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr_actor,
            device=device,
            config=self.config
        )
        
        self.target_critic = DecPOMDPTD3TwinCritic(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            lr=lr_critic,
            device=device,
            config=self.config
        )
        
        # 硬更新目标网络
        self.hard_update_target_networks()
        
        # 更新计数器
        self.update_counter = 0
    
    def select_action(self, observation: np.ndarray, manager_id: str,
                     fo_constraints: Optional[np.ndarray] = None,
                     fo_satisfaction: Optional[float] = None,
                     add_noise: bool = True,
                     noise_scale: float = 0.1) -> np.ndarray:
        """
        选择动作
        
        Args:
            observation: 原始观测
            manager_id: Manager ID
            fo_constraints: FlexOffer约束
            fo_satisfaction: FlexOffer满意度
            add_noise: 是否添加探索噪声
            noise_scale: 噪声规模
            
        Returns:
            选择的动作
        """
        # Dec-POMDP观测适配
        adapted_obs = self.observation_adapter.adapt_observation_for_fomatd3(
            observation, manager_id, fo_constraints, fo_satisfaction
        )
        
        # 通过Actor网络选择动作
        with torch.no_grad():
            action = self.actor(adapted_obs)
            
            # TD3探索噪声
            if add_noise:
                noise = torch.normal(0, noise_scale, size=action.shape).to(self.device)
                action = action + noise
                action = torch.clamp(action, -1, 1)
        
        # 移除批次维度（单个动作选择）
        if action.shape[0] == 1:
            action = action.squeeze(0)
        
        return action.cpu().numpy()
    
    def hard_update_target_networks(self):
        """硬更新目标网络"""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def soft_update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'agent_id': self.agent_id,
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params,
            'policy_freq': self.policy_freq,
            'policy_noise': self.policy_noise,
            'observation_adapter': 'FOMAtd3DecPOMDPAdapter',
            'twin_critic': True,
            'dec_pomdp_enabled': True
        }
    
    def save_models(self, checkpoint_dir: str):
        """保存模型"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(self.actor.state_dict(), 
                   os.path.join(checkpoint_dir, f"dec_pomdp_td3_actor_{self.agent_id}.pt"))
        torch.save(self.critic.state_dict(), 
                   os.path.join(checkpoint_dir, f"dec_pomdp_td3_critic_{self.agent_id}.pt"))
        torch.save(self.target_actor.state_dict(), 
                   os.path.join(checkpoint_dir, f"dec_pomdp_td3_target_actor_{self.agent_id}.pt"))
        torch.save(self.target_critic.state_dict(), 
                   os.path.join(checkpoint_dir, f"dec_pomdp_td3_target_critic_{self.agent_id}.pt"))
    
    def load_models(self, checkpoint_dir: str) -> bool:
        """加载模型"""
        try:
            self.actor.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, f"dec_pomdp_td3_actor_{self.agent_id}.pt"),
                map_location=self.device))
            self.critic.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, f"dec_pomdp_td3_critic_{self.agent_id}.pt"),
                map_location=self.device))
            self.target_actor.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, f"dec_pomdp_td3_target_actor_{self.agent_id}.pt"),
                map_location=self.device))
            self.target_critic.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, f"dec_pomdp_td3_target_critic_{self.agent_id}.pt"),
                map_location=self.device))
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False 
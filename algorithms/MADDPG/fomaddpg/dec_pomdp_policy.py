#!/usr/bin/env python3
"""
FOMADDPG Dec-POMDP策略网络

为FOMADDPG算法提供支持Dec-POMDP的Actor-Critic网络架构。
针对确定性策略梯度算法的特点，设计专门的网络结构。

核心特性：
1. Dec-POMDP感知的Actor网络（确定性策略）
2. 集中式训练的Critic网络（多智能体价值评估）
3. 信息融合层（私有+公共+他者信息整合）
4. 目标网络软更新机制
5. DDPG特定的网络结构优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from .dec_pomdp_adapter import FOMaddpgDecPOMDPAdapter

class DecPOMDPActor(nn.Module):
    """
    Dec-POMDP感知的Actor网络（FOMADDPG专用）
    
    专门为确定性策略梯度算法设计的Actor网络，
    支持分层观测信息处理和确定性动作输出。
    """
    
    def __init__(self, 
                 private_dim: int = 40,    # 增强私有观测维度
                 public_dim: int = 18,     # 公共观测维度
                 others_dim: int = 15,     # 他者观测维度
                 action_dim: int = 36,     # 动作维度
                 hidden_dim: int = 256,    # 隐藏层维度
                 max_action: float = 1.0,  # 最大动作值
                 device: str = "cpu"):
        super(DecPOMDPActor, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        
        # 私有信息编码器（最重要）
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 公共信息编码器（次重要）
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 他者信息编码器（辅助信息）
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),  # 更高的dropout，因为他者信息不太可靠
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU()
        )
        
        # 信息融合网络（DDPG关键组件）
        fusion_input_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 确定性策略输出层（DDPG特性）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 确定性策略使用tanh激活
        )
        
        # 动作缩放层
        self.action_scale = nn.Parameter(torch.ones(action_dim) * max_action, requires_grad=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, 
                private_obs: torch.Tensor, 
                public_obs: torch.Tensor, 
                others_obs: torch.Tensor,
                enable_others: bool = True) -> torch.Tensor:
        """
        前向传播 - 确定性策略输出
        
        Args:
            private_obs: 私有观测 [batch_size, private_dim]
            public_obs: 公共观测 [batch_size, public_dim]
            others_obs: 他者观测 [batch_size, others_dim]
            enable_others: 是否启用他者信息
            
        Returns:
            确定性动作 [batch_size, action_dim]
        """
        batch_size = private_obs.shape[0]
        
        # 编码各层信息
        private_features = self.private_encoder(private_obs)  # [batch, hidden//2]
        public_features = self.public_encoder(public_obs)     # [batch, hidden//4]
        
        if enable_others:
            others_features = self.others_encoder(others_obs)  # [batch, hidden//8]
        else:
            others_features = torch.zeros(batch_size, self.hidden_dim // 8).to(self.device)
        
        # 信息融合
        fused_features = torch.cat([private_features, public_features, others_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        # 确定性策略输出
        raw_actions = self.policy_head(fused_representation)
        
        # 应用动作缩放
        scaled_actions = raw_actions * self.action_scale
        
        return scaled_actions
    
    def get_features(self, 
                    private_obs: torch.Tensor, 
                    public_obs: torch.Tensor, 
                    others_obs: torch.Tensor,
                    enable_others: bool = True) -> Dict[str, torch.Tensor]:
        """
        获取特征表示（用于分析和调试）
        
        Returns:
            Dict包含各层特征和融合表示
        """
        batch_size = private_obs.shape[0]
        
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        if enable_others:
            others_features = self.others_encoder(others_obs)
        else:
            others_features = torch.zeros(batch_size, self.hidden_dim // 8).to(self.device)
        
        fused_features = torch.cat([private_features, public_features, others_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        return {
            'private_features': private_features,
            'public_features': public_features,
            'others_features': others_features,
            'fused_representation': fused_representation
        }

class DecPOMDPCritic(nn.Module):
    """
    Dec-POMDP感知的Critic网络（FOMADDPG专用）
    
    支持集中式训练的价值网络，能够处理多智能体的联合状态-动作信息。
    """
    
    def __init__(self,
                 state_dim: int,           # 单智能体状态维度
                 action_dim: int,          # 单智能体动作维度
                 n_agents: int = 4,        # 智能体数量
                 hidden_dim: int = 256,    # 隐藏层维度
                 device: str = "cpu"):
        super(DecPOMDPCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # 集中式输入维度：所有智能体的状态和动作
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents
        total_input_dim = total_state_dim + total_action_dim
        
        # 状态编码网络
        self.state_encoder = nn.Sequential(
            nn.Linear(total_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 动作编码网络
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 状态-动作融合网络
        fusion_input_dim = hidden_dim + hidden_dim // 2
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Q值输出头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # 输出单个Q值
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, 
                global_states: torch.Tensor, 
                global_actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - Q值估计
        
        Args:
            global_states: 所有智能体状态 [batch_size, n_agents * state_dim]
            global_actions: 所有智能体动作 [batch_size, n_agents * action_dim]
            
        Returns:
            Q值 [batch_size, 1]
        """
        # 分别编码状态和动作
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        
        # 状态-动作融合
        fused_features = torch.cat([state_features, action_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        # Q值输出
        q_value = self.q_head(fused_representation)
        
        return q_value
    
    def get_features(self, 
                    global_states: torch.Tensor, 
                    global_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取特征表示（用于分析）
        
        Returns:
            Dict包含状态特征、动作特征和融合表示
        """
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        fused_features = torch.cat([state_features, action_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        return {
            'state_features': state_features,
            'action_features': action_features,
            'fused_representation': fused_representation
        }

class DecPOMDPFOMaddpgPolicy:
    """
    完整的FOMADDPG Dec-POMDP策略类
    
    集成Actor-Critic网络、观测适配器和训练逻辑，
    专门为FOMADDPG算法的Dec-POMDP适配设计。
    """
    
    def __init__(self, 
                 agent_id: int,
                 dec_pomdp_config: DecPOMDPConfig,
                 state_dim: int = 73,      # 适配后状态维度
                 action_dim: int = 36,     # 动作维度
                 n_agents: int = 4,        # 智能体数量
                 hidden_dim: int = 256,    # 隐藏层维度
                 max_action: float = 1.0,  # 最大动作值
                 lr_actor: float = 1e-4,   # Actor学习率
                 lr_critic: float = 1e-3,  # Critic学习率
                 tau: float = 0.005,       # 软更新系数
                 device: str = "cpu"):
        
        self.agent_id = agent_id
        self.config = dec_pomdp_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.tau = tau
        self.device = torch.device(device)
        
        # 创建观测适配器
        self.obs_adapter = FOMaddpgDecPOMDPAdapter(dec_pomdp_config, device)
        
        # 获取适配后的观测维度
        adapted_dims = self.obs_adapter.get_adapted_dimensions()
        private_dim = adapted_dims['private_dim']      # 40
        public_dim = adapted_dims['public_dim']        # 18
        others_dim = adapted_dims['others_dim']        # 15
        
        # 创建Actor网络
        self.actor = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(self.device)
        
        # 创建Actor目标网络
        self.actor_target = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(self.device)
        
        # 创建Critic网络
        self.critic = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(self.device)
        
        # 创建Critic目标网络
        self.critic_target = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 训练统计
        self.train_step = 0
    
    def select_action(self, observation: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        选择动作（确定性策略+噪声探索）
        
        Args:
            observation: 原始观测
            noise_scale: 噪声比例
            
        Returns:
            选择的动作
        """
        # 适配观测
        adapted_obs = self.obs_adapter.adapt_observation_for_fomaddpg(
            observation, f"manager_{self.agent_id}"
        )
        
        # 提取各层观测
        private_obs = adapted_obs['private']
        public_obs = adapted_obs['public']
        others_obs = adapted_obs['others']
        
        # 确定性策略输出
        with torch.no_grad():
            action = self.actor(private_obs, public_obs, others_obs, 
                              enable_others=self.config.enable_other_manager_info)
            action = action.cpu().numpy()[0]
        
        # 添加探索噪声
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def update_networks(self, tau: Optional[float] = None):
        """更新目标网络"""
        if tau is None:
            tau = self.tau
        
        self.soft_update(self.actor_target, self.actor, tau)
        self.soft_update(self.critic_target, self.critic, tau)
    
    def save_models(self, filepath_prefix: str):
        """保存模型"""
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pt")
        torch.save(self.actor_target.state_dict(), f"{filepath_prefix}_actor_target.pt")
        torch.save(self.critic_target.state_dict(), f"{filepath_prefix}_critic_target.pt")
    
    def load_models(self, filepath_prefix: str):
        """加载模型"""
        self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pt", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{filepath_prefix}_actor_target.pt", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f"{filepath_prefix}_critic_target.pt", map_location=self.device))
    
    def get_network_info(self) -> Dict[str, int]:
        """获取网络参数信息"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params,
            'agent_id': self.agent_id,
            'train_step': self.train_step
        } 
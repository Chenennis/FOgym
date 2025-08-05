import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Actor(nn.Module):
    """Actor网络 - 为FlexOffer优化"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # FlexOffer特定的网络结构
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # 批归一化层 - 有助于FlexOffer约束的稳定性
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout层 - 提高泛化能力
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        x = self.fc1(state)
        # 只在batch size > 1时使用批归一化
        if x.size(0) > 1:
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = F.relu(self.bn2(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        
        # 应用FlexOffer约束 - 确保动作在有效范围内
        return self.max_action * x

class Critic(nn.Module):
    """Critic网络 - 支持多智能体状态-动作价值评估"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.n_agents = n_agents
        
        # 输入维度为所有智能体的状态和动作
        total_input_dim = state_dim * n_agents + action_dim * n_agents
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, states, actions):
        """
        前向传播
        
        Args:
            states: 所有智能体的状态 [batch_size, n_agents * state_dim]
            actions: 所有智能体的动作 [batch_size, n_agents * action_dim]
        """
        # 拼接所有智能体的状态和动作
        x = torch.cat([states, actions], dim=1)
        
        x = self.fc1(x)
        # 只在batch size > 1时使用批归一化
        if x.size(0) > 1:
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = F.relu(self.bn2(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value

class FOMaddpgPolicy:
    """
    FlexOffer Multi-Agent DDPG策略类
    
    专门为FlexOffer系统设计的多智能体DDPG策略，
    支持Manager级别的协作学习和设备级别的精确控制。
    """
    
    def __init__(self, 
                 agent_id: int,
                 state_dim: int, 
                 action_dim: int,
                 n_agents: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: str = "cpu"):
        """
        初始化FOMADDPG策略
        
        Args:
            agent_id: 智能体ID
            state_dim: 状态维度
            action_dim: 动作维度  
            n_agents: 智能体数量
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            hidden_dim: 隐藏层维度
            max_action: 最大动作值
            device: 计算设备
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.device = torch.device(device)
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # 创建Critic网络
        self.critic = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # FlexOffer特定参数
        self.fo_constraint_weight = 0.1  # FlexOffer约束权重
        self.coordination_weight = 0.05   # 协调权重
        
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            noise_scale: 噪声比例
            
        Returns:
            选择的动作
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # 添加探索噪声
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update_critic(self, 
                      states: torch.Tensor,
                      actions: torch.Tensor, 
                      rewards: torch.Tensor,
                      next_states: torch.Tensor,
                      next_actions: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float = 0.99) -> float:
        """
        更新Critic网络
        
        Args:
            states: 当前状态批次 [batch_size, n_agents * state_dim]
            actions: 当前动作批次 [batch_size, n_agents * action_dim]
            rewards: 奖励批次 [batch_size, 1]
            next_states: 下一状态批次
            next_actions: 下一动作批次
            dones: 完成标志批次
            gamma: 折扣因子
            
        Returns:
            Critic损失值
        """
        # 计算目标Q值
        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        # 计算当前Q值
        current_q = self.critic(states, actions)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 添加FlexOffer约束损失
        fo_constraint_loss = self._compute_fo_constraint_loss(actions)
        total_loss = critic_loss + self.fo_constraint_weight * fo_constraint_loss
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, 
                     states: torch.Tensor,
                     all_actions: torch.Tensor,
                     agent_actions: torch.Tensor) -> float:
        """
        更新Actor网络
        
        Args:
            states: 状态批次
            all_actions: 所有智能体的动作
            agent_actions: 当前智能体的动作
            
        Returns:
            Actor损失值
        """
        # 计算策略损失
        policy_loss = -self.critic(states, all_actions).mean()
        
        # 添加协调损失 - 鼓励Manager间协作
        coordination_loss = self._compute_coordination_loss(agent_actions, all_actions)
        total_loss = policy_loss + self.coordination_weight * coordination_loss
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return policy_loss.item()
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算FlexOffer约束损失
        
        Args:
            actions: 动作张量
            
        Returns:
            约束损失
        """
        # 简化实现：确保动作在合理范围内
        constraint_violation = torch.relu(torch.abs(actions) - self.max_action)
        return constraint_violation.mean()
    
    def _compute_coordination_loss(self, agent_actions: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        计算协调损失 - 鼓励Manager间协作
        
        Args:
            agent_actions: 当前智能体动作
            all_actions: 所有智能体动作
            
        Returns:
            协调损失
        """
        # 计算动作间的相关性，鼓励适度协调
        if all_actions.size(1) > self.action_dim:
            other_actions = all_actions[:, self.action_dim:]  # 其他智能体的动作
            # 计算动作差异，适度的差异有利于探索
            action_diff = torch.abs(agent_actions.unsqueeze(1) - other_actions.view(-1, self.n_agents-1, self.action_dim))
            # 鼓励适度协调（不是完全一致）
            coordination_loss = torch.relu(0.5 - action_diff.mean())  # 目标差异为0.5
            return coordination_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 更新目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic) 
"""
FlexOffer Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (FOMATD3)

This module implements the main FOMATD3 algorithm for multi-agent reinforcement learning
in FlexOffer systems. FOMATD3 extends MATD3 with FlexOffer-specific constraints and 
multi-agent coordination mechanisms.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from collections import deque

from .fomatd3_policy import FOMATd3Policy


class FOReplayBuffer:
    """FlexOffer-specific经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, n_agents: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0
        
        # 存储缓冲区
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, n_agents, action_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, n_agents), dtype=bool)
        
        # FlexOffer特定信息
        self.fo_constraints = np.zeros((capacity, n_agents, action_dim))  # FlexOffer约束
        self.fo_satisfaction = np.zeros((capacity, n_agents))  # FlexOffer满意度
    
    def add(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_state: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
            fo_satisfaction: np.ndarray = None):
        """添加经验到缓冲区"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = dones
        
        if fo_constraints is not None:
            self.fo_constraints[self.ptr] = fo_constraints
        if fo_satisfaction is not None:
            self.fo_satisfaction[self.ptr] = fo_satisfaction
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """从缓冲区采样批次数据"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.fo_constraints[indices],
            self.fo_satisfaction[indices]
        )
    
    def __len__(self):
        return self.size


class FOMATD3:
    """FlexOffer Multi-Agent Twin Delayed Deep Deterministic Policy Gradient"""
    
    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 hidden_dim: int = 256, max_action: float = 1.0,
                 gamma: float = 0.99, tau: float = 0.005,
                 noise_scale: float = 0.1, noise_clip: float = 0.2,
                 buffer_capacity: int = 100000, batch_size: int = 64,
                 policy_delay: int = 2, device: str = "cpu"):
        """
        初始化FOMATD3算法
        
        Args:
            n_agents: 智能体数量
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            hidden_dim: 隐藏层维度
            max_action: 最大动作值
            gamma: 折扣因子
            tau: 软更新参数
            noise_scale: 噪声尺度
            noise_clip: 噪声裁剪
            buffer_capacity: 经验回放缓冲区容量
            batch_size: 批次大小
            policy_delay: 策略延迟更新频率
            device: 计算设备
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.device = device
        
        # 创建每个智能体的策略
        self.agents = []
        for i in range(n_agents):
            agent = FOMATd3Policy(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device
            )
            self.agents.append(agent)
        
        # 经验回放缓冲区
        self.replay_buffer = FOReplayBuffer(buffer_capacity, state_dim, action_dim, n_agents)
        
        # 训练计数器
        self.total_iterations = 0
        
        # FlexOffer约束权重
        self.fo_constraint_weight = 0.1
        self.fo_satisfaction_weight = 0.2
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """为所有智能体选择动作"""
        actions = np.zeros((self.n_agents, self.action_dim))
        
        for i, agent in enumerate(self.agents):
            # 为每个智能体传入正确的状态切片
            if len(states.shape) == 2:  # 多智能体状态: (n_agents, state_dim)
                agent_state = states[i]  # 选择第i个智能体的状态
            else:  # 单智能体状态或全局状态: (state_dim,)
                agent_state = states  # 使用全局状态
            
            actions[i] = agent.select_action(agent_state, add_noise)
        
        return actions
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                        next_states: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
                        fo_satisfaction: np.ndarray = None):
        """存储经验到回放缓冲区"""
        # 🔧 修复：将多智能体状态展平为全局状态
        if len(states.shape) == 2:  # 多智能体状态: (n_agents, obs_dim)
            global_state = states.flatten()  # 展平为全局状态
            global_next_state = next_states.flatten()
        else:  # 已经是全局状态
            global_state = states
            global_next_state = next_states
            
        self.replay_buffer.add(global_state, actions, rewards, global_next_state, dones, 
                              fo_constraints, fo_satisfaction)
    
    def update(self) -> Optional[Dict[str, float]]:
        """更新所有智能体的策略"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.total_iterations += 1
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones, fo_constraints, fo_satisfaction = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        fo_constraints = torch.FloatTensor(fo_constraints).to(self.device)
        fo_satisfaction = torch.FloatTensor(fo_satisfaction).to(self.device)
        
        # 更新每个智能体的Critic
        critic_losses = []
        for i, agent in enumerate(self.agents):
            critic_loss = self._update_critic(agent, i, states, actions, rewards, 
                                            next_states, dones, fo_constraints, fo_satisfaction)
            critic_losses.append(critic_loss)
        
        # 延迟更新Actor
        actor_losses = []
        if self.total_iterations % self.policy_delay == 0:
            for i, agent in enumerate(self.agents):
                actor_loss = self._update_actor(agent, i, states, actions, fo_constraints)
                actor_losses.append(actor_loss)
                
                # 更新目标网络
                agent.update_target_networks()
        
        return {
            'critic_loss': np.mean(critic_losses) if critic_losses else 0.0,
            'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
            'total_iterations': self.total_iterations
        }
    
    def _update_critic(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                      actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor,
                      dones: torch.Tensor, fo_constraints: torch.Tensor, 
                      fo_satisfaction: torch.Tensor) -> float:
        """更新Critic网络"""
        with torch.no_grad():
            # 计算目标动作
            next_actions = torch.zeros_like(actions)
            for i, next_agent in enumerate(self.agents):
                next_action = next_agent.target_actor(next_states)
                
                # 添加目标策略噪声
                noise = torch.randn_like(next_action) * self.noise_scale
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = torch.clamp(next_action + noise, -self.max_action, self.max_action)
                next_actions[:, i] = next_action
            
            # 展平动作用于Critic输入
            next_actions_flat = next_actions.view(next_actions.size(0), -1)
            
            # 计算目标Q值
            target_q1, target_q2 = agent.target_critic(next_states, next_actions_flat)
            target_q = torch.min(target_q1, target_q2)
            
            # 计算目标值，包含FlexOffer约束奖励
            fo_reward = self._compute_fo_reward(fo_satisfaction[:, agent_idx], fo_constraints[:, agent_idx])
            target_q = rewards[:, agent_idx] + fo_reward + (1 - dones[:, agent_idx]) * self.gamma * target_q
        
        # 当前Q值
        current_actions_flat = actions.view(actions.size(0), -1)
        current_q1, current_q2 = agent.critic(states, current_actions_flat)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化Critic
        agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic.optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                     actions: torch.Tensor, fo_constraints: torch.Tensor) -> float:
        """更新Actor网络"""
        # 计算当前策略动作
        policy_actions = torch.zeros_like(actions)
        for i, policy_agent in enumerate(self.agents):
            if i == agent_idx:
                policy_actions[:, i] = agent.actor(states)
            else:
                with torch.no_grad():
                    policy_actions[:, i] = policy_agent.actor(states)
        
        # 展平动作
        policy_actions_flat = policy_actions.view(policy_actions.size(0), -1)
        
        # Actor损失：最大化Q值
        actor_loss = -agent.critic.Q1(states, policy_actions_flat).mean()
        
        # 添加FlexOffer约束损失
        fo_constraint_loss = self._compute_fo_constraint_loss(
            policy_actions[:, agent_idx], fo_constraints[:, agent_idx]
        )
        
        total_actor_loss = actor_loss + self.fo_constraint_weight * fo_constraint_loss
        
        # 优化Actor
        agent.actor.optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor.optimizer.step()
        
        return total_actor_loss.item()
    
    def _compute_fo_reward(self, fo_satisfaction: torch.Tensor, fo_constraints: torch.Tensor) -> torch.Tensor:
        """计算FlexOffer约束奖励"""
        # 基于FlexOffer满意度的奖励
        satisfaction_reward = self.fo_satisfaction_weight * fo_satisfaction
        
        # 约束违反惩罚
        constraint_penalty = -0.1 * torch.sum(torch.clamp(fo_constraints - 1.0, min=0.0), dim=-1)
        
        return satisfaction_reward + constraint_penalty
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """计算FlexOffer约束损失"""
        # 动作应该满足FlexOffer约束
        constraint_violation = torch.clamp(torch.abs(actions) - torch.abs(constraints), min=0.0)
        return torch.mean(constraint_violation)
    
    def save_models(self, checkpoint_dir: str):
        """保存所有智能体的模型"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
            agent.save_models(agent_dir)
    
    def load_models(self, checkpoint_dir: str):
        """加载所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
            if os.path.exists(agent_dir):
                agent.load_models(agent_dir)
    
    def set_eval_mode(self):
        """设置为评估模式"""
        for agent in self.agents:
            agent.actor.eval()
            agent.critic.eval()
            agent.target_actor.eval()
            agent.target_critic.eval()
    
    def set_train_mode(self):
        """设置为训练模式"""
        for agent in self.agents:
            agent.actor.train()
            agent.critic.train()
            agent.target_actor.train()
            agent.target_critic.train()
    
    def get_action_info(self) -> Dict[str, Any]:
        """获取动作信息，用于调试"""
        return {
            'n_agents': self.n_agents,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
            'noise_scale': self.noise_scale,
            'total_iterations': self.total_iterations
        } 
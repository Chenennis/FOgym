import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .fomaddpg_policy import FOMaddpgPolicy

class ReplayBuffer:
    """经验回放缓冲区 - 支持多智能体经验存储"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, 
             states: np.ndarray, 
             actions: np.ndarray, 
             rewards: np.ndarray, 
             next_states: np.ndarray, 
             dones: np.ndarray):
        """添加经验到缓冲区"""
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """从缓冲区采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)

class FOMADDPG:
    """
    FlexOffer Multi-Agent Deep Deterministic Policy Gradient (FOMADDPG)
    
    专门为FlexOffer系统设计的多智能体DDPG算法。
    支持Manager级别的协作学习和设备级别的精确控制。
    
    主要特点：
    - 设备级状态转移建模
    - Manager间协作机制
    - FlexOffer约束感知的奖励设计
    - 分布式训练和集中式执行
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_scale: float = 0.1,
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256,
                 device: str = "cpu"):
        """
        初始化FOMADDPG算法
        
        Args:
            n_agents: 智能体数量（Manager数量）
            state_dim: 单个智能体状态维度
            action_dim: 单个智能体动作维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            hidden_dim: 网络隐藏层维度
            max_action: 最大动作值
            gamma: 折扣因子
            tau: 软更新系数
            noise_scale: 探索噪声比例
            buffer_capacity: 经验回放缓冲区容量
            batch_size: 批次大小
            device: 计算设备
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # 创建多个智能体策略
        self.agents = []
        for i in range(n_agents):
            agent = FOMaddpgPolicy(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                hidden_dim=hidden_dim,
                max_action=max_action,
                device=device
            )
            self.agents.append(agent)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # FlexOffer特定参数
        self.fo_generation_mode = True  # FlexOffer生成模式
        self.manager_coordination_weight = 0.1  # Manager协调权重
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        为所有智能体选择动作
        
        Args:
            states: 所有智能体的状态 [n_agents, state_dim]
            add_noise: 是否添加探索噪声
            
        Returns:
            所有智能体的动作 [n_agents, action_dim]
        """
        actions = []
        noise_scale = self.noise_scale if add_noise else 0.0
        
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i], noise_scale)
            actions.append(action)
        
        return np.array(actions)
    
    def store_experience(self, 
                        states: np.ndarray, 
                        actions: np.ndarray, 
                        rewards: np.ndarray, 
                        next_states: np.ndarray, 
                        dones: np.ndarray):
        """
        存储经验到回放缓冲区
        
        Args:
            states: 当前状态 [n_agents, state_dim]
            actions: 动作 [n_agents, action_dim]
            rewards: 奖励 [n_agents]
            next_states: 下一状态 [n_agents, state_dim]
            dones: 完成标志 [n_agents]
        """
        # 展平状态和动作以适应集中式训练
        flat_states = states.flatten()
        flat_actions = actions.flatten()
        flat_next_states = next_states.flatten()
        
        # 使用平均奖励作为全局奖励
        global_reward = np.mean(rewards)
        global_done = np.any(dones)
        
        self.replay_buffer.push(
            flat_states, 
            flat_actions, 
            global_reward, 
            flat_next_states, 
            global_done
        )
    
    def update(self) -> Dict[str, float]:
        """
        更新所有智能体的策略
        
        Returns:
            训练统计信息
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # 计算下一状态的动作（使用目标网络）
        next_actions = []
        for i, agent in enumerate(self.agents):
            # 提取每个智能体的下一状态
            agent_next_state = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
            with torch.no_grad():
                next_action = agent.actor_target(agent_next_state)
            next_actions.append(next_action)
        
        next_actions = torch.cat(next_actions, dim=1)
        
        # 更新每个智能体
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i, agent in enumerate(self.agents):
            # 提取当前智能体的状态
            agent_states = states[:, i*self.state_dim:(i+1)*self.state_dim]
            agent_actions = actions[:, i*self.action_dim:(i+1)*self.action_dim]
            
            # 更新Critic
            critic_loss = agent.update_critic(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=next_actions,
                dones=dones,
                gamma=self.gamma
            )
            
            # 更新Actor
            # 创建当前智能体的动作，其他智能体使用当前策略
            current_actions = actions.clone()
            current_actions[:, i*self.action_dim:(i+1)*self.action_dim] = agent.actor(agent_states)
            
            actor_loss = agent.update_actor(
                states=states,
                all_actions=current_actions,
                agent_actions=agent_actions
            )
            
            # 软更新目标网络
            agent.soft_update(agent.actor_target, agent.actor, self.tau)
            agent.soft_update(agent.critic_target, agent.critic, self.tau)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        self.training_step += 1
        
        # 记录训练统计
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'training_step': self.training_step
        }
    
    def generate_flexoffers(self, states: np.ndarray) -> Dict[str, Any]:
        """
        基于当前状态生成FlexOffer
        
        Args:
            states: 当前状态 [n_agents, state_dim]
            
        Returns:
            FlexOffer系统字典
        """
        # 选择动作（不添加噪声，用于推理）
        actions = self.select_actions(states, add_noise=False)
        
        # 将动作转换为FlexOffer参数
        fo_systems = {}
        
        for i in range(self.n_agents):
            manager_id = f"manager_{i+1}"
            agent_action = actions[i]
            
            # 将动作映射到FlexOffer参数
            # 这里是简化实现，实际应该根据具体的FlexOffer模型进行映射
            fo_systems[manager_id] = self._action_to_flexoffer(agent_action, manager_id)
        
        return fo_systems
    
    def _action_to_flexoffer(self, action: np.ndarray, manager_id: str) -> Dict[str, Any]:
        """
        将智能体动作转换为FlexOffer系统
        
        Args:
            action: 智能体动作
            manager_id: Manager ID
            
        Returns:
            FlexOffer系统字典
        """
        # 简化实现：将动作映射为FlexOffer参数
        # 实际实现应该根据具体的FlexOffer模型进行详细映射
        
        device_fos = {}
        
        # 假设每个Manager管理多个设备，动作维度对应不同设备
        devices_per_manager = len(action) // 2  # 假设每个设备需要2个动作参数
        
        for device_idx in range(devices_per_manager):
            device_id = f"device_{manager_id}_{device_idx}"
            
            # 提取设备相关的动作参数
            start_idx = device_idx * 2
            power_action = action[start_idx] if start_idx < len(action) else 0.0
            flexibility_action = action[start_idx + 1] if start_idx + 1 < len(action) else 0.0
            
            # 创建简化的FlexOffer系统
            # 这里应该根据实际的DFO/SFO模型进行创建
            device_fo = {
                'device_id': device_id,
                'power_range': (max(0, power_action - 0.5), max(0, power_action + 0.5)),
                'flexibility': max(0, min(1, flexibility_action)),
                'time_horizon': 24,
                'energy_bounds': self._compute_energy_bounds(power_action, flexibility_action)
            }
            
            device_fos[device_id] = device_fo
        
        return device_fos
    
    def _compute_energy_bounds(self, power_action: float, flexibility_action: float) -> List[Tuple[float, float]]:
        """
        基于动作计算能量边界
        
        Args:
            power_action: 功率动作
            flexibility_action: 灵活性动作
            
        Returns:
            24小时的能量边界列表
        """
        bounds = []
        base_power = max(0, power_action)
        flexibility = max(0, min(1, flexibility_action))
        
        for hour in range(24):
            # 简化的能量边界计算
            min_power = base_power * (1 - flexibility)
            max_power = base_power * (1 + flexibility)
            bounds.append((min_power, max_power))
        
        return bounds
    
    def train_episode(self, env, max_steps: int = 24) -> Dict[str, float]:
        """
        训练一个episode
        
        Args:
            env: 多智能体环境
            max_steps: 最大步数（对应24小时）
            
        Returns:
            Episode统计信息
        """
        states = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            # 选择动作
            actions = self.select_actions(states, add_noise=True)
            
            # 执行动作
            next_states, rewards, dones, infos = env.step(actions)
            
            # 存储经验
            self.store_experience(states, actions, rewards, next_states, dones)
            
            # 更新策略
            if len(self.replay_buffer) >= self.batch_size:
                update_info = self.update()
            
            # 更新状态
            states = next_states
            episode_reward += np.mean(rewards)
            episode_steps += 1
            
            # 检查是否完成
            if np.any(dones):
                break
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_episodes': len(self.episode_rewards)
        }
    
    def save_models(self, filepath_prefix: str):
        """保存所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            agent.save(filepath)
    
    def load_models(self, filepath_prefix: str):
        """加载所有智能体的模型"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            try:
                agent.load(filepath)
            except FileNotFoundError:
                print(f"警告: 无法找到模型文件 {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'training_steps': self.training_step,
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        } 
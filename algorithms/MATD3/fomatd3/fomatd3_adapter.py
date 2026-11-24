#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from collections import deque

logger = logging.getLogger(__name__)


def map_normalized_fo_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Map normalized Actor actions into FO environment units, preserving gradients."""
    width = raw_actions.shape[-1]
    count = width // 5
    if count == 0:
        return torch.zeros_like(raw_actions)
    blocks = raw_actions[..., :count * 5].reshape(*raw_actions.shape[:-1], count, 5)
    mapped = torch.stack((
        blocks[..., 0].clamp(-1., 1.),
        blocks[..., 1].clamp(-1., 1.),
        0.1 + 0.45 * (blocks[..., 2].clamp(-1., 1.) + 1.),
        1.0 + 0.5 * (blocks[..., 3].clamp(-1., 1.) + 1.),
        0.1 + 0.95 * (blocks[..., 4].clamp(-1., 1.) + 1.),
    ), dim=-1).reshape(*raw_actions.shape[:-1], count * 5)
    if count * 5 < width:
        mapped = torch.cat((mapped, torch.zeros_like(raw_actions[..., count * 5:])), dim=-1)
    return mapped


class FOTd3Agent:
    
    def __init__(self, agent_id: str, obs_dim: int, action_dim: int, global_state_dim: int,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, hidden_dim: int = 256,
                 max_action: float = 1.0, gamma: float = 0.99, tau: float = 0.005,
                 device: str = "cpu", valid_action_dim: Optional[int] = None):
        self.valid_action_dim = action_dim if valid_action_dim is None else valid_action_dim
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = torch.device(device)
        
        # Actor network: only receive own observation (original MATD3 architecture)
        self.actor, self.actor_optimizer = self._build_actor_network(obs_dim, action_dim, hidden_dim, lr_actor)
        self.actor_target, _ = self._build_actor_network(obs_dim, action_dim, hidden_dim, lr_actor)
        
        # Critic network: receive global state + own action (original MATD3 architecture)
        critic_input_dim = global_state_dim + action_dim
        self.critic, self.critic_optimizer = self._build_critic_network(critic_input_dim, hidden_dim, lr_critic)
        self.critic_target, _ = self._build_critic_network(critic_input_dim, hidden_dim, lr_critic)
        
        # initialize target networks
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        # training statistics
        self.actor_loss_history = deque(maxlen=100)
        self.critic_loss_history = deque(maxlen=100)
        
    def _build_actor_network(self, input_dim: int, output_dim: int, hidden_dim: int, lr: float):
        """build Actor network"""
        class ActorNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                
                # weight initialization
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)
                
            def forward(self, state):
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
                return torch.tanh(self.fc3(x))
                
        network = ActorNetwork().to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        return network, optimizer
        
    def _build_critic_network(self, input_dim: int, hidden_dim: int, lr: float):
        """build Critic network (Twin Q network)"""
        class CriticNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                # Q1 network
                self.fc1_q1 = nn.Linear(input_dim, hidden_dim)
                self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3_q1 = nn.Linear(hidden_dim, 1)
                
                # Q2 network  
                self.fc1_q2 = nn.Linear(input_dim, hidden_dim)
                self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3_q2 = nn.Linear(hidden_dim, 1)
                
                # weight initialization
                for layer in [self.fc1_q1, self.fc2_q1, self.fc3_q1, 
                             self.fc1_q2, self.fc2_q2, self.fc3_q2]:
                    nn.init.xavier_uniform_(layer.weight)
                    
            def forward(self, state_action):
                # Q1
                q1 = F.relu(self.fc1_q1(state_action))
                q1 = F.relu(self.fc2_q1(q1))
                q1 = self.fc3_q1(q1)
                
                # Q2
                q2 = F.relu(self.fc1_q2(state_action))
                q2 = F.relu(self.fc2_q2(q2))
                q2 = self.fc3_q2(q2)
                
                return q1.squeeze(-1), q2.squeeze(-1)
                
            def Q1(self, state_action):
                q1 = F.relu(self.fc1_q1(state_action))
                q1 = F.relu(self.fc2_q1(q1))
                return self.fc3_q1(q1).squeeze(-1)
                
        network = CriticNetwork().to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        return network, optimizer
        
    def select_action(self, obs: np.ndarray, add_noise: bool = True, noise_scale: float = 0.1) -> np.ndarray:
        """select action"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            action = self.actor(obs_tensor)
            
        action = action.cpu().numpy()
        if len(action.shape) > 1:
            action = action[0]
            
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
        
    def update_critic(self, global_state: torch.Tensor, own_action: torch.Tensor, 
                     reward: torch.Tensor, next_global_state: torch.Tensor, 
                     next_own_action: torch.Tensor, done: torch.Tensor,
                     noise_clip: float = 0.2, target_noise: float = 0.2) -> float:
        """update Critic network - original MATD3 architecture: global state + own action"""
        with torch.no_grad():
            # target policy noise (only for own action)
            noise = torch.randn_like(next_own_action) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_action_noisy = torch.clamp(next_own_action + noise, -self.max_action, self.max_action)
            physical_next_action = torch.zeros_like(next_action_noisy)
            physical_next_action[:, :self.valid_action_dim] = map_normalized_fo_actions(
                next_action_noisy[:, :self.valid_action_dim]
            )
            
            # target Q value calculation (global state + own action)
            next_state_action = torch.cat([next_global_state, physical_next_action], dim=1)
            target_q1, target_q2 = self.critic_target(next_state_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
            
        # current Q value (global state + own action)
        current_state_action = torch.cat([global_state, own_action], dim=1)
        current_q1, current_q2 = self.critic(current_state_action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_loss_history.append(critic_loss.item())
        return critic_loss.item()
        
    def update_actor(self, global_state: torch.Tensor, own_obs: torch.Tensor) -> float:
        """update Actor network"""
        # calculate actor loss
        raw_action = self.actor(own_obs)
        action = torch.zeros_like(raw_action)
        action[:, :self.valid_action_dim] = map_normalized_fo_actions(
            raw_action[:, :self.valid_action_dim]
        )
        state_action = torch.cat([global_state, action], dim=1)
        actor_loss = -self.critic.Q1(state_action).mean()
        
        # optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        self.actor_loss_history.append(actor_loss.item())
        return actor_loss.item()
        
    def soft_update_targets(self):
        """soft update target networks"""
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
        
    def _soft_update(self, target, source, tau):
        """soft update"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def _hard_update(self, target, source):
        """hard update"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class FOMATD3Adapter:
    """FOMATD3 algorithm FO Pipeline adapter - strictly follow original MATD3 architecture"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_scale: float = 0.1,
                 noise_clip: float = 0.2,
                 target_noise: float = 0.2,
                 policy_delay: int = 2,
                 device: str = "cpu",
                 observation_dims: Optional[Dict[str, int]] = None,
                 action_dims: Optional[Dict[str, int]] = None,
                 **kwargs):
        
        self.state_dim = state_dim  # dimension of single agent's observation
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_ids = [f"manager_{i+1}" for i in range(num_agents)]
        self.observation_dims = {
            mid: (observation_dims or {}).get(mid, state_dim) for mid in self.agent_ids
        }
        self.action_dims = {
            mid: (action_dims or {}).get(mid, action_dim) for mid in self.agent_ids
        }
        if any(not (0 < dim <= state_dim) for dim in self.observation_dims.values()) or any(
            not (0 < dim <= action_dim) for dim in self.action_dims.values()
        ):
            raise ValueError("Invalid manager observation/action dimension")
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.target_noise = target_noise
        self.device = device
        
        # global state dimension (for critic)
        self.global_state_dim = state_dim * num_agents
        
        # create agents
        self.agents = {}
        manager_ids = self.agent_ids
        
        for manager_id in manager_ids:
            self.agents[manager_id] = FOTd3Agent(
                agent_id=manager_id,
                obs_dim=state_dim,
                action_dim=action_dim,
                global_state_dim=self.global_state_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                hidden_dim=hidden_dim,
                gamma=gamma,
                tau=tau,
                device=device,
                valid_action_dim=self.action_dims[manager_id]
            )
        
        # experience replay buffer
        self.replay_buffer = self._create_replay_buffer(buffer_capacity)
        
        # training statistics
        self.total_iterations = 0
        self.training_history = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': []
        }
        
        # add training_iterations attribute
        self.training_iterations = 0
        
        # manager reward statistics
        self._manager_rewards = {agent_id: [] for agent_id in manager_ids}
        
        # compatibility parameter object
        class Args:
            def __init__(self, policy_delay, noise_clip, noise_scale, target_noise):
                self.policy_delay = policy_delay
                self.noise_clip = noise_clip
                self.noise_scale = noise_scale
                self.target_noise = target_noise
                
        self.args = Args(policy_delay, noise_clip, noise_scale, target_noise)
        
        # initialize experience cache variables (for handling next_states)
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
        
        # add reward normalization mechanism
        self.reward_normalizer = self._create_reward_normalizer()
        
        # add training improvement parameters
        self.training_improvement_config = {
            'reward_clipping': True,
            'reward_clip_range': (-10.0, 10.0),  # reward clipping range
            'adaptive_noise': True,
            'noise_decay_rate': 0.995,
            'min_noise': 0.01,
            'gradient_clip_value': 1.0,
            'buffer_warm_up': 100,  
        }
        
    def _create_replay_buffer(self, capacity: int):
        """create experience replay buffer"""
        return {
            'states': np.zeros((capacity, self.num_agents, self.state_dim)),
            'actions': np.zeros((capacity, self.num_agents, self.action_dim)),
            'rewards': np.zeros((capacity, self.num_agents)),
            'next_states': np.zeros((capacity, self.num_agents, self.state_dim)),
            'dones': np.zeros((capacity, self.num_agents), dtype=bool),
            'ptr': 0,
            'size': 0,
            'capacity': capacity
        }
        
    def _create_reward_normalizer(self):
        """create reward normalizer"""
        return {
            'running_min': float('inf'),
            'running_max': float('-inf'),
            'running_mean': 0.0,  
            'running_var': 1.0,  
            'count': 0,
            'epsilon': 1e-8
        }
        
    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Use a fixed positive reward scale for consistent Bellman targets.

        A replayed transition must retain the same scaled reward regardless of
        which other transitions were previously sampled.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        if not np.isfinite(rewards).all():
            raise ValueError("MATD3 received non-finite rewards")
        self.reward_normalizer['count'] += rewards.size
        return rewards / 1000.0

    def _update_exploration_noise(self):
        """dynamic adjust exploration noise"""
        if self.training_improvement_config['adaptive_noise']:
            current_noise = self.noise_scale
            decay_rate = self.training_improvement_config['noise_decay_rate']
            min_noise = self.training_improvement_config['min_noise']
            
            new_noise = max(min_noise, current_noise * decay_rate)
            self.noise_scale = new_noise
            
            # update noise for all agents
            for agent in self.agents.values():
                agent.noise_scale = new_noise
        
    def reset_buffers(self):
        """reset buffers"""
        # Off-policy replay must persist across episode boundaries.
        # Only clear any per-episode transition cache.
        # reset experience cache variables
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
        
    def _stack_observations(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        for i, mid in enumerate(self.agent_ids):
            value = np.asarray(obs[mid], dtype=np.float32).reshape(-1)
            if value.size != self.observation_dims[mid]:
                raise ValueError(f"{mid} observation has {value.size} entries; expected {self.observation_dims[mid]}")
            padded[i, :value.size] = value
        return padded

    def _stack_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        for i, mid in enumerate(self.agent_ids):
            value = np.asarray(actions[mid], dtype=np.float32).reshape(-1)
            if value.size != self.action_dims[mid]:
                raise ValueError(f"{mid} action has {value.size} entries; expected {self.action_dims[mid]}")
            padded[i, :value.size] = value
        return padded

    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        actions = {}
        action_log_probs = {}  # TD3 not used, keep compatibility
        values = {}           # TD3 not used, keep compatibility
        
        states = self._stack_observations(obs)
        for i, manager_id in enumerate(self.agent_ids):
            observation = states[i]
            if manager_id in self.agents:
                raw_action = self.agents[manager_id].select_action(
                    observation, 
                    add_noise=not deterministic, 
                    noise_scale=self.noise_scale
                )
                
                fo_action = self._map_action_to_fo_params(
                    raw_action[:self.action_dims[manager_id]]
                )
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.zeros(self.action_dims[manager_id]) 
                values[manager_id] = np.zeros(1)  
                
                logger.debug(f"Manager {manager_id} TD3 FlexOffer action: {fo_action.shape} dim, "
                           f"first 5 parameters: {fo_action[:5]}")
            else:
                logger.warning(f"unknown manager_id: {manager_id}")
                actions[manager_id] = np.random.uniform(-1, 1, self.action_dim)
                action_log_probs[manager_id] = np.zeros(self.action_dim)
                values[manager_id] = np.zeros(1)
        
        return actions, action_log_probs, values
        
    def collect_step(self,
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     next_obs: Dict[str, np.ndarray],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """Store the executed (state, action, reward, next_state, done) transition."""
        manager_ids = list(self.agents.keys())
        states = self._stack_observations(obs)
        next_states = self._stack_observations(next_obs)
        actions_array = self._stack_actions(actions)
        rewards_array = np.asarray([rewards[mid] for mid in manager_ids], dtype=np.float32)
        dones_array = np.asarray([dones[mid] for mid in manager_ids], dtype=bool)

        ptr = self.replay_buffer['ptr']
        self.replay_buffer['states'][ptr] = states
        self.replay_buffer['actions'][ptr] = actions_array
        self.replay_buffer['rewards'][ptr] = rewards_array
        self.replay_buffer['next_states'][ptr] = next_states
        self.replay_buffer['dones'][ptr] = dones_array
        self.replay_buffer['ptr'] = (ptr + 1) % self.replay_buffer['capacity']
        self.replay_buffer['size'] = min(
            self.replay_buffer['size'] + 1, self.replay_buffer['capacity']
        )

    def train_on_batch(self) -> Dict[str, Any]:
        """train a batch - strictly follow original MATD3 architecture (with reward normalization improvement)"""
        # check warm-up period
        warm_up_steps = self.training_improvement_config['buffer_warm_up']
        if self.replay_buffer['size'] < max(self.batch_size, warm_up_steps):
            return {'status': 'warming_up', 'buffer_size': self.replay_buffer['size']}
            
        self.total_iterations += 1
        self.training_iterations = self.total_iterations

        # sample batch
        batch_indices = np.random.choice(self.replay_buffer['size'], self.batch_size, replace=False)
        
        states = torch.FloatTensor(self.replay_buffer['states'][batch_indices]).to(self.device)  # [batch, agents, obs_dim]
        actions = torch.FloatTensor(self.replay_buffer['actions'][batch_indices]).to(self.device)  # [batch, agents, action_dim]
        
        # reward normalization
        raw_rewards = self.replay_buffer['rewards'][batch_indices]  # [batch, agents]
        normalized_rewards = np.zeros_like(raw_rewards)
        for agent_idx in range(self.num_agents):
            agent_rewards = raw_rewards[:, agent_idx]
            normalized_rewards[:, agent_idx] = self._normalize_rewards(agent_rewards)
        
        rewards = torch.FloatTensor(normalized_rewards).to(self.device)  # [batch, agents]
        next_states = torch.FloatTensor(self.replay_buffer['next_states'][batch_indices]).to(self.device)  # [batch, agents, obs_dim]
        dones = torch.FloatTensor(self.replay_buffer['dones'][batch_indices]).to(self.device)  # [batch, agents]
        
        # build global state (concatenate all agent observations)
        global_states = states.view(self.batch_size, -1)  # [batch, agents*obs_dim]
        next_global_states = next_states.view(self.batch_size, -1)  # [batch, agents*obs_dim]
        
        # generate action for next state (for target Q calculation)
        next_actions = []
        manager_ids = list(self.agents.keys())
        
        for i, manager_id in enumerate(manager_ids):
            agent_next_states = next_states[:, i, :]  # [batch, obs_dim]
            with torch.no_grad():
                next_action = self.agents[manager_id].actor_target(agent_next_states)
                next_actions.append(next_action)
        
        # train each agent - original MATD3 architecture
        training_stats = {}
        
        for i, manager_id in enumerate(manager_ids):
            agent_states = states[:, i, :]  # [batch, obs_dim] 
            agent_actions = actions[:, i, :]  # [batch, action_dim]
            agent_rewards = rewards[:, i]  # [batch]
            agent_dones = dones[:, i]  # [batch]
            agent_next_actions = next_actions[i]  # [batch, action_dim]
            
            # update Critic - original MATD3: global state + own action
            critic_loss = self.agents[manager_id].update_critic(
                global_states, agent_actions, agent_rewards,
                next_global_states, agent_next_actions, agent_dones,
                noise_clip=self.noise_clip, target_noise=self.target_noise
            )
            
            # delay update Actor
            if self.total_iterations % self.policy_delay == 0:
                actor_loss = self.agents[manager_id].update_actor(global_states, agent_states)
                self.agents[manager_id].soft_update_targets()
                training_stats[f'{manager_id}_actor_loss'] = actor_loss
                training_stats['actor_updated'] = True
            else:
                training_stats['actor_updated'] = False
                
            training_stats[f'{manager_id}_critic_loss'] = critic_loss
        
        # dynamic adjust exploration noise
        if self.total_iterations % 100 == 0:  # adjust every 100 steps
            self._update_exploration_noise()
            
        # Aggregate the actual per-agent losses for the pipeline's metrics.
        critic_losses = [training_stats[f'{mid}_critic_loss'] for mid in manager_ids]
        actor_losses = [
            training_stats[f'{mid}_actor_loss'] for mid in manager_ids
            if f'{mid}_actor_loss' in training_stats
        ]
        training_stats.update({
            'status': 'updated',
            'policy_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'value_loss': float(np.mean(critic_losses)),
            'total_iterations': self.total_iterations,
            'buffer_size': self.replay_buffer['size'],
            'current_noise_scale': self.noise_scale,
            'reward_stats': {
                'mean': self.reward_normalizer['running_mean'],
                'std': np.sqrt(self.reward_normalizer['running_var']),
                'count': self.reward_normalizer['count']
            }
        })
            
        return training_stats
        
    def compute_returns(self):
        pass
        
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        stats = {}
        for manager_id, agent in self.agents.items():
            if agent.actor_loss_history:
                stats[f'{manager_id}_avg_actor_loss'] = np.mean(agent.actor_loss_history)
            if agent.critic_loss_history:
                stats[f'{manager_id}_avg_critic_loss'] = np.mean(agent.critic_loss_history)
        return stats
        
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """get manager reward statistics summary"""
        summary = {}
        for manager_id, rewards in self._manager_rewards.items():
            if rewards:
                summary[f'{manager_id}_avg_reward'] = np.mean(rewards)
                summary[f'{manager_id}_min_reward'] = np.min(rewards)
                summary[f'{manager_id}_max_reward'] = np.max(rewards)
                summary[f'{manager_id}_total_reward'] = np.sum(rewards)
                summary[f'{manager_id}_num_episodes'] = len(rewards)
            else:
                summary[f'{manager_id}_avg_reward'] = 0.0
                summary[f'{manager_id}_min_reward'] = 0.0
                summary[f'{manager_id}_max_reward'] = 0.0
                summary[f'{manager_id}_total_reward'] = 0.0
                summary[f'{manager_id}_num_episodes'] = 0 
        return summary

    def save_models(self, save_path: str):
        """Save online and target networks, optimizer states, and TD3 counters."""
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            'version': 2,
            'manager_ids': list(self.agent_ids),
            'observation_dims': dict(self.observation_dims),
            'action_dims': dict(self.action_dims),
            'total_iterations': self.total_iterations,
            'training_iterations': self.training_iterations,
            'noise_scale': self.noise_scale,
            'reward_normalizer': dict(self.reward_normalizer),
            'agents': {
                mid: {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'actor_target': agent.actor_target.state_dict(),
                    'critic_target': agent.critic_target.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }
                for mid, agent in self.agents.items()
            },
        }
        torch.save(checkpoint, os.path.join(save_path, 'matd3_checkpoint.pt'))

    def load_models(self, load_path: str):
        """Load a complete MATD3 model checkpoint; fail visibly if incompatible."""
        path = os.path.join(load_path, 'matd3_checkpoint.pt')
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get('version') != 2:
            raise ValueError('Unsupported MATD3 checkpoint version')
        if (checkpoint['manager_ids'] != self.agent_ids
                or checkpoint['observation_dims'] != self.observation_dims
                or checkpoint['action_dims'] != self.action_dims):
            raise ValueError('MATD3 checkpoint manager dimensions do not match')
        for mid, agent in self.agents.items():
            saved = checkpoint['agents'][mid]
            agent.actor.load_state_dict(saved['actor'])
            agent.critic.load_state_dict(saved['critic'])
            agent.actor_target.load_state_dict(saved['actor_target'])
            agent.critic_target.load_state_dict(saved['critic_target'])
            agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.total_iterations = checkpoint['total_iterations']
        self.training_iterations = checkpoint['training_iterations']
        self.noise_scale = checkpoint['noise_scale']
        self.reward_normalizer = checkpoint['reward_normalizer']
        logger.info('MATD3 model checkpoint loaded from %s', path)

    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        raw_tensor = torch.as_tensor(raw_action, dtype=torch.float32)
        return map_normalized_fo_actions(raw_tensor).cpu().numpy()

    def _generate_default_fo_action(self) -> np.ndarray:
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
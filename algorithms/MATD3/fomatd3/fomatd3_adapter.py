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


class FOTd3Agent:
    
    def __init__(self, agent_id: str, obs_dim: int, action_dim: int, global_state_dim: int,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, hidden_dim: int = 256,
                 max_action: float = 1.0, gamma: float = 0.99, tau: float = 0.005,
                 device: str = "cpu"):
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
            
            # target Q value calculation (global state + own action)
            next_state_action = torch.cat([next_global_state, next_action_noisy], dim=1)
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
        action = self.actor(own_obs)
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
                 **kwargs):
        
        self.state_dim = state_dim  # dimension of single agent's observation
        self.action_dim = action_dim
        self.num_agents = num_agents
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
        manager_ids = [f"manager_{i+1}" for i in range(num_agents)]
        
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
                device=device
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
        """normalize rewards - scaling method, keep relative relationship"""

        fixed_scale_normalized = rewards / 1000.0
        
        # dynamic range normalization (keep relative relationship)
        # update running max and min
        if self.reward_normalizer['count'] == 0:
            self.reward_normalizer['running_min'] = np.min(rewards)
            self.reward_normalizer['running_max'] = np.max(rewards)
        else:
            self.reward_normalizer['running_min'] = min(
                self.reward_normalizer['running_min'], np.min(rewards)
            )
            self.reward_normalizer['running_max'] = max(
                self.reward_normalizer['running_max'], np.max(rewards)
            )
        
        self.reward_normalizer['count'] += len(rewards)
        
        # prevent division by zero
        reward_range = self.reward_normalizer['running_max'] - self.reward_normalizer['running_min']
        if reward_range > self.reward_normalizer['epsilon']:
            # normalize to [0,1] range, keep relative relationship
            range_normalized = (rewards - self.reward_normalizer['running_min']) / reward_range
            # further scale to appropriate range
            range_normalized = range_normalized * 0.1  # scale to [0,0.1]
        else:
            # if reward change is small, use fixed scaling
            range_normalized = fixed_scale_normalized
        
        # choose normalization method:
        # if reward change range is large, use dynamic range normalization
        # if reward change is small, use fixed scaling
        if reward_range > 10.0:  # if reward change is larger than 10, use dynamic normalization
            normalized_rewards = range_normalized
        else:  # if reward change is small, use fixed scaling
            normalized_rewards = fixed_scale_normalized
        
        return normalized_rewards
    
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
        self.replay_buffer['ptr'] = 0
        self.replay_buffer['size'] = 0
        
        # reset experience cache variables
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
        
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        actions = {}
        action_log_probs = {}  # TD3 not used, keep compatibility
        values = {}           # TD3 not used, keep compatibility
        
        for manager_id, observation in obs.items():
            if manager_id in self.agents:
                raw_action = self.agents[manager_id].select_action(
                    observation, 
                    add_noise=not deterministic, 
                    noise_scale=self.noise_scale
                )
                
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.zeros(self.action_dim) 
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
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """collect experience step """
        try:
            manager_ids = list(obs.keys())
            
            # convert to array format
            states = np.array([obs[mid] for mid in manager_ids])
            actions_array = np.array([actions[mid] for mid in manager_ids])
            rewards_array = np.array([rewards[mid] for mid in manager_ids])
            dones_array = np.array([dones[mid] for mid in manager_ids])
            
            # use cache mechanism to handle next_states
            if hasattr(self, '_prev_states') and self._prev_states is not None:
                # if there is previous state, store it as experience
                ptr = self.replay_buffer['ptr']
                self.replay_buffer['states'][ptr] = self._prev_states
                self.replay_buffer['actions'][ptr] = self._prev_actions
                self.replay_buffer['rewards'][ptr] = self._prev_rewards
                self.replay_buffer['next_states'][ptr] = states  # current state as previous step's next_states
                self.replay_buffer['dones'][ptr] = self._prev_dones
                
                self.replay_buffer['ptr'] = (ptr + 1) % self.replay_buffer['capacity']
                self.replay_buffer['size'] = min(self.replay_buffer['size'] + 1, self.replay_buffer['capacity'])
            
            # save current state for next storage
            self._prev_states = states.copy()
            self._prev_actions = actions_array.copy()
            self._prev_rewards = rewards_array.copy()
            self._prev_dones = dones_array.copy()
            
        except Exception as e:
            logger.error(f"FOMATD3 experience collection failed: {e}")
            
    def train_on_batch(self) -> Dict[str, Any]:
        """train a batch - strictly follow original MATD3 architecture (with reward normalization improvement)"""
        # check warm-up period
        warm_up_steps = self.training_improvement_config['buffer_warm_up']
        if self.replay_buffer['size'] < max(self.batch_size, warm_up_steps):
            return {'status': 'warming_up', 'buffer_size': self.replay_buffer['size']}
            
        self.total_iterations += 1
        
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
            
        # add training statistics
        training_stats.update({
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
        """save models"""
        os.makedirs(save_path, exist_ok=True)
        for manager_id, agent in self.agents.items():
            torch.save(agent.actor.state_dict(), f"{save_path}/{manager_id}_actor.pt")
            torch.save(agent.critic.state_dict(), f"{save_path}/{manager_id}_critic.pt")
            
    def load_models(self, load_path: str):
        """load models"""
        for manager_id, agent in self.agents.items():
            actor_path = f"{load_path}/{manager_id}_actor.pt"
            critic_path = f"{load_path}/{manager_id}_critic.pt"
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                agent.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                agent.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                agent._hard_update(agent.actor_target, agent.actor)
                agent._hard_update(agent.critic_target, agent.critic)
                logger.info(f"model of {manager_id} loaded")
            else:
                logger.warning(f"model file of {manager_id} not found")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:

        fo_action = np.zeros_like(raw_action)
        
        num_devices = len(raw_action) // 5 if len(raw_action) >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < len(raw_action):
                # start_flex: [-1, 1] → [-1, 1] (keep unchanged)
                fo_action[base_idx] = np.clip(raw_action[base_idx], -1.0, 1.0)
                
                # end_flex: [-1, 1] → [-1, 1] (keep unchanged)
                fo_action[base_idx + 1] = np.clip(raw_action[base_idx + 1], -1.0, 1.0)
                
                # energy_min_factor: [-1, 1] → [0.1, 1.0]
                fo_action[base_idx + 2] = 0.1 + 0.45 * (raw_action[base_idx + 2] + 1.0)
                
                # energy_max_factor: [-1, 1] → [1.0, 2.0]  
                fo_action[base_idx + 3] = 1.0 + 0.5 * (raw_action[base_idx + 3] + 1.0)
                
                # priority_weight: [-1, 1] → [0.1, 2.0]
                fo_action[base_idx + 4] = 0.1 + 0.95 * (raw_action[base_idx + 4] + 1.0)
        
        return fo_action
    
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
#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from collections import deque
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from .dec_pomdp_policy import DecPOMDPFOMaddpgPolicy

class DecPOMDPReplayBuffer:
    
    def __init__(self, 
                 capacity: int = 1000000,
                 n_agents: int = 4,
                 state_dim: int = 73,
                 action_dim: int = 36):
        self.capacity = capacity
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # experience buffer
        self.experiences = deque(maxlen=capacity)
        
        # Dec-POMDP specific buffer
        self.private_observations = deque(maxlen=capacity)
        self.public_observations = deque(maxlen=capacity)
        self.others_observations = deque(maxlen=capacity)
        
        # information quality record
        self.observation_quality = deque(maxlen=capacity)
        self.noise_levels = deque(maxlen=capacity)
        
        self.position = 0
        self.size = 0
    
    def push(self,
             states: np.ndarray,           # current state [n_agents, state_dim]
             actions: np.ndarray,          # action [n_agents, action_dim]
             rewards: np.ndarray,          # reward [n_agents]
             next_states: np.ndarray,      # next state [n_agents, state_dim]
             dones: np.ndarray,            # done flag [n_agents]
             private_obs: np.ndarray,      # private observation [n_agents, private_dim]
             public_obs: np.ndarray,       # public observation [n_agents, public_dim]
             others_obs: np.ndarray,       # others observation [n_agents, others_dim]
             obs_quality: float = 1.0,     # observation quality
             noise_level: float = 0.0):    # noise level
        """add experience to buffer"""
        
        # standard experience
        experience = (states, actions, rewards, next_states, dones)
        self.experiences.append(experience)
        
        # Dec-POMDP specific information
        self.private_observations.append(private_obs)
        self.public_observations.append(public_obs)
        self.others_observations.append(others_obs)
        self.observation_quality.append(obs_quality)
        self.noise_levels.append(noise_level)
        
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """sample batch data"""
        if self.size < batch_size:
            return None
        
        indices = random.sample(range(self.size), batch_size)
        
        # standard experience
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Dec-POMDP specific
        batch_private_obs = []
        batch_public_obs = []
        batch_others_obs = []
        batch_obs_quality = []
        batch_noise_levels = []
        
        for idx in indices:
            states, actions, rewards, next_states, dones = self.experiences[idx]
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
            
            batch_private_obs.append(self.private_observations[idx])
            batch_public_obs.append(self.public_observations[idx])
            batch_others_obs.append(self.others_observations[idx])
            batch_obs_quality.append(self.observation_quality[idx])
            batch_noise_levels.append(self.noise_levels[idx])
        
        return {
            'states': torch.FloatTensor(np.array(batch_states)),
            'actions': torch.FloatTensor(np.array(batch_actions)),
            'rewards': torch.FloatTensor(np.array(batch_rewards)),
            'next_states': torch.FloatTensor(np.array(batch_next_states)),
            'dones': torch.FloatTensor(np.array(batch_dones)),
            'private_obs': torch.FloatTensor(np.array(batch_private_obs)),
            'public_obs': torch.FloatTensor(np.array(batch_public_obs)),
            'others_obs': torch.FloatTensor(np.array(batch_others_obs)),
            'obs_quality': torch.FloatTensor(batch_obs_quality),
            'noise_levels': torch.FloatTensor(batch_noise_levels)
        }
    
    def __len__(self):
        return self.size

class DecPOMDPFOMaddpgTrainer:
    
    def __init__(self,
                 dec_pomdp_config: DecPOMDPConfig,
                 n_agents: int = 4,
                 state_dim: int = 73,
                 action_dim: int = 36,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 256,
                 buffer_capacity: int = 1000000,
                 device: str = "cpu"):
        
        self.config = dec_pomdp_config
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # create agent policies
        self.agents = []
        for i in range(n_agents):
            agent = DecPOMDPFOMaddpgPolicy(
                agent_id=i,
                dec_pomdp_config=dec_pomdp_config,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dim,
                max_action=max_action,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                tau=tau,
                device=device
            )
            self.agents.append(agent)
        
        # Dec-POMDP experience replay buffer
        self.replay_buffer = DecPOMDPReplayBuffer(
            capacity=buffer_capacity,
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # training parameters
        self.exploration_noise = 0.1    # initial exploration noise
        self.noise_decay = 0.999        # noise decay
        self.min_noise = 0.01           # minimum noise
        
        # Dec-POMDP specific parameters
        self.uncertainty_weight = 0.1   # uncertainty weight
        self.collaboration_weight = 0.05 # collaboration weight
        self.observation_quality_threshold = 0.7  # observation quality threshold
        
        # training statistics
        self.train_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        self.exploration_rates = []
    
    def select_actions(self, 
                      observations: Dict[str, np.ndarray],
                      add_noise: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        select actions for all agents
        
        Args:
            observations: Dec-POMDP observations for each agent
            add_noise: whether to add exploration noise
            
        Returns:
            (actions, info): actions and information dictionary
        """
        actions = []
        action_info = {
            'exploration_noise': self.exploration_noise,
            'agent_actions': {},
            'observation_quality': []
        }
        
        noise_scale = self.exploration_noise if add_noise else 0.0
        
        for i, agent in enumerate(self.agents):
            # get single agent observation
            obs = observations[f'agent_{i}']
            
            # select action
            action = agent.select_action(obs, noise_scale)
            actions.append(action)
            
            action_info['agent_actions'][f'agent_{i}'] = action
        
        # calculate average observation quality
        avg_quality = self._estimate_observation_quality(observations)
        action_info['observation_quality'] = avg_quality
        
        return np.array(actions), action_info
    
    def store_experience(self,
                        states: np.ndarray,
                        actions: np.ndarray,
                        rewards: np.ndarray,
                        next_states: np.ndarray,
                        dones: np.ndarray,
                        observations: Dict[str, Any],
                        obs_quality: float = 1.0):
        """store experience"""
        
        # extract Dec-POMDP observations
        private_obs = []
        public_obs = []
        others_obs = []
        
        for i in range(self.n_agents):
            agent_obs = observations.get(f'agent_{i}', {})
            private_obs.append(agent_obs.get('private', np.zeros(40)))
            public_obs.append(agent_obs.get('public', np.zeros(18)))
            others_obs.append(agent_obs.get('others', np.zeros(15)))
        
        # store to replay buffer
        self.replay_buffer.push(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            private_obs=np.array(private_obs),
            public_obs=np.array(public_obs),
            others_obs=np.array(others_obs),
            obs_quality=obs_quality,
            noise_level=self.exploration_noise
        )
    
    def update(self) -> Dict[str, float]:
        """
        update all agent policies
        
        Returns:
            training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # sample experience
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}
        
        # move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # training statistics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q_value = 0.0
        
        # update policy for each agent
        for i, agent in enumerate(self.agents):
            
            # update Critic
            critic_loss, q_val = self._update_critic(agent, batch, i)
            total_critic_loss += critic_loss
            total_q_value += q_val
            
            # update Actor
            actor_loss = self._update_actor(agent, batch, i)
            total_actor_loss += actor_loss
            
            # update target network
            agent.update_networks(self.tau)
        
        # update exploration noise
        self._update_exploration_noise()
        
        # record statistics
        self.train_step += 1
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        avg_q_value = total_q_value / self.n_agents
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.q_values.append(avg_q_value)
        self.exploration_rates.append(self.exploration_noise)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'q_value': avg_q_value,
            'exploration_noise': self.exploration_noise,
            'train_step': self.train_step
        }
    
    def _update_critic(self, 
                      agent: DecPOMDPFOMaddpgPolicy, 
                      batch: Dict[str, torch.Tensor], 
                      agent_idx: int) -> Tuple[float, float]:
        """update Critic network"""
        
        # extract batch data
        states = batch['states']  # [batch_size, n_agents, state_dim]
        actions = batch['actions']  # [batch_size, n_agents, action_dim]
        rewards = batch['rewards'][:, agent_idx].unsqueeze(1)  # [batch_size, 1]
        next_states = batch['next_states']
        dones = batch['dones'][:, agent_idx].unsqueeze(1)
        
        # observation quality information
        obs_quality = batch['obs_quality'].unsqueeze(1)  # [batch_size, 1]
        
        # flatten to centralized input
        global_states = states.view(states.shape[0], -1)  # [batch_size, n_agents*state_dim]
        global_actions = actions.view(actions.shape[0], -1)  # [batch_size, n_agents*action_dim]
        global_next_states = next_states.view(next_states.shape[0], -1)
        
        # current Q value
        current_q = agent.critic(global_states, global_actions)
        
        # target Q value calculation
        with torch.no_grad():
            # get next state action for all agents
            next_actions = []
            for j, other_agent in enumerate(self.agents):
                # use target Actor network
                next_private = batch['private_obs'][:, j]  # [batch_size, private_dim]
                next_public = batch['public_obs'][:, j]    # [batch_size, public_dim]
                next_others = batch['others_obs'][:, j]    # [batch_size, others_dim]
                
                next_action = other_agent.actor_target(
                    next_private, next_public, next_others,
                    enable_others=self.config.enable_other_manager_info
                )
                next_actions.append(next_action)
            
            global_next_actions = torch.cat(next_actions, dim=1)
            
            # target Q value
            target_q = agent.critic_target(global_next_states, global_next_actions)
            
            # Dec-POMDP uncertainty adjustment
            uncertainty_factor = self._compute_uncertainty_factor(obs_quality)
            target_q = target_q * uncertainty_factor
            
            # Bellman equation
            target_q = rewards + (self.gamma * target_q * (1 - dones))
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Dec-POMDP specific loss adjustment
        quality_weight = torch.clamp(obs_quality, 0.1, 1.0)
        weighted_loss = critic_loss * quality_weight.mean()
        
        # backpropagation
        agent.critic_optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()
        
        return weighted_loss.item(), current_q.mean().item()
    
    def _update_actor(self, 
                     agent: DecPOMDPFOMaddpgPolicy, 
                     batch: Dict[str, torch.Tensor], 
                     agent_idx: int) -> float:
        """update Actor network"""
        
        # get current agent observation
        private_obs = batch['private_obs'][:, agent_idx]  # [batch_size, private_dim]
        public_obs = batch['public_obs'][:, agent_idx]    # [batch_size, public_dim]
        others_obs = batch['others_obs'][:, agent_idx]    # [batch_size, others_dim]
        obs_quality = batch['obs_quality']
        
        # calculate current agent action
        agent_actions = agent.actor(
            private_obs, public_obs, others_obs,
            enable_others=self.config.enable_other_manager_info
        )
        
        # build all agent actions (other use batch actions)
        all_actions = []
        for j in range(self.n_agents):
            if j == agent_idx:
                all_actions.append(agent_actions)
            else:
                all_actions.append(batch['actions'][:, j])
        
        global_actions = torch.cat(all_actions, dim=1)
        global_states = batch['states'].view(batch['states'].shape[0], -1)
        
        # Actor loss (deterministic policy gradient)
        actor_loss = -agent.critic(global_states, global_actions).mean()
        
        # Dec-POMDP specific loss
        
        # 1. uncertainty loss
        uncertainty_loss = self._compute_uncertainty_loss(
            agent_actions, obs_quality
        )
        
        # 2. collaboration loss
        collaboration_loss = self._compute_collaboration_loss(
            agent_actions, all_actions, agent_idx
        )
        
        # total loss
        total_loss = (actor_loss + 
                     self.uncertainty_weight * uncertainty_loss +
                     self.collaboration_weight * collaboration_loss)
        
        # backpropagation
        agent.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()
        
        return total_loss.item()
    
    def _compute_uncertainty_factor(self, obs_quality: torch.Tensor) -> torch.Tensor:
        """compute uncertainty factor"""
        # lower observation quality, higher uncertainty, smaller discount factor
        uncertainty_factor = torch.clamp(obs_quality, 0.5, 1.0)
        return uncertainty_factor
    
    def _compute_uncertainty_loss(self, 
                                 actions: torch.Tensor, 
                                 obs_quality: torch.Tensor) -> torch.Tensor:
        """compute uncertainty loss"""
        # when observation quality is low, encourage more conservative actions
        if not self.config.enable_observation_noise:
            return torch.tensor(0.0, device=self.device)
        
        # action variance loss (when quality is low, encourage smaller action variance)
        action_variance = torch.var(actions, dim=1).mean()
        quality_factor = torch.clamp(1.0 - obs_quality.mean(), 0.0, 1.0)
        
        uncertainty_loss = action_variance * quality_factor
        return uncertainty_loss
    
    def _compute_collaboration_loss(self, 
                                   agent_actions: torch.Tensor,
                                   all_actions: List[torch.Tensor],
                                   agent_idx: int) -> torch.Tensor:
        """compute collaboration loss"""
        if not self.config.enable_other_manager_info:
            return torch.tensor(0.0, device=self.device)
        
        # compute similarity between agent actions and other agent actions
        collaboration_loss = 0.0
        num_others = 0
        
        for j, other_actions in enumerate(all_actions):
            if j != agent_idx:
                # action similarity (cosine similarity)
                similarity = F.cosine_similarity(
                    agent_actions, other_actions, dim=1
                ).mean()
                
                # encourage moderate collaboration (not exactly the same, but has some similarity)
                target_similarity = 0.3  # target similarity
                collaboration_loss += torch.abs(similarity - target_similarity)
                num_others += 1
        
        if num_others > 0:
            collaboration_loss /= num_others
        
        return collaboration_loss
    
    def _estimate_observation_quality(self, observations: Dict[str, Any]) -> float:
        """estimate observation quality"""
        total_quality = 0.0
        count = 0
        
        for i in range(self.n_agents):
            obs = observations.get(f'agent_{i}', {})
            
            # estimate quality based on observation completeness
            private_quality = 1.0 if 'private' in obs else 0.5
            public_quality = 1.0 if 'public' in obs else 0.5
            others_quality = 1.0 if 'others' in obs and self.config.enable_other_manager_info else 0.8
            
            agent_quality = (private_quality + public_quality + others_quality) / 3.0
            total_quality += agent_quality
            count += 1
        
        return total_quality / count if count > 0 else 1.0
    
    def _update_exploration_noise(self):
        """update exploration noise"""
        self.exploration_noise = max(
            self.min_noise,
            self.exploration_noise * self.noise_decay
        )
    
    def save_models(self, filepath_prefix: str):
        """save all agent models"""
        for i, agent in enumerate(self.agents):
            agent.save_models(f"{filepath_prefix}_agent_{i}")
    
    def load_models(self, filepath_prefix: str):
        """load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load_models(f"{filepath_prefix}_agent_{i}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'train_step': self.train_step,
            'exploration_noise': self.exploration_noise,
            'buffer_size': len(self.replay_buffer),
            'recent_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'recent_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'recent_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0.0,
            'uncertainty_weight': self.uncertainty_weight,
            'collaboration_weight': self.collaboration_weight
        } 
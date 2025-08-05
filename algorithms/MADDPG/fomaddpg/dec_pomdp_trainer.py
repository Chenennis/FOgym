#!/usr/bin/env python3
"""
FOMADDPG Dec-POMDP Trainer
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from collections import deque
import random
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from .dec_pomdp_policy import DecPOMDPFOMaddpgPolicy

class DecPOMDPReplayBuffer:
    """
    Dec-POMDP aware experience replay buffer
    
    Multi-agent experience storage specifically designed for FOMADDPG,
    supporting storage and sampling of layered observation information.
    """
    
    def __init__(self, 
                 capacity: int = 1000000,
                 n_agents: int = 4,
                 state_dim: int = 73,
                 action_dim: int = 36):
        self.capacity = capacity
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Experience storage
        self.experiences = deque(maxlen=capacity)
        
        # Dec-POMDP specific storage
        self.private_observations = deque(maxlen=capacity)
        self.public_observations = deque(maxlen=capacity)
        self.others_observations = deque(maxlen=capacity)
        
        # Information quality records
        self.observation_quality = deque(maxlen=capacity)
        self.noise_levels = deque(maxlen=capacity)
        
        self.position = 0
        self.size = 0
    
    def push(self,
             states: np.ndarray,           # Current states [n_agents, state_dim]
             actions: np.ndarray,          # Actions [n_agents, action_dim]
             rewards: np.ndarray,          # Rewards [n_agents]
             next_states: np.ndarray,      # Next states [n_agents, state_dim]
             dones: np.ndarray,            # Done flags [n_agents]
             private_obs: np.ndarray,      # Private observations [n_agents, private_dim]
             public_obs: np.ndarray,       # Public observations [n_agents, public_dim]
             others_obs: np.ndarray,       # Others' observations [n_agents, others_dim]
             obs_quality: float = 1.0,     # Observation quality
             noise_level: float = 0.0):    # Noise level
        """Add experience to buffer"""
        
        # Standard experience
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
        """Sample batch data"""
        if self.size < batch_size:
            return None
        
        indices = random.sample(range(self.size), batch_size)
        
        # Basic experience
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
            # Get experience from deque by index
            idx_adjusted = idx if idx < len(self.experiences) else -1
            states, actions, rewards, next_states, dones = list(self.experiences)[idx_adjusted]
            
            # Basic experience
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
            
            # Dec-POMDP specific
            batch_private_obs.append(list(self.private_observations)[idx_adjusted])
            batch_public_obs.append(list(self.public_observations)[idx_adjusted])
            batch_others_obs.append(list(self.others_observations)[idx_adjusted])
            batch_obs_quality.append(list(self.observation_quality)[idx_adjusted])
            batch_noise_levels.append(list(self.noise_levels)[idx_adjusted])
        
        # Convert to tensors
        batch = {
            'states': torch.FloatTensor(np.array(batch_states)),
            'actions': torch.FloatTensor(np.array(batch_actions)),
            'rewards': torch.FloatTensor(np.array(batch_rewards)),
            'next_states': torch.FloatTensor(np.array(batch_next_states)),
            'dones': torch.FloatTensor(np.array(batch_dones)),
            'private_obs': torch.FloatTensor(np.array(batch_private_obs)),
            'public_obs': torch.FloatTensor(np.array(batch_public_obs)),
            'others_obs': torch.FloatTensor(np.array(batch_others_obs)),
            'obs_quality': torch.FloatTensor(np.array(batch_obs_quality)),
            'noise_levels': torch.FloatTensor(np.array(batch_noise_levels))
        }
        
        return batch
    
    def __len__(self):
        return self.size

class DecPOMDPFOMaddpgTrainer:
    """
    Dec-POMDP FOMADDPG Trainer
    
    Manages training of multiple FOMADDPG agents in a Dec-POMDP setting.
    Handles experience collection, policy updates, and coordination.
    """
    
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
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Create agents
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
        
        # Create replay buffer
        self.replay_buffer = DecPOMDPReplayBuffer(
            capacity=buffer_capacity,
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Training parameters
        self.exploration_noise = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.02
        
        # Training statistics
        self.training_iterations = 0
        self.actor_losses = []
        self.critic_losses = []
        self.uncertainty_losses = []
        self.collaboration_losses = []
        self.rewards_history = []
        
        # Agent-specific statistics
        self.agent_stats = [{
            'actor_loss': [],
            'critic_loss': [],
            'rewards': [],
            'uncertainty': [],
            'collaboration': []
        } for _ in range(n_agents)]
    
    def select_actions(self, 
                      observations: Dict[str, np.ndarray],
                      add_noise: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select actions for all agents
        
        Args:
            observations: Dictionary of observations for each agent
            add_noise: Whether to add exploration noise
            
        Returns:
            Tuple of actions array and additional info
        """
        actions = np.zeros((self.n_agents, self.action_dim))
        info = {}
        
        noise_scale = self.exploration_noise if add_noise else 0.0
        
        for i, agent in enumerate(self.agents):
            agent_id = f"agent_{i}"
            if agent_id in observations:
                obs = observations[agent_id]
                actions[i] = agent.select_action(obs, noise_scale)
                
                # Estimate observation quality
                obs_quality = self._estimate_observation_quality({
                    'raw': obs,
                    'agent_id': agent_id
                })
                info[agent_id] = {'obs_quality': obs_quality}
        
        return actions, info
    
    def store_experience(self,
                        states: np.ndarray,
                        actions: np.ndarray,
                        rewards: np.ndarray,
                        next_states: np.ndarray,
                        dones: np.ndarray,
                        observations: Dict[str, Any],
                        obs_quality: float = 1.0):
        """
        Store experience in replay buffer
        
        Args:
            states: Current states [n_agents, state_dim]
            actions: Actions [n_agents, action_dim]
            rewards: Rewards [n_agents]
            next_states: Next states [n_agents, state_dim]
            dones: Done flags [n_agents]
            observations: Dictionary of observations
            obs_quality: Observation quality
        """
        # Extract Dec-POMDP specific observations
        private_obs = np.zeros((self.n_agents, 40))  # Default private observation dimension
        public_obs = np.zeros((self.n_agents, 18))   # Default public observation dimension
        others_obs = np.zeros((self.n_agents, 15))   # Default others observation dimension
        
        # Extract layered observations if available
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            if agent_id in observations and 'layered' in observations[agent_id]:
                layered = observations[agent_id]['layered']
                private_obs[i] = layered.get('private', np.zeros(40))
                public_obs[i] = layered.get('public', np.zeros(18))
                others_obs[i] = layered.get('others', np.zeros(15))
        
        # Store in replay buffer
        self.replay_buffer.push(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            private_obs=private_obs,
            public_obs=public_obs,
            others_obs=others_obs,
            obs_quality=obs_quality,
            noise_level=self.exploration_noise
        )
    
    def update(self) -> Dict[str, float]:
        """
        Update all agent policies
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'uncertainty_loss': 0.0,
                'collaboration_loss': 0.0
            }
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'uncertainty_loss': 0.0,
                'collaboration_loss': 0.0
            }
        
        # Move batch to device
        for key, value in batch.items():
            batch[key] = value.to(self.device)
        
        # Update each agent
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_uncertainty_loss = 0.0
        total_collaboration_loss = 0.0
        
        # Collect next actions from all agents' target networks
        next_actions = []
        for i, agent in enumerate(self.agents):
            next_state_i = batch['next_states'][:, i]
            next_private_i = batch['private_obs'][:, i]
            next_public_i = batch['public_obs'][:, i]
            next_others_i = batch['others_obs'][:, i]
            
            with torch.no_grad():
                # Use target policy to select next actions
                next_action_i = agent.actor_target(
                    private_obs=next_private_i,
                    public_obs=next_public_i,
                    others_obs=next_others_i,
                    enable_others=self.config.enable_other_manager_info
                )
            next_actions.append(next_action_i)
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Update critic
            critic_loss, uncertainty_loss = self._update_critic(agent, batch, i)
            
            # Update actor
            actor_loss = self._update_actor(agent, batch, i)
            
            # Compute collaboration loss
            collaboration_loss = self._compute_collaboration_loss(
                agent_actions=batch['actions'][:, i],
                all_actions=[batch['actions'][:, j] for j in range(self.n_agents)],
                agent_idx=i
            )
            
            # Update target networks
            agent.update_networks(self.tau)
            
            # Accumulate losses
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_uncertainty_loss += uncertainty_loss
            total_collaboration_loss += collaboration_loss
            
            # Record agent-specific statistics
            self.agent_stats[i]['actor_loss'].append(actor_loss)
            self.agent_stats[i]['critic_loss'].append(critic_loss)
            self.agent_stats[i]['uncertainty'].append(uncertainty_loss)
            self.agent_stats[i]['collaboration'].append(collaboration_loss.item())
        
        # Update exploration noise
        self._update_exploration_noise()
        
        # Record global statistics
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        avg_uncertainty_loss = total_uncertainty_loss / self.n_agents
        avg_collaboration_loss = total_collaboration_loss / self.n_agents
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.uncertainty_losses.append(avg_uncertainty_loss)
        self.collaboration_losses.append(avg_collaboration_loss)
        
        self.training_iterations += 1
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'uncertainty_loss': avg_uncertainty_loss,
            'collaboration_loss': avg_collaboration_loss,
            'training_iterations': self.training_iterations
        }
    
    def _update_critic(self, 
                      agent: DecPOMDPFOMaddpgPolicy, 
                      batch: Dict[str, torch.Tensor], 
                      agent_idx: int) -> Tuple[float, float]:
        """
        Update critic network for an agent
        
        Args:
            agent: Agent to update
            batch: Batch of experiences
            agent_idx: Agent index
            
        Returns:
            Tuple of (critic_loss, uncertainty_loss)
        """
        # Get agent's observations
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards'][:, agent_idx].unsqueeze(1)
        next_states = batch['next_states']
        dones = batch['dones'][:, agent_idx].unsqueeze(1)
        obs_quality = batch['obs_quality']
        
        # Flatten states and actions for critic input
        flat_states = states.reshape(states.shape[0], -1)
        flat_actions = actions.reshape(actions.shape[0], -1)
        flat_next_states = next_states.reshape(next_states.shape[0], -1)
        
        # Collect next actions from all agents' target networks
        next_actions = []
        for i, target_agent in enumerate(self.agents):
            next_private_i = batch['private_obs'][:, i]
            next_public_i = batch['public_obs'][:, i]
            next_others_i = batch['others_obs'][:, i]
            
            with torch.no_grad():
                next_action_i = target_agent.actor_target(
                    private_obs=next_private_i,
                    public_obs=next_public_i,
                    others_obs=next_others_i,
                    enable_others=self.config.enable_other_manager_info
                )
            next_actions.append(next_action_i)
        
        # Combine next actions
        flat_next_actions = torch.cat(next_actions, dim=1)
        
        # Compute target Q value
        with torch.no_grad():
            target_q = agent.critic_target(flat_next_states, flat_next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Compute current Q value
        current_q = agent.critic(flat_states, flat_actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Compute uncertainty loss
        uncertainty_factor = self._compute_uncertainty_factor(obs_quality)
        uncertainty_loss = self._compute_uncertainty_loss(actions[:, agent_idx], obs_quality)
        
        # Total loss with uncertainty weighting
        total_loss = critic_loss + uncertainty_loss * uncertainty_factor
        
        # Update critic
        agent.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()
        
        return critic_loss.item(), uncertainty_loss.item()
    
    def _update_actor(self, 
                     agent: DecPOMDPFOMaddpgPolicy, 
                     batch: Dict[str, torch.Tensor], 
                     agent_idx: int) -> float:
        """
        Update actor network for an agent
        
        Args:
            agent: Agent to update
            batch: Batch of experiences
            agent_idx: Agent index
            
        Returns:
            Actor loss value
        """
        # Get agent's observations
        states = batch['states']
        actions = batch['actions']
        private_obs = batch['private_obs'][:, agent_idx]
        public_obs = batch['public_obs'][:, agent_idx]
        others_obs = batch['others_obs'][:, agent_idx]
        
        # Flatten states for critic input
        flat_states = states.reshape(states.shape[0], -1)
        
        # Current policy actions
        current_actions = agent.actor(
            private_obs=private_obs,
            public_obs=public_obs,
            others_obs=others_obs,
            enable_others=self.config.enable_other_manager_info
        )
        
        # Create action inputs for critic where only this agent's actions are from the current policy
        all_actions = actions.clone()
        all_actions[:, agent_idx] = current_actions
        flat_actions = all_actions.reshape(all_actions.shape[0], -1)
        
        # Compute policy loss (negative of Q value)
        policy_loss = -agent.critic(flat_states, flat_actions).mean()
        
        # Compute collaboration loss
        collaboration_loss = self._compute_collaboration_loss(
            agent_actions=current_actions,
            all_actions=[actions[:, j] for j in range(self.n_agents)],
            agent_idx=agent_idx
        )
        
        # Total loss with collaboration term
        total_loss = policy_loss + self.config.collaboration_weight * collaboration_loss
        
        # Update actor
        agent.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_optimizer.step()
        
        return policy_loss.item()
    
    def _compute_uncertainty_factor(self, obs_quality: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty factor based on observation quality"""
        # Scale uncertainty factor inversely with observation quality
        return torch.mean(1.0 - obs_quality) * self.config.uncertainty_weight
    
    def _compute_uncertainty_loss(self, 
                                 actions: torch.Tensor, 
                                 obs_quality: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware loss component
        
        Higher uncertainty (lower obs_quality) should lead to more conservative actions
        """
        # Compute action magnitude penalty weighted by uncertainty
        action_magnitude = torch.norm(actions, dim=1)
        uncertainty = 1.0 - obs_quality
        
        # Higher uncertainty should penalize large actions more
        uncertainty_loss = torch.mean(action_magnitude * uncertainty)
        
        return uncertainty_loss
    
    def _compute_collaboration_loss(self, 
                                   agent_actions: torch.Tensor,
                                   all_actions: List[torch.Tensor],
                                   agent_idx: int) -> torch.Tensor:
        """
        Compute collaboration loss to encourage coordination between agents
        
        Args:
            agent_actions: Actions of the current agent
            all_actions: Actions of all agents
            agent_idx: Index of the current agent
            
        Returns:
            Collaboration loss tensor
        """
        if not self.config.enable_collaboration or len(all_actions) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        # Compute average action of other agents
        other_actions = []
        for i, actions in enumerate(all_actions):
            if i != agent_idx:
                other_actions.append(actions)
        
        if not other_actions:
            return torch.tensor(0.0, device=self.device)
        
        other_actions_tensor = torch.stack(other_actions, dim=0)
        avg_other_actions = torch.mean(other_actions_tensor, dim=0)
        
        # Compute action difference
        action_diff = agent_actions - avg_other_actions
        
        # Compute collaboration loss
        # Moderate difference is good (not too similar, not too different)
        target_diff = 0.3  # Target difference magnitude
        actual_diff = torch.norm(action_diff, dim=1)
        collaboration_loss = torch.mean(torch.abs(actual_diff - target_diff))
        
        return collaboration_loss
    
    def _estimate_observation_quality(self, observations: Dict[str, Any]) -> float:
        """
        Estimate observation quality based on various factors
        
        Args:
            observations: Dictionary containing observation information
            
        Returns:
            Estimated observation quality [0.0, 1.0]
        """
        # Default quality
        quality = 1.0
        
        # If observation has explicit quality information, use it
        if 'quality' in observations:
            return observations['quality']
        
        # If Dec-POMDP config specifies observation noise
        if hasattr(self.config, 'enable_observation_noise') and self.config.enable_observation_noise:
            quality *= (1.0 - self.config.noise_level * 0.5)
        
        # If Dec-POMDP config specifies information loss
        if hasattr(self.config, 'enable_info_missing') and self.config.enable_info_missing:
            quality *= 0.9  # Reduce quality due to potential missing information
        
        return quality
    
    def _update_exploration_noise(self):
        """Update exploration noise with decay"""
        self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
    
    def save_models(self, filepath_prefix: str):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            agent.save_models(f"{filepath_prefix}_agent_{i}")
    
    def load_models(self, filepath_prefix: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load_models(f"{filepath_prefix}_agent_{i}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'uncertainty_losses': self.uncertainty_losses,
            'collaboration_losses': self.collaboration_losses,
            'training_iterations': self.training_iterations,
            'exploration_noise': self.exploration_noise,
            'agent_stats': self.agent_stats,
            'rewards_history': self.rewards_history
        } 
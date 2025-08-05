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
    """FlexOffer-specific experience replay buffer"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, n_agents: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0
        
        # Storage buffers
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, n_agents, action_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, n_agents), dtype=bool)
        
        # FlexOffer specific information
        self.fo_constraints = np.zeros((capacity, n_agents, action_dim))  # FlexOffer constraints
        self.fo_satisfaction = np.zeros((capacity, n_agents))  # FlexOffer satisfaction
    
    def add(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_state: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
            fo_satisfaction: np.ndarray = None):
        """Add experience to buffer"""
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
        """Sample batch data from buffer"""
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
        Initialize FOMATD3 algorithm
        
        Args:
            n_agents: Number of agents
            state_dim: State space dimension
            action_dim: Action space dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Hidden layer dimension
            max_action: Maximum action value
            gamma: Discount factor
            tau: Soft update parameter
            noise_scale: Exploration noise scale
            noise_clip: Noise clipping value
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            policy_delay: Actor update delay steps
            device: Computation device
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
        self.device = torch.device(device)
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            agent = FOMATd3Policy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                max_action=max_action,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                device=device
            )
            self.agents.append(agent)
        
        # Create replay buffer
        self.buffer = FOReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        
        # Training parameters
        self.total_steps = 0
        self.update_count = 0
        
        # FlexOffer specific parameters
        self.fo_constraint_weight = 0.1  # Weight for FlexOffer constraint loss
        self.fo_satisfaction_weight = 0.2  # Weight for FlexOffer satisfaction reward
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select actions for all agents
        
        Args:
            states: Environment states
            add_noise: Whether to add exploration noise
            
        Returns:
            actions: Selected actions for all agents
        """
        actions = np.zeros((self.n_agents, self.action_dim))
        
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states, add_noise)
            actions[i] = action
        
        return actions
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                        next_states: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
                        fo_satisfaction: np.ndarray = None):
        """
        Store experience in replay buffer
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Episode termination flags
            fo_constraints: FlexOffer constraints (optional)
            fo_satisfaction: FlexOffer satisfaction metrics (optional)
        """
        self.buffer.add(
            state=states,
            actions=actions,
            rewards=rewards,
            next_state=next_states,
            dones=dones,
            fo_constraints=fo_constraints,
            fo_satisfaction=fo_satisfaction
        )
        self.total_steps += 1
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update all agent policies
        
        Returns:
            Dictionary containing training statistics
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, fo_constraints, fo_satisfaction = self.buffer.sample(self.batch_size)
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        fo_constraints = torch.FloatTensor(fo_constraints).to(self.device)
        fo_satisfaction = torch.FloatTensor(fo_satisfaction).to(self.device)
        
        # Update statistics
        critic_losses = []
        actor_losses = []
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Always update critic
            critic_loss = self._update_critic(
                agent=agent,
                agent_idx=i,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                fo_constraints=fo_constraints,
                fo_satisfaction=fo_satisfaction
            )
            critic_losses.append(critic_loss)
            
            # Delayed policy update
            actor_loss = 0
            if self.total_steps % self.policy_delay == 0:
                actor_loss = self._update_actor(
                    agent=agent,
                    agent_idx=i,
                    states=states,
                    actions=actions,
                    fo_constraints=fo_constraints
                )
                actor_losses.append(actor_loss)
        
        # Update counter
        self.update_count += 1
        
        # Return training statistics
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses) if critic_losses else 0,
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0,
            "update_count": self.update_count
        }
    
    def _update_critic(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                      actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor,
                      dones: torch.Tensor, fo_constraints: torch.Tensor, 
                      fo_satisfaction: torch.Tensor) -> float:
        """
        Update critic networks for an agent
        
        Args:
            agent: Agent policy
            agent_idx: Agent index
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            fo_constraints: Batch of FlexOffer constraints
            fo_satisfaction: Batch of FlexOffer satisfaction metrics
            
        Returns:
            critic_loss: Critic loss value
        """
        with torch.no_grad():
            # Select next actions with noise for target policy smoothing
            next_actions = torch.zeros_like(actions)
            for i, target_agent in enumerate(self.agents):
                noise = torch.randn_like(actions[:, i]) * self.noise_scale
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = target_agent.actor_target(next_states)
                next_action = torch.clamp(next_action + noise, -self.max_action, self.max_action)
                next_actions[:, i] = next_action
            
            # Compute target Q values
            target_q1, target_q2 = agent.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add FlexOffer satisfaction reward if available
            fo_reward = self._compute_fo_reward(fo_satisfaction, fo_constraints)
            target_q = rewards[:, agent_idx].unsqueeze(1) + fo_reward + \
                      self.gamma * (1 - dones[:, agent_idx].unsqueeze(1)) * target_q
        
        # Compute current Q values
        current_q1, current_q2 = agent.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                     actions: torch.Tensor, fo_constraints: torch.Tensor) -> float:
        """
        Update actor network for an agent
        
        Args:
            agent: Agent policy
            agent_idx: Agent index
            states: Batch of states
            actions: Batch of actions
            fo_constraints: Batch of FlexOffer constraints
            
        Returns:
            actor_loss: Actor loss value
        """
        # Create action batch with current agent's actions
        actions_copy = actions.clone()
        actions_copy[:, agent_idx] = agent.actor(states)
        
        # Compute actor loss
        actor_loss = -agent.critic.Q1(states, actions_copy).mean()
        
        # Add FlexOffer constraint loss
        if fo_constraints is not None:
            constraint_loss = self._compute_fo_constraint_loss(
                actions_copy[:, agent_idx], fo_constraints[:, agent_idx]
            )
            actor_loss += self.fo_constraint_weight * constraint_loss
        
        # Update actor
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        
        # Update target networks
        agent.update_targets(self.tau)
        
        return actor_loss.item()
    
    def _compute_fo_reward(self, fo_satisfaction: torch.Tensor, fo_constraints: torch.Tensor) -> torch.Tensor:
        """Compute additional reward based on FlexOffer satisfaction"""
        if fo_satisfaction is None or fo_satisfaction.shape[0] == 0:
            return torch.zeros(self.batch_size, 1).to(self.device)
        
        # Simple reward based on satisfaction level
        fo_reward = fo_satisfaction.mean(dim=1, keepdim=True) * self.fo_satisfaction_weight
        return fo_reward
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """Compute loss for FlexOffer constraint violations"""
        # Simple L2 distance between actions and constraints
        constraint_loss = F.mse_loss(actions, constraints)
        return constraint_loss
    
    def save_models(self, checkpoint_dir: str):
        """Save all agent models"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(checkpoint_dir, f"agent_{i}"))
    
    def load_models(self, checkpoint_dir: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(checkpoint_dir, f"agent_{i}"))
    
    def set_eval_mode(self):
        """Set all agents to evaluation mode"""
        for agent in self.agents:
            agent.actor.eval()
            agent.critic.eval()
            agent.actor_target.eval()
            agent.critic_target.eval()
    
    def set_train_mode(self):
        """Set all agents to training mode"""
        for agent in self.agents:
            agent.actor.train()
            agent.critic.train()
            agent.actor_target.train()
            agent.critic_target.train()
    
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about action space"""
        return {
            "n_agents": self.n_agents,
            "action_dim": self.action_dim,
            "max_action": self.max_action
        } 
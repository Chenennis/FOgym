import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Actor(nn.Module):
    """Actor network - optimized for FlexOffer"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # FlexOffer specific network structure
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Batch normalization layers - helps with FlexOffer constraint stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer - improves generalization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward propagation"""
        x = self.fc1(state)
        # Only use batch normalization when batch size > 1
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
        
        # Apply FlexOffer constraints - ensure actions are within valid range
        return self.max_action * x

class Critic(nn.Module):
    """Critic network - supports multi-agent state-action value evaluation"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.n_agents = n_agents
        
        # Input dimension is all agents' states and actions
        total_input_dim = state_dim * n_agents + action_dim * n_agents
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, states, actions):
        """
        Forward propagation
        
        Args:
            states: All agents' states [batch_size, n_agents * state_dim]
            actions: All agents' actions [batch_size, n_agents * action_dim]
        """
        # Concatenate all agents' states and actions
        x = torch.cat([states, actions], dim=1)
        
        x = self.fc1(x)
        # Only use batch normalization when batch size > 1
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
    FlexOffer Multi-Agent DDPG Policy Class
    
    A multi-agent DDPG policy specifically designed for the FlexOffer system,
    supporting collaborative learning at the Manager level and precise control at the device level.
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
        Initialize FOMADDPG policy
        
        Args:
            agent_id: Agent ID
            state_dim: State dimension
            action_dim: Action dimension  
            n_agents: Number of agents
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Hidden layer dimension
            max_action: Maximum action value
            device: Computation device
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.device = torch.device(device)
        
        # Create Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Create Critic network
        self.critic = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # FlexOffer specific parameters
        self.fo_constraint_weight = 0.1  # FlexOffer constraint weight
        self.coordination_weight = 0.05   # Coordination weight
        
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Select action
        
        Args:
            state: Current state
            noise_scale: Noise scale
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
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
        Update Critic network
        
        Args:
            states: Current state batch [batch_size, n_agents * state_dim]
            actions: Current action batch [batch_size, n_agents * action_dim]
            rewards: Reward batch [batch_size, 1]
            next_states: Next state batch
            next_actions: Next action batch
            dones: Done flags batch
            gamma: Discount factor
            
        Returns:
            Critic loss value
        """
        # Calculate target Q value
        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        # Calculate current Q value
        current_q = self.critic(states, actions)
        
        # Calculate Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Add FlexOffer constraint loss
        fo_constraint_loss = self._compute_fo_constraint_loss(actions)
        total_loss = critic_loss + self.fo_constraint_weight * fo_constraint_loss
        
        # Update Critic
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
        Update Actor network
        
        Args:
            states: State batch
            all_actions: Actions of all agents
            agent_actions: Actions of current agent
            
        Returns:
            Actor loss value
        """
        # Calculate policy loss
        policy_loss = -self.critic(states, all_actions).mean()
        
        # Add coordination loss - encourage cooperation between Managers
        coordination_loss = self._compute_coordination_loss(agent_actions, all_actions)
        total_loss = policy_loss + self.coordination_weight * coordination_loss
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return policy_loss.item()
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Calculate FlexOffer constraint loss
        
        Args:
            actions: Action tensor
            
        Returns:
            Constraint loss
        """
        # Simplified implementation: ensure actions are within reasonable range
        constraint_violation = torch.relu(torch.abs(actions) - self.max_action)
        return constraint_violation.mean()
    
    def _compute_coordination_loss(self, agent_actions: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Calculate coordination loss - encourage cooperation between Managers
        
        Args:
            agent_actions: Current agent actions
            all_actions: All agents' actions
            
        Returns:
            Coordination loss
        """
        # Calculate correlation between actions, encourage moderate coordination
        if all_actions.size(1) > self.action_dim:
            other_actions = all_actions[:, self.action_dim:]  # Actions of other agents
            # Calculate action difference, moderate differences are beneficial for exploration
            action_diff = torch.abs(agent_actions.unsqueeze(1) - other_actions.view(-1, self.n_agents-1, self.action_dim))
            # Encourage moderate coordination (not complete consistency)
            coordination_loss = torch.relu(0.5 - action_diff.mean())  # Target difference is 0.5
            return coordination_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float = 0.005):
        """Soft update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic) 
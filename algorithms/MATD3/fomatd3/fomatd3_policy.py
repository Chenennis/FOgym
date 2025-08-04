"""
FlexOffer Multi-Agent Twin Delayed Deep Deterministic Policy Gradient Policy Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union


class FOActorNetwork(nn.Module):
    """FlexOffer-specific Actor Network for MATD3"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-4, device: str = "cpu", name: str = "fo_actor"):
        super(FOActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.name = name
        
        # FlexOffer constraint-aware network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # FlexOffer output layer
        self.fo_output = nn.Linear(hidden_dim // 2, action_dim)
        
        # Batch normalization layers (supporting dynamic batch size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Move to specified device
        self.to(device)
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fo_output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = state.size(0)
        
        x = F.relu(self.fc1(state))
        
        # Dynamically handle batch normalization
        if batch_size > 1:
            x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        
        if batch_size > 1:
            x = self.bn2(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # FlexOffer constraint output (using tanh to ensure output in [-1, 1] range)
        fo_actions = torch.tanh(self.fo_output(x))
        
        return fo_actions
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        torch.save(self.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return True
        return False


class FOCriticNetwork(nn.Module):
    """FlexOffer-specific Twin Critic Network for MATD3"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int,
                 hidden_dim: int = 256, lr: float = 1e-3, device: str = "cpu", 
                 name: str = "fo_critic"):
        super(FOCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim * n_agents
        self.hidden_dim = hidden_dim
        self.device = device
        self.name = name
        
        # Input dimension: state + all agents' actions
        input_dim = state_dim + self.action_dim
        
        # Q1 network
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network (for twin critic)
        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        
        # Batch normalization
        self.q1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.q2_bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Move to specified device
        self.to(device)
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.q1_fc1, self.q1_fc2, self.q1_fc3, 
                      self.q2_fc1, self.q2_fc2, self.q2_fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both Q networks
        
        Args:
            state: Environment state tensor
            actions: Actions tensor from all agents
            
        Returns:
            q1_value, q2_value: Q values from both networks
        """
        batch_size = state.size(0)
        
        # Concatenate state and actions
        sa = torch.cat([state, actions], dim=1)
        
        # Q1 network
        q1 = F.relu(self.q1_fc1(sa))
        if batch_size > 1:
            q1 = self.q1_bn1(q1)
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc2(q1))
        q1_value = self.q1_fc3(q1)
        
        # Q2 network
        q2 = F.relu(self.q2_fc1(sa))
        if batch_size > 1:
            q2 = self.q2_bn1(q2)
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc2(q2))
        q2_value = self.q2_fc3(q2)
        
        return q1_value, q2_value
    
    def Q1(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Q1 network only (used for actor updates)
        
        Args:
            state: Environment state tensor
            actions: Actions tensor from all agents
            
        Returns:
            q1_value: Q value from Q1 network
        """
        batch_size = state.size(0)
        
        # Concatenate state and actions
        sa = torch.cat([state, actions], dim=1)
        
        # Q1 network
        q1 = F.relu(self.q1_fc1(sa))
        if batch_size > 1:
            q1 = self.q1_bn1(q1)
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc2(q1))
        q1_value = self.q1_fc3(q1)
        
        return q1_value
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        torch.save(self.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return True
        return False


class FOMATd3Policy:
    """FlexOffer Multi-Agent TD3 Policy"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 max_action: float = 1.0, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 device: str = "cpu"):
        """
        Initialize FOMATD3 Policy
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            max_action: Maximum action value
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: Computation device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        
        # Actor network
        self.actor = FOActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr_actor,
            device=device,
            name="actor"
        )
        
        # Actor target network
        self.actor_target = FOActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr_actor,
            device=device,
            name="actor_target"
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic network
        self.critic = FOCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=1,  # Single agent perspective
            hidden_dim=hidden_dim,
            lr=lr_critic,
            device=device,
            name="critic"
        )
        
        # Critic target network
        self.critic_target = FOCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=1,  # Single agent perspective
            hidden_dim=hidden_dim,
            lr=lr_critic,
            device=device,
            name="critic_target"
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = self.critic.optimizer
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action based on current policy
        
        Args:
            state: Environment state
            add_noise: Whether to add exploration noise
            
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze()
        
        if add_noise:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update_targets(self, tau: float):
        """
        Update target networks with soft update
        
        Args:
            tau: Soft update parameter
        """
        self.soft_update(self.actor_target, self.actor, tau)
        self.soft_update(self.critic_target, self.critic, tau)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float):
        """
        Soft update for target networks: θ_target = τ*θ_local + (1-τ)*θ_target
        
        Args:
            target_net: Target network to update
            source_net: Source network
            tau: Soft update parameter
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, checkpoint_dir: str):
        """Save all networks"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.actor.save_checkpoint(checkpoint_dir)
        self.critic.save_checkpoint(checkpoint_dir)
        self.actor_target.save_checkpoint(checkpoint_dir)
        self.critic_target.save_checkpoint(checkpoint_dir)
    
    def load(self, checkpoint_dir: str):
        """Load all networks"""
        if os.path.exists(checkpoint_dir):
            self.actor.load_checkpoint(checkpoint_dir)
            self.critic.load_checkpoint(checkpoint_dir)
            self.actor_target.load_checkpoint(checkpoint_dir)
            self.critic_target.load_checkpoint(checkpoint_dir) 
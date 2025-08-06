import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class FOActorNetwork(nn.Module):
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 use_batch_norm: bool = True):
        super(FOActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_batch_norm = use_batch_norm
        
        # Actor network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Batch normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # FlexOffer constraint layer
        self.fo_constraint_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize final layer with smaller weights for stable initial actions
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through actor network
        
        Args:
            state: State tensor [batch_size, state_dim]
            fo_constraints: FlexOffer constraints [batch_size, action_dim]
            
        Returns:
            actions: Continuous actions [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn2(x)
        
        # Generate base actions
        actions = torch.tanh(self.fc3(x)) * self.max_action
        
        # Apply FlexOffer constraints if provided
        if fo_constraints is not None:
            constraint_adjustment = torch.tanh(self.fo_constraint_layer(x))
            actions = actions * fo_constraints + constraint_adjustment * (1 - fo_constraints)
        
        return actions


class FOCriticNetwork(nn.Module):
    """
    Critic network for FOSQDDPG algorithm
    Estimates Q-values for state-action pairs in multi-agent setting
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 use_batch_norm: bool = True):
        super(FOCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.use_batch_norm = use_batch_norm
        
        # Total input dimension: all agents' states and actions
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents
        input_dim = total_state_dim + total_action_dim
        
        # Critic network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Batch normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # FlexOffer value estimation layer
        self.fo_value_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize final layer with smaller weights
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, 
                states: torch.Tensor, 
                actions: torch.Tensor,
                fo_satisfaction: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Args:
            states: All agents' states [batch_size, n_agents * state_dim]
            actions: All agents' actions [batch_size, n_agents * action_dim]
            fo_satisfaction: FlexOffer satisfaction scores [batch_size, n_agents]
            
        Returns:
            q_values: Q-values [batch_size, 1]
        """
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=1)
        
        x = F.relu(self.fc1(x))
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn2(x)
        
        # Base Q-value
        q_value = self.fc3(x)
        
        # Add FlexOffer satisfaction bonus if provided
        if fo_satisfaction is not None:
            fo_bonus = self.fo_value_layer(x)
            satisfaction_weight = torch.mean(fo_satisfaction, dim=1, keepdim=True)
            q_value = q_value + fo_bonus * satisfaction_weight
        
        return q_value


class FOSQDDPGPolicy:
    """
    FOSQDDPG Policy class combining Actor and Critic networks
    with FlexOffer-specific enhancements
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 tau: float = 0.005,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.tau = tau
        self.device = device
        
        # Create networks
        self.actor = FOActorNetwork(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.critic = FOCriticNetwork(state_dim, action_dim, n_agents, hidden_dim).to(device)
        
        # Create target networks
        self.actor_target = FOActorNetwork(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.critic_target = FOCriticNetwork(state_dim, action_dim, n_agents, hidden_dim).to(device)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, state: np.ndarray, fo_constraints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select action using current policy
        
        Args:
            state: Current state [state_dim]
            fo_constraints: FlexOffer constraints [action_dim]
            
        Returns:
            action: Selected action [action_dim]
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        fo_constraints_tensor = None
        if fo_constraints is not None:
            fo_constraints_tensor = torch.FloatTensor(fo_constraints.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor, fo_constraints_tensor)
        
        return action.cpu().data.numpy().flatten()
    
    def update_target_networks(self):
        """Soft update of target networks"""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_models(self, filepath: str):
        """Save actor and critic models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, f"{filepath}_fosqddpg_policy.pt")
    
    def load_models(self, filepath: str):
        """Load actor and critic models"""
        checkpoint = torch.load(f"{filepath}_fosqddpg_policy.pt", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict()) 
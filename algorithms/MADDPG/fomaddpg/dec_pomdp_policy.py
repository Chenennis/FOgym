#!/usr/bin/env python3
"""
FOMADDPG Dec-POMDP Policy Network

Provides Actor-Critic network architecture supporting Dec-POMDP for FOMADDPG algorithm.
Designed with specialized network structures for deterministic policy gradient algorithms.

Core features:
1. Dec-POMDP aware Actor network (deterministic policy)
2. Centralized training Critic network (multi-agent value evaluation)
3. Information fusion layer (private+public+others information integration)
4. Target network soft update mechanism
5. DDPG-specific network structure optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from .dec_pomdp_adapter import FOMaddpgDecPOMDPAdapter

class DecPOMDPActor(nn.Module):
    """
    Dec-POMDP aware Actor network (FOMADDPG specific)

    """
    
    def __init__(self, 
                 private_dim: int = 40,    # Enhanced private observation dimension
                 public_dim: int = 18,     # Public observation dimension
                 others_dim: int = 15,     # Others' observation dimension
                 action_dim: int = 36,     # Action dimension
                 hidden_dim: int = 256,    # Hidden layer dimension
                 max_action: float = 1.0,  # Maximum action value
                 device: str = "cpu"):
        super(DecPOMDPActor, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        
        # Private information encoder (most important)
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Public information encoder (second most important)
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Others' information encoder (auxiliary information)
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),  # Higher dropout because others' information is less reliable
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU()
        )
        
        # Information fusion network (key DDPG component)
        fusion_input_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Deterministic policy output layer (DDPG feature)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, 
                private_obs: torch.Tensor, 
                public_obs: torch.Tensor, 
                others_obs: torch.Tensor,
                enable_others: bool = True) -> torch.Tensor:
        """
        Forward pass through the Dec-POMDP Actor network
        
        Args:
            private_obs: Private observation tensor
            public_obs: Public observation tensor
            others_obs: Others' observation tensor
            enable_others: Whether to use others' information
            
        Returns:
            Deterministic action tensor in range [-max_action, max_action]
        """
        # Process each observation component
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        if enable_others:
            others_features = self.others_encoder(others_obs)
        else:
            # If others' information is disabled, use zeros
            others_features = torch.zeros(
                private_features.shape[0], 
                self.others_encoder[-2].out_features, 
                device=self.device
            )
        
        # Fuse all features
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fusion_output = self.fusion_network(fused_features)
        
        # Generate deterministic action
        action = self.policy_head(fusion_output)
        
        # Scale to action range
        return self.max_action * action
    
    def get_features(self, 
                    private_obs: torch.Tensor, 
                    public_obs: torch.Tensor, 
                    others_obs: torch.Tensor,
                    enable_others: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get intermediate features from the network
        
        Useful for debugging and visualization
        """
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        if enable_others:
            others_features = self.others_encoder(others_obs)
        else:
            others_features = torch.zeros(
                private_features.shape[0], 
                self.others_encoder[-2].out_features, 
                device=self.device
            )
        
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fusion_output = self.fusion_network(fused_features)
        
        return {
            'private_features': private_features,
            'public_features': public_features,
            'others_features': others_features,
            'fused_features': fused_features,
            'fusion_output': fusion_output
        }

class DecPOMDPCritic(nn.Module):
    """
    Dec-POMDP Critic network for centralized training
    
    Evaluates state-action values for all agents
    """
    
    def __init__(self,
                 state_dim: int,           # Single agent state dimension
                 action_dim: int,          # Single agent action dimension
                 n_agents: int = 4,        # Number of agents
                 hidden_dim: int = 256,    # Hidden layer dimension
                 device: str = "cpu"):
        super(DecPOMDPCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)
        
        # Global state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * n_agents, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Global action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * n_agents, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, 
                global_states: torch.Tensor, 
                global_actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dec-POMDP Critic network
        
        Args:
            global_states: States of all agents [batch_size, n_agents * state_dim]
            global_actions: Actions of all agents [batch_size, n_agents * action_dim]
            
        Returns:
            Q-value tensor [batch_size, 1]
        """
        # Process global states and actions
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        
        # Concatenate state and action features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Compute Q-value
        q_value = self.q_network(combined_features)
        
        return q_value
    
    def get_features(self, 
                    global_states: torch.Tensor, 
                    global_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate features from the network
        
        Useful for debugging and visualization
        """
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        return {
            'state_features': state_features,
            'action_features': action_features,
            'combined_features': combined_features
        }

class DecPOMDPFOMaddpgPolicy:
    """
    Dec-POMDP FOMADDPG Policy
    
    Integrates Actor and Critic networks with Dec-POMDP observation adapter
    """
    
    def __init__(self, 
                 agent_id: int,
                 dec_pomdp_config: DecPOMDPConfig,
                 state_dim: int = 73,      # Adapted state dimension
                 action_dim: int = 36,     # Action dimension
                 n_agents: int = 4,        # Number of agents
                 hidden_dim: int = 256,    # Hidden layer dimension
                 max_action: float = 1.0,  # Maximum action value
                 lr_actor: float = 1e-4,   # Actor learning rate
                 lr_critic: float = 1e-3,  # Critic learning rate
                 tau: float = 0.005,       # Soft update coefficient
                 device: str = "cpu"):
        
        self.agent_id = agent_id
        self.config = dec_pomdp_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.tau = tau
        self.device = torch.device(device)
        
        # Create observation adapter
        self.obs_adapter = FOMaddpgDecPOMDPAdapter(dec_pomdp_config, device=device)
        
        # Get adapted dimensions
        adapted_dims = self.obs_adapter.get_adapted_dimensions()
        private_dim = adapted_dims['private']
        public_dim = adapted_dims['public']
        others_dim = adapted_dims['others']
        
        # Create Actor network
        self.actor = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(device)
        
        # Create target Actor network
        self.actor_target = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(device)
        
        # Create Critic network
        self.critic = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        # Create target Critic network
        self.critic_target = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        # Initialize target networks with same weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Setup optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, observation: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        Select action based on observation
        
        Args:
            observation: Raw observation
            noise_scale: Scale of exploration noise
            
        Returns:
            Selected action
        """
        # Convert to tensor
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.FloatTensor(observation).to(self.device)
        else:
            observation_tensor = observation
        
        # Adapt observation for Dec-POMDP
        adapted_obs = self.obs_adapter.adapt_observation_for_fomaddpg(
            observation_tensor, f"agent_{self.agent_id}"
        )
        
        # Get components
        private_obs = adapted_obs['private']
        public_obs = adapted_obs['public']
        others_obs = adapted_obs['others']
        
        # Use Actor network to select action
        with torch.no_grad():
            action = self.actor(
                private_obs=private_obs,
                public_obs=public_obs,
                others_obs=others_obs,
                enable_others=self.config.enable_other_manager_info
            ).cpu().numpy()
        
        # Add exploration noise if needed
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action.flatten()
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def update_networks(self, tau: Optional[float] = None):
        """Update target networks"""
        tau_value = tau if tau is not None else self.tau
        self.soft_update(self.actor_target, self.actor, tau_value)
        self.soft_update(self.critic_target, self.critic, tau_value)
    
    def save_models(self, filepath_prefix: str):
        """Save models to files"""
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor_{self.agent_id}.pt")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic_{self.agent_id}.pt")
    
    def load_models(self, filepath_prefix: str):
        """Load models from files"""
        self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor_{self.agent_id}.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic_{self.agent_id}.pt", map_location=self.device))
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
    
    def get_network_info(self) -> Dict[str, int]:
        """Get information about network parameters"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params
        } 
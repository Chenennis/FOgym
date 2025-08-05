#!/usr/bin/env python3
"""
Dec-POMDP-aware FOMAPPO policy network
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPFOMAPPOPolicy:
    """Dec-POMDP-aware FOMAPPO policy class"""
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.device = device
        self.dec_pomdp_config = dec_pomdp_config
        
        # Basic parameters
        self.lr = getattr(args, 'lr', 0.0003)
        self.critic_lr = getattr(args, 'critic_lr', 0.0003)
        self.opti_eps = getattr(args, 'opti_eps', 1e-5)
        self.weight_decay = getattr(args, 'weight_decay', 0)
        
        # Observation space dimension (based on Dec-POMDP architecture)
        # Get total dimension from obs_space
        self.total_obs_dim = obs_space.shape[0]
        
        # Calculate each part dimension based on total dimension, maintaining proportions
        total_parts = 72  # Original total
        self.private_dim = int(self.total_obs_dim * (39/total_parts))  # Private information layer dimension
        self.public_dim = int(self.total_obs_dim * (18/total_parts))   # Public information layer dimension
        self.others_dim = self.total_obs_dim - self.private_dim - self.public_dim  # Limited other-agent information layer dimension
        
        # Ensure total dimension is correct
        assert self.private_dim + self.public_dim + self.others_dim == self.total_obs_dim, f"Dimension mismatch: {self.private_dim} + {self.public_dim} + {self.others_dim} != {self.total_obs_dim}"
        
        # Action space
        self.act_space = act_space
        self.action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 10
        
        # Create Dec-POMDP-aware network
        self.actor = DecPOMDPActor(args, self.private_dim, self.public_dim, self.others_dim, 
                                   self.action_dim, dec_pomdp_config, device)
        self.critic = DecPOMDPCritic(args, self.private_dim, self.public_dim, self.others_dim, 
                                     dec_pomdp_config, device)
        
        # Optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    
    def parse_observation(self, obs):
        """Parse Dec-POMDP observation space, safely handling observations with different dimensions"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Process batch dimension
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Get actual observation dimension
        actual_dim = obs.shape[1]
        
        # Check if dimension matches expectation
        if actual_dim != self.total_obs_dim:
            logger.warning(f"Observation dimension mismatch: expected {self.total_obs_dim} dimensions, got {actual_dim} dimensions. Attempting safe handling.")
            
            # Calculate safe slice indices
            safe_private_end = min(self.private_dim, actual_dim)
            safe_public_end = min(self.private_dim + self.public_dim, actual_dim)
            
            # Safely separate three layers of observation information
            private_obs = obs[:, :safe_private_end]
            public_obs = obs[:, min(self.private_dim, actual_dim):safe_public_end]
            others_obs = obs[:, min(safe_public_end, actual_dim):]
            
            # If dimension is insufficient, pad with zeros
            if safe_private_end < self.private_dim:
                padding = torch.zeros(obs.shape[0], self.private_dim - safe_private_end, device=self.device)
                private_obs = torch.cat([private_obs, padding], dim=1)
                
            if safe_public_end - self.private_dim < self.public_dim:
                padding = torch.zeros(obs.shape[0], self.public_dim - (safe_public_end - self.private_dim), device=self.device)
                public_obs = torch.cat([public_obs, padding], dim=1)
                
            if actual_dim - safe_public_end < self.others_dim:
                padding = torch.zeros(obs.shape[0], self.others_dim - (actual_dim - safe_public_end), device=self.device)
                others_obs = torch.cat([others_obs, padding], dim=1)
        else:
            # Normal separation of three layers of observation information
            private_obs = obs[:, :self.private_dim]
            public_obs = obs[:, self.private_dim:self.private_dim + self.public_dim]
            others_obs = obs[:, self.private_dim + self.public_dim:]
        
        return private_obs, public_obs, others_obs
    
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False):
        """Get actions and value predictions"""
        # Parse observation
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        # Actor forward pass
        actions, action_log_probs, rnn_states_actor = self.actor(
            private_obs, public_obs, others_obs, rnn_states_actor, masks, 
            available_actions, deterministic
        )
        
        # Critic forward pass
        values, rnn_states_critic = self.critic(
            private_obs, public_obs, others_obs, rnn_states_critic, masks
        )
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions"""
        private_obs, public_obs, others_obs = self.parse_observation(cent_obs)
        values, _ = self.critic(private_obs, public_obs, others_obs, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, 
                         masks, available_actions=None, active_masks=None):
        """Evaluate action log probabilities, entropy, and value function"""
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        # Actor evaluation
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            private_obs, public_obs, others_obs, rnn_states_actor, action, 
            masks, available_actions, active_masks
        )
        
        # Critic evaluation
        values, _ = self.critic(private_obs, public_obs, others_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """Only calculate actions (for inference)"""
        private_obs, public_obs, others_obs = self.parse_observation(obs)
        
        actions, _, rnn_states_actor = self.actor(
            private_obs, public_obs, others_obs, rnn_states_actor, masks,
            available_actions, deterministic
        )
        
        return actions, rnn_states_actor


class DecPOMDPActor(nn.Module):
    """Dec-POMDP-aware Actor network"""
    
    def __init__(self, args, private_dim, public_dim, others_dim, action_dim, 
                 dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        super(DecPOMDPActor, self).__init__()
        
        self.device = device
        self.config = dec_pomdp_config
        self.hidden_size = getattr(args, 'hidden_size', 256)
        self.action_dim = action_dim
        
        # Record input dimension
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        logger.info(f"Actor network input dimension: private={private_dim}, public={public_dim}, others={others_dim}")
        
        # Private information processing network (no noise, high weight)
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU()
        )
        
        # Public information processing network (standard processing)
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # Others information processing network (with uncertainty processing)
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # Information fusion network
        fusion_input_dim = (self.hidden_size // 2) + (self.hidden_size // 4) + (self.hidden_size // 4)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # Action output network
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.action_dim)
        )
        
        # Action distribution parameters
        self.action_std = nn.Parameter(torch.ones(self.action_dim) * 0.1)
        
        self.to(device)
    
    def forward(self, private_obs, public_obs, others_obs, rnn_states, masks, 
                available_actions=None, deterministic=False):
        """Forward pass"""
        
        # Process private information (most reliable)
        private_features = self.private_encoder(private_obs)
        
        # Process public information (standard reliability)
        public_features = self.public_encoder(public_obs)
        
        # Process others information (consider uncertainty)
        others_features = self.others_encoder(others_obs)
        
        # Process others information availability
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        # Fuse all information
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        # Generate action distribution
        action_mean = self.action_head(fused_output)
        action_std = self.action_std.expand_as(action_mean)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = action_dist.sample()
        
        action_log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # RNN state processing (simplified implementation)
        new_rnn_states = rnn_states
        
        return actions, action_log_probs, new_rnn_states
    
    def evaluate_actions(self, private_obs, public_obs, others_obs, rnn_states, action, 
                         masks, available_actions=None, active_masks=None):
        """Evaluate given action log probabilities and entropy"""
        
        # Recalculate action distribution
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        others_features = self.others_encoder(others_obs)
        
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        action_mean = self.action_head(fused_output)
        action_std = self.action_std.expand_as(action_mean)
        
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        action_log_probs = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_probs, dist_entropy


class DecPOMDPCritic(nn.Module):
    """Dec-POMDP-aware Critic network"""
    
    def __init__(self, args, private_dim, public_dim, others_dim, 
                 dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        super(DecPOMDPCritic, self).__init__()
        
        self.device = device
        self.config = dec_pomdp_config
        self.hidden_size = getattr(args, 'hidden_size', 256)
        
        # Record input dimension
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        logger.info(f"Critic network input dimension: private={private_dim}, public={public_dim}, others={others_dim}")
        
        # Similar architecture to Actor
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU()
        )
        
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )
        
        # Information fusion network
        fusion_input_dim = (self.hidden_size // 2) + (self.hidden_size // 4) + (self.hidden_size // 4)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # Value function output
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.to(device)
    
    def forward(self, private_obs, public_obs, others_obs, rnn_states, masks):
        """Forward pass"""
        
        # Process each layer of observation information
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        others_features = self.others_encoder(others_obs)
        
        # Process others information availability
        if not self.config.enable_other_manager_info:
            others_features = torch.zeros_like(others_features)
        
        # Fuse information
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        fused_output = self.fusion_network(fused_features)
        
        # Calculate value function
        values = self.value_head(fused_output)
        
        # RNN state processing (simplified implementation)
        new_rnn_states = rnn_states
        
        return values, new_rnn_states 
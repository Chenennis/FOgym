#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import sys
import os

# add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig
from .dec_pomdp_adapter import FOMaddpgDecPOMDPAdapter

class DecPOMDPActor(nn.Module):
    
    def __init__(self, 
                 private_dim: int = 40,    # enhanced private observation dimension
                 public_dim: int = 18,     # public observation dimension
                 others_dim: int = 15,     # other observation dimension
                 action_dim: int = 36,     # action dimension
                 hidden_dim: int = 256,    # hidden layer dimension
                 max_action: float = 1.0,  # maximum action value
                 device: str = "cpu"):
        super(DecPOMDPActor, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        
        # private information encoder (most important)
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # public information encoder (second most important)
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # other information encoder (auxiliary information)
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),  # higher dropout, because other information is not reliable
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU()
        )
        
        # information fusion network (DDPG key component)
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
        
        # deterministic policy output layer (DDPG feature)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # deterministic policy uses tanh activation
        )
        
        # action scaling layer
        self.action_scale = nn.Parameter(torch.ones(action_dim) * max_action, requires_grad=False)
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, 
                private_obs: torch.Tensor, 
                public_obs: torch.Tensor, 
                others_obs: torch.Tensor,
                enable_others: bool = True) -> torch.Tensor:
        """
        forward propagation - deterministic policy output
        
        Args:
            private_obs: private observation [batch_size, private_dim]
            public_obs: public observation [batch_size, public_dim]
            others_obs: other observation [batch_size, others_dim]
            enable_others: whether to enable other information
            
        Returns:
            deterministic action [batch_size, action_dim]
        """
        batch_size = private_obs.shape[0]
        
        # encode each layer of information
        private_features = self.private_encoder(private_obs)  # [batch, hidden//2]
        public_features = self.public_encoder(public_obs)     # [batch, hidden//4]
        
        if enable_others:
            others_features = self.others_encoder(others_obs)  # [batch, hidden//8]
        else:
            others_features = torch.zeros(batch_size, self.hidden_dim // 8).to(self.device)
        
        # information fusion
        fused_features = torch.cat([private_features, public_features, others_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        # deterministic policy output
        raw_actions = self.policy_head(fused_representation)
        
        # apply action scaling
        scaled_actions = raw_actions * self.action_scale
        
        return scaled_actions
    
    def get_features(self, 
                    private_obs: torch.Tensor, 
                    public_obs: torch.Tensor, 
                    others_obs: torch.Tensor,
                    enable_others: bool = True) -> Dict[str, torch.Tensor]:
        """
        get feature representation (for analysis and debugging)
        
        Returns:
            Dict containing each layer feature and fused representation
        """
        batch_size = private_obs.shape[0]
        
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        if enable_others:
            others_features = self.others_encoder(others_obs)
        else:
            others_features = torch.zeros(batch_size, self.hidden_dim // 8).to(self.device)
        
        fused_features = torch.cat([private_features, public_features, others_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        return {
            'private_features': private_features,
            'public_features': public_features,
            'others_features': others_features,
            'fused_representation': fused_representation
        }

class DecPOMDPCritic(nn.Module):
    
    def __init__(self,
                 state_dim: int,           # single agent state dimension
                 action_dim: int,          # single agent action dimension
                 n_agents: int = 4,        # number of agents
                 hidden_dim: int = 256,    # hidden layer dimension
                 device: str = "cpu"):
        super(DecPOMDPCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # centralized input dimension: all agents' states and actions
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents
        total_input_dim = total_state_dim + total_action_dim
        
        # state encoding network
        self.state_encoder = nn.Sequential(
            nn.Linear(total_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # action encoding network
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # state-action fusion network
        fusion_input_dim = hidden_dim + hidden_dim // 2
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Q value output head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # output single Q value
        )
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, 
                global_states: torch.Tensor, 
                global_actions: torch.Tensor) -> torch.Tensor:
        """
        forward propagation - Q value estimation
        
        Args:
            global_states: all agents' states [batch_size, n_agents * state_dim]
            global_actions: all agents' actions [batch_size, n_agents * action_dim]
            
        Returns:
            Q value [batch_size, 1]
        """
        # encode state and action separately
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        
        # state-action fusion
        fused_features = torch.cat([state_features, action_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        # Q value output
        q_value = self.q_head(fused_representation)
        
        return q_value
    
    def get_features(self, 
                    global_states: torch.Tensor, 
                    global_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        get feature representation (for analysis)
        
        Returns:
            Dict containing state features, action features and fused representation
        """
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        fused_features = torch.cat([state_features, action_features], dim=1)
        fused_representation = self.fusion_network(fused_features)
        
        return {
            'state_features': state_features,
            'action_features': action_features,
            'fused_representation': fused_representation
        }

class DecPOMDPFOMaddpgPolicy:
    
    def __init__(self, 
                 agent_id: int,
                 dec_pomdp_config: DecPOMDPConfig,
                 state_dim: int = 73,      # adapted state dimension
                 action_dim: int = 36,     # action dimension
                 n_agents: int = 4,        # number of agents
                 hidden_dim: int = 256,    # hidden layer dimension
                 max_action: float = 1.0,  # maximum action value
                 lr_actor: float = 1e-4,   # Actor learning rate
                 lr_critic: float = 1e-3,  # Critic learning rate
                 tau: float = 0.005,       # soft update coefficient
                 device: str = "cpu"):
        
        self.agent_id = agent_id
        self.config = dec_pomdp_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.tau = tau
        self.device = torch.device(device)
        
        # create observation adapter
        self.obs_adapter = FOMaddpgDecPOMDPAdapter(dec_pomdp_config, device)
        
        # get adapted observation dimension
        adapted_dims = self.obs_adapter.get_adapted_dimensions()
        private_dim = adapted_dims['private_dim']      # 40
        public_dim = adapted_dims['public_dim']        # 18
        others_dim = adapted_dims['others_dim']        # 15
        
        # create Actor network
        self.actor = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(self.device)
        
        # create Actor target network
        self.actor_target = DecPOMDPActor(
            private_dim=private_dim,
            public_dim=public_dim,
            others_dim=others_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_action=max_action,
            device=device
        ).to(self.device)
        
        # create Critic network
        self.critic = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(self.device)
        
        # create Critic target network
        self.critic_target = DecPOMDPCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            device=device
        ).to(self.device)
        
        # initialize target network
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # training statistics
        self.train_step = 0
    
    def select_action(self, observation: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        select action (deterministic policy + noise exploration)
        
        Args:
            observation: original observation
            noise_scale: noise ratio
            
        Returns:
            selected action
        """
        # adapt observation
        adapted_obs = self.obs_adapter.adapt_observation_for_fomaddpg(
            observation, f"manager_{self.agent_id}"
        )
        
        # extract each layer of observation
        private_obs = adapted_obs['private']
        public_obs = adapted_obs['public']
        others_obs = adapted_obs['others']
        
        # deterministic policy output
        with torch.no_grad():
            action = self.actor(private_obs, public_obs, others_obs, 
                              enable_others=self.config.enable_other_manager_info)
            action = action.cpu().numpy()[0]
        
        # add exploration noise
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def update_networks(self, tau: Optional[float] = None):
        """update target network"""
        if tau is None:
            tau = self.tau
        
        self.soft_update(self.actor_target, self.actor, tau)
        self.soft_update(self.critic_target, self.critic, tau)
    
    def save_models(self, filepath_prefix: str):
        """save models"""
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pt")
        torch.save(self.actor_target.state_dict(), f"{filepath_prefix}_actor_target.pt")
        torch.save(self.critic_target.state_dict(), f"{filepath_prefix}_critic_target.pt")
    
    def load_models(self, filepath_prefix: str):
        """load models"""
        self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pt", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{filepath_prefix}_actor_target.pt", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f"{filepath_prefix}_critic_target.pt", map_location=self.device))
    
    def get_network_info(self) -> Dict[str, int]:
        """get network parameter information"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params,
            'agent_id': self.agent_id,
            'train_step': self.train_step
        } 
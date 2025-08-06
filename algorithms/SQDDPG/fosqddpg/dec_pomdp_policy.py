# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any


class DecPOMDPFOSQDDPGActor(nn.Module):
    """FOSQDDPG Actor network, supporting Shapley value fairness allocation"""
    
    def __init__(self, private_dim=40, public_dim=18, others_dim=15, 
                 action_dim=36, hidden_dim=256, max_action=1.0,
                 enable_other_manager_info=True):
        super(DecPOMDPFOSQDDPGActor, self).__init__()
        
        self.enable_other_manager_info = enable_other_manager_info
        self.max_action = max_action
        
        # encoder network
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        
        if enable_other_manager_info:
            self.others_encoder = nn.Sequential(
                nn.Linear(others_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU()
            )
            fusion_dim = 64 + 16 + 16  # 96
        else:
            fusion_dim = 64 + 16  # 80
        
        # Shapley feature encoder
        self.shapley_encoder = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU()
        )
        
        # fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim + 4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
        # action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # fairness adjustment layer
        self.fairness_adjustment = nn.Sequential(
            nn.Linear(hidden_dim + 4, action_dim),
            nn.Tanh()
        )
    
    def forward(self, private_obs, public_obs, others_obs=None, 
                fairness_score=None, shapley_weight=None):
        """forward propagation"""
        batch_size = private_obs.size(0)
        
        # encode each layer observation
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        
        if self.enable_other_manager_info and others_obs is not None:
            others_features = self.others_encoder(others_obs)
            combined_features = torch.cat([private_features, public_features, others_features], dim=1)
        else:
            combined_features = torch.cat([private_features, public_features], dim=1)
        
        # process Shapley features
        if fairness_score is not None and shapley_weight is not None:
            shapley_features = torch.cat([fairness_score, shapley_weight], dim=1)
            shapley_encoded = self.shapley_encoder(shapley_features)
        else:
            shapley_encoded = torch.zeros(batch_size, 4, device=private_obs.device)
        
        # fuse all features
        total_features = torch.cat([combined_features, shapley_encoded], dim=1)
        fused_features = self.fusion_network(total_features)
        
        # generate base actions
        base_actions = self.action_head(fused_features)
        
        # fairness adjustment
        fairness_input = torch.cat([fused_features, shapley_encoded], dim=1)
        fairness_adjustment = self.fairness_adjustment(fairness_input)
        
        # fairness weighted
        fairness_factor = fairness_score.mean() if fairness_score is not None else 1.0
        adjusted_actions = (fairness_factor * base_actions + 
                          (1 - fairness_factor) * fairness_adjustment * 0.3)
        
        return adjusted_actions * self.max_action


class DecPOMDPFOSQDDPGCritic(nn.Module):
    """FOSQDDPG Critic network, supporting Shapley value decomposition"""
    
    def __init__(self, n_agents=4, private_dim=40, public_dim=18, others_dim=15,
                 action_dim=36, hidden_dim=256, enable_other_manager_info=True):
        super(DecPOMDPFOSQDDPGCritic, self).__init__()
        
        self.n_agents = n_agents
        
        # calculate total dimension
        others_dim_actual = others_dim if enable_other_manager_info else 0
        total_state_dim = n_agents * (private_dim + public_dim + others_dim_actual)
        total_action_dim = n_agents * action_dim
        
        # global encoder
        self.global_state_encoder = nn.Sequential(
            nn.Linear(total_state_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )
        
        self.global_action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        
        # fairness encoder
        self.fairness_encoder = nn.Sequential(
            nn.Linear(n_agents * 2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        
        # Q network
        q_input_dim = hidden_dim + hidden_dim // 2 + 16
        self.q_network = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Shapley decomposition network
        self.shapley_decomposition = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
            nn.Softmax(dim=1)
        )
    
    def forward(self, global_states, global_actions, fairness_features=None):
        """forward propagation"""
        batch_size = global_states.size(0)
        
        # encode global features
        state_features = self.global_state_encoder(global_states)
        action_features = self.global_action_encoder(global_actions)
        
        # process fairness features
        if fairness_features is not None:
            fairness_encoded = self.fairness_encoder(fairness_features)
        else:
            fairness_encoded = torch.zeros(batch_size, 16, device=global_states.device)
        
        # feature fusion
        combined_features = torch.cat([state_features, action_features, fairness_encoded], dim=1)
        
        # calculate Q value and Shapley decomposition
        q_value = self.q_network(combined_features)
        shapley_contributions = self.shapley_decomposition(combined_features)
        
        return {
            'q_value': q_value,
            'shapley_contributions': shapley_contributions,
            'state_features': state_features,
            'action_features': action_features
        }


class DecPOMDPFOSQDDPGPolicy:
    """FOSQDDPG policy manager"""
    
    def __init__(self, n_agents=4, private_dim=40, public_dim=18, others_dim=15,
                 action_dim=36, hidden_dim=256, max_action=1.0, device="cpu",
                 enable_other_manager_info=True, **kwargs):
        
        self.device = torch.device(device)
        self.enable_other_manager_info = enable_other_manager_info
        
        # create network
        self.actor = DecPOMDPFOSQDDPGActor(
            private_dim, public_dim, others_dim, action_dim, 
            hidden_dim, max_action, enable_other_manager_info
        ).to(self.device)
        
        self.actor_target = DecPOMDPFOSQDDPGActor(
            private_dim, public_dim, others_dim, action_dim,
            hidden_dim, max_action, enable_other_manager_info
        ).to(self.device)
        
        self.critic = DecPOMDPFOSQDDPGCritic(
            n_agents, private_dim, public_dim, others_dim,
            action_dim, hidden_dim, enable_other_manager_info
        ).to(self.device)
        
        self.critic_target = DecPOMDPFOSQDDPGCritic(
            n_agents, private_dim, public_dim, others_dim,
            action_dim, hidden_dim, enable_other_manager_info
        ).to(self.device)
        
        # copy parameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
    
    def select_action(self, adapted_obs: Dict[str, torch.Tensor], 
                     deterministic=False) -> np.ndarray:
        """select action"""
        self.actor.eval()
        
        with torch.no_grad():
            private_obs = adapted_obs['private'].unsqueeze(0)
            public_obs = adapted_obs['public'].unsqueeze(0)
            others_obs = adapted_obs['others'].unsqueeze(0) if self.enable_other_manager_info else None
            fairness_score = adapted_obs.get('fairness_score', torch.zeros(1, 1)).unsqueeze(0)
            shapley_weight = adapted_obs.get('shapley_weight', torch.zeros(1, 1)).unsqueeze(0)
            
            action = self.actor(private_obs, public_obs, others_obs, 
                              fairness_score, shapley_weight)
        
        self.actor.train()
        return action.cpu().numpy().flatten()
    
    def update_target_networks(self):
        """update target networks"""
        tau = 0.005
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_network_info(self) -> Dict[str, Any]:
        """get network information"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params,
            'device': str(self.device),
            'enable_other_manager_info': self.enable_other_manager_info,
            'shapley_integration': True,
            'fairness_weighting': True
        } 
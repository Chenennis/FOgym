#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPObservationAdapter:
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimension (based on implemented architecture)
        self.private_dim = 39  # private information layer dimension
        self.public_dim = 18   # public information layer dimension  
        self.others_dim = 15   # limited other information layer dimension
        
        # total observation dimension
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72-dimensional
        
        # observation processing weights
        self.private_weight = 1.0  # private information completely trusted
        self.public_weight = 1.0   # public information completely trusted
        self.others_weight = 0.8   # other information partially trusted (configurable noise)
        
        # history observation cache
        self.observation_history = {}
        self.max_history_len = 10
        
    def parse_observation(self, observation: np.ndarray, manager_id: str) -> Dict[str, np.ndarray]:
        if len(observation) != self.total_obs_dim:
            raise ValueError(f"observation dimension mismatch: expected {self.total_obs_dim}, actual {len(observation)}")
        
        # separate three layers of observation information
        private_obs = observation[:self.private_dim]
        public_obs = observation[self.private_dim:self.private_dim + self.public_dim]
        others_obs = observation[self.private_dim + self.public_dim:]
        
        return {
            'private': private_obs,
            'public': public_obs, 
            'others': others_obs
        }
    
    def enhance_private_observation(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:

        enhanced_private = private_obs.copy()
        
        # add history trend information (if there is history)
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-3:]  # last 3 steps
            if len(recent_obs) >= 2:
                # calculate private information trend
                trend = recent_obs[-1][:self.private_dim] - recent_obs[-2][:self.private_dim]
                trend_norm = np.linalg.norm(trend)
                
                # add trend strength as private information enhancement
                enhanced_private = np.append(enhanced_private, min(1.0, trend_norm))
            else:
                enhanced_private = np.append(enhanced_private, 0.0)
        else:
            enhanced_private = np.append(enhanced_private, 0.0)
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: np.ndarray) -> np.ndarray:

        processed_public = public_obs.copy()
        
        # verify the reasonability of public information
        if np.any(np.isnan(processed_public)) or np.any(np.isinf(processed_public)):
            print(f"warning: public observation information contains invalid values")
            processed_public = np.nan_to_num(processed_public, nan=0.0, posinf=0.0, neginf=0.0)
        
        return processed_public
    
    def process_others_observation(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:

        if not self.config.enable_other_manager_info:
            # if other information is disabled, return zero vector
            return np.zeros_like(others_obs)
        
        processed_others = others_obs.copy()
        
        # apply observation noise (if enabled)
        if self.config.enable_observation_noise:
            noise_level = self.config.noise_level
            noise = np.random.normal(0, noise_level, processed_others.shape)
            processed_others += noise
        
        # apply information quality weights
        processed_others *= self.others_weight
        
        # simulate information loss (randomly set some information to zero)
        if self.config.enable_observation_noise:
            loss_prob = self.config.noise_level * 0.5  # information loss probability
            loss_mask = np.random.random(processed_others.shape) > loss_prob
            processed_others *= loss_mask
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: np.ndarray, 
                              public_obs: np.ndarray, 
                              others_obs: np.ndarray,
                              enhanced: bool = True) -> np.ndarray:
        if enhanced:
            # enhanced mode: add inter-layer interaction information
            
            # calculate the correlation between private and public information
            private_public_corr = np.dot(private_obs[:min(len(private_obs), len(public_obs))], 
                                       public_obs[:min(len(private_obs), len(public_obs))])
            private_public_corr = np.tanh(private_public_corr)  # normalize to [-1,1]
            
            # calculate the correlation between private and other information
            private_others_corr = np.dot(private_obs[:min(len(private_obs), len(others_obs))], 
                                       others_obs[:min(len(private_obs), len(others_obs))])
            private_others_corr = np.tanh(private_others_corr)  # normalize to [-1,1]
            
            # add interaction features
            interaction_features = np.array([private_public_corr, private_others_corr])
            
            # reconstruct observation: private + public + other + interaction
            reconstructed = np.concatenate([private_obs, public_obs, others_obs, interaction_features])
        else:
            # standard mode: directly concatenate
            reconstructed = np.concatenate([private_obs, public_obs, others_obs])
        
        return reconstructed
    
    def adapt_observation_for_fomappo(self, observation: np.ndarray, manager_id: str) -> Dict[str, torch.Tensor]:
        # parse layered observation
        parsed_obs = self.parse_observation(observation, manager_id)
        
        # process each layer of observation
        enhanced_private = self.enhance_private_observation(parsed_obs['private'], manager_id)
        processed_public = self.process_public_observation(parsed_obs['public'])
        processed_others = self.process_others_observation(parsed_obs['others'], manager_id)
        
        # reconstruct full observation
        reconstructed_obs = self.reconstruct_observation(
            enhanced_private, processed_public, processed_others, enhanced=True
        )
        
        # update observation history
        self._update_observation_history(manager_id, observation)
        
        # convert to PyTorch tensor
        adapted_obs = {
            'full_observation': torch.FloatTensor(reconstructed_obs).to(self.device),
            'private_features': torch.FloatTensor(enhanced_private).to(self.device),
            'public_features': torch.FloatTensor(processed_public).to(self.device),
            'others_features': torch.FloatTensor(processed_others).to(self.device),
            'layer_weights': torch.FloatTensor([
                self.private_weight, self.public_weight, self.others_weight
            ]).to(self.device)
        }
        
        return adapted_obs
    
    def _update_observation_history(self, manager_id: str, observation: np.ndarray):
        """update observation history"""
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.copy())
        
        # limit history length
        if len(self.observation_history[manager_id]) > self.max_history_len:
            self.observation_history[manager_id] = self.observation_history[manager_id][-self.max_history_len:]
    
    def get_observation_stats(self, manager_id: str) -> Dict[str, float]:
        """get observation statistics"""
        if manager_id not in self.observation_history or len(self.observation_history[manager_id]) == 0:
            return {}
        
        recent_obs = np.array(self.observation_history[manager_id])
        
        stats = {
            'mean': np.mean(recent_obs),
            'std': np.std(recent_obs),
            'min': np.min(recent_obs),
            'max': np.max(recent_obs),
            'history_length': len(self.observation_history[manager_id])
        }
        
        return stats
    
    def reset_history(self, manager_id: Optional[str] = None):
        """reset observation history"""
        if manager_id is None:
            self.observation_history.clear()
        else:
            if manager_id in self.observation_history:
                del self.observation_history[manager_id]


class DecPOMDPAwareNetwork(nn.Module):
    """
    Dec-POMDP perception network layer
    
    specifically designed for processing layered observation information
    """
    
    def __init__(self, private_dim: int, public_dim: int, others_dim: int, 
                 hidden_dim: int = 128, output_dim: int = 64):
        super(DecPOMDPAwareNetwork, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        # layered processing network
        self.private_net = nn.Sequential(
            nn.Linear(private_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.public_net = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.others_net = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )
        
        # fusion network
        fusion_input_dim = hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 8
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim // 4, num_heads=4, batch_first=True)
        
    def forward(self, private_obs: torch.Tensor, public_obs: torch.Tensor, others_obs: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        
        Args:
            private_obs: private observation [batch_size, private_dim]
            public_obs: public observation [batch_size, public_dim]
            others_obs: other observation [batch_size, others_dim]
            
        Returns:
            fused feature representation [batch_size, output_dim]
        """
        # layered feature extraction
        private_features = self.private_net(private_obs)
        public_features = self.public_net(public_obs)
        others_features = self.others_net(others_obs)
        
        # apply attention mechanism (optional)
        if private_features.dim() == 2:
            # add sequence dimension for attention mechanism
            attention_input = torch.stack([private_features, public_features], dim=1)  # [batch_size, 2, hidden_dim//4]
            attended_features, _ = self.attention(attention_input, attention_input, attention_input)
            private_attended = attended_features[:, 0, :]  # [batch_size, hidden_dim//4]
            public_attended = attended_features[:, 1, :]   # [batch_size, hidden_dim//4]
        else:
            private_attended = private_features
            public_attended = public_features
        
        # feature fusion
        fused_features = torch.cat([private_attended, public_attended, others_features], dim=-1)
        output = self.fusion_net(fused_features)
        
        return output 
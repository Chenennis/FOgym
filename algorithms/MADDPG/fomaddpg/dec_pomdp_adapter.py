#!/usr/bin/env python3
"""
FOMADDPG Dec-POMDP observation space adapter

Provide observation space processing capabilities for FOMADDPG algorithm.
Optimized for Actor-Critic deterministic policy gradient algorithm.

Core functions:
1. Observation space hierarchical parsing 
2. Observation processing optimization for deterministic policy
3. Observation enhancement for continuous action space
4. Multi-Agent collaboration information DDPG adaptation
5. Observation consistency processing for target network update
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import sys
import os

# add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMaddpgDecPOMDPAdapter:
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimension (consistent with FOMAPPO)
        self.private_dim = 39  # private information layer dimension
        self.public_dim = 18   # public information layer dimension  
        self.others_dim = 15   # limited other information layer dimension
        
        # total observation dimension
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72 dimensions
        
        # FOMADDPG specific observation processing weights
        self.private_weight = 1.0   # private information completely trusted
        self.public_weight = 1.0    # public information completely trusted
        self.others_weight = 0.7    # other information in DDPG is slightly more trusted 
        
        # deterministic policy specific parameters
        self.deterministic_mode = True  # DDPG uses deterministic policy
        self.action_smoothing_factor = 0.95  # action smoothing factor
        
        # history observation cache (for consistency of target network update)
        self.observation_history = {}
        self.max_history_len = 5  # DDPG usually needs a short history
        
        # multi-agent collaboration specific cache
        self.global_observation_cache = None
        self.local_observation_cache = {}
        
    def parse_observation(self, observation: Union[np.ndarray, torch.Tensor], 
                         manager_id: str) -> Dict[str, torch.Tensor]:

        # convert to torch tensor
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        # process batch dimension
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        if observation.shape[-1] != self.total_obs_dim:
            raise ValueError(f"observation dimension mismatch: expected {self.total_obs_dim}, actual {observation.shape[-1]}")
        
        # separate three layers of observation information
        private_obs = observation[..., :self.private_dim]
        public_obs = observation[..., self.private_dim:self.private_dim + self.public_dim]
        others_obs = observation[..., self.private_dim + self.public_dim:]
        
        return {
            'private': private_obs,
            'public': public_obs, 
            'others': others_obs
        }
    
    def enhance_private_observation(self, private_obs: torch.Tensor, 
                                  manager_id: str) -> torch.Tensor:

        enhanced_private = private_obs.clone()
        
        # for deterministic policy, use smoother history information processing
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-2:]  # only use the last 2 steps
            if len(recent_obs) >= 2:
                # calculate trend, but use a smoother way
                prev_private = recent_obs[-1][:self.private_dim] if recent_obs[-1].shape[0] >= self.private_dim else torch.zeros_like(enhanced_private[0])
                trend = enhanced_private[0] - prev_private
                trend_norm = torch.norm(trend).item()
                
                # use exponential smoothing
                smoothed_trend = min(1.0, trend_norm) * self.action_smoothing_factor
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[smoothed_trend]]).to(self.device)], dim=-1)
            else:
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        else:
            enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: torch.Tensor) -> torch.Tensor:
        # cache global observation information, ensure multi-agent consistency
        processed_public = public_obs.clone()
        
        # verify the reasonability of public information
        if torch.any(torch.isnan(processed_public)) or torch.any(torch.isinf(processed_public)):
            print(f"warning: public observation information contains invalid values")
            processed_public = torch.nan_to_num(processed_public, nan=0.0, posinf=0.0, neginf=0.0)
        
        # update global observation cache
        self.global_observation_cache = processed_public.clone()
        
        return processed_public
    
    def process_others_observation(self, others_obs: torch.Tensor, 
                                 manager_id: str) -> torch.Tensor:
        """
        process limited other observation information - DDPG multi-agent optimization
        """
        if not self.config.enable_other_manager_info:
            # if other information is disabled, return zero vector
            return torch.zeros_like(others_obs)
        
        processed_others = others_obs.clone()
        
        # DDPG specific noise processing 
        if self.config.enable_observation_noise:
            noise_level = self.config.noise_level * 0.7  # DDPG uses smaller noise
            noise = torch.randn_like(processed_others) * noise_level
            processed_others += noise
        
        # apply information quality weights
        processed_others *= self.others_weight
        
        # information loss processing in DDPG 
        if self.config.enable_observation_noise and hasattr(self.config, 'enable_info_missing'):
            if getattr(self.config, 'enable_info_missing', False):
                loss_prob = self.config.noise_level * 0.3  # lower information loss probability
                loss_mask = torch.rand_like(processed_others) > loss_prob
                processed_others *= loss_mask.float()
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: torch.Tensor, 
                              public_obs: torch.Tensor, 
                              others_obs: torch.Tensor,
                              manager_id: str,
                              enhanced: bool = True) -> torch.Tensor:
        if enhanced:
            private_flat = private_obs.view(private_obs.shape[0], -1)
            public_flat = public_obs.view(public_obs.shape[0], -1)
            
            # ensure dimension matching
            min_dim = min(private_flat.shape[1], public_flat.shape[1])
            private_flat = private_flat[:, :min_dim]
            public_flat = public_flat[:, :min_dim]
            
            # calculate cosine similarity
            private_norm = torch.norm(private_flat, dim=1, keepdim=True) + 1e-8
            public_norm = torch.norm(public_flat, dim=1, keepdim=True) + 1e-8
            
            private_public_corr = torch.sum(private_flat * public_flat, dim=1) / (private_norm.squeeze() * public_norm.squeeze())
            private_public_corr = torch.tanh(private_public_corr).unsqueeze(1)  # normalize to [-1,1]
            
            # calculate observation stability indicator (DDPG specific)
            if manager_id in self.observation_history and len(self.observation_history[manager_id]) > 0:
                prev_obs = self.observation_history[manager_id][-1]
                current_obs = torch.cat([private_obs, public_obs, others_obs], dim=-1)
                
                # ensure dimension matching
                if prev_obs.shape[1] == current_obs.shape[1]:
                    stability = 1.0 - torch.norm(current_obs - prev_obs, dim=1).mean().item()
                    stability = max(0.0, min(1.0, stability))  # limit to [0,1]
                else:
                    stability = 0.5  
            else:
                stability = 0.5
            
            stability_feature = torch.tensor([[stability]], device=self.device).expand(private_obs.shape[0], 1)
            
            # add DDPG specific interaction features
            interaction_features = torch.cat([private_public_corr, stability_feature], dim=1)
            
            reconstructed = torch.cat([private_obs, public_obs, others_obs, interaction_features], dim=-1)
        else:
            reconstructed = torch.cat([private_obs, public_obs, others_obs], dim=-1)
        
        # cache observation for stability calculation
        self.local_observation_cache[manager_id] = reconstructed.clone()
        
        return reconstructed
    
    def adapt_observation_for_fomaddpg(self, observation: Union[np.ndarray, torch.Tensor], 
                                     manager_id: str) -> Dict[str, torch.Tensor]:

        # 1. parse observation
        parsed_obs = self.parse_observation(observation, manager_id)
        
        # 2. enhance each layer of observation
        enhanced_private = self.enhance_private_observation(parsed_obs['private'], manager_id)
        processed_public = self.process_public_observation(parsed_obs['public'])
        processed_others = self.process_others_observation(parsed_obs['others'], manager_id)
        
        # 3. reconstruct complete observation
        fused_observation = self.reconstruct_observation(
            enhanced_private, processed_public, processed_others, manager_id, enhanced=True
        )
        
        # 4. update history record
        self._update_observation_history(manager_id, fused_observation)
        
        return {
            'private': enhanced_private,
            'public': processed_public,
            'others': processed_others,
            'fused': fused_observation,
            'raw_parsed': parsed_obs
        }
    
    def _update_observation_history(self, manager_id: str, observation: torch.Tensor):
        """update observation history"""
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.clone())
        
        # keep history length limit
        if len(self.observation_history[manager_id]) > self.max_history_len:
            self.observation_history[manager_id].pop(0)
    
    def get_observation_stats(self, manager_id: str) -> Dict[str, float]:
        """get observation statistics"""
        if manager_id not in self.observation_history or len(self.observation_history[manager_id]) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'stability': 0.0,
                'history_length': 0
            }
        
        recent_obs = torch.stack(self.observation_history[manager_id])
        
        return {
            'mean': recent_obs.mean().item(),
            'std': recent_obs.std().item(),
            'stability': 1.0 - recent_obs.std().item() if recent_obs.std().item() < 1.0 else 0.0,
            'history_length': len(self.observation_history[manager_id])
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """reset observation history"""
        if manager_id is None:
            self.observation_history.clear()
            self.local_observation_cache.clear()
            self.global_observation_cache = None
        else:
            if manager_id in self.observation_history:
                del self.observation_history[manager_id]
            if manager_id in self.local_observation_cache:
                del self.local_observation_cache[manager_id]
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """get adapted observation dimension information"""
        return {
            'private_dim': self.private_dim + 1,  # +1 for trend
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'interaction_dim': 2,  # private_public_corr + stability
            'total_enhanced_dim': self.private_dim + 1 + self.public_dim + self.others_dim + 2
        }
    
    def enable_deterministic_mode(self, deterministic: bool = True):
        """enable/disable deterministic mode"""
        self.deterministic_mode = deterministic
        if deterministic:
            self.action_smoothing_factor = 0.98  # smoother
            self.others_weight = 0.8  # increase other information weight
        else:
            self.action_smoothing_factor = 0.95  # normal
            self.others_weight = 0.7  # normal weight 
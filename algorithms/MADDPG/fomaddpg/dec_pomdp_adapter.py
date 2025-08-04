#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMaddpgDecPOMDPAdapter:
    """
    FOMADDPG Dec-POMDP observation space adapter
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimensions (consistent with FOMAPPO)
        self.private_dim = 39  # Private information layer dimension
        self.public_dim = 18   # Public information layer dimension
        self.others_dim = 15   # Limited other agents' information layer dimension
        
        # Total observation dimension
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72 dimensions
        
        # FOMADDPG specific observation processing weights
        self.private_weight = 1.0   # Private information fully trusted
        self.public_weight = 1.0    # Public information fully trusted
        self.others_weight = 0.7    # Other agents' information slightly more trusted in DDPG (original 0.8→0.7)
        
        # Deterministic policy specific parameters
        self.deterministic_mode = True  # DDPG uses deterministic policy
        self.action_smoothing_factor = 0.95  # Action smoothing factor
        
        # Observation history cache (for consistency in target network updates)
        self.observation_history = {}
        self.max_history_len = 5  # DDPG typically needs shorter history
        
        # Multi-agent collaboration specific cache
        self.global_observation_cache = None
        self.local_observation_cache = {}
        
    def parse_observation(self, observation: Union[np.ndarray, torch.Tensor], 
                         manager_id: str) -> Dict[str, torch.Tensor]:
        """
        Parse the layered structure of Dec-POMDP observation space
        """
        # Convert to torch tensor
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        # Handle batch dimension
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        if observation.shape[-1] != self.total_obs_dim:
            raise ValueError(f"Observation dimension mismatch: expected {self.total_obs_dim}, got {observation.shape[-1]}")
        
        # Separate three layers of observation information
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
        """
        Enhance private observation information - optimized for deterministic policy
        
        DDPG characteristics:
        1. Deterministic policy needs more stable observations
        2. Reduce the impact of observation noise
        3. Maintain temporal consistency
        """
        enhanced_private = private_obs.clone()
        
        # For deterministic policy, use smoother historical information processing
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-2:]  # Only use the most recent 2 steps
            if len(recent_obs) >= 2:
                # Calculate trend, but in a smoother way
                prev_private = recent_obs[-1][:self.private_dim] if recent_obs[-1].shape[0] >= self.private_dim else torch.zeros_like(enhanced_private[0])
                trend = enhanced_private[0] - prev_private
                trend_norm = torch.norm(trend).item()
                
                # Use exponential smoothing
                smoothed_trend = min(1.0, trend_norm) * self.action_smoothing_factor
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[smoothed_trend]]).to(self.device)], dim=-1)
            else:
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        else:
            # No history yet
            enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        
        # Update observation history
        self._update_observation_history(manager_id, private_obs[0])
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: torch.Tensor) -> torch.Tensor:
        """
        Process public observation information
        
        For DDPG, public information is fully trusted and can be used directly
        """
        # For DDPG, we can directly use public information
        processed_public = public_obs.clone()
        
        # Cache global observation for multi-agent coordination
        if self.global_observation_cache is None:
            self.global_observation_cache = processed_public.detach()
        else:
            # Update with exponential moving average
            alpha = 0.8
            self.global_observation_cache = alpha * self.global_observation_cache + (1 - alpha) * processed_public.detach()
        
        return processed_public
    
    def process_others_observation(self, others_obs: torch.Tensor, 
                                 manager_id: str) -> torch.Tensor:
        """
        Process other agents' observation information
        
        For DDPG:
        1. Other agents' information is more trusted than in stochastic policy algorithms
        2. Continuous action space requires better coordination
        3. Apply appropriate weighting
        """
        # For DDPG, we can use a higher weight for other agents' information
        processed_others = others_obs.clone()
        
        # Apply weight
        processed_others = processed_others * self.others_weight
        
        # Cache for coordination
        if manager_id not in self.local_observation_cache:
            self.local_observation_cache[manager_id] = {}
        
        self.local_observation_cache[manager_id]['others'] = processed_others.detach()
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: torch.Tensor, 
                              public_obs: torch.Tensor, 
                              others_obs: torch.Tensor,
                              manager_id: str,
                              enhanced: bool = True) -> torch.Tensor:
        """
        Reconstruct complete observation from three layers
        
        For DDPG:
        1. Apply appropriate weights to each layer
        2. Ensure stability for deterministic policy
        3. Maintain temporal consistency
        
        Args:
            private_obs: Private observation layer
            public_obs: Public observation layer
            others_obs: Other agents' observation layer
            manager_id: Manager ID
            enhanced: Whether to use enhanced observation
            
        Returns:
            Reconstructed complete observation
        """
        # Apply weights to each layer
        weighted_private = private_obs * self.private_weight
        weighted_public = public_obs * self.public_weight
        weighted_others = others_obs * self.others_weight
        
        # Reconstruct complete observation
        if enhanced and self.deterministic_mode:
            # For deterministic policy, we need more stable observation
            if manager_id in self.observation_history and len(self.observation_history[manager_id]) > 0:
                # Get history
                history = self.observation_history[manager_id]
                
                # If we have enough history, apply temporal smoothing
                if len(history) >= 2:
                    prev_obs = history[-1]
                    prev2_obs = history[-2]
                    
                    # Calculate trend
                    trend = prev_obs - prev2_obs
                    
                    # Apply temporal smoothing - weighted average of current and trend
                    alpha = 0.8  # Weight for current observation
                    beta = 0.2   # Weight for trend
                    
                    # Only apply to private information which is most volatile
                    smoothed_private = alpha * weighted_private + beta * trend[:self.private_dim].unsqueeze(0)
                    
                    # Reconstruct with smoothed private information
                    reconstructed = torch.cat([smoothed_private, weighted_public, weighted_others], dim=-1)
                else:
                    # Not enough history, use weighted concatenation
                    reconstructed = torch.cat([weighted_private, weighted_public, weighted_others], dim=-1)
            else:
                # No history, use weighted concatenation
                reconstructed = torch.cat([weighted_private, weighted_public, weighted_others], dim=-1)
        else:
            # For non-enhanced mode, simply concatenate weighted observations
            reconstructed = torch.cat([weighted_private, weighted_public, weighted_others], dim=-1)
        
        return reconstructed
    
    def adapt_observation_for_fomaddpg(self, observation: Union[np.ndarray, torch.Tensor], 
                                     manager_id: str) -> Dict[str, torch.Tensor]:
        """
        Adapt observation for FOMADDPG
        
        Main entry point for observation adaptation in FOMADDPG
        
        Args:
            observation: Raw observation
            manager_id: Manager ID
            
        Returns:
            Dictionary containing adapted observation information
        """
        # Parse observation into three layers
        parsed_obs = self.parse_observation(observation, manager_id)
        
        # Process each layer
        enhanced_private = self.enhance_private_observation(parsed_obs['private'], manager_id)
        processed_public = self.process_public_observation(parsed_obs['public'])
        processed_others = self.process_others_observation(parsed_obs['others'], manager_id)
        
        # Reconstruct complete observation
        reconstructed = self.reconstruct_observation(
            enhanced_private, 
            processed_public, 
            processed_others,
            manager_id
        )
        
        # Return both the reconstructed observation and individual components
        return {
            'reconstructed': reconstructed,
            'private': enhanced_private,
            'public': processed_public,
            'others': processed_others
        }
    
    def _update_observation_history(self, manager_id: str, observation: torch.Tensor):
        """
        Update observation history for a manager
        """
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.detach().clone())
        
        # Limit history length
        if len(self.observation_history[manager_id]) > self.max_history_len:
            self.observation_history[manager_id].pop(0)
    
    def get_observation_stats(self, manager_id: str) -> Dict[str, float]:
        """
        Get observation statistics for a manager
        
        Returns statistics about observation history and processing
        """
        stats = {}
        
        if manager_id in self.observation_history:
            history = self.observation_history[manager_id]
            stats['history_length'] = len(history)
            
            if len(history) >= 2:
                # Calculate observation variance
                history_tensor = torch.stack(history)
                stats['observation_variance'] = torch.var(history_tensor, dim=0).mean().item()
                
                # Calculate temporal difference
                diffs = torch.abs(history_tensor[1:] - history_tensor[:-1])
                stats['mean_temporal_difference'] = diffs.mean().item()
        else:
            stats['history_length'] = 0
            stats['observation_variance'] = 0.0
            stats['mean_temporal_difference'] = 0.0
        
        return stats
    
    def reset_history(self, manager_id: Optional[str] = None):
        """
        Reset observation history
        
        Args:
            manager_id: If provided, reset only for this manager; otherwise reset all
        """
        if manager_id is not None:
            if manager_id in self.observation_history:
                self.observation_history[manager_id] = []
            if manager_id in self.local_observation_cache:
                self.local_observation_cache[manager_id] = {}
        else:
            self.observation_history = {}
            self.local_observation_cache = {}
            self.global_observation_cache = None
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of adapted observation components
        """
        return {
            'private': self.private_dim + 1,  # +1 for trend information
            'public': self.public_dim,
            'others': self.others_dim,
            'reconstructed': self.private_dim + 1 + self.public_dim + self.others_dim
        }
    
    def enable_deterministic_mode(self, deterministic: bool = True):
        """
        Enable or disable deterministic mode
        
        In deterministic mode, observations are processed for stability
        """
        self.deterministic_mode = deterministic
        if deterministic:
            self.others_weight = 0.7  # Higher weight for other agents' information
        else:
            self.others_weight = 0.5  # Lower weight for exploration 
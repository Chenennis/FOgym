#!/usr/bin/env python3
"""
Dec-POMDP Observation Space Adapter

Provides Dec-POMDP observation space processing capabilities for the FOMAPPO algorithm.
Supports separation, processing, and recombination of three-layer observation information.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPObservationAdapter:
    """
    Dec-POMDP Observation Space Adapter
    
    Processes layered observation information, providing structured observation inputs for the FOMAPPO algorithm
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimensions (based on implemented architecture)
        self.private_dim = 39  # Private information layer dimension
        self.public_dim = 18   # Public information layer dimension
        self.others_dim = 15   # Limited other-agent information layer dimension
        
        # Total observation dimension
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72 dimensions
        
        # Observation processing weights
        self.private_weight = 1.0  # Private information fully trusted
        self.public_weight = 1.0   # Public information fully trusted
        self.others_weight = 0.8   # Other-agent information partially trusted (configurable noise)
        
        # Observation history cache
        self.observation_history = {}
        self.max_history_len = 10
        
    def parse_observation(self, observation: np.ndarray, manager_id: str) -> Dict[str, np.ndarray]:
        """
        Parse the layered structure of Dec-POMDP observation space
        
        Args:
            observation: Complete observation vector (72 dimensions)
            manager_id: Manager identifier
            
        Returns:
            Dict containing:
                - 'private': Private information layer (39 dimensions)
                - 'public': Public information layer (18 dimensions)
                - 'others': Limited other-agent information layer (15 dimensions)
        """
        if len(observation) != self.total_obs_dim:
            raise ValueError(f"Observation dimension mismatch: expected {self.total_obs_dim}, got {len(observation)}")
        
        # Separate three layers of observation information
        private_obs = observation[:self.private_dim]
        public_obs = observation[self.private_dim:self.private_dim + self.public_dim]
        others_obs = observation[self.private_dim + self.public_dim:]
        
        return {
            'private': private_obs,
            'public': public_obs, 
            'others': others_obs
        }
    
    def enhance_private_observation(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """
        Enhance private observation information

        """
        enhanced_private = private_obs.copy()
        
        # Ensure private observation is standardized
        # Already standardized in Manager, keep as is
        
        # Add historical trend information (if history exists)
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-3:]  # Last 3 steps
            if len(recent_obs) >= 3:
                # Calculate trend features
                prev_private = recent_obs[-1][:self.private_dim]
                prev2_private = recent_obs[-2][:self.private_dim]
                
                # Short-term trend (t-1 to t)
                short_trend = enhanced_private - prev_private
                
                # Long-term trend (t-2 to t)
                long_trend = enhanced_private - prev2_private
                
                # Add trend direction as feature
                trend_direction = np.sign(short_trend).mean()
                
                # Add trend magnitude as feature
                trend_magnitude = np.abs(short_trend).mean()
                
                # Append trend features
                enhanced_private = np.append(enhanced_private, [trend_direction, trend_magnitude])
        
        # Update observation history
        self._update_observation_history(manager_id, private_obs)
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: np.ndarray) -> np.ndarray:

        processed_public = public_obs.copy()
        
        # Public information is shared across all agents, no need for agent-specific processing
        # Ensure standardization
        if self.config.enable_observation_normalization:
            # Simple standardization (zero mean, unit variance)
            # Already standardized in Manager, keep as is
            pass
            
        return processed_public
    
    def process_others_observation(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """
        Process other-agent observation information
        
        Other-agent information processing principles:
        1. Apply configurable noise (based on dec_pomdp_config)
        2. Apply information loss (based on dec_pomdp_config)
        3. Weight appropriately
        """
        if not self.config.enable_other_manager_info:
            # If other-agent information is disabled, return zeros
            return np.zeros_like(others_obs)
        
        processed_others = others_obs.copy()
        
        # Apply noise if enabled
        if self.config.enable_observation_noise:
            noise_level = self.config.noise_level
            noise = np.random.normal(0, noise_level, size=processed_others.shape)
            processed_others += noise
        
        # Apply information loss if enabled
        if hasattr(self.config, 'enable_info_missing') and self.config.enable_info_missing:
            loss_prob = self.config.noise_level * 2  # Higher probability of information loss
            loss_mask = np.random.random(processed_others.shape) > loss_prob
            processed_others *= loss_mask
        
        # Apply weight
        processed_others *= self.others_weight
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: np.ndarray, 
                              public_obs: np.ndarray, 
                              others_obs: np.ndarray,
                              enhanced: bool = True) -> np.ndarray:
        """
        Reconstruct complete observation from three layers
        
        Args:
            private_obs: Private observation layer
            public_obs: Public observation layer
            others_obs: Other-agent observation layer
            enhanced: Whether to use enhanced mode
            
        Returns:
            Reconstructed complete observation vector
        """
        # Apply weights to each layer
        weighted_private = private_obs * self.private_weight
        weighted_public = public_obs * self.public_weight
        weighted_others = others_obs * self.others_weight
        
        # Concatenate layers
        reconstructed = np.concatenate([weighted_private, weighted_public, weighted_others])
        
        # Apply enhancement if enabled
        if enhanced and self.config.enable_observation_enhancement:
            # Add global features
            if hasattr(self.config, 'global_features') and self.config.global_features is not None:
                reconstructed = np.concatenate([reconstructed, self.config.global_features])
        
        return reconstructed
    
    def adapt_observation_for_fomappo(self, observation: np.ndarray, manager_id: str) -> Dict[str, torch.Tensor]:
        """
        Adapt observation for FOMAPPO algorithm
        
        Main entry point for observation adaptation in FOMAPPO
        
        Args:
            observation: Raw observation
            manager_id: Manager identifier
            
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
            processed_others
        )
        
        # Convert to PyTorch tensors
        private_tensor = torch.FloatTensor(enhanced_private).to(self.device)
        public_tensor = torch.FloatTensor(processed_public).to(self.device)
        others_tensor = torch.FloatTensor(processed_others).to(self.device)
        reconstructed_tensor = torch.FloatTensor(reconstructed).to(self.device)
        
        # Return both the reconstructed observation and individual components
        return {
            'reconstructed': reconstructed_tensor,
            'private': private_tensor,
            'public': public_tensor,
            'others': others_tensor,
            'raw_parsed': {
                'private': torch.FloatTensor(parsed_obs['private']).to(self.device),
                'public': torch.FloatTensor(parsed_obs['public']).to(self.device),
                'others': torch.FloatTensor(parsed_obs['others']).to(self.device)
            }
        }
    
    def _update_observation_history(self, manager_id: str, observation: np.ndarray):
        """Update observation history for a manager"""
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.copy())
        
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
                history_array = np.array(history)
                stats['observation_variance'] = np.var(history_array).mean()
                
                # Calculate temporal difference
                diffs = np.abs(history_array[1:] - history_array[:-1])
                stats['mean_temporal_difference'] = diffs.mean()
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
        else:
            self.observation_history = {}

class DecPOMDPAwareNetwork(nn.Module):
    """
    Dec-POMDP aware neural network
    
    Processes layered Dec-POMDP observations with specialized processing for each layer
    """
    
    def __init__(self, private_dim: int, public_dim: int, others_dim: int, 
                 hidden_dim: int = 128, output_dim: int = 64):
        super(DecPOMDPAwareNetwork, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        # Private information encoder
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Public information encoder
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Other-agent information encoder
        self.others_encoder = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU()
        )
        
        # Fusion network
        fusion_input_dim = (hidden_dim // 2) + (hidden_dim // 4) + (hidden_dim // 8)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, private_obs: torch.Tensor, public_obs: torch.Tensor, others_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dec-POMDP aware network
        
        Args:
            private_obs: Private observation layer
            public_obs: Public observation layer
            others_obs: Other-agent observation layer
            
        Returns:
            Fused representation
        """
        # Process each layer
        private_features = self.private_encoder(private_obs)
        public_features = self.public_encoder(public_obs)
        others_features = self.others_encoder(others_obs)
        
        # Concatenate features
        fused_features = torch.cat([private_features, public_features, others_features], dim=-1)
        
        # Apply fusion network
        output = self.fusion_network(fused_features)
        
        return output 
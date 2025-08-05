#!/usr/bin/env python3
"""
FOMATD3 Dec-POMDP Observation Space Adapter

Provides Dec-POMDP observation space processing capabilities for the FOMATD3 algorithm.
Optimized for the characteristics of the Twin Delayed DDPG algorithm.

Core Features:
1. Hierarchical observation space parsing (reuse Dec-POMDP architecture)
2. TD3 specific observation processing optimization
3. Observation enhancement of twin critic network
4. Target policy smoothing observation consistency
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMAtd3DecPOMDPAdapter:
    """
    FOMATD3 Dec-POMDP Observation Space Adapter
    
    Specifically designed for the FOMATD3 algorithm, this Dec-POMDP observation processor supports Twin Delayed DDPG and FlexOffer constraint-based observation space management.
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimensions
        self.private_dim = 39      # Private observation base dimension
        self.public_dim = 18       # Public observation dimension
        self.others_dim = 15       # Other observation dimension
        self.total_obs_dim = 72    # Total observation dimension
        
        # TD3 specific parameters
        self.twin_critic_mode = True
        self.target_smoothing_factor = 0.8  # Target policy smoothing factor
        self.delay_update_steps = 2         # Delay update steps
        self.observation_history_length = 3 # Observation history length (TD3 optimization)
        
        # FlexOffer constraint integration
        self.fo_constraint_dim = 36        # FlexOffer constraint dimension
        self.fo_satisfaction_weight = 0.2  # FlexOffer satisfaction weight
        
        # Observation processing cache
        self._observation_cache = {}
        self._history_buffer = {}
        
        # Initialize observation history cache
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """Initialize observation history buffer"""
        for manager_id in [f"manager_{i}" for i in range(4)]:
            self._history_buffer[manager_id] = {
                'private': [],
                'public': [],
                'others': [],
                'full_obs': []
            }
    
    def adapt_observation_for_fomatd3(self, 
                                     observation: np.ndarray, 
                                     manager_id: str,
                                     fo_constraints: Optional[np.ndarray] = None,
                                     fo_satisfaction: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Adapt observation space for FOMATD3
        
        Args:
            observation: Original observation [obs_dim]
            manager_id: Manager ID (e.g., "manager_0")
            fo_constraints: FlexOffer constraint [constraint_dim]
            fo_satisfaction: FlexOffer satisfaction scalar
            
        Returns:
            Adapted hierarchical observation dictionary
        """
        # Parse Dec-POMDP observation
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints(private_obs, fo_constraints, fo_satisfaction)
        else:
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # TD3 specific observation enhancement
        enhanced_private = self._enhance_private_obs_for_td3(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_td3(public_obs)
        enhanced_others = self._enhance_others_obs_for_td3(others_obs, manager_id)
        
        # Observation noise processing
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private')
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public')
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others')
        
        # Convert to tensor
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device)
        }
        
        # Update history cache
        self._update_history_buffer(manager_id, adapted_obs)
        
        return adapted_obs
    
    def _parse_dec_pomdp_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse Dec-POMDP observation structure"""
        if len(observation) < self.total_obs_dim:
            observation = np.pad(observation, (0, self.total_obs_dim - len(observation)))
        elif len(observation) > self.total_obs_dim:
            observation = observation[:self.total_obs_dim]
        
        # Hierarchical parsing
        private_start = 0
        private_end = self.private_dim
        public_start = private_end
        public_end = public_start + self.public_dim
        others_start = public_end
        others_end = others_start + self.others_dim
        
        private_obs = observation[private_start:private_end]
        public_obs = observation[public_start:public_end]
        others_obs = observation[others_start:others_end]
        
        return private_obs, public_obs, others_obs
    
    def _integrate_fo_constraints(self, 
                                 private_obs: np.ndarray, 
                                 fo_constraints: np.ndarray,
                                 fo_satisfaction: Optional[float] = None) -> np.ndarray:
        """Integrate FlexOffer constraints into private observations"""
        # FlexOffer constraint feature extraction
        constraint_features = self._extract_fo_constraint_features(fo_constraints)
        
        # FlexOffer satisfaction processing
        satisfaction_feature = fo_satisfaction if fo_satisfaction is not None else 0.8
        
        # Add trend information (TD3 optimization: more focus on long-term trends)
        constraint_trend = np.mean(constraint_features) - np.mean(private_obs[:10])  # Compare first 10 dimensions
        
        # Expand private observations: 39 + 1(trend) = 40 dimensions
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40]  # Ensure dimension consistency
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """Extract FlexOffer constraint features"""
        if len(fo_constraints) == 0:
            return np.zeros(5)  # Default constraint features
        
        # Statistical features
        constraint_features = np.array([
            np.mean(fo_constraints),     # Average constraint value
            np.std(fo_constraints),      # Constraint variance
            np.min(fo_constraints),      # Minimum constraint
            np.max(fo_constraints),      # Maximum constraint
            np.sum(fo_constraints > 0.5) / len(fo_constraints)  # Activation ratio
        ])
        
        return constraint_features
    
    def _enhance_private_obs_for_td3(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """Enhance private observations for TD3"""
        # Get history observations for smoothing
        history = self._history_buffer[manager_id]['private']
        
        if len(history) > 0:
            # TD3 target policy smoothing: combine historical information
            recent_history = history[-2:] if len(history) >= 2 else history
            if recent_history:
                avg_history = np.mean(recent_history, axis=0)
                # Smooth current observation
                smoothed_obs = (self.target_smoothing_factor * private_obs + 
                               (1 - self.target_smoothing_factor) * avg_history)
                return smoothed_obs
        
        return private_obs
    
    def _enhance_public_obs_for_td3(self, public_obs: np.ndarray) -> np.ndarray:
        if self.twin_critic_mode:
            noise_scale = 0.02  # Small noise
            noise = np.random.normal(0, noise_scale, public_obs.shape)
            robust_obs = public_obs + noise
            return robust_obs
        
        return public_obs
    
    def _enhance_others_obs_for_td3(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """Enhance others observations for TD3"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        conservative_weight = 0.7  
        
        # Get history others observations
        history = self._history_buffer[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.delay_update_steps:], axis=0) if len(history) >= self.delay_update_steps else np.mean(history, axis=0)
            conservative_obs = conservative_weight * others_obs + (1 - conservative_weight) * recent_avg
            return conservative_obs
        
        return others_obs * conservative_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default') -> np.ndarray:
        """Add observation noise"""
        if not self.config.enable_observation_noise:
            return observation
        
        # TD3 specific noise settings
        noise_scales = {
            'private': self.config.noise_level * 0.8,  # Private observation noise is smaller
            'public': self.config.noise_level * 0.5,   # Public observation noise is smaller
            'others': self.config.noise_level * 1.2    # Others observation noise is larger
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # Generate noise
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # Add noise
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _update_history_buffer(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor]):
        """Update observation history buffer"""
        history = self._history_buffer[manager_id]
        
        # Convert to numpy and add to history
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # Maintain history length
        for key in history:
            if len(history[key]) > self.observation_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """Get adapted observation dimension information"""
        return {
            'private_dim': 40,  # 39 + 1(trend)
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  # 73
            'history_length': self.observation_history_length,
            'fo_constraint_dim': self.fo_constraint_dim
        }
    
    def get_td3_specific_info(self) -> Dict[str, Any]:
        """Get TD3 specific adaptation information"""
        return {
            'twin_critic_mode': self.twin_critic_mode,
            'target_smoothing_factor': self.target_smoothing_factor,
            'delay_update_steps': self.delay_update_steps,
            'observation_history_length': self.observation_history_length,
            'conservative_weight': 0.7,
            'fo_integration': True
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """Reset observation history"""
        if manager_id is not None:
            self._init_history_buffers()
        else:
            if manager_id in self._history_buffer:
                for key in self._history_buffer[manager_id]:
                    self._history_buffer[manager_id][key].clear()
    
    def get_smoothed_observation(self, 
                                manager_id: str, 
                                current_obs: Dict[str, torch.Tensor],
                                smoothing_factor: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Get smoothed observation (TD3 target policy smoothing)"""
        if smoothing_factor is None:
            smoothing_factor = self.target_smoothing_factor
        
        history = self._history_buffer[manager_id]
        
        if len(history['full_obs']) == 0:
            return current_obs
        
        # Get recent observations
        recent_obs = history['full_obs'][-1]
        
        # Smoothing processing
        smoothed_obs = {}
        for key in current_obs:
            if key in ['private', 'public', 'others']:
                current_tensor = current_obs[key]
                if len(history[key]) > 0:
                    recent_tensor = torch.FloatTensor(history[key][-1]).to(self.device)
                    smoothed_tensor = (smoothing_factor * current_tensor + 
                                     (1 - smoothing_factor) * recent_tensor)
                    smoothed_obs[key] = smoothed_tensor
                else:
                    smoothed_obs[key] = current_tensor
            else:
                smoothed_obs[key] = current_obs[key]
        
        return smoothed_obs 
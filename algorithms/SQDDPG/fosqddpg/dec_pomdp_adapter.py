#!/usr/bin/env python3
"""
FOSQDDPG Dec-POMDP Observation Space Adapter

Provides Dec-POMDP observation space processing capabilities for the FOSQDDPG algorithm.
Specifically designed for Shapley value fairness allocation and FlexOffer constraint-based observation space management.

Core Features:
1. Hierarchical observation space parsing (reuse Dec-POMDP architecture)
2. Observation enhancement with Shapley value calculation
3. Fairness-aware observation processing
4. FlexOffer constraint-based observation integration
5. Fairness weight allocation for coalition information
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOSqddpgDecPOMDPAdapter:
    """
    FOSQDDPG Dec-POMDP Observation Space Adapter
    
    Dec-POMDP observation processor specifically designed for FOSQDDPG algorithm,
    supporting Shapley value fairness allocation and FlexOffer constraint-based observation space management.
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimensions
        self.private_dim = 39      # Private observation base dimension
        self.public_dim = 18       # Public observation dimension
        self.others_dim = 15       # Others observation dimension
        self.total_obs_dim = 72    # Total observation dimension
        
        # FOSQDDPG specific parameters
        self.shapley_mode = True
        self.fairness_weight = 0.3              # Fairness weight
        self.credit_assignment_factor = 0.2     # Credit assignment factor
        self.coalition_history_length = 5       # Coalition history length
        
        # FlexOffer constraint integration (enhanced version)
        self.fo_constraint_dim = 36             # FlexOffer constraint dimension
        self.fo_fairness_weight = 0.25          # FlexOffer fairness weight
        self.fo_shapley_integration = True      # Shapley value integration switch
        
        # Observation processing cache
        self._observation_cache = {}
        self._coalition_history = {}
        self._fairness_scores = {}
        
        # Initialize history buffers
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """Initialize observation history buffers"""
        for manager_id in [f"manager_{i}" for i in range(4)]:
            self._coalition_history[manager_id] = {
                'private': [],
                'public': [],
                'others': [],
                'full_obs': [],
                'shapley_values': [],
                'fairness_scores': []
            }
            self._fairness_scores[manager_id] = 1.0  # Initial fairness score
    
    def adapt_observation_for_fosqddpg(self, 
                                      observation: np.ndarray, 
                                      manager_id: str,
                                      fo_constraints: Optional[np.ndarray] = None,
                                      fo_satisfaction: Optional[float] = None,
                                      coalition_info: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Adapt observation space for FOSQDDPG
        
        Args:
            observation: Raw observation [obs_dim]
            manager_id: Manager ID (e.g., "manager_0")
            fo_constraints: FlexOffer constraints [constraint_dim]
            fo_satisfaction: FlexOffer satisfaction scalar
            coalition_info: Coalition information dictionary
            
        Returns:
            Adapted hierarchical observation dictionary, including Shapley value information
        """
        # Parse Dec-POMDP observation
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        # FlexOffer constraint integration (Shapley value aware)
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints_with_shapley(
                private_obs, fo_constraints, fo_satisfaction, manager_id
            )
        else:
            # If no FlexOffer constraints, pad to 40 dimensions
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # FOSQDDPG specific observation enhancement
        enhanced_private = self._enhance_private_obs_for_fosqddpg(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_fosqddpg(public_obs, coalition_info)
        enhanced_others = self._enhance_others_obs_for_fosqddpg(others_obs, manager_id, coalition_info)
        
        # Observation noise processing (fairness weighted)
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private', manager_id=manager_id)
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public', manager_id=manager_id)
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others', manager_id=manager_id)
        
        # Convert to tensors
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device),
            'fairness_score': torch.FloatTensor([self._fairness_scores[manager_id]]).to(self.device),
            'shapley_weight': torch.FloatTensor([self._compute_shapley_weight(manager_id)]).to(self.device)
        }
        
        # Update history cache (including Shapley value information)
        self._update_coalition_history(manager_id, adapted_obs, coalition_info)
        
        return adapted_obs
    
    def _parse_dec_pomdp_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse Dec-POMDP observation structure"""
        if len(observation) < self.total_obs_dim:
            # Pad insufficient dimensions
            observation = np.pad(observation, (0, self.total_obs_dim - len(observation)))
        elif len(observation) > self.total_obs_dim:
            # Truncate excess dimensions
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
    
    def _integrate_fo_constraints_with_shapley(self, 
                                              private_obs: np.ndarray, 
                                              fo_constraints: np.ndarray,
                                              fo_satisfaction: Optional[float] = None,
                                              manager_id: str = None) -> np.ndarray:
        """Integrate FlexOffer constraints into private observation (Shapley value aware)"""
        # FlexOffer constraint feature extraction
        constraint_features = self._extract_fo_constraint_features(fo_constraints)
        
        # FlexOffer satisfaction processing (fairness weighted)
        satisfaction_feature = fo_satisfaction if fo_satisfaction is not None else 0.8
        fairness_score = self._fairness_scores.get(manager_id, 1.0)
        weighted_satisfaction = satisfaction_feature * fairness_score
        
        # Shapley value integrated constraint trend
        if self.fo_shapley_integration:
            shapley_weight = self._compute_shapley_weight(manager_id)
            constraint_trend = (np.mean(constraint_features) * shapley_weight - 
                              np.mean(private_obs[:10]) * (1 - shapley_weight))
        else:
            constraint_trend = np.mean(constraint_features) - np.mean(private_obs[:10])
        
        # Extend private observation: 39 + 1(Shapley trend) = 40 dimensions
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40]  # Ensure consistent dimensions
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """Extract FlexOffer constraint features (fairness aware)"""
        if len(fo_constraints) == 0:
            return np.zeros(5)  # Default constraint features
        
        # Statistical features
        constraint_features = np.array([
            np.mean(fo_constraints),     # Mean constraint value
            np.std(fo_constraints),      # Constraint variance
            np.min(fo_constraints),      # Minimum constraint
            np.max(fo_constraints),      # Maximum constraint
            np.sum(fo_constraints > 0.5) / len(fo_constraints)  # Activation ratio
        ])
        
        return constraint_features
    
    def _enhance_private_obs_for_fosqddpg(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """Enhance private observation for FOSQDDPG (Shapley value integration)"""
        # Get historical observations for Shapley value calculation
        history = self._coalition_history[manager_id]['private']
        
        if len(history) > 0:
            # Shapley value weighted historical information
            shapley_values = self._coalition_history[manager_id]['shapley_values']
            if shapley_values:
                recent_shapley = np.mean(shapley_values[-3:]) if len(shapley_values) >= 3 else np.mean(shapley_values)
                recent_history = history[-2:] if len(history) >= 2 else history
                if recent_history:
                    avg_history = np.mean(recent_history, axis=0)
                    # Shapley value weighted fusion
                    shapley_enhanced_obs = (recent_shapley * private_obs + 
                                          (1 - recent_shapley) * avg_history)
                    return shapley_enhanced_obs
        
        return private_obs
    
    def _enhance_public_obs_for_fosqddpg(self, public_obs: np.ndarray, coalition_info: Optional[Dict] = None) -> np.ndarray:
        """Enhance public observation for FOSQDDPG (coalition aware)"""
        # Coalition information integration
        if coalition_info and self.shapley_mode:
            coalition_strength = coalition_info.get('coalition_strength', 1.0)
            coalition_fairness = coalition_info.get('fairness_index', 1.0)
            
            # Coalition adjustment for public observation
            coalition_factor = 0.1 * coalition_strength * coalition_fairness
            enhanced_obs = public_obs * (1 + coalition_factor)
            return enhanced_obs
        
        return public_obs
    
    def _enhance_others_obs_for_fosqddpg(self, others_obs: np.ndarray, manager_id: str, coalition_info: Optional[Dict] = None) -> np.ndarray:
        """Enhance others observation for FOSQDDPG (fairness weighted)"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        # Apply fairness weight
        fairness_weight = self.fairness_weight
        
        # Adjust fairness weight if coalition information is available
        if coalition_info:
            member_fairness = coalition_info.get('member_fairness', {})
            if manager_id in member_fairness:
                individual_fairness = member_fairness[manager_id]
                fairness_weight *= individual_fairness
        
        # Get historical others observations
        history = self._coalition_history[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.coalition_history_length:], axis=0) if len(history) >= self.coalition_history_length else np.mean(history, axis=0)
            # Fairness weighted update
            fair_weighted_obs = fairness_weight * others_obs + (1 - fairness_weight) * recent_avg
            return fair_weighted_obs
        
        return others_obs * fairness_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default', manager_id: str = None) -> np.ndarray:
        """Add observation noise (fairness adjusted)"""
        if not self.config.enable_observation_noise:
            return observation
        
        # FOSQDDPG specific noise settings (fairness adjusted)
        fairness_factor = self._fairness_scores.get(manager_id, 1.0) if manager_id else 1.0
        
        noise_scales = {
            'private': self.config.noise_level * 0.7 * fairness_factor,     # Private observation noise (fairness adjusted)
            'public': self.config.noise_level * 0.4,                        # Public observation noise is smaller
            'others': self.config.noise_level * 1.1 * (2 - fairness_factor) # Others observation noise (inverse fairness adjusted)
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # Generate noise
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # Add noise
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _compute_shapley_weight(self, manager_id: str) -> float:
        """Compute Shapley value weight"""
        history = self._coalition_history[manager_id]['shapley_values']
        if not history:
            return 0.25  # Default Shapley weight
        
        # Use recent Shapley values
        recent_values = history[-3:] if len(history) >= 3 else history
        return np.mean(recent_values)
    
    def _update_coalition_history(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor], coalition_info: Optional[Dict]):
        """Update coalition history buffer"""
        history = self._coalition_history[manager_id]
        
        # Convert to numpy and add to history
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # Add Shapley values and fairness information
        if coalition_info:
            history['shapley_values'].append(coalition_info.get('shapley_value', 0.25))
            fairness_score = coalition_info.get('fairness_score', 1.0)
            history['fairness_scores'].append(fairness_score)
            # Update current fairness score
            self._fairness_scores[manager_id] = fairness_score
        else:
            # Default values
            history['shapley_values'].append(0.25)
            history['fairness_scores'].append(1.0)
        
        # Maintain history length
        for key in history:
            if len(history[key]) > self.coalition_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """Get adapted observation dimension information"""
        return {
            'private_dim': 40,  # 39 + 1(Shapley trend)
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  # 73
            'coalition_history_length': self.coalition_history_length,
            'fo_constraint_dim': self.fo_constraint_dim,
            'fairness_features': 2  # fairness_score + shapley_weight
        }
    
    def get_fosqddpg_specific_info(self) -> Dict[str, Any]:
        """Get FOSQDDPG specific adaptation information"""
        return {
            'shapley_mode': self.shapley_mode,
            'fairness_weight': self.fairness_weight,
            'credit_assignment_factor': self.credit_assignment_factor,
            'coalition_history_length': self.coalition_history_length,
            'fo_fairness_weight': self.fo_fairness_weight,
            'fo_shapley_integration': self.fo_shapley_integration,
            'fairness_scores': dict(self._fairness_scores)
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """Reset coalition history"""
        if manager_id is None:
            self._init_history_buffers()
        else:
            if manager_id in self._coalition_history:
                for key in self._coalition_history[manager_id]:
                    self._coalition_history[manager_id][key].clear()
                self._fairness_scores[manager_id] = 1.0
    
    def update_fairness_scores(self, fairness_updates: Dict[str, float]):
        """Update fairness scores"""
        for manager_id, score in fairness_updates.items():
            if manager_id in self._fairness_scores:
                # Smooth update of fairness score
                self._fairness_scores[manager_id] = (0.7 * self._fairness_scores[manager_id] + 
                                                   0.3 * score)
    
    def get_coalition_enhanced_observation(self, 
                                          manager_id: str, 
                                          current_obs: Dict[str, torch.Tensor],
                                          coalition_members: List[str],
                                          coalition_strength: float = 1.0) -> Dict[str, torch.Tensor]:
        """Get coalition enhanced observation (FOSQDDPG feature)"""
        enhanced_obs = current_obs.copy()
        
        # Coalition strength weighting
        coalition_factor = coalition_strength * self.credit_assignment_factor
        
        # Enhance private observation (coalition influence)
        enhanced_obs['private'] = enhanced_obs['private'] * (1 + coalition_factor * 0.1)
        
        # Enhance others observation (coalition member information)
        if len(coalition_members) > 1:
            coalition_size_factor = len(coalition_members) / 4.0  # Maximum 4 Managers
            enhanced_obs['others'] = enhanced_obs['others'] * (1 + coalition_size_factor * 0.05)
        
        # Add coalition specific information
        enhanced_obs['coalition_strength'] = torch.FloatTensor([coalition_strength]).to(self.device)
        enhanced_obs['coalition_size'] = torch.FloatTensor([len(coalition_members)]).to(self.device)
        
        return enhanced_obs 
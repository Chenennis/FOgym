#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOSqddpgDecPOMDPAdapter:
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimension
        self.private_dim = 39      # private observation base dimension
        self.public_dim = 18       # public observation dimension
        self.others_dim = 15       # others observation dimension
        self.total_obs_dim = 72    # total observation dimension
        
        # FOSQDDPG specific parameters
        self.shapley_mode = True
        self.fairness_weight = 0.3              # fairness weight
        self.credit_assignment_factor = 0.2     # credit assignment factor
        self.coalition_history_length = 5       # coalition history length
        
        # FlexOffer constraint integration (enhanced version)
        self.fo_constraint_dim = 36             # FlexOffer constraint dimension
        self.fo_fairness_weight = 0.25          # FlexOffer fairness weight
        self.fo_shapley_integration = True      # Shapley value integration switch
        
        # observation processing cache
        self._observation_cache = {}
        self._coalition_history = {}
        self._fairness_scores = {}
        
        # initialize history cache
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """initialize observation history buffer"""
        for manager_id in [f"manager_{i}" for i in range(4)]:
            self._coalition_history[manager_id] = {
                'private': [],
                'public': [],
                'others': [],
                'full_obs': [],
                'shapley_values': [],
                'fairness_scores': []
            }
            self._fairness_scores[manager_id] = 1.0  # initial fairness score
    
    def adapt_observation_for_fosqddpg(self, 
                                      observation: np.ndarray, 
                                      manager_id: str,
                                      fo_constraints: Optional[np.ndarray] = None,
                                      fo_satisfaction: Optional[float] = None,
                                      coalition_info: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        adapt observation space for FOSQDDPG
        
        Args:
            observation: original observation [obs_dim]
            manager_id: Manager ID (e.g., "manager_0")
            fo_constraints: FlexOffer constraint [constraint_dim]
            fo_satisfaction: FlexOffer satisfaction scalar
            coalition_info: coalition information dictionary
            
        Returns:
            adapted observation dictionary, including Shapley value information
        """
        # parse Dec-POMDP observation
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        # FlexOffer constraint integration (Shapley value aware)
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints_with_shapley(
                private_obs, fo_constraints, fo_satisfaction, manager_id
            )
        else:
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # FOSQDDPG specific observation enhancement
        enhanced_private = self._enhance_private_obs_for_fosqddpg(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_fosqddpg(public_obs, coalition_info)
        enhanced_others = self._enhance_others_obs_for_fosqddpg(others_obs, manager_id, coalition_info)
        
        # observation noise processing (fairness weight)
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private', manager_id=manager_id)
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public', manager_id=manager_id)
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others', manager_id=manager_id)
        
        # convert to tensor
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device),
            'fairness_score': torch.FloatTensor([self._fairness_scores[manager_id]]).to(self.device),
            'shapley_weight': torch.FloatTensor([self._compute_shapley_weight(manager_id)]).to(self.device)
        }
        
        # update history cache (including Shapley value information)
        self._update_coalition_history(manager_id, adapted_obs, coalition_info)
        
        return adapted_obs
    
    def _parse_dec_pomdp_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """parse Dec-POMDP observation structure"""
        if len(observation) < self.total_obs_dim:
            # pad insufficient dimensions
            observation = np.pad(observation, (0, self.total_obs_dim - len(observation)))
        elif len(observation) > self.total_obs_dim:
            # truncate excess dimensions
            observation = observation[:self.total_obs_dim]
        
        # hierarchical parsing
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
        """integrate FlexOffer constraint into private observation (Shapley value aware)"""
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
        
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40] 
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """extract FlexOffer constraint features (fairness aware)"""
        if len(fo_constraints) == 0:
            return np.zeros(5)  # default constraint features
        
        # statistical features
        constraint_features = np.array([
            np.mean(fo_constraints),     # average constraint value
            np.std(fo_constraints),      # constraint variance
            np.min(fo_constraints),      # minimum constraint
            np.max(fo_constraints),      # maximum constraint
            np.sum(fo_constraints > 0.5) / len(fo_constraints)  # activation ratio
        ])
        
        return constraint_features
    
    def _enhance_private_obs_for_fosqddpg(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """enhance private observation for FOSQDDPG (Shapley value integration)"""
        # get history observation for Shapley value calculation
        history = self._coalition_history[manager_id]['private']
        
        if len(history) > 0:
            # Shapley value weighted history information
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
        """enhance public observation for FOSQDDPG (coalition aware)"""
        # coalition information integration
        if coalition_info and self.shapley_mode:
            coalition_strength = coalition_info.get('coalition_strength', 1.0)
            coalition_fairness = coalition_info.get('fairness_index', 1.0)
            
            # coalition adjustment for public observation
            coalition_factor = 0.1 * coalition_strength * coalition_fairness
            enhanced_obs = public_obs * (1 + coalition_factor)
            return enhanced_obs
        
        return public_obs
    
    def _enhance_others_obs_for_fosqddpg(self, others_obs: np.ndarray, manager_id: str, coalition_info: Optional[Dict] = None) -> np.ndarray:
        """enhance others observation for FOSQDDPG (fairness weighted)"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        # fairness weight application
        fairness_weight = self.fairness_weight
        
        # if there is coalition information, adjust fairness weight
        if coalition_info:
            member_fairness = coalition_info.get('member_fairness', {})
            if manager_id in member_fairness:
                individual_fairness = member_fairness[manager_id]
                fairness_weight *= individual_fairness
        
        # get history others observation
        history = self._coalition_history[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.coalition_history_length:], axis=0) if len(history) >= self.coalition_history_length else np.mean(history, axis=0)
            # fairness weighted update
            fair_weighted_obs = fairness_weight * others_obs + (1 - fairness_weight) * recent_avg
            return fair_weighted_obs
        
        return others_obs * fairness_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default', manager_id: str = None) -> np.ndarray:
        """add observation noise (fairness adjustment)"""
        if not self.config.enable_observation_noise:
            return observation
        
        # FOSQDDPG specific noise setting (fairness adjustment)
        fairness_factor = self._fairness_scores.get(manager_id, 1.0) if manager_id else 1.0
        
        noise_scales = {
            'private': self.config.noise_level * 0.7 * fairness_factor,     # private observation noise (fairness adjustment)
            'public': self.config.noise_level * 0.4,                        # public observation noise (smaller)
            'others': self.config.noise_level * 1.1 * (2 - fairness_factor) # others observation noise (reverse fairness adjustment)
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # generate noise
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # add noise
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _compute_shapley_weight(self, manager_id: str) -> float:
        """calculate Shapley value weight"""
        history = self._coalition_history[manager_id]['shapley_values']
        if not history:
            return 0.25  # default Shapley weight
        
        # use recent Shapley values
        recent_values = history[-3:] if len(history) >= 3 else history
        return np.mean(recent_values)
    
    def _update_coalition_history(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor], coalition_info: Optional[Dict]):
        """update coalition history buffer"""
        history = self._coalition_history[manager_id]
        
        # convert to numpy and add to history
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # add Shapley value and fairness information
        if coalition_info:
            history['shapley_values'].append(coalition_info.get('shapley_value', 0.25))
            fairness_score = coalition_info.get('fairness_score', 1.0)
            history['fairness_scores'].append(fairness_score)
            # update current fairness score
            self._fairness_scores[manager_id] = fairness_score
        else:
            # default value
            history['shapley_values'].append(0.25)
            history['fairness_scores'].append(1.0)
        
        # maintain history length
        for key in history:
            if len(history[key]) > self.coalition_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """get adapted observation dimension information"""
        return {
            'private_dim': 40, 
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  
            'coalition_history_length': self.coalition_history_length,
            'fo_constraint_dim': self.fo_constraint_dim,
            'fairness_features': 2  # fairness_score + shapley_weight
        }
    
    def get_fosqddpg_specific_info(self) -> Dict[str, Any]:
        """get FOSQDDPG specific adaptation information"""
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
        """reset coalition history"""
        if manager_id is None:
            self._init_history_buffers()
        else:
            if manager_id in self._coalition_history:
                for key in self._coalition_history[manager_id]:
                    self._coalition_history[manager_id][key].clear()
                self._fairness_scores[manager_id] = 1.0
    
    def update_fairness_scores(self, fairness_updates: Dict[str, float]):
        """update fairness scores"""
        for manager_id, score in fairness_updates.items():
            if manager_id in self._fairness_scores:
                # smooth update fairness scores
                self._fairness_scores[manager_id] = (0.7 * self._fairness_scores[manager_id] + 
                                                   0.3 * score)
    
    def get_coalition_enhanced_observation(self, 
                                          manager_id: str, 
                                          current_obs: Dict[str, torch.Tensor],
                                          coalition_members: List[str],
                                          coalition_strength: float = 1.0) -> Dict[str, torch.Tensor]:
        """get coalition enhanced observation (FOSQDDPG specific feature)"""
        enhanced_obs = current_obs.copy()
        
        # coalition strength weighted
        coalition_factor = coalition_strength * self.credit_assignment_factor
        
        # enhance private observation (coalition influence)
        enhanced_obs['private'] = enhanced_obs['private'] * (1 + coalition_factor * 0.1)
        
        # enhance others observation (coalition member information)
        if len(coalition_members) > 1:
            coalition_size_factor = len(coalition_members) / 4.0  # maximum 4 managers
            enhanced_obs['others'] = enhanced_obs['others'] * (1 + coalition_size_factor * 0.05)
        
        # add coalition specific information
        enhanced_obs['coalition_strength'] = torch.FloatTensor([coalition_strength]).to(self.device)
        enhanced_obs['coalition_size'] = torch.FloatTensor([len(coalition_members)]).to(self.device)
        
        return enhanced_obs 
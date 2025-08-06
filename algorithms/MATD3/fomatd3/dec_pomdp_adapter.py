#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMAtd3DecPOMDPAdapter:

    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP observation space dimension
        self.private_dim = 39      # private observation base dimension
        self.public_dim = 18       # public observation dimension
        self.others_dim = 15       # others observation dimension
        self.total_obs_dim = 72    # total observation dimension
        
        # TD3 specific parameters
        self.twin_critic_mode = True
        self.target_smoothing_factor = 0.8  # target policy smoothing factor
        self.delay_update_steps = 2         # delay update steps
        self.observation_history_length = 3 # observation history length (TD3 optimization)
        
        # FlexOffer constraint integration
        self.fo_constraint_dim = 36        # FlexOffer constraint dimension
        self.fo_satisfaction_weight = 0.2  # FlexOffer satisfaction weight
        
        # observation processing cache
        self._observation_cache = {}
        self._history_buffer = {}
        
        # initialize observation history cache
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """initialize observation history buffer"""
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

        # parse Dec-POMDP observation
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        # FlexOffer constraint integration (always ensure 40 dimensions)
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints(private_obs, fo_constraints, fo_satisfaction)
        else:
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # TD3 specific observation enhancement
        enhanced_private = self._enhance_private_obs_for_td3(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_td3(public_obs)
        enhanced_others = self._enhance_others_obs_for_td3(others_obs, manager_id)
        
        # observation noise processing
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private')
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public')
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others')
        
        # convert to tensor
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device)
        }
        
        # update history cache
        self._update_history_buffer(manager_id, adapted_obs)
        
        return adapted_obs
    
    def _parse_dec_pomdp_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """parse Dec-POMDP observation structure"""
        if len(observation) < self.total_obs_dim:
            # fill missing dimensions
            observation = np.pad(observation, (0, self.total_obs_dim - len(observation)))
        elif len(observation) > self.total_obs_dim:
            # truncate extra dimensions
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
    
    def _integrate_fo_constraints(self, 
                                 private_obs: np.ndarray, 
                                 fo_constraints: np.ndarray,
                                 fo_satisfaction: Optional[float] = None) -> np.ndarray:
        """integrate FlexOffer constraints into private observation"""
        # FlexOffer constraint feature extraction
        constraint_features = self._extract_fo_constraint_features(fo_constraints)
        
        # FlexOffer satisfaction processing
        satisfaction_feature = fo_satisfaction if fo_satisfaction is not None else 0.8
        
        # add trend information (TD3 optimization: more focus on long-term trend)
        constraint_trend = np.mean(constraint_features) - np.mean(private_obs[:10])  # compare first 10 dimensions
        
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40]  # ensure dimension consistency
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """extract FlexOffer constraint features"""
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
    
    def _enhance_private_obs_for_td3(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """enhance private observation for TD3"""
        # get history observation for smoothing
        history = self._history_buffer[manager_id]['private']
        
        if len(history) > 0:
            # TD3 target policy smoothing: combine history information
            recent_history = history[-2:] if len(history) >= 2 else history
            if recent_history:
                avg_history = np.mean(recent_history, axis=0)
                # smooth current observation
                smoothed_obs = (self.target_smoothing_factor * private_obs + 
                               (1 - self.target_smoothing_factor) * avg_history)
                return smoothed_obs
        
        return private_obs
    
    def _enhance_public_obs_for_td3(self, public_obs: np.ndarray) -> np.ndarray:
        """enhance public observation for TD3"""
        # TD3 twin critic feature: increase observation robustness
        # add small amount of noise to improve generalization
        if self.twin_critic_mode:
            noise_scale = 0.02  
            noise = np.random.normal(0, noise_scale, public_obs.shape)
            robust_obs = public_obs + noise
            return robust_obs
        
        return public_obs
    
    def _enhance_others_obs_for_td3(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """enhance others observation for TD3"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        # TD3 delay update feature: more conservative handling of others information
        conservative_weight = 0.7  # more conservative than DDPG
        
        # get history others observation
        history = self._history_buffer[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.delay_update_steps:], axis=0) if len(history) >= self.delay_update_steps else np.mean(history, axis=0)
            # conservative update
            conservative_obs = conservative_weight * others_obs + (1 - conservative_weight) * recent_avg
            return conservative_obs
        
        return others_obs * conservative_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default') -> np.ndarray:
        """add observation noise"""
        if not self.config.enable_observation_noise:
            return observation
        
        # TD3 specific noise setting
        noise_scales = {
            'private': self.config.noise_level * 0.8,  # private observation noise is smaller
            'public': self.config.noise_level * 0.5,   # public observation noise is smaller
            'others': self.config.noise_level * 1.2    # others observation noise is larger
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # generate noise
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # add noise
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _update_history_buffer(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor]):
        """update observation history buffer"""
        history = self._history_buffer[manager_id]
        
        # convert to numpy and add to history
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # maintain history length
        for key in history:
            if len(history[key]) > self.observation_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """get adapted observation dimension information"""
        return {
            'private_dim': 40,  
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  
            'history_length': self.observation_history_length,
            'fo_constraint_dim': self.fo_constraint_dim
        }
    
    def get_td3_specific_info(self) -> Dict[str, Any]:
        """get TD3 specific adaptation information"""
        return {
            'twin_critic_mode': self.twin_critic_mode,
            'target_smoothing_factor': self.target_smoothing_factor,
            'delay_update_steps': self.delay_update_steps,
            'observation_history_length': self.observation_history_length,
            'conservative_weight': 0.7,
            'fo_integration': True
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """reset observation history"""
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
        """get smoothed observation (TD3 target policy smoothing)"""
        if smoothing_factor is None:
            smoothing_factor = self.target_smoothing_factor
        
        history = self._history_buffer[manager_id]
        
        if len(history['full_obs']) == 0:
            return current_obs
        
        # get recent observation
        recent_obs = history['full_obs'][-1]
        
        # smooth processing
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
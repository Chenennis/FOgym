"""
Dec-POMDP Observation Space Configuration File
Defines the observation space architecture for decentralized partially observable Markov decision processes
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DecPOMDPConfig:
    """Dec-POMDP Configuration Class"""
    
    # Observation noise configuration
    enable_observation_noise: bool = True  # Observation noise switch
    noise_level: float = 0.05  # Noise standard deviation (5% - slight noise)
    observation_noise_std: float = 0.05  # Observation noise standard deviation (compatible attribute)
    
    # Network quality configuration
    network_quality: str = "normal"  # Network quality level
    enable_dynamic_noise: bool = True  # Dynamic noise switch
    
    # Information sharing limitation configuration
    enable_other_manager_info: bool = True  # Whether other Manager information can be observed
    limited_other_info_features: Optional[List[str]] = None  # Limited features of other Manager information
    
    # Information transmission delay configuration
    enable_info_delay: bool = False  # Information delay switch
    max_delay_steps: int = 1  # Maximum delay steps
    
    # Information loss configuration
    enable_info_missing: bool = False  # Information loss switch
    missing_probability: float = 0.1  # Information loss probability
    
    def __post_init__(self):
        if self.limited_other_info_features is None:
            self.limited_other_info_features = [
                'user_count_ratio',      
                'device_count_ratio',    
                'energy_consumption_level',  
                'satisfaction_level',    
                'is_active',            
            ]

class DecPOMDPObservationSpace:
    """Dec-POMDP Observation Space Definition"""
    
    def __init__(self, config: Optional[DecPOMDPConfig] = None):
        self.config = config if config is not None else DecPOMDPConfig()
        
    def get_observation_definition(self) -> Dict[str, Any]:
        """
        Get observation space mathematical definition
        
        Returns:
            Dict containing definitions of various components of the observation space
        """
        return {
            'observation_space_formula': 'O_i = [O_private_i, O_public, O_limited_others_i]',
            'components': {
                'O_private_i': {
                    'description': 'Complete private information of Manager i (no noise)',
                    'includes': [
                        'self_device_states',     
                        'self_user_preferences',  
                        'self_manager_features',  
                        'self_markov_history',    
                    ],
                    'noise_level': 0.0,  # No noise for private information
                },
                'O_public': {
                    'description': 'Public environment information (no noise, visible to all Managers)',
                    'includes': [
                        'time_features',          # Time features (hour, weekday, etc.)
                        'price_features',         # Price information and trends
                        'weather_features',       # Weather information and trends
                        'market_basic_info',      # Basic market information (peak/valley periods, etc.)
                    ],
                    'noise_level': 0.0,  # No noise for public information
                },
                'O_limited_others_i': {
                    'description': 'Limited aggregated information of other Managers (configurable noise)',
                    'includes': self.config.limited_other_info_features,
                    'noise_level': self.config.noise_level if self.config.enable_observation_noise else 0.0,
                    'available': self.config.enable_other_manager_info,
                },
            },
            'total_dimension_formula': 'dim(O_i) = dim(O_private_i) + dim(O_public) + dim(O_limited_others_i)',
        }
    
    def compute_limited_other_manager_info(self, manager_info: Dict[str, List[float]], 
                                         current_manager_id: str) -> np.ndarray:
        """
        Compute limited aggregated information of other Managers
        
        Args:
            manager_info: Complete information of all Managers
            current_manager_id: ID of the current Manager
            
        Returns:
            Limited aggregated information vector of other Managers
        """
        if not self.config.enable_other_manager_info:
            return np.array([])
        
        limited_features = []
        
        # Calculate global statistics for normalization
        all_user_counts = [info[0] for info in manager_info.values()]  # User counts
        all_device_counts = [info[1] for info in manager_info.values()]  # Device counts
        all_energies = [info[3] for info in manager_info.values()]  # Cumulative energy consumption
        all_satisfactions = [info[4] for info in manager_info.values()]  # User satisfaction
        
        total_users = sum(all_user_counts)
        total_devices = sum(all_device_counts)
        max_energy = max(all_energies) if all_energies else 1.0
        avg_satisfaction = np.mean(all_satisfactions) if all_satisfactions else 0.5
        
        for other_id, other_info in manager_info.items():
            if other_id == current_manager_id:
                continue
                
            # Extract basic information of other Managers
            user_count = other_info[0]
            device_count = other_info[1]
            cumulative_cost = other_info[2]
            cumulative_energy = other_info[3]
            satisfaction = other_info[4]
            
            # Calculate limited aggregated features
            manager_limited_features = []
            
            # Check if limited feature list exists
            config_features = self.config.limited_other_info_features
            if config_features is not None:
                if 'user_count_ratio' in config_features:
                    user_ratio = user_count / max(1, total_users)
                    manager_limited_features.append(user_ratio)
                
                if 'device_count_ratio' in config_features:
                    device_ratio = device_count / max(1, total_devices)
                    manager_limited_features.append(device_ratio)
                
                if 'energy_consumption_level' in config_features:
                    energy_level = cumulative_energy / max(1, max_energy)
                    if energy_level < 0.33:
                        energy_level_discrete = 0.0  
                    elif energy_level < 0.67:
                        energy_level_discrete = 0.5  
                    else:
                        energy_level_discrete = 1.0  
                    manager_limited_features.append(energy_level_discrete)
                
                if 'satisfaction_level' in config_features:
                    if satisfaction < 0.33:
                        satisfaction_level = 0.0  
                    elif satisfaction < 0.67:
                        satisfaction_level = 0.5  
                    else:
                        satisfaction_level = 1.0  
                    manager_limited_features.append(satisfaction_level)
                
                if 'is_active' in config_features:
                    is_active = 1.0 if cumulative_energy > np.mean(all_energies) else 0.0
                    manager_limited_features.append(is_active)
            
            limited_features.extend(manager_limited_features)
        
        limited_features_array = np.array(limited_features, dtype=np.float32)
        
        # Apply observation noise (if enabled)
        if self.config.enable_observation_noise and self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, size=limited_features_array.shape)
            limited_features_array = limited_features_array + noise
            
            # Ensure feature values are within reasonable range
            limited_features_array = np.clip(limited_features_array, -2.0, 2.0)
        
        return limited_features_array
    
    def apply_information_delay(self, current_observation: np.ndarray, 
                              observation_history: List[np.ndarray]) -> np.ndarray:
        """
        Apply information transmission delay
        
        Args:
            current_observation: Current observation
            observation_history: Observation history
            
        Returns:
            Potentially delayed observation
        """
        if not self.config.enable_info_delay:
            return current_observation
            
        if len(observation_history) < self.config.max_delay_steps:
            return current_observation
            
        # Randomly select delay steps
        delay_steps = np.random.randint(0, self.config.max_delay_steps + 1)
        
        if delay_steps == 0:
            return current_observation
        else:
            # Return delayed observation
            delayed_idx = min(delay_steps, len(observation_history))
            return observation_history[-delayed_idx]
    
    def apply_information_missing(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply information loss
        
        Args:
            observation: Original observation
            
        Returns:
            Observation with potentially missing information
        """
        if not self.config.enable_info_missing:
            return observation
            
        # Randomly decide which features are missing
        missing_mask = np.random.random(observation.shape) < self.config.missing_probability
        
        # Set missing features to 0 or special value
        observation_with_missing = observation.copy()
        observation_with_missing[missing_mask] = 0.0
        
        return observation_with_missing

# Default configuration instance
DEFAULT_DEC_POMDP_CONFIG = DecPOMDPConfig()
DEFAULT_OBSERVATION_SPACE = DecPOMDPObservationSpace(DEFAULT_DEC_POMDP_CONFIG) 
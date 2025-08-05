import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DecPOMDPConfig:
    """Dec-POMDP configuration class"""
    
    # observation noise configuration
    enable_observation_noise: bool = True  # observation noise switch
    noise_level: float = 0.05  # noise standard deviation (5% - mild noise)
    observation_noise_std: float = 0.05  # observation noise standard deviation (compatible attribute)
    
    # network quality configuration
    network_quality: str = "normal"  # network quality level
    enable_dynamic_noise: bool = True  # dynamic noise switch
    
    # information sharing limit configuration
    enable_other_manager_info: bool = True  # whether can observe other Manager information
    limited_other_info_features: Optional[List[str]] = None  # limited other Manager information features
    
    # information delay configuration
    enable_info_delay: bool = False  # information delay switch
    max_delay_steps: int = 1  # maximum delay steps
    
    # information missing configuration
    enable_info_missing: bool = False  # information missing switch
    missing_probability: float = 0.1  # information missing probability
    
    def __post_init__(self):
        """post-initialization processing"""
        if self.limited_other_info_features is None:
            # default limited other Manager information: only provide aggregated metrics, not detailed states
            self.limited_other_info_features = [
                'user_count_ratio',      # user count ratio (rather than absolute number)
                'device_count_ratio',    # device count ratio (rather than absolute number)
                'energy_consumption_level',  # energy consumption level (low/medium/high, rather than exact value)
                'satisfaction_level',    # satisfaction level (low/medium/high, rather than exact value)
                'is_active',            # whether active (boolean value)
            ]

class DecPOMDPObservationSpace:
    """Dec-POMDP observation space definition"""
    
    def __init__(self, config: Optional[DecPOMDPConfig] = None):
        self.config = config if config is not None else DecPOMDPConfig()
        
    def get_observation_definition(self) -> Dict[str, Any]:
        """
        get observation space mathematical definition
        
        Returns:
            Dict containing definitions of each component of the observation space
        """
        return {
            'observation_space_formula': 'O_i = [O_private_i, O_public, O_limited_others_i]',
            'components': {
                'O_private_i': {
                    'description': 'Manager i\'s private complete information (no noise)',
                    'includes': [
                        'self_device_states',     # self all device states
                        'self_user_preferences',  # self user preference aggregation
                        'self_manager_features',  # self Manager features
                        'self_markov_history',    # self Markov history
                    ],
                    'noise_level': 0.0,  # private information no noise
                },
                'O_public': {
                    'description': 'public environment information (no noise, all Managers visible)',
                    'includes': [
                        'time_features',          # time features (hour, weekday, etc.)
                        'price_features',         # price information and trend
                        'weather_features',       # weather information and trend
                        'market_basic_info',      # basic market information (peak/valley periods, etc.)
                    ],
                    'noise_level': 0.0,  # public information no noise
                },
                'O_limited_others_i': {
                    'description': 'limited aggregated information of other Managers (configurable noise)',
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
        compute limited aggregated information of other Managers
        
        Args:
            manager_info: complete information of all Managers
            current_manager_id: ID of current Manager
            
        Returns:
            limited aggregated information vector of other Managers
        """
        if not self.config.enable_other_manager_info:
            return np.array([])
        
        limited_features = []
        
        # calculate global statistics for relativeization
        all_user_counts = [info[0] for info in manager_info.values()]  # user count
        all_device_counts = [info[1] for info in manager_info.values()]  # device count
        all_energies = [info[3] for info in manager_info.values()]  # cumulative energy consumption
        all_satisfactions = [info[4] for info in manager_info.values()]  # user satisfaction
        
        total_users = sum(all_user_counts)
        total_devices = sum(all_device_counts)
        max_energy = max(all_energies) if all_energies else 1.0
        avg_satisfaction = np.mean(all_satisfactions) if all_satisfactions else 0.5
        
        for other_id, other_info in manager_info.items():
            if other_id == current_manager_id:
                continue
                
            # extract basic information of other Managers
            user_count = other_info[0]
            device_count = other_info[1]
            cumulative_cost = other_info[2]
            cumulative_energy = other_info[3]
            satisfaction = other_info[4]
            
            # calculate limited aggregated features
            manager_limited_features = []
            
            # check if the limited feature list exists
            config_features = self.config.limited_other_info_features
            if config_features is not None:
                if 'user_count_ratio' in config_features:
                    # user count ratio (rather than absolute number)
                    user_ratio = user_count / max(1, total_users)
                    manager_limited_features.append(user_ratio)
                
                if 'device_count_ratio' in config_features:
                    # device count ratio (rather than absolute number)
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
                    # whether active (based on whether energy consumption is higher than average level)
                    is_active = 1.0 if cumulative_energy > np.mean(all_energies) else 0.0
                    manager_limited_features.append(is_active)
            
            limited_features.extend(manager_limited_features)
        
        # convert to numpy array
        limited_features_array = np.array(limited_features, dtype=np.float32)
        
        # apply observation noise (if enabled)
        if self.config.enable_observation_noise and self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, size=limited_features_array.shape)
            limited_features_array = limited_features_array + noise
            
            # ensure feature values are within reasonable range
            limited_features_array = np.clip(limited_features_array, -2.0, 2.0)
        
        return limited_features_array
    
    def apply_information_delay(self, current_observation: np.ndarray, 
                              observation_history: List[np.ndarray]) -> np.ndarray:

        if not self.config.enable_info_delay:
            return current_observation
            
        if len(observation_history) < self.config.max_delay_steps:
            return current_observation
            
        # randomly select delay steps
        delay_steps = np.random.randint(0, self.config.max_delay_steps + 1)
        
        if delay_steps == 0:
            return current_observation
        else:
            # return delayed observation
            delayed_idx = min(delay_steps, len(observation_history))
            return observation_history[-delayed_idx]
    
    def apply_information_missing(self, observation: np.ndarray) -> np.ndarray:

        if not self.config.enable_info_missing:
            return observation
            
        # randomly determine which features are missing
        missing_mask = np.random.random(observation.shape) < self.config.missing_probability
        
        # set missing features to 0 or special value
        observation_with_missing = observation.copy()
        observation_with_missing[missing_mask] = 0.0
        
        return observation_with_missing

# default configuration instance
DEFAULT_DEC_POMDP_CONFIG = DecPOMDPConfig()
DEFAULT_OBSERVATION_SPACE = DecPOMDPObservationSpace(DEFAULT_DEC_POMDP_CONFIG) 
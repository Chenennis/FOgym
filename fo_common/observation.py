"""Global observation space management"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import json
import os
import gymnasium as gym
from gymnasium import spaces

# Import related functionality
from fo_common.feature_extraction import (
    extract_generate_features,
    extract_aggregate_features,
    extract_trading_features,
    extract_schedule_features,
    compute_cross_module_correlations,
    compute_global_metrics
)
from fo_common.dim_reduction import FeatureProcessor
from fo_common.config import default_global_observation_config, get_observation_dimension

# Create logger
logger = logging.getLogger(__name__)

class GlobalObservationManager:
    """Global observation manager, integrating observation spaces across modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize global observation manager
        
        Args:
            config: Configuration dictionary specifying module observation weights and processing methods, uses default config if None
        """
        self.config = config or default_global_observation_config
        self.feature_extractors = {
            "generate": extract_generate_features,
            "aggregate": extract_aggregate_features,
            "trading": extract_trading_features,
            "schedule": extract_schedule_features
        }
        self.feature_processors = {}
        self.observation_cache = {}
        self.module_envs = {}
        self.observation_space = None
        
        # Initialize feature processors
        self._init_feature_processors()
        
        # Calculate observation space
        self._init_observation_space()
        
        logger.info(f"Global observation manager initialized, observation space dimension: {self.get_observation_dim()}")
        
    def _init_feature_processors(self) -> None:
        """Initialize feature processors"""
        for module, module_config in self.config.items():
            if module != "global" and module_config.get("enabled", True):
                dim_reduction_method = module_config.get("dim_reduction", "none")
                self.feature_processors[module] = FeatureProcessor(method=dim_reduction_method)
                
    def _init_observation_space(self) -> None:
        """Initialize observation space"""
        observation_dim = get_observation_dimension(self.config)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
        
    def register_module(self, module_name: str, env_instance: Optional[Any] = None, 
                       feature_extraction_fn: Optional[Callable] = None, 
                       weight: float = 1.0) -> None:
        """
        Register a module and its environment instance
        
        Args:
            module_name: Module name
            env_instance: Environment instance, can be gym.Env or other type
            feature_extraction_fn: Feature extraction function, uses predefined function if None
            weight: Module weight
        """
        if module_name not in self.config:
            logger.warning(f"Module {module_name} not defined in configuration, using default config")
            self.config[module_name] = {
                "enabled": True,
                "weight": weight,
                "features": [],
                "dim_reduction": "none"
            }
        else:
            self.config[module_name]["weight"] = weight
            
        self.module_envs[module_name] = env_instance
        
        if feature_extraction_fn is not None:
            self.feature_extractors[module_name] = feature_extraction_fn
            
        logger.info(f"Registered module {module_name}, weight: {weight}")
        
    def update_observation(self, module_name: str, observation: Union[np.ndarray, Dict[str, Any]]) -> None:
        """
        Update observation for a specific module
        
        Args:
            module_name: Module name
            observation: Observation data
        """
        if module_name not in self.config or not self.config[module_name].get("enabled", True):
            return
            
        self.observation_cache[module_name] = observation
        logger.debug(f"Updated observation for module {module_name}")
        
    def get_global_observation(self) -> np.ndarray:
        """
        Get global observation vector
        
        Returns:
            Global observation vector
        """
        if not self.observation_cache:
            logger.warning("Observation cache is empty, returning zero vector")
            return np.zeros(self.get_observation_dim(), dtype=np.float32)
            
        # Extract features
        features = {}
        for module_name, observation in self.observation_cache.items():
            if module_name in self.feature_extractors and self.config.get(module_name, {}).get("enabled", True):
                try:
                    extractor = self.feature_extractors[module_name]
                    module_config = self.config[module_name]
                    
                    # Extract features
                    module_features = extractor(observation, module_config)
                    
                    # Apply dimensionality reduction (if configured)
                    if module_name in self.feature_processors:
                        processor = self.feature_processors[module_name]
                        if not processor.is_fitted and len(module_features) > 0:
                            # First time fitting the processor
                            processor.fit(module_features)
                        
                        if processor.is_fitted:
                            module_features = processor.transform(module_features)
                            
                    features[module_name] = module_features
                except Exception as e:
                    logger.error(f"Error processing observation for module {module_name}: {e}")
        
        # Calculate cross-module correlations
        try:
            correlations = compute_cross_module_correlations(self.observation_cache, self.config)
        except Exception as e:
            logger.error(f"Error calculating cross-module correlations: {e}")
            # Default all correlations to 0.5
            correlations = np.array([0.5] * 6, dtype=np.float32)
        
        # Calculate global metrics
        try:
            global_metrics = compute_global_metrics(self.observation_cache, self.config)
        except Exception as e:
            logger.error(f"Error calculating global metrics: {e}")
            # Default all metrics to 0.7
            global_features_count = len(self.config.get("global", {}).get("features", []))
            global_metrics = np.array([0.7] * global_features_count, dtype=np.float32)
        
        # Combine all features
        all_features = []
        
        # Add features from each module (with configured weights)
        for module_name, module_config in self.config.items():
            if module_name != "global" and module_config.get("enabled", True):
                if module_name in features:
                    # Apply weight
                    weight = module_config.get("weight", 1.0)
                    weighted_features = features[module_name] * weight
                    all_features.append(weighted_features)
        
        # Add correlation features
        all_features.append(correlations)
        
        # Add global metrics
        if self.config.get("global", {}).get("enabled", True):
            all_features.append(global_metrics)
        
        # Merge all features
        if all_features:
            try:
                # First filter out empty arrays
                valid_features = [f for f in all_features if len(f) > 0]
                if valid_features:
                    global_observation = np.concatenate(valid_features)
                else:
                    global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
            except Exception as e:
                logger.error(f"Error merging features: {e}")
                global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
        else:
            global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
            
        # Ensure dimensions match
        expected_dim = self.get_observation_dim()
        if len(global_observation) != expected_dim:
            logger.warning(f"Global observation dimension mismatch, expected {expected_dim}, got {len(global_observation)}")
            if len(global_observation) < expected_dim:
                # Zero padding
                padded = np.zeros(expected_dim, dtype=np.float32)
                padded[:len(global_observation)] = global_observation
                global_observation = padded
            else:
                # Truncate
                global_observation = global_observation[:expected_dim]
            logger.info(f"Adjusted global observation dimension to {len(global_observation)}")
        
        # Ensure correct data type
        if not isinstance(global_observation, np.ndarray) or global_observation.dtype != np.float32:
            global_observation = np.array(global_observation, dtype=np.float32)
        
        return global_observation
    
    def get_observation_space(self) -> gym.Space:
        """Get observation space"""
        return self.observation_space
    
    def get_observation_dim(self) -> int:
        """Get observation space dimension"""
        return self.observation_space.shape[0]
    
    def reset(self) -> None:
        """Reset observation cache"""
        self.observation_cache = {}
        
    def save_config(self, path: str) -> None:
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def load_config(self, path: str) -> None:
        """Load configuration from file"""
        if not os.path.exists(path):
            logger.warning(f"Configuration file {path} does not exist, using default configuration")
            return
            
        try:
            with open(path, 'r') as f:
                self.config = json.load(f)
                
            # Re-initialize
            self._init_feature_processors()
            self._init_observation_space()
            
            logger.info(f"Successfully loaded configuration from file {path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
    def _extract_features(self, module_name: str, observation: np.ndarray) -> np.ndarray:
        """Extract key features from module observation"""
        if module_name not in self.feature_extractors:
            logger.warning(f"Module {module_name} does not have a corresponding feature extractor")
            return np.array([])
            
        try:
            extractor = self.feature_extractors[module_name]
            module_config = self.config.get(module_name, {})
            return extractor(observation, module_config)
        except Exception as e:
            logger.error(f"Error extracting features for module {module_name}: {e}")
            return np.array([])
            
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information"""
        info = {}
        
        for module_name, module_config in self.config.items():
            if module_name != "global" and module_config.get("enabled", True):
                feature_count = 0
                if module_name in self.feature_processors and self.feature_processors[module_name].is_fitted:
                    feature_count = self.feature_processors[module_name].get_output_dim()
                    
                info[module_name] = {
                    "enabled": module_config.get("enabled", True),
                    "weight": module_config.get("weight", 1.0),
                    "features": module_config.get("features", []),
                    "dim_reduction": module_config.get("dim_reduction", "none"),
                    "feature_count": feature_count
                }
                
        return info 
import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging

from .dec_pomdp_config import DecPOMDPConfig
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """algorithm basic configuration"""
    name: str
    type: str  # 'ppo', 'ddpg', 'td3', 'sqddpg'
    
    # network configuration
    state_dim: int = 73
    action_dim: int = 36
    hidden_dim: int = 256
    n_agents: int = 4
    
    # training configuration
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    max_action: float = 1.0
    
    # specific attributes
    stochastic: bool = True
    has_value_network: bool = True
    has_replay_buffer: bool = False
    has_twin_critic: bool = False
    has_shapley_computation: bool = False
    
    # device configuration
    device: str = "cpu"
    
    # Dec-POMDP configuration
    enable_dec_pomdp: bool = True
    private_dim: int = 40
    public_dim: int = 18
    others_dim: int = 15


@dataclass
class EnvironmentConfig:
    """environment configuration"""
    name: str = "FlexOffer-v1"
    
    # time configuration
    time_horizon: int = 24  # hours
    time_step: float = 0.25  # 15 minutes step
    total_steps: int = 96  # 24 * 4
    
    # device configuration
    n_users: int = 36
    n_devices: int = 118
    device_types: List[str] = field(default_factory=lambda: [
        'battery', 'heat_pump', 'ev', 'pv', 'dishwasher'
    ])
    
    # Manager configuration
    n_managers: int = 4
    manager_assignment: Dict[int, List[int]] = field(default_factory=lambda: {
        0: list(range(0, 9)),   # Manager 0: Users 0-8
        1: list(range(9, 18)),  # Manager 1: Users 9-17
        2: list(range(18, 27)), # Manager 2: Users 18-26
        3: list(range(27, 36))  # Manager 3: Users 27-35
    })
    
    # reward configuration
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'user_satisfaction': 0.4,
        'system_efficiency': 0.3,
        'cost_optimization': 0.2,
        'fairness': 0.1
    })


@dataclass
class TrainingConfig:
    """training configuration"""
    # basic training parameters
    total_episodes: int = 1000
    max_episode_steps: int = 96
    save_interval: int = 100
    eval_interval: int = 50
    
    # experience replay configuration
    buffer_capacity: int = 100000
    min_buffer_size: int = 1000
    
    # exploration configuration
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01
    
    # network update configuration
    policy_update_interval: int = 2  # TD3 feature
    target_update_freq: int = 1
    
    # early stopping configuration
    patience: int = 100
    min_improvement: float = 0.01


@dataclass
class LoggingConfig:
    """logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # file logging
    enable_file_logging: bool = True
    log_file: str = "flexoffer.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # console logging
    enable_console_logging: bool = True
    console_level: str = "INFO"
    
    # special logging
    enable_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    
    enable_wandb: bool = False
    wandb_project: str = "flexoffer"
    wandb_entity: str = ""


class ConfigManager:
    """unified configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._configs: Dict[str, Any] = {}
        self._load_default_configs()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def _load_default_configs(self):
        """load default configuration"""
        # Dec-POMDP configuration
        self._configs['dec_pomdp'] = DecPOMDPConfig()
        
        # basic configuration
        self._configs['base'] = Config()
        
        # algorithm configuration
        self._configs['algorithms'] = {
            'FOMAPPO': AlgorithmConfig(
                name='FOMAPPO',
                type='ppo',
                stochastic=True,
                has_value_network=True,
                has_replay_buffer=False
            ),
            'FOMADDPG': AlgorithmConfig(
                name='FOMADDPG',
                type='ddpg',
                stochastic=False,
                has_value_network=True,
                has_replay_buffer=True
            ),
            'FOMATD3': AlgorithmConfig(
                name='FOMATD3',
                type='td3',
                stochastic=False,
                has_value_network=True,
                has_replay_buffer=True,
                has_twin_critic=True
            ),
            'FOSQDDPG': AlgorithmConfig(
                name='FOSQDDPG',
                type='sqddpg',
                stochastic=False,
                has_value_network=True,
                has_replay_buffer=True,
                has_shapley_computation=True
            )
        }
        
        # environment configuration
        self._configs['environment'] = EnvironmentConfig()
        
        # training configuration
        self._configs['training'] = TrainingConfig()
        
        # logging configuration
        self._configs['logging'] = LoggingConfig()
    
    def get_config(self, config_name: str) -> Any:
        """get configuration"""
        if config_name not in self._configs:
            raise ValueError(f"configuration '{config_name}' not found")
        return self._configs[config_name]
    
    def get_algorithm_config(self, algorithm_name: str) -> AlgorithmConfig:
        """get algorithm configuration"""
        algorithms = self._configs.get('algorithms', {})
        if algorithm_name not in algorithms:
            raise ValueError(f"algorithm '{algorithm_name}' configuration not found")
        return algorithms[algorithm_name]
    
    def set_config(self, config_name: str, config: Any):
        """set configuration"""
        self._configs[config_name] = config
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """update configuration"""
        if config_name not in self._configs:
            raise ValueError(f"configuration '{config_name}' not found")
        
        config = self._configs[config_name]
        if hasattr(config, '__dict__'):
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"configuration '{config_name}' has no attribute '{key}'")
        else:
            # dictionary type configuration
            config.update(updates)
    
    def save_to_file(self, file_path: str, format: str = 'auto'):
        """save configuration to file"""
        if format == 'auto':
            format = 'yaml' if file_path.endswith('.yaml') or file_path.endswith('.yml') else 'json'
        
        # convert to serializable dictionary
        serializable_configs = {}
        for name, config in self._configs.items():
            if hasattr(config, '__dict__'):
                serializable_configs[name] = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
            else:
                serializable_configs[name] = config
        
        # add metadata
        serializable_configs['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'source': 'FlexOffer ConfigManager'
        }
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(serializable_configs, f, default_flow_style=False, indent=2)
                else:
                    json.dump(serializable_configs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"configuration saved to: {file_path}")
        except Exception as e:
            logger.error(f"failed to save configuration: {e}")
            raise
    
    def load_from_file(self, file_path: str):
        """load configuration from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    loaded_configs = yaml.safe_load(f)
                else:
                    loaded_configs = json.load(f)
            
            # skip metadata
            if '_metadata' in loaded_configs:
                del loaded_configs['_metadata']
            
            # update configuration
            for name, config_data in loaded_configs.items():
                if name in self._configs:
                    if isinstance(config_data, dict):
                        self.update_config(name, config_data)
                    else:
                        self._configs[name] = config_data
                else:
                    self._configs[name] = config_data
            
            logger.info(f"configuration loaded from: {file_path}")
        except Exception as e:
            logger.error(f"failed to load configuration: {e}")
            raise
    
    def validate_config(self, config_name: str) -> bool:
        """validate configuration"""
        if config_name not in self._configs:
            logger.error(f"configuration '{config_name}' not found")
            return False
        
        config = self._configs[config_name]
        
        try:
            if config_name == 'algorithms':
                return self._validate_algorithm_configs(config)
            elif config_name == 'environment':
                return self._validate_environment_config(config)
            elif config_name == 'training':
                return self._validate_training_config(config)
            else:
                # basic validation
                return config is not None
        except Exception as e:
            logger.error(f"error validating configuration '{config_name}': {e}")
            return False
    
    def _validate_algorithm_configs(self, algorithms: Dict[str, AlgorithmConfig]) -> bool:
        """validate algorithm configuration"""
        required_algorithms = ['FOMAPPO', 'FOMADDPG', 'FOMATD3', 'FOSQDDPG']
        
        for algo_name in required_algorithms:
            if algo_name not in algorithms:
                logger.error(f"missing algorithm configuration: {algo_name}")
                return False
            
            config = algorithms[algo_name]
            if config.state_dim <= 0 or config.action_dim <= 0:
                logger.error(f"invalid dimension configuration for algorithm '{algo_name}'")
                return False
        
        return True
    
    def _validate_environment_config(self, env_config: EnvironmentConfig) -> bool:
        """validate environment configuration"""
        if env_config.n_users <= 0 or env_config.n_devices <= 0:
            logger.error("invalid user or device count in environment configuration")
            return False
        
        if env_config.n_managers <= 0:
            logger.error("invalid manager count in environment configuration")
            return False
        
        return True
    
    def _validate_training_config(self, training_config: TrainingConfig) -> bool:
        """validate training configuration"""
        if training_config.total_episodes <= 0:
            logger.error("invalid training episode count in training configuration")
            return False
        
        if training_config.buffer_capacity <= training_config.min_buffer_size:
            logger.error("invalid buffer size in training configuration")
            return False
        
        return True
    
    def print_config_summary(self):
        """print configuration summary"""
        print("🔧 FlexOffer configuration summary")
        print("=" * 50)
        
        for name, config in self._configs.items():
            print(f"\n📋 {name.upper()} configuration:")
            
            if name == 'algorithms':
                for algo_name, algo_config in config.items():
                    print(f"   🤖 {algo_name}: {algo_config.type} ({'stochastic' if algo_config.stochastic else 'deterministic'})")
            elif hasattr(config, '__dict__'):
                important_attrs = self._get_important_attributes(name)
                for attr in important_attrs:
                    if hasattr(config, attr):
                        value = getattr(config, attr)
                        print(f"   📊 {attr}: {value}")
            else:
                print(f"   📊 type: {type(config).__name__}")
        
        print("=" * 50)
    
    def _get_important_attributes(self, config_name: str) -> List[str]:
        """get important configuration attributes"""
        important_attrs = {
            'environment': ['n_users', 'n_devices', 'n_managers', 'time_horizon'],
            'training': ['total_episodes', 'batch_size', 'buffer_capacity'],
            'logging': ['level', 'enable_file_logging', 'enable_tensorboard'],
            'dec_pomdp': ['observation_noise_std', 'network_quality', 'enable_dynamic_noise']
        }
        return important_attrs.get(config_name, [])
    
    def export_config_template(self, file_path: str):
        """export configuration template"""
        template = {
            "algorithms": {
                "FOMAPPO": {
                    "lr_actor": 1e-4,
                    "lr_critic": 1e-3,
                    "hidden_dim": 256,
                    "batch_size": 64
                }
            },
            "environment": {
                "time_horizon": 24,
                "n_users": 36,
                "n_devices": 118
            },
            "training": {
                "total_episodes": 1000,
                "buffer_capacity": 100000
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"configuration template exported to: {file_path}")


# global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """get global configuration manager"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def get_algorithm_config(algorithm_name: str) -> AlgorithmConfig:
    """get algorithm configuration"""
    return get_config_manager().get_algorithm_config(algorithm_name)


def get_dec_pomdp_config() -> DecPOMDPConfig:
    """get Dec-POMDP configuration"""
    return get_config_manager().get_config('dec_pomdp')


def get_environment_config() -> EnvironmentConfig:
    """get environment configuration"""
    return get_config_manager().get_config('environment')


def get_training_config() -> TrainingConfig:
    """get training configuration"""
    return get_config_manager().get_config('training') 
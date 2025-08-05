"""
FlexOffer统一配置管理系统

本模块提供统一的配置管理接口，整合所有配置相关功能，
包括Dec-POMDP配置、算法配置、设备配置等。
"""

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
    """算法基础配置"""
    name: str
    type: str  # 'ppo', 'ddpg', 'td3', 'sqddpg'
    
    # 网络配置
    state_dim: int = 73
    action_dim: int = 36
    hidden_dim: int = 256
    n_agents: int = 4
    
    # 训练配置
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    max_action: float = 1.0
    
    # 特定属性
    stochastic: bool = True
    has_value_network: bool = True
    has_replay_buffer: bool = False
    has_twin_critic: bool = False
    has_shapley_computation: bool = False
    
    # 设备配置
    device: str = "cpu"
    
    # Dec-POMDP配置
    enable_dec_pomdp: bool = True
    private_dim: int = 40
    public_dim: int = 18
    others_dim: int = 15


@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str = "FlexOffer-v1"
    
    # 时间配置
    time_horizon: int = 24  # 小时
    time_step: float = 0.25  # 15分钟步长
    total_steps: int = 96  # 24 * 4
    
    # 设备配置
    n_users: int = 36
    n_devices: int = 118
    device_types: List[str] = field(default_factory=lambda: [
        'battery', 'heat_pump', 'ev', 'pv', 'dishwasher'
    ])
    
    # Manager配置
    n_managers: int = 4
    manager_assignment: Dict[int, List[int]] = field(default_factory=lambda: {
        0: list(range(0, 9)),   # Manager 0: Users 0-8
        1: list(range(9, 18)),  # Manager 1: Users 9-17
        2: list(range(18, 27)), # Manager 2: Users 18-26
        3: list(range(27, 36))  # Manager 3: Users 27-35
    })
    
    # 奖励配置
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'user_satisfaction': 0.4,
        'system_efficiency': 0.3,
        'cost_optimization': 0.2,
        'fairness': 0.1
    })


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    total_episodes: int = 1000
    max_episode_steps: int = 96
    save_interval: int = 100
    eval_interval: int = 50
    
    # 经验重放配置
    buffer_capacity: int = 100000
    min_buffer_size: int = 1000
    
    # 探索配置
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01
    
    # 网络更新配置
    policy_update_interval: int = 2  # TD3特性
    target_update_freq: int = 1
    
    # 早停配置
    patience: int = 100
    min_improvement: float = 0.01


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件日志
    enable_file_logging: bool = True
    log_file: str = "flexoffer.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # 控制台日志
    enable_console_logging: bool = True
    console_level: str = "INFO"
    
    # 特殊日志
    enable_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    
    enable_wandb: bool = False
    wandb_project: str = "flexoffer"
    wandb_entity: str = ""


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._configs: Dict[str, Any] = {}
        self._load_default_configs()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def _load_default_configs(self):
        """加载默认配置"""
        # Dec-POMDP配置
        self._configs['dec_pomdp'] = DecPOMDPConfig()
        
        # 基础配置
        self._configs['base'] = Config()
        
        # 算法配置
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
        
        # 环境配置
        self._configs['environment'] = EnvironmentConfig()
        
        # 训练配置
        self._configs['training'] = TrainingConfig()
        
        # 日志配置
        self._configs['logging'] = LoggingConfig()
    
    def get_config(self, config_name: str) -> Any:
        """获取配置"""
        if config_name not in self._configs:
            raise ValueError(f"配置'{config_name}'不存在")
        return self._configs[config_name]
    
    def get_algorithm_config(self, algorithm_name: str) -> AlgorithmConfig:
        """获取算法配置"""
        algorithms = self._configs.get('algorithms', {})
        if algorithm_name not in algorithms:
            raise ValueError(f"算法'{algorithm_name}'配置不存在")
        return algorithms[algorithm_name]
    
    def set_config(self, config_name: str, config: Any):
        """设置配置"""
        self._configs[config_name] = config
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """更新配置"""
        if config_name not in self._configs:
            raise ValueError(f"配置'{config_name}'不存在")
        
        config = self._configs[config_name]
        if hasattr(config, '__dict__'):
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"配置'{config_name}'没有属性'{key}'")
        else:
            # 字典类型配置
            config.update(updates)
    
    def save_to_file(self, file_path: str, format: str = 'auto'):
        """保存配置到文件"""
        if format == 'auto':
            format = 'yaml' if file_path.endswith('.yaml') or file_path.endswith('.yml') else 'json'
        
        # 转换为可序列化的字典
        serializable_configs = {}
        for name, config in self._configs.items():
            if hasattr(config, '__dict__'):
                serializable_configs[name] = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
            else:
                serializable_configs[name] = config
        
        # 添加元数据
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
            
            logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def load_from_file(self, file_path: str):
        """从文件加载配置"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    loaded_configs = yaml.safe_load(f)
                else:
                    loaded_configs = json.load(f)
            
            # 跳过元数据
            if '_metadata' in loaded_configs:
                del loaded_configs['_metadata']
            
            # 更新配置
            for name, config_data in loaded_configs.items():
                if name in self._configs:
                    if isinstance(config_data, dict):
                        self.update_config(name, config_data)
                    else:
                        self._configs[name] = config_data
                else:
                    self._configs[name] = config_data
            
            logger.info(f"配置已从文件加载: {file_path}")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def validate_config(self, config_name: str) -> bool:
        """验证配置"""
        if config_name not in self._configs:
            logger.error(f"配置'{config_name}'不存在")
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
                # 基础验证
                return config is not None
        except Exception as e:
            logger.error(f"验证配置'{config_name}'时出错: {e}")
            return False
    
    def _validate_algorithm_configs(self, algorithms: Dict[str, AlgorithmConfig]) -> bool:
        """验证算法配置"""
        required_algorithms = ['FOMAPPO', 'FOMADDPG', 'FOMATD3', 'FOSQDDPG']
        
        for algo_name in required_algorithms:
            if algo_name not in algorithms:
                logger.error(f"缺少算法配置: {algo_name}")
                return False
            
            config = algorithms[algo_name]
            if config.state_dim <= 0 or config.action_dim <= 0:
                logger.error(f"算法'{algo_name}'维度配置无效")
                return False
        
        return True
    
    def _validate_environment_config(self, env_config: EnvironmentConfig) -> bool:
        """验证环境配置"""
        if env_config.n_users <= 0 or env_config.n_devices <= 0:
            logger.error("环境配置中用户数或设备数无效")
            return False
        
        if env_config.n_managers <= 0:
            logger.error("Manager数量配置无效")
            return False
        
        return True
    
    def _validate_training_config(self, training_config: TrainingConfig) -> bool:
        """验证训练配置"""
        if training_config.total_episodes <= 0:
            logger.error("训练轮数配置无效")
            return False
        
        if training_config.buffer_capacity <= training_config.min_buffer_size:
            logger.error("缓冲区配置无效")
            return False
        
        return True
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("🔧 FlexOffer配置摘要")
        print("=" * 50)
        
        for name, config in self._configs.items():
            print(f"\n📋 {name.upper()}配置:")
            
            if name == 'algorithms':
                for algo_name, algo_config in config.items():
                    print(f"   🤖 {algo_name}: {algo_config.type} ({'随机' if algo_config.stochastic else '确定性'})")
            elif hasattr(config, '__dict__'):
                important_attrs = self._get_important_attributes(name)
                for attr in important_attrs:
                    if hasattr(config, attr):
                        value = getattr(config, attr)
                        print(f"   📊 {attr}: {value}")
            else:
                print(f"   📊 类型: {type(config).__name__}")
        
        print("=" * 50)
    
    def _get_important_attributes(self, config_name: str) -> List[str]:
        """获取重要配置属性"""
        important_attrs = {
            'environment': ['n_users', 'n_devices', 'n_managers', 'time_horizon'],
            'training': ['total_episodes', 'batch_size', 'buffer_capacity'],
            'logging': ['level', 'enable_file_logging', 'enable_tensorboard'],
            'dec_pomdp': ['observation_noise_std', 'network_quality', 'enable_dynamic_noise']
        }
        return important_attrs.get(config_name, [])
    
    def export_config_template(self, file_path: str):
        """导出配置模板"""
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
        
        print(f"配置模板已导出到: {file_path}")


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def get_algorithm_config(algorithm_name: str) -> AlgorithmConfig:
    """快捷方式：获取算法配置"""
    return get_config_manager().get_algorithm_config(algorithm_name)


def get_dec_pomdp_config() -> DecPOMDPConfig:
    """快捷方式：获取Dec-POMDP配置"""
    return get_config_manager().get_config('dec_pomdp')


def get_environment_config() -> EnvironmentConfig:
    """快捷方式：获取环境配置"""
    return get_config_manager().get_config('environment')


def get_training_config() -> TrainingConfig:
    """快捷方式：获取训练配置"""
    return get_config_manager().get_config('training') 
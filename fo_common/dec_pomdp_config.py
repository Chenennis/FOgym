"""
Dec-POMDP观测空间配置文件
定义分布式部分可观测马尔可夫决策过程的观测空间架构
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DecPOMDPConfig:
    """Dec-POMDP配置类"""
    
    # 观测噪声配置
    enable_observation_noise: bool = True  # 观测噪声开关
    noise_level: float = 0.05  # 噪声标准差 (5% - 轻微噪声)
    observation_noise_std: float = 0.05  # 观测噪声标准差（兼容属性）
    
    # 网络质量配置
    network_quality: str = "normal"  # 网络质量级别
    enable_dynamic_noise: bool = True  # 动态噪声开关
    
    # 信息共享限制配置
    enable_other_manager_info: bool = True  # 是否能观测其他Manager信息
    limited_other_info_features: Optional[List[str]] = None  # 限制的其他Manager信息特征
    
    # 信息传递延迟配置
    enable_info_delay: bool = False  # 信息延迟开关
    max_delay_steps: int = 1  # 最大延迟步数
    
    # 信息缺失配置
    enable_info_missing: bool = False  # 信息缺失开关
    missing_probability: float = 0.1  # 信息缺失概率
    
    def __post_init__(self):
        """初始化后处理"""
        if self.limited_other_info_features is None:
            # 默认限制的其他Manager信息：只提供聚合指标，不提供详细状态
            self.limited_other_info_features = [
                'user_count_ratio',      # 用户数量占比（而非绝对数量）
                'device_count_ratio',    # 设备数量占比（而非绝对数量）
                'energy_consumption_level',  # 能耗水平（低/中/高，而非精确值）
                'satisfaction_level',    # 满意度水平（低/中/高，而非精确值）
                'is_active',            # 是否活跃（布尔值）
            ]

class DecPOMDPObservationSpace:
    """Dec-POMDP观测空间定义"""
    
    def __init__(self, config: Optional[DecPOMDPConfig] = None):
        self.config = config if config is not None else DecPOMDPConfig()
        
    def get_observation_definition(self) -> Dict[str, Any]:
        """
        获取观测空间数学定义
        
        Returns:
            Dict包含观测空间的各个组成部分的定义
        """
        return {
            'observation_space_formula': 'O_i = [O_private_i, O_public, O_limited_others_i]',
            'components': {
                'O_private_i': {
                    'description': 'Manager i的私有完整信息（无噪声）',
                    'includes': [
                        'self_device_states',     # 自身所有设备状态
                        'self_user_preferences',  # 自身用户偏好聚合
                        'self_manager_features',  # 自身Manager特征
                        'self_markov_history',    # 自身马尔可夫历史
                    ],
                    'noise_level': 0.0,  # 私有信息无噪声
                },
                'O_public': {
                    'description': '公共环境信息（无噪声，所有Manager可见）',
                    'includes': [
                        'time_features',          # 时间特征（小时、工作日等）
                        'price_features',         # 电价信息和趋势
                        'weather_features',       # 天气信息和趋势
                        'market_basic_info',      # 基础市场信息（峰谷时段等）
                    ],
                    'noise_level': 0.0,  # 公共信息无噪声
                },
                'O_limited_others_i': {
                    'description': '其他Manager的有限聚合信息（可配置噪声）',
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
        计算其他Manager的有限聚合信息
        
        Args:
            manager_info: 所有Manager的完整信息
            current_manager_id: 当前Manager的ID
            
        Returns:
            有限的其他Manager聚合信息向量
        """
        if not self.config.enable_other_manager_info:
            return np.array([])
        
        limited_features = []
        
        # 计算全局统计用于相对化
        all_user_counts = [info[0] for info in manager_info.values()]  # 用户数量
        all_device_counts = [info[1] for info in manager_info.values()]  # 设备数量
        all_energies = [info[3] for info in manager_info.values()]  # 累计能耗
        all_satisfactions = [info[4] for info in manager_info.values()]  # 用户满意度
        
        total_users = sum(all_user_counts)
        total_devices = sum(all_device_counts)
        max_energy = max(all_energies) if all_energies else 1.0
        avg_satisfaction = np.mean(all_satisfactions) if all_satisfactions else 0.5
        
        for other_id, other_info in manager_info.items():
            if other_id == current_manager_id:
                continue
                
            # 提取其他Manager的基础信息
            user_count = other_info[0]
            device_count = other_info[1]
            cumulative_cost = other_info[2]
            cumulative_energy = other_info[3]
            satisfaction = other_info[4]
            
            # 计算有限的聚合特征
            manager_limited_features = []
            
            # 检查限制特征列表是否存在
            config_features = self.config.limited_other_info_features
            if config_features is not None:
                if 'user_count_ratio' in config_features:
                    # 用户数量占比（而非绝对数量）
                    user_ratio = user_count / max(1, total_users)
                    manager_limited_features.append(user_ratio)
                
                if 'device_count_ratio' in config_features:
                    # 设备数量占比（而非绝对数量）
                    device_ratio = device_count / max(1, total_devices)
                    manager_limited_features.append(device_ratio)
                
                if 'energy_consumption_level' in config_features:
                    # 能耗水平（低/中/高，而非精确值）
                    energy_level = cumulative_energy / max(1, max_energy)
                    if energy_level < 0.33:
                        energy_level_discrete = 0.0  # 低
                    elif energy_level < 0.67:
                        energy_level_discrete = 0.5  # 中
                    else:
                        energy_level_discrete = 1.0  # 高
                    manager_limited_features.append(energy_level_discrete)
                
                if 'satisfaction_level' in config_features:
                    # 满意度水平（低/中/高，而非精确值）
                    if satisfaction < 0.33:
                        satisfaction_level = 0.0  # 低
                    elif satisfaction < 0.67:
                        satisfaction_level = 0.5  # 中
                    else:
                        satisfaction_level = 1.0  # 高
                    manager_limited_features.append(satisfaction_level)
                
                if 'is_active' in config_features:
                    # 是否活跃（基于能耗是否高于平均水平）
                    is_active = 1.0 if cumulative_energy > np.mean(all_energies) else 0.0
                    manager_limited_features.append(is_active)
            
            limited_features.extend(manager_limited_features)
        
        # 转换为numpy数组
        limited_features_array = np.array(limited_features, dtype=np.float32)
        
        # 应用观测噪声（如果启用）
        if self.config.enable_observation_noise and self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, size=limited_features_array.shape)
            limited_features_array = limited_features_array + noise
            
            # 确保特征值在合理范围内
            limited_features_array = np.clip(limited_features_array, -2.0, 2.0)
        
        return limited_features_array
    
    def apply_information_delay(self, current_observation: np.ndarray, 
                              observation_history: List[np.ndarray]) -> np.ndarray:
        """
        应用信息传递延迟
        
        Args:
            current_observation: 当前观测
            observation_history: 观测历史
            
        Returns:
            可能延迟的观测
        """
        if not self.config.enable_info_delay:
            return current_observation
            
        if len(observation_history) < self.config.max_delay_steps:
            return current_observation
            
        # 随机选择延迟步数
        delay_steps = np.random.randint(0, self.config.max_delay_steps + 1)
        
        if delay_steps == 0:
            return current_observation
        else:
            # 返回延迟的观测
            delayed_idx = min(delay_steps, len(observation_history))
            return observation_history[-delayed_idx]
    
    def apply_information_missing(self, observation: np.ndarray) -> np.ndarray:
        """
        应用信息缺失
        
        Args:
            observation: 原始观测
            
        Returns:
            可能缺失部分信息的观测
        """
        if not self.config.enable_info_missing:
            return observation
            
        # 随机决定哪些特征缺失
        missing_mask = np.random.random(observation.shape) < self.config.missing_probability
        
        # 将缺失的特征设为0或特殊值
        observation_with_missing = observation.copy()
        observation_with_missing[missing_mask] = 0.0
        
        return observation_with_missing

# 默认配置实例
DEFAULT_DEC_POMDP_CONFIG = DecPOMDPConfig()
DEFAULT_OBSERVATION_SPACE = DecPOMDPObservationSpace(DEFAULT_DEC_POMDP_CONFIG) 
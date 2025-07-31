"""全局观测空间管理"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import json
import os
import gymnasium as gym
from gymnasium import spaces

# 导入相关功能
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

# 创建日志记录器
logger = logging.getLogger(__name__)

class GlobalObservationManager:
    """全局观测管理器，整合各模块观测空间"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化全局观测管理器
        
        Args:
            config: 配置字典，指定各模块观测权重和处理方式，如果为None则使用默认配置
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
        
        # 初始化特征处理器
        self._init_feature_processors()
        
        # 计算观测空间
        self._init_observation_space()
        
        logger.info(f"全局观测管理器初始化完成，观测空间维度: {self.get_observation_dim()}")
        
    def _init_feature_processors(self) -> None:
        """初始化特征处理器"""
        for module, module_config in self.config.items():
            if module != "global" and module_config.get("enabled", True):
                dim_reduction_method = module_config.get("dim_reduction", "none")
                self.feature_processors[module] = FeatureProcessor(method=dim_reduction_method)
                
    def _init_observation_space(self) -> None:
        """初始化观测空间"""
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
        注册一个模块及其环境实例
        
        Args:
            module_name: 模块名称
            env_instance: 环境实例，可以是gym.Env或其他类型
            feature_extraction_fn: 特征提取函数，如果None则使用预定义函数
            weight: 模块权重
        """
        if module_name not in self.config:
            logger.warning(f"模块 {module_name} 未在配置中定义，将使用默认配置")
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
            
        logger.info(f"注册模块 {module_name}，权重: {weight}")
        
    def update_observation(self, module_name: str, observation: Union[np.ndarray, Dict[str, Any]]) -> None:
        """
        更新特定模块的观测
        
        Args:
            module_name: 模块名称
            observation: 观测数据
        """
        if module_name not in self.config or not self.config[module_name].get("enabled", True):
            return
            
        self.observation_cache[module_name] = observation
        logger.debug(f"更新模块 {module_name} 的观测")
        
    def get_global_observation(self) -> np.ndarray:
        """
        获取全局观测向量
        
        Returns:
            全局观测向量
        """
        if not self.observation_cache:
            logger.warning("观测缓存为空，返回全零向量")
            return np.zeros(self.get_observation_dim(), dtype=np.float32)
            
        # 提取特征
        features = {}
        for module_name, observation in self.observation_cache.items():
            if module_name in self.feature_extractors and self.config.get(module_name, {}).get("enabled", True):
                try:
                    extractor = self.feature_extractors[module_name]
                    module_config = self.config[module_name]
                    
                    # 提取特征
                    module_features = extractor(observation, module_config)
                    
                    # 应用降维（如果已配置）
                    if module_name in self.feature_processors:
                        processor = self.feature_processors[module_name]
                        if not processor.is_fitted and len(module_features) > 0:
                            # 首次拟合处理器
                            processor.fit(module_features)
                        
                        if processor.is_fitted:
                            module_features = processor.transform(module_features)
                            
                    features[module_name] = module_features
                except Exception as e:
                    logger.error(f"处理模块 {module_name} 的观测时出错: {e}")
        
        # 计算模块间相关性
        try:
            correlations = compute_cross_module_correlations(self.observation_cache, self.config)
        except Exception as e:
            logger.error(f"计算模块间相关性时出错: {e}")
            # 默认所有相关性为0.5
            correlations = np.array([0.5] * 6, dtype=np.float32)
        
        # 计算全局指标
        try:
            global_metrics = compute_global_metrics(self.observation_cache, self.config)
        except Exception as e:
            logger.error(f"计算全局指标时出错: {e}")
            # 默认所有指标为0.7
            global_features_count = len(self.config.get("global", {}).get("features", []))
            global_metrics = np.array([0.7] * global_features_count, dtype=np.float32)
        
        # 组合所有特征
        all_features = []
        
        # 添加各模块特征（按配置权重）
        for module_name, module_config in self.config.items():
            if module_name != "global" and module_config.get("enabled", True):
                if module_name in features:
                    # 应用权重
                    weight = module_config.get("weight", 1.0)
                    weighted_features = features[module_name] * weight
                    all_features.append(weighted_features)
        
        # 添加相关性特征
        all_features.append(correlations)
        
        # 添加全局指标
        if self.config.get("global", {}).get("enabled", True):
            all_features.append(global_metrics)
        
        # 合并所有特征
        if all_features:
            try:
                # 先过滤掉空数组
                valid_features = [f for f in all_features if len(f) > 0]
                if valid_features:
                    global_observation = np.concatenate(valid_features)
                else:
                    global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
            except Exception as e:
                logger.error(f"合并特征时出错: {e}")
                global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
        else:
            global_observation = np.zeros(self.get_observation_dim(), dtype=np.float32)
            
        # 确保维度匹配
        expected_dim = self.get_observation_dim()
        if len(global_observation) != expected_dim:
            logger.warning(f"全局观测维度不匹配，期望 {expected_dim}，实际 {len(global_observation)}")
            if len(global_observation) < expected_dim:
                # 零填充
                padded = np.zeros(expected_dim, dtype=np.float32)
                padded[:len(global_observation)] = global_observation
                global_observation = padded
            else:
                # 截断
                global_observation = global_observation[:expected_dim]
            logger.info(f"已调整全局观测维度为 {len(global_observation)}")
        
        # 确保数据类型正确
        if not isinstance(global_observation, np.ndarray) or global_observation.dtype != np.float32:
            global_observation = np.array(global_observation, dtype=np.float32)
        
        return global_observation
    
    def get_observation_space(self) -> gym.Space:
        """获取观测空间"""
        return self.observation_space
    
    def get_observation_dim(self) -> int:
        """获取观测空间维度"""
        return self.observation_space.shape[0]
    
    def reset(self) -> None:
        """重置观测缓存"""
        self.observation_cache = {}
        
    def save_config(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def load_config(self, path: str) -> None:
        """从文件加载配置"""
        if not os.path.exists(path):
            logger.warning(f"配置文件 {path} 不存在，使用默认配置")
            return
            
        try:
            with open(path, 'r') as f:
                self.config = json.load(f)
                
            # 重新初始化
            self._init_feature_processors()
            self._init_observation_space()
            
            logger.info(f"从文件 {path} 加载配置成功")
        except Exception as e:
            logger.error(f"加载配置时出错: {e}")
            
    def _extract_features(self, module_name: str, observation: np.ndarray) -> np.ndarray:
        """从模块观测中提取关键特征"""
        if module_name not in self.feature_extractors:
            logger.warning(f"模块 {module_name} 没有对应的特征提取器")
            return np.array([])
            
        try:
            extractor = self.feature_extractors[module_name]
            module_config = self.config.get(module_name, {})
            return extractor(observation, module_config)
        except Exception as e:
            logger.error(f"提取 {module_name} 模块特征时出错: {e}")
            return np.array([])
            
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
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
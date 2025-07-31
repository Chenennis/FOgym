"""
FlexOffer ModelBased Pipeline

提供基于物理模型的FlexOffer生成、聚合、交易和分解流程，不使用强化学习概念。
"""

from .config import PipelineConfig, ModelBasedConfig, load_config
from .model_based_controller import ModelBasedController, DeviceModel, BatteryModel, HeatPumpModel
from .model_based_pipeline import ModelBasedPipeline, run_pipeline, FlexOffer, Manager

__all__ = [
    'PipelineConfig', 
    'ModelBasedConfig',
    'load_config',
    'ModelBasedController',
    'DeviceModel',
    'BatteryModel',
    'HeatPumpModel',
    'ModelBasedPipeline',
    'run_pipeline',
    'FlexOffer',
    'Manager'
]

__version__ = '0.1.0'
__author__ = 'Your Name' 
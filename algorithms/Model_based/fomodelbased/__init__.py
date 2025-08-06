"""
FlexOffer ModelBased Pipeline

provide a model-based FlexOffer generation, aggregation, trading and decomposition pipeline, without using reinforcement learning concepts.
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
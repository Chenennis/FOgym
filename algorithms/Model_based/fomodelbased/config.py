from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
import json

@dataclass
class ModelBasedConfig:
    """FOModelBased configuration"""
    time_horizon: int = 24  # default 24 hours
    time_step: int = 1      # time step (hour)
    optimization_type: str = "battery_type_0.55"  # optimization type
    heat_pump_strategy: str = "simple"  # heat pump strategy
    use_convex_optimization: bool = True  # whether to use convex optimization

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # basic settings
    time_horizon: int = 24  # default 24 hours
    time_step: int = 1      # default 1 hour
    
    # algorithm selection - consistent with main pipeline
    aggregation_method: str = "LP"  # LP or DP
    trading_method: str = "bidding"  # bidding or market-clearing
    disaggregation_method: str = "proportional"  # average or proportional
    
    seed: Optional[int] = None
    
    # device and user configuration
    num_managers: int = 4   # number of managers
    users_per_manager: List[int] = None  # number of users per manager
    device_config_file: str = "data/device_config_36users.csv"  # device configuration file
    
    # price data configuration
    price_data_file: str = "data/grid_price.csv"  # price data file
    
    # output configuration
    results_dir: str = "results"  # results directory
    
    # model configuration
    model_config: ModelBasedConfig = None
    
    def __post_init__(self):
        if self.users_per_manager is None:
            self.users_per_manager = [6, 10, 8, 12] 
        
        if self.model_config is None:
            self.model_config = ModelBasedConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """create configuration object from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # special handling for model_config
        if "model_config" in config_dict and isinstance(config_dict["model_config"], dict):
            model_config = ModelBasedConfig()
            for key, value in config_dict["model_config"].items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            config.model_config = model_config
            
        return config
    
    @classmethod
    def from_json_file(cls, json_path: str) -> "PipelineConfig":
        """load configuration from JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "model_config" and value is not None:
                config_dict[key] = {k: v for k, v in value.__dict__.items()}
            else:
                config_dict[key] = value
                
        return config_dict
    
    def to_json_file(self, json_path: str) -> None:
        """save configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """load configuration"""
    if config_path and os.path.exists(config_path):
        return PipelineConfig.from_json_file(config_path)
    
    return PipelineConfig()  # return default configuration

def generate_default_config_file(output_path: str) -> None:
    """generate default configuration file"""
    default_config = PipelineConfig()
    default_config.to_json_file(output_path)
    print(f"default configuration saved to: {output_path}")

if __name__ == "__main__":
    # generate default configuration file
    generate_default_config_file("default_modelbased_config.json")

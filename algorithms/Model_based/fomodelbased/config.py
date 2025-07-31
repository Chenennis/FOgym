"""
FlexOffer ModelBased Pipeline配置

提供与主Pipeline兼容的配置选项，确保可以使用相同的参数进行比较
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
import json

@dataclass
class ModelBasedConfig:
    """FOModelBased配置"""
    time_horizon: int = 24  # 默认24小时
    time_step: int = 1      # 时间步长（小时）
    optimization_type: str = "battery_type_0.55"  # 优化类型
    heat_pump_strategy: str = "simple"  # 热泵策略
    use_convex_optimization: bool = True  # 是否使用凸优化

@dataclass
class PipelineConfig:
    """Pipeline配置"""
    # 基本设置
    time_horizon: int = 24  # 默认24小时
    time_step: int = 1      # 默认1小时
    
    # 算法选择 - 与主pipeline保持一致
    aggregation_method: str = "LP"  # LP或DP
    trading_method: str = "bidding"  # bidding或market-clearing
    disaggregation_method: str = "proportional"  # average或proportional
    
    # 随机种子，用于保证实验可重复性
    seed: Optional[int] = None
    
    # 设备和用户配置
    num_managers: int = 4   # Manager数量
    users_per_manager: List[int] = None  # 每个Manager的用户数
    device_config_file: str = "data/device_config_36users.csv"  # 设备配置文件
    
    # 价格数据配置
    price_data_file: str = "data/grid_price.csv"  # 电价数据文件
    
    # 输出配置
    results_dir: str = "results"  # 结果目录
    
    # 模型配置
    model_config: ModelBasedConfig = None
    
    def __post_init__(self):
        if self.users_per_manager is None:
            self.users_per_manager = [6, 10, 8, 12]  # 默认36用户分布
        
        if self.model_config is None:
            self.model_config = ModelBasedConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """从字典创建配置对象"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # 特殊处理model_config
        if "model_config" in config_dict and isinstance(config_dict["model_config"], dict):
            model_config = ModelBasedConfig()
            for key, value in config_dict["model_config"].items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            config.model_config = model_config
            
        return config
    
    @classmethod
    def from_json_file(cls, json_path: str) -> "PipelineConfig":
        """从JSON文件加载配置"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"配置文件不存在: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "model_config" and value is not None:
                config_dict[key] = {k: v for k, v in value.__dict__.items()}
            else:
                config_dict[key] = value
                
        return config_dict
    
    def to_json_file(self, json_path: str) -> None:
        """保存配置到JSON文件"""
        config_dict = self.to_dict()
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """加载配置"""
    if config_path and os.path.exists(config_path):
        return PipelineConfig.from_json_file(config_path)
    
    return PipelineConfig()  # 返回默认配置

def generate_default_config_file(output_path: str) -> None:
    """生成默认配置文件"""
    default_config = PipelineConfig()
    default_config.to_json_file(output_path)
    print(f"默认配置已保存到: {output_path}")

if __name__ == "__main__":
    # 生成默认配置文件
    generate_default_config_file("default_modelbased_config.json")

"""全局观测空间配置"""

from typing import Dict, Any, List

# 默认全局观测配置
default_global_observation_config = {
    "generate": {
        "enabled": True,
        "weight": 1.0,
        "features": ["time", "user_demand", "device_stats"],
        "dim_reduction": "none"
    },
    "aggregate": {
        "enabled": True,
        "weight": 0.8,
        "features": ["energy_bounds", "flexibility"],
        "dim_reduction": "pca"
    },
    "trading": {
        "enabled": True,
        "weight": 0.9,
        "features": ["price_trends", "trade_stats"],
        "dim_reduction": "none"
    },
    "schedule": {
        "enabled": True,
        "weight": 0.7,
        "features": ["efficiency", "cost_optimization"],
        "dim_reduction": "none"
    },
    "global": {
        "enabled": True,
        "features": ["efficiency", "economic", "reliability", "environmental"],
        "dim_reduction": "none"
    }
}

# 特征尺寸配置
feature_dimensions = {
    "generate": {
        "time": 1,         # 时间段分类 (早中晚夜)
        "user_demand": 2,  # 总需求和未来需求预测
        "device_stats": 5  # 平均SOC, 总可用功率等统计信息
    },
    "aggregate": {
        "energy_bounds": 4,  # 最小/最大能量统计
        "flexibility": 2     # 灵活性指标
    },
    "trading": {
        "price_trends": 3,   # 价格趋势指标
        "trade_stats": 4     # 交易统计信息
    },
    "schedule": {
        "efficiency": 1,           # 调度效率指标
        "cost_optimization": 2     # 成本优化指标
    },
    "global": {
        "efficiency": 1,
        "economic": 1,
        "reliability": 1,
        "environmental": 1
    },
    "cross_module": {
        "time_sync": 1,
        "energy_flow": 2,
        "value_flow": 2,
        "consistency": 1
    }
}

def get_observation_dimension(config: Dict[str, Any]) -> int:
    """计算基于配置的观测空间维度
    
    Args:
        config: 全局观测配置
        
    Returns:
        观测空间总维度
    """
    total_dim = 0
    
    # 计算每个模块的维度
    for module, module_config in config.items():
        if module == "global" or not module_config.get("enabled", True):
            continue
            
        for feature in module_config.get("features", []):
            if feature in feature_dimensions.get(module, {}):
                total_dim += feature_dimensions[module][feature]
    
    # 添加全局特征
    if config.get("global", {}).get("enabled", True):
        for feature in config.get("global", {}).get("features", []):
            if feature in feature_dimensions.get("global", {}):
                total_dim += feature_dimensions["global"][feature]
    
    # 添加跨模块相关性
    total_dim += sum(feature_dimensions["cross_module"].values())
    
    return total_dim 
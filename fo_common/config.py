"""Global observation space configuration"""

from typing import Dict, Any, List

# Default global observation configuration
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

# Feature dimension configuration
feature_dimensions = {
    "generate": {
        "time": 1,         # Time period classification (morning, noon, evening, night)
        "user_demand": 2,  # Total demand and future demand forecast
        "device_stats": 5  # Average SOC, total available power and other statistics
    },
    "aggregate": {
        "energy_bounds": 4,  # Min/max energy statistics
        "flexibility": 2     # Flexibility indicators
    },
    "trading": {
        "price_trends": 3,   # Price trend indicators
        "trade_stats": 4     # Trading statistics
    },
    "schedule": {
        "efficiency": 1,           # Scheduling efficiency indicator
        "cost_optimization": 2     # Cost optimization indicator
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
    """Calculate observation space dimension based on configuration
    
    Args:
        config: Global observation configuration
        
    Returns:
        Total observation space dimension
    """
    total_dim = 0
    
    # Calculate dimension for each module
    for module, module_config in config.items():
        if module == "global" or not module_config.get("enabled", True):
            continue
            
        for feature in module_config.get("features", []):
            if feature in feature_dimensions.get(module, {}):
                total_dim += feature_dimensions[module][feature]
    
    # Add global features
    if config.get("global", {}).get("enabled", True):
        for feature in config.get("global", {}).get("features", []):
            if feature in feature_dimensions.get("global", {}):
                total_dim += feature_dimensions["global"][feature]
    
    # Add cross-module correlations
    total_dim += sum(feature_dimensions["cross_module"].values())
    
    return total_dim 
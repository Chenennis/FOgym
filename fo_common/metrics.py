"""全局性能指标计算"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

def calculate_system_efficiency(observations: Dict[str, Any]) -> float:
    """
    计算系统效率指标
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        系统效率分数 (0-1)
    """
    efficiency_values = []
    
    # 从生成模块获取效率
    if "generate" in observations:
        gen_obs = observations["generate"]
        gen_efficiency = 0.8  # 默认值
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
            # 根据设备状态计算生成效率
            device_states = gen_obs[30:]
            # 简化：使用设备状态的均值作为效率指标
            gen_efficiency = min(max(np.mean(device_states), 0.0), 1.0)
            
        efficiency_values.append(gen_efficiency)
    
    # 从交易模块获取效率
    if "trading" in observations:
        trade_obs = observations["trading"]
        trade_efficiency = 0.7  # 默认值
        
        if isinstance(trade_obs, dict) and 'trades' in trade_obs:
            trades = trade_obs['trades']
            # 使用交易成功率作为效率指标
            trade_efficiency = min(max(trades.get('success_rate', 0.7), 0.0), 1.0)
            
        efficiency_values.append(trade_efficiency)
    
    # 从调度模块获取效率
    if "schedule" in observations:
        sched_obs = observations["schedule"]
        sched_efficiency = 0.9  # 默认值
        
        if isinstance(sched_obs, dict) and 'efficiency' in sched_obs:
            sched_efficiency = min(max(sched_obs['efficiency'], 0.0), 1.0)
            
        efficiency_values.append(sched_efficiency)
    
    # 计算整体效率
    if efficiency_values:
        # 简单平均
        system_efficiency = sum(efficiency_values) / len(efficiency_values)
    else:
        system_efficiency = 0.8  # 默认值
        
    return system_efficiency

def calculate_economic_score(observations: Dict[str, Any]) -> float:
    """
    计算经济性得分
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        经济性得分 (0-1)
    """
    costs = []
    revenues = []
    
    # 从生成模块获取成本
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
            # 假设索引24是电价，用作成本指标
            costs.append(gen_obs[24] * 100)  # 缩放假设
    
    # 从交易模块获取收益
    if "trading" in observations:
        trade_obs = observations["trading"]
        
        if isinstance(trade_obs, dict) and 'price' in trade_obs:
            revenues.append(trade_obs['price'])
    
    # 从调度模块获取成本
    if "schedule" in observations:
        sched_obs = observations["schedule"]
        
        if isinstance(sched_obs, dict) and 'cost' in sched_obs:
            if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                costs.append(sched_obs['cost']['value'])
    
    # 计算经济性得分
    if costs and revenues:
        total_cost = sum(costs)
        total_revenue = sum(revenues)
        
        # 计算利润率
        profit_margin = (total_revenue - total_cost) / max(total_revenue, 0.01)
        
        # 归一化到[0,1]
        economic_score = min(max((profit_margin + 1.0) / 2.0, 0.0), 1.0)
    else:
        economic_score = 0.6  # 默认中等偏上
        
    return economic_score

def calculate_reliability_score(observations: Dict[str, Any]) -> float:
    """
    计算系统可靠性得分
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        可靠性得分 (0-1)
    """
    reliability_factors = []
    
    # 从生成模块获取可靠性
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
            # 假设设备状态稳定性反映可靠性
            device_states = gen_obs[30:]
            device_reliability = 1.0 - min(np.std(device_states) / 2.0, 1.0)
            reliability_factors.append(device_reliability)
    
    # 从交易模块获取可靠性
    if "trading" in observations:
        trade_obs = observations["trading"]
        
        if isinstance(trade_obs, dict) and 'trades' in trade_obs:
            trades = trade_obs['trades']
            trade_reliability = min(max(trades.get('success_rate', 0.7), 0.0), 1.0)
            reliability_factors.append(trade_reliability)
    
    # 计算整体可靠性
    if reliability_factors:
        reliability_score = sum(reliability_factors) / len(reliability_factors)
    else:
        reliability_score = 0.75  # 默认较高可靠性
        
    return reliability_score

def calculate_environmental_score(observations: Dict[str, Any]) -> float:
    """
    计算环境友好性得分
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        环境友好性得分 (0-1)
    """
    # 默认中等偏上环保性
    environmental_score = 0.7
    
    # 从生成模块获取环保性
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 28:
            # 假设索引28是环保性偏好
            environmental_score = min(max(gen_obs[28], 0.0), 1.0)
            
    return environmental_score

def calculate_cross_module_consistency(observations: Dict[str, Any]) -> float:
    """
    计算跨模块一致性得分
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        一致性得分 (0-1)
    """
    # 默认为中等一致性
    consistency_score = 0.5
    
    # 如果多于一个模块，计算模块间一致性
    if len(observations) > 1:
        # 这里只是一个简化的示例，实际应该基于具体的模块状态计算
        # 例如，计算电价、时间等信息的一致性
        
        # 检查时间一致性
        time_values = []
        
        if "generate" in observations:
            gen_obs = observations["generate"]
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) >= 24:
                # 获取小时（从one-hot编码）
                gen_hour = np.argmax(gen_obs[:24])
                time_values.append(gen_hour)
                
        if "trading" in observations:
            trade_obs = observations["trading"]
            if isinstance(trade_obs, dict) and 'time' in trade_obs:
                if hasattr(trade_obs['time'], 'hour'):
                    time_values.append(trade_obs['time'].hour)
        
        # 如果有多个时间值，计算一致性
        if len(time_values) > 1:
            # 计算标准差并归一化
            time_std = np.std(time_values)
            time_consistency = max(0.0, 1.0 - time_std / 12.0)  # 最多12小时差异视为完全不一致
            consistency_score = time_consistency
            
    return consistency_score

def calculate_global_metrics(observations: Dict[str, Any]) -> Dict[str, float]:
    """
    计算所有全局指标
    
    Args:
        observations: 各模块的观测字典
        
    Returns:
        包含所有指标的字典
    """
    try:
        metrics = {
            "efficiency": calculate_system_efficiency(observations),
            "economic": calculate_economic_score(observations),
            "reliability": calculate_reliability_score(observations),
            "environmental": calculate_environmental_score(observations),
            "consistency": calculate_cross_module_consistency(observations)
        }
        
        # 添加综合得分（加权平均）
        weights = {
            "efficiency": 0.25,
            "economic": 0.25,
            "reliability": 0.2,
            "environmental": 0.15,
            "consistency": 0.15
        }
        
        weighted_sum = sum(metrics[k] * weights[k] for k in metrics)
        total_weight = sum(weights.values())
        metrics["overall"] = weighted_sum / total_weight
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算全局指标时出错: {e}")
        # 返回默认值
        return {
            "efficiency": 0.8,
            "economic": 0.6,
            "reliability": 0.75,
            "environmental": 0.7,
            "consistency": 0.5,
            "overall": 0.7
        } 
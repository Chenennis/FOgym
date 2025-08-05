"""global performance metrics calculation"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# create logger
logger = logging.getLogger(__name__)

def calculate_system_efficiency(observations: Dict[str, Any]) -> float:
    """
    calculate system efficiency metrics
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        system efficiency score (0-1)
    """
    efficiency_values = []
    
    # get efficiency from generate module
    if "generate" in observations:
        gen_obs = observations["generate"]
        gen_efficiency = 0.8  # default value
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
            # calculate generation efficiency based on device states
            device_states = gen_obs[30:]
            # use the mean of device states as efficiency metric
            gen_efficiency = min(max(np.mean(device_states), 0.0), 1.0)
            
        efficiency_values.append(gen_efficiency)
    
    # get efficiency from trading module
    if "trading" in observations:
        trade_obs = observations["trading"]
        trade_efficiency = 0.7  # default value
        
        if isinstance(trade_obs, dict) and 'trades' in trade_obs:
            trades = trade_obs['trades']
            # use trade success rate as efficiency metric
            trade_efficiency = min(max(trades.get('success_rate', 0.7), 0.0), 1.0)
            
        efficiency_values.append(trade_efficiency)
    
    # get efficiency from schedule module
    if "schedule" in observations:
        sched_obs = observations["schedule"]
        sched_efficiency = 0.9  # default value
        
        if isinstance(sched_obs, dict) and 'efficiency' in sched_obs:
            sched_efficiency = min(max(sched_obs['efficiency'], 0.0), 1.0)
            
        efficiency_values.append(sched_efficiency)
    
    # calculate overall efficiency
    if efficiency_values:
        # average
        system_efficiency = sum(efficiency_values) / len(efficiency_values)
    else:
        system_efficiency = 0.8  # default value
        
    return system_efficiency

def calculate_economic_score(observations: Dict[str, Any]) -> float:
    """
    calculate economic score
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        economic score (0-1)
    """
    costs = []
    revenues = []
    
    # get cost from generate module
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
            # assume index 24 is price, used as cost metric
            costs.append(gen_obs[24] * 100)  # scale assumption
    
    # get revenue from trading module
    if "trading" in observations:
        trade_obs = observations["trading"]
        
        if isinstance(trade_obs, dict) and 'price' in trade_obs:
            revenues.append(trade_obs['price'])
    
    # get cost from schedule module
    if "schedule" in observations:
        sched_obs = observations["schedule"]
        
        if isinstance(sched_obs, dict) and 'cost' in sched_obs:
            if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                costs.append(sched_obs['cost']['value'])
    
    # calculate economic score
    if costs and revenues:
        total_cost = sum(costs)
        total_revenue = sum(revenues)
        
        # calculate profit margin
        profit_margin = (total_revenue - total_cost) / max(total_revenue, 0.01)
        
        # normalize to [0,1]
        economic_score = min(max((profit_margin + 1.0) / 2.0, 0.0), 1.0)
    else:
        economic_score = 0.6  
        
    return economic_score

def calculate_reliability_score(observations: Dict[str, Any]) -> float:
    """
    calculate system reliability score
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        reliability score (0-1)
    """
    reliability_factors = []
    
    # get reliability from generate module
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
            # assume device state stability reflects reliability
            device_states = gen_obs[30:]
            device_reliability = 1.0 - min(np.std(device_states) / 2.0, 1.0)
            reliability_factors.append(device_reliability)
    
    # get reliability from trading module
    if "trading" in observations:
        trade_obs = observations["trading"]
        
        if isinstance(trade_obs, dict) and 'trades' in trade_obs:
            trades = trade_obs['trades']
            trade_reliability = min(max(trades.get('success_rate', 0.7), 0.0), 1.0)
            reliability_factors.append(trade_reliability)
    
    # calculate overall reliability
    if reliability_factors:
        reliability_score = sum(reliability_factors) / len(reliability_factors)
    else:
        reliability_score = 0.75  # default high reliability
        
    return reliability_score

def calculate_environmental_score(observations: Dict[str, Any]) -> float:
    """
    calculate environmental friendliness score
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        environmental friendliness score (0-1)
    """
    environmental_score = 0.7
    
    # get environmental friendliness from generate module
    if "generate" in observations:
        gen_obs = observations["generate"]
        
        if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 28:
            # assume index 28 is environmental preference
            environmental_score = min(max(gen_obs[28], 0.0), 1.0)
            
    return environmental_score

def calculate_cross_module_consistency(observations: Dict[str, Any]) -> float:
    """
    calculate cross-module consistency score
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        consistency score (0-1)
    """
    # default medium consistency
    consistency_score = 0.5
    
    # if more than one module, calculate cross-module consistency
    if len(observations) > 1:
        
        # check time consistency
        time_values = []
        
        if "generate" in observations:
            gen_obs = observations["generate"]
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) >= 24:
                # get hour (from one-hot encoding)
                gen_hour = np.argmax(gen_obs[:24])
                time_values.append(gen_hour)
                
        if "trading" in observations:
            trade_obs = observations["trading"]
            if isinstance(trade_obs, dict) and 'time' in trade_obs:
                if hasattr(trade_obs['time'], 'hour'):
                    time_values.append(trade_obs['time'].hour)
        
        # if there are multiple time values, calculate consistency
        if len(time_values) > 1:
            # calculate standard deviation and normalize
            time_std = np.std(time_values)
            time_consistency = max(0.0, 1.0 - time_std / 12.0)  # maximum 12 hours difference considered completely inconsistent
            consistency_score = time_consistency
            
    return consistency_score

def calculate_global_metrics(observations: Dict[str, Any]) -> Dict[str, float]:
    """
    calculate all global metrics
    
    Args:
        observations: observation dictionary of each module
        
    Returns:
        dictionary containing all metrics
    """
    try:
        metrics = {
            "efficiency": calculate_system_efficiency(observations),
            "economic": calculate_economic_score(observations),
            "reliability": calculate_reliability_score(observations),
            "environmental": calculate_environmental_score(observations),
            "consistency": calculate_cross_module_consistency(observations)
        }
        
        # add overall score (weighted average)
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
        logger.error(f"error calculating global metrics: {e}")
        # return default values
        return {
            "efficiency": 0.8,
            "economic": 0.6,
            "reliability": 0.75,
            "environmental": 0.7,
            "consistency": 0.5,
            "overall": 0.7
        } 
"""feature extraction function"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

# create logger
logger = logging.getLogger(__name__)

def extract_generate_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """extract key features from the generate module
    
    Args:
        observation: original observation vector
        config: feature configuration
        
    Returns:
        extracted feature vector
    """
    features = []
    
    try:
        # extract time feature (from one-hot compression to time period classification)
        if "time" in config["features"]:
            if len(observation) >= 24:
                hour_onehot = observation[:24]
                hour = np.argmax(hour_onehot)
                # map hour to time period (morning, noon, evening, night)
                time_period = hour // 6  # 0-5, 6-11, 12-17, 18-23
                features.append(time_period / 4.0)  # normalize to [0,1]
            else:
                logger.warning("observation vector length is not enough, cannot extract time feature")
                features.append(0.0)
        
        # extract user demand feature
        if "user_demand" in config["features"]:
            # assume: user preference is in index 25-28, we use them to calculate total demand
            if len(observation) >= 29:
                # calculate basic demand from user preference 
                preference_sum = sum(observation[25:29])
                normalized_demand = min(preference_sum / 2.0, 1.0)  # normalize
                
                # add current and predicted demand
                features.append(normalized_demand)
                features.append(normalized_demand * 1.1)  # simple prediction, assume 10% increase
            else:
                logger.warning("observation vector length is not enough, cannot extract user demand feature")
                features.extend([0.0, 0.0])
        
        # extract device statistics feature
        if "device_stats" in config["features"]:
            # take the part after index 30 as device state, calculate average and other statistics
            if len(observation) > 30:
                device_states = observation[30:]
                
                # calculate basic statistics
                mean_value = np.mean(device_states)
                max_value = np.max(device_states)
                min_value = np.min(device_states)
                std_value = np.std(device_states)
                median_value = np.median(device_states)
                
                # normalize, ensure the result is within [0,1]
                features.extend([
                    min(max(mean_value, 0.0), 1.0),
                    min(max(max_value/10.0, 0.0), 1.0),
                    min(max(min_value+0.5, 0.0), 1.0),
                    min(max(std_value/2.0, 0.0), 1.0),
                    min(max(median_value, 0.0), 1.0)
                ])
            else:
                logger.warning("observation vector length is not enough, cannot extract device statistics feature")
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    except Exception as e:
        logger.error(f"error extracting features from generate module: {e}")
        # return all 0 vectors as backup
        expected_dim = 0
        if "time" in config["features"]:
            expected_dim += 1
        if "user_demand" in config["features"]:
            expected_dim += 2
        if "device_stats" in config["features"]:
            expected_dim += 5
        features = [0.0] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_aggregate_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """extract key features from the aggregate module
    
    Args:
        observation: original aggregate information
        config: feature configuration
        
    Returns:
        extracted feature vector
    """
    features = []
    
    try:
        # extract energy boundary information
        if "energy_bounds" in config["features"]:
            # assume observation is some representation of energy boundary information
            if isinstance(observation, dict) and 'energy_min' in observation and 'energy_max' in observation:
                e_min = observation['energy_min']
                e_max = observation['energy_max']
                
                if isinstance(e_min, (list, np.ndarray)) and isinstance(e_max, (list, np.ndarray)):
                    # calculate statistics
                    min_e_min = min(e_min)
                    max_e_min = max(e_min)
                    min_e_max = min(e_max)
                    max_e_max = max(e_max)
                    
                    # normalize
                    features.extend([
                        min(max((min_e_min + 100) / 200, 0.0), 1.0),
                        min(max((max_e_min + 100) / 200, 0.0), 1.0),
                        min(max((min_e_max) / 200, 0.0), 1.0),
                        min(max((max_e_max) / 200, 0.0), 1.0)
                    ])
                else:
                    features.extend([0.5, 0.5, 0.5, 0.5])  # default value
            else:
                # if there is no energy boundary information, use default value
                features.extend([0.5, 0.5, 0.5, 0.5])
            
        # extract flexibility metrics
        if "flexibility" in config["features"]:
            if isinstance(observation, dict) and 'flexibility' in observation:
                flex = observation['flexibility']
                time_flex = flex.get('time_flexibility', 0.5)
                power_flex = flex.get('power_flexibility', 0.5)
                
                features.extend([
                    min(max(time_flex, 0.0), 1.0),
                    min(max(power_flex, 0.0), 1.0)
                ])
            else:
                # default flexibility metrics
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"error extracting features from aggregate module: {e}")
        # return all 0.5 vectors as backup (middle value)
        expected_dim = 0
        if "energy_bounds" in config["features"]:
            expected_dim += 4
        if "flexibility" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_trading_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """extract key features from the trading module
    
    Args:
        observation: original trading state
        config: feature configuration
        
    Returns:
        extracted feature vector
    """
    features = []
    
    try:
        # extract price trend feature
        if "price_trends" in config["features"]:
            if isinstance(observation, dict) and 'prices' in observation:
                prices = observation['prices']
                if len(prices) >= 3:
                    # calculate simple trend indicators
                    current_price = prices[-1]
                    prev_price = prices[-2]
                    earliest_price = prices[0]
                    
                    # short-term trend (normalize to [-1,1], then normalize to [0,1])
                    short_trend = min(max((current_price - prev_price) / max(prev_price, 0.01), -1.0), 1.0)
                    short_trend = (short_trend + 1.0) / 2.0  # normalize to [0,1]
                    
                    # long-term trend
                    long_trend = min(max((current_price - earliest_price) / max(earliest_price, 0.01), -1.0), 1.0)
                    long_trend = (long_trend + 1.0) / 2.0  # normalize to [0,1]
                    
                    # price volatility (standard deviation/mean)
                    volatility = min(np.std(prices) / max(np.mean(prices), 0.01), 1.0)
                    
                    features.extend([short_trend, long_trend, volatility])
                else:
                    features.extend([0.5, 0.5, 0.5])  # default middle value
            else:
                # if there is no price information, use default value
                features.extend([0.5, 0.5, 0.5])
                
        # 提取交易统计特征
        if "trade_stats" in config["features"]:
            if isinstance(observation, dict) and 'trades' in observation:
                trades = observation['trades']
                
                # calculate success rate
                success_rate = trades.get('success_rate', 0.5)
                
                # calculate volume
                volume = min(trades.get('volume', 50) / 100.0, 1.0)
                
                # calculate average price deviation (the difference between the actual transaction price and the target price)
                price_deviation = min(max((trades.get('price_deviation', 0) + 0.2) / 0.4, 0.0), 1.0)
                
                # calculate transaction frequency
                frequency = min(trades.get('frequency', 0.5), 1.0)
                
                features.extend([success_rate, volume, price_deviation, frequency])
            else:
                # default trade statistics
                features.extend([0.5, 0.5, 0.5, 0.5])
    
    except Exception as e:
        logger.error(f"error extracting features from trading module: {e}")
        # return all 0.5 vectors as backup (middle value)
        expected_dim = 0
        if "price_trends" in config["features"]:
            expected_dim += 3
        if "trade_stats" in config["features"]:
            expected_dim += 4
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_schedule_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """extract key features from the schedule module
    
    Args:
        observation: original schedule state
        config: feature configuration
        
    Returns:
        extracted feature vector
    """
    features = []
    
    try:
        # extract efficiency metrics
        if "efficiency" in config["features"]:
            if isinstance(observation, dict) and 'efficiency' in observation:
                efficiency = min(max(observation['efficiency'], 0.0), 1.0)
                features.append(efficiency)
            else:
                # default efficiency
                features.append(0.7)  # optimistic default value
                
        # extract cost optimization metrics
        if "cost_optimization" in config["features"]:
            if isinstance(observation, dict) and 'cost' in observation:
                cost_data = observation['cost']
                
                # cost optimization potential (the ratio of actual cost to optimal cost)
                optimization_potential = min(max(cost_data.get('potential', 0.5), 0.0), 1.0)
                
                # cost trend (the direction of recent cost changes)
                # normalize to [0,1], 0 means cost increase, 1 means cost decrease
                cost_trend = min(max((cost_data.get('trend', 0) + 1.0) / 2.0, 0.0), 1.0)
                
                features.extend([optimization_potential, cost_trend])
            else:
                # default cost metrics
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"error extracting features from schedule module: {e}")
        # return all 0.5 vectors as backup (middle value)
        expected_dim = 0
        if "efficiency" in config["features"]:
            expected_dim += 1
        if "cost_optimization" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def compute_cross_module_correlations(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """compute cross-module correlation features
    
    Args:
        observations: observation dictionary of each module
        config: feature configuration
        
    Returns:
        correlation feature vector
    """
    correlations = []
    
    try:
        # time synchronization feature
        if all(["generate" in observations, "trading" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # get time from generate module (assume the first 24 elements of 24-dimensional one-hot)
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) >= 24:
                gen_hour = np.argmax(gen_obs[:24])
                
                # get time from trading module (assume in the dictionary or array)
                trade_hour = None
                if isinstance(trade_obs, dict) and 'time' in trade_obs:
                    trade_hour = trade_obs['time'].hour if hasattr(trade_obs['time'], 'hour') else 0
                elif isinstance(trade_obs, np.ndarray) and len(trade_obs) > 0:
                    # assume the first element is related to time
                    trade_hour = int(trade_obs[0] * 24) if trade_obs[0] <= 1 else 0
                
                if trade_hour is not None:
                    # calculate time difference indicator, normalize to [0,1]
                    # 0 means completely out of sync, 1 means completely synchronized
                    time_diff = abs(gen_hour - trade_hour)
                    time_sync = 1.0 - min(time_diff / 12.0, 1.0)  # maximum difference of 12 hours is considered completely out of sync
                    correlations.append(time_sync)
                else:
                    correlations.append(0.5)  # default medium synchronization degree
            else:
                correlations.append(0.5)
        else:
            correlations.append(0.5)
            
        # energy flow vector (assume: generate-aggregate-trading-schedule energy flow)
        if all(["generate" in observations, "trading" in observations]):
            # simplified calculation: energy balance between generate and trading modules
            gen_energy = 0.5  # default generated energy
            trade_energy = 0.5  # default traded energy
            
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # extract energy generation from generate module (assume)
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                # use the average value of device state as the energy generation indicator
                gen_energy = min(max(np.mean(gen_obs[30:]) / 2.0, 0.0), 1.0)
            
            # extract energy demand from trading module (assume)
            if isinstance(trade_obs, dict) and 'demand' in trade_obs:
                trade_energy = min(max(trade_obs['demand'] / 100.0, 0.0), 1.0)
            
            # calculate energy matching degree (0=severe mismatch, 1=perfect match)
            energy_match = 1.0 - min(abs(gen_energy - trade_energy), 1.0)
            
            # calculate energy flow direction (0=consumption>generation, 1=generation>consumption)
            energy_direction = 1.0 if gen_energy > trade_energy else 0.0
            
            correlations.extend([energy_match, energy_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # value flow vector
        # assume value flow is based on price and cost
        if all(["generate" in observations, "trading" in observations, "schedule" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            sched_obs = observations["schedule"]
            
            # get generation cost (assume)
            gen_cost = 0.5
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                # assume index 24 is price
                gen_cost = min(max(gen_obs[24], 0.0), 1.0)
            
            # get trading price (assume)
            trade_price = 0.5
            if isinstance(trade_obs, dict) and 'price' in trade_obs:
                trade_price = min(max(trade_obs['price'] / 100.0, 0.0), 1.0)
            
            # get schedule cost (assume)
            sched_cost = 0.5
            if isinstance(sched_obs, dict) and 'cost' in sched_obs:
                if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                    sched_cost = min(max(sched_obs['cost']['value'] / 100.0, 0.0), 1.0)
            
            # calculate value flow indicators
            value_efficiency = min(max(trade_price / (gen_cost + sched_cost + 0.01), 0.0), 1.0)
            value_direction = min(max((trade_price - gen_cost) / max(gen_cost, 0.01), 0.0), 1.0)
            
            correlations.extend([value_efficiency, value_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # state consistency indicators
        # calculate the consistency of the state vectors of each module
        enabled_modules = [
            module for module, data in observations.items() 
            if data is not None and module in config and config[module].get("enabled", True)
        ]
        
        if len(enabled_modules) > 1:
            # consistency measure: average cosine angle of normalized state vectors of each module
            consistency = 0.5  # default medium consistency
            
            # calculation, actually need to calculate based on the specific module state
            correlations.append(consistency)
        else:
            correlations.append(0.5)
    
    except Exception as e:
        logger.error(f"error computing cross-module correlations: {e}")
        # default correlation vector
        correlations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
    return np.array(correlations[:6], dtype=np.float32)  # return at most 6 correlation features

def compute_global_metrics(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """compute global optimization metrics
    
    Args:
        observations: observation dictionary of each module
        config: feature configuration
        
    Returns:
        global metrics vector
    """
    metrics = []
    
    try:
        global_config = config.get("global", {})
        enabled_features = global_config.get("features", [])
        
        # system efficiency metrics
        if "efficiency" in enabled_features:
            # calculate comprehensive efficiency based on each module
            efficiency_values = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                # assume calculate generation efficiency (e.g., based on device state)
                gen_efficiency = 0.8  # default value
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # use the average value of device state to calculate efficiency
                    gen_efficiency = min(max(np.mean(gen_obs[30:]), 0.0), 1.0)
                efficiency_values.append(gen_efficiency)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                # assume calculate trading efficiency
                trade_efficiency = 0.7  # default value
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_efficiency = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                efficiency_values.append(trade_efficiency)
            
            if "schedule" in observations:
                sched_obs = observations["schedule"]
                # assume calculate schedule efficiency
                sched_efficiency = 0.9  # default value
                if isinstance(sched_obs, dict) and 'efficiency' in sched_obs:
                    sched_efficiency = min(max(sched_obs['efficiency'], 0.0), 1.0)
                efficiency_values.append(sched_efficiency)
            
            # calculate overall efficiency
            if efficiency_values:
                system_efficiency = sum(efficiency_values) / len(efficiency_values)
                metrics.append(system_efficiency)
            else:
                metrics.append(0.8)  # default higher efficiency
                
        # economic metrics
        if "economic" in enabled_features:
            # calculate economic metrics based on cost and price
            economic_score = 0.6  # default medium-high
            
            # use generation cost, trading price and schedule cost
            costs = []
            revenues = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                    # assume index 24 is price, used as cost indicator
                    costs.append(gen_obs[24] * 100)  # scale assumption
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                if isinstance(trade_obs, dict):
                    if 'price' in trade_obs:
                        revenues.append(trade_obs['price'])
            
            if "schedule" in observations:
                sched_obs = observations["schedule"]
                if isinstance(sched_obs, dict) and 'cost' in sched_obs:
                    if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                        costs.append(sched_obs['cost']['value'])
            
            if costs and revenues:
                total_cost = sum(costs)
                total_revenue = sum(revenues)
                profit_margin = (total_revenue - total_cost) / max(total_revenue, 0.01)
                economic_score = min(max((profit_margin + 1.0) / 2.0, 0.0), 1.0)  # normalize to [0,1]
            
            metrics.append(economic_score)
            
        # reliability metrics
        if "reliability" in enabled_features:
            # calculate system reliability metrics
            reliability_score = 0.75  # default higher reliability
            
            # based on device state and trading success rate
            reliability_factors = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # assume device state stability reflects reliability
                    device_reliability = 1.0 - min(np.std(gen_obs[30:]) / 2.0, 1.0)
                    reliability_factors.append(device_reliability)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_reliability = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                    reliability_factors.append(trade_reliability)
            
            # calculate overall reliability
            if reliability_factors:
                reliability_score = sum(reliability_factors) / len(reliability_factors)
            
            metrics.append(reliability_score)
            
        # environmental metrics
        if "environmental" in enabled_features:
            # calculate environmental impact metrics
            environmental_score = 0.7  # default good
            
            # based on the proportion of renewable energy use
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 25:
                    # assume the index 28 is the environmental preference
                    environmental_score = min(max(gen_obs[28], 0.0), 1.0)
            
            metrics.append(environmental_score)
    
    except Exception as e:
        logger.error(f"error computing global metrics: {e}")
        # default global metrics
        expected_dim = 0
        if "efficiency" in enabled_features:
            expected_dim += 1
        if "economic" in enabled_features:
            expected_dim += 1
        if "reliability" in enabled_features:
            expected_dim += 1
        if "environmental" in enabled_features:
            expected_dim += 1
        metrics = [0.7] * expected_dim  # default good metrics
        
    return np.array(metrics, dtype=np.float32) 
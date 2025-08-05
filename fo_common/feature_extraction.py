"""Feature extraction functionality"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Create logger
logger = logging.getLogger(__name__)

def extract_generate_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Extract key features from generation module
    
    Args:
        observation: Original observation vector
        config: Feature configuration
        
    Returns:
        Extracted feature vector
    """
    features = []
    
    try:
        # Extract time features (compress one-hot to time period classification)
        if "time" in config["features"]:
            if len(observation) >= 24:
                hour_onehot = observation[:24]
                hour = np.argmax(hour_onehot)
                # Map hour to time period (morning, noon, evening, night)
                time_period = hour // 6  # 0-5, 6-11, 12-17, 18-23
                features.append(time_period / 4.0)  # Normalize to [0,1]
            else:
                logger.warning("Observation vector too short, cannot extract time features")
                features.append(0.0)
        
        # Extract user demand features
        if "user_demand" in config["features"]:
            # Assumption: User preferences at indices 25-28, we use them to calculate total demand
            if len(observation) >= 29:
                # Calculate basic demand from user preferences (simplified calculation)
                preference_sum = sum(observation[25:29])
                normalized_demand = min(preference_sum / 2.0, 1.0)  # Normalize
                
                # Add current and predicted demand
                features.append(normalized_demand)
                features.append(normalized_demand * 1.1)  # Simple prediction, assume 10% increase
            else:
                logger.warning("Observation vector too short, cannot extract user demand features")
                features.extend([0.0, 0.0])
        
        # Extract device statistics features
        if "device_stats" in config["features"]:
            # Simplified: Use portion after index 30 as device states, calculate averages and other statistics
            if len(observation) > 30:
                device_states = observation[30:]
                
                # Calculate basic statistics
                mean_value = np.mean(device_states)
                max_value = np.max(device_states)
                min_value = np.min(device_states)
                std_value = np.std(device_states)
                median_value = np.median(device_states)
                
                # Normalize, ensure results are in [0,1] range
                features.extend([
                    min(max(mean_value, 0.0), 1.0),
                    min(max(max_value/10.0, 0.0), 1.0),
                    min(max(min_value+0.5, 0.0), 1.0),
                    min(max(std_value/2.0, 0.0), 1.0),
                    min(max(median_value, 0.0), 1.0)
                ])
            else:
                logger.warning("Observation vector too short, cannot extract device statistics features")
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    except Exception as e:
        logger.error(f"Error extracting generation module features: {e}")
        # Return all-zero vector as fallback
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
    """Extract key features from aggregation module
    
    Args:
        observation: Original aggregation information
        config: Feature configuration
        
    Returns:
        Extracted feature vector
    """
    features = []
    
    try:
        # Aggregation module might not have traditional observation, but DFO/SFO systems
        # Below is simplified code to handle this case
        
        # Extract energy bounds information
        if "energy_bounds" in config["features"]:
            # Assume observation is some representation of energy bounds information
            if isinstance(observation, dict) and 'energy_min' in observation and 'energy_max' in observation:
                e_min = observation['energy_min']
                e_max = observation['energy_max']
                
                if isinstance(e_min, (list, np.ndarray)) and isinstance(e_max, (list, np.ndarray)):
                    # Calculate statistics
                    min_e_min = min(e_min)
                    max_e_min = max(e_min)
                    min_e_max = min(e_max)
                    max_e_max = max(e_max)
                    
                    # Normalize
                    features.extend([
                        min(max((min_e_min + 100) / 200, 0.0), 1.0),
                        min(max((max_e_min + 100) / 200, 0.0), 1.0),
                        min(max((min_e_max) / 200, 0.0), 1.0),
                        min(max((max_e_max) / 200, 0.0), 1.0)
                    ])
                else:
                    features.extend([0.5, 0.5, 0.5, 0.5])  # Default values
            else:
                # If no energy bounds information, use default values
                features.extend([0.5, 0.5, 0.5, 0.5])
            
        # Extract flexibility metrics
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
                # Default flexibility metrics
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"Error extracting aggregation module features: {e}")
        # Return all-0.5 vector as fallback (middle values)
        expected_dim = 0
        if "energy_bounds" in config["features"]:
            expected_dim += 4
        if "flexibility" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_trading_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Extract key features from trading module
    
    Args:
        observation: Original trading state
        config: Feature configuration
        
    Returns:
        Extracted feature vector
    """
    features = []
    
    try:
        # Extract price trend features
        if "price_trends" in config["features"]:
            if isinstance(observation, dict) and 'prices' in observation:
                prices = observation['prices']
                if len(prices) >= 3:
                    # Calculate simple trend indicators
                    current_price = prices[-1]
                    prev_price = prices[-2]
                    earliest_price = prices[0]
                    
                    # Short-term trend (normalized to [-1,1], then to [0,1])
                    short_trend = min(max((current_price - prev_price) / max(prev_price, 0.01), -1.0), 1.0)
                    short_trend = (short_trend + 1.0) / 2.0  # Normalize to [0,1]
                    
                    # Long-term trend
                    long_trend = min(max((current_price - earliest_price) / max(earliest_price, 0.01), -1.0), 1.0)
                    long_trend = (long_trend + 1.0) / 2.0  # Normalize to [0,1]
                    
                    # Price volatility (std/mean)
                    volatility = min(np.std(prices) / max(np.mean(prices), 0.01), 1.0)
                    
                    features.extend([short_trend, long_trend, volatility])
                else:
                    features.extend([0.5, 0.5, 0.5])  # Default middle values
            else:
                # If no price information, use default values
                features.extend([0.5, 0.5, 0.5])
                
        # Extract trade statistics features
        if "trade_stats" in config["features"]:
            if isinstance(observation, dict) and 'trades' in observation:
                trades = observation['trades']
                
                # Calculate success rate
                success_rate = trades.get('success_rate', 0.5)
                
                # Calculate volume
                volume = min(trades.get('volume', 50) / 100.0, 1.0)
                
                # Calculate average price deviation (difference between actual and target price)
                price_deviation = min(max((trades.get('price_deviation', 0) + 0.2) / 0.4, 0.0), 1.0)
                
                # Calculate trade frequency
                frequency = min(trades.get('frequency', 0.5), 1.0)
                
                features.extend([success_rate, volume, price_deviation, frequency])
            else:
                # Default trade statistics
                features.extend([0.5, 0.5, 0.5, 0.5])
    
    except Exception as e:
        logger.error(f"Error extracting trading module features: {e}")
        # Return all-0.5 vector as fallback (middle values)
        expected_dim = 0
        if "price_trends" in config["features"]:
            expected_dim += 3
        if "trade_stats" in config["features"]:
            expected_dim += 4
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_schedule_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Extract key features from scheduling module
    
    Args:
        observation: Original scheduling state
        config: Feature configuration
        
    Returns:
        Extracted feature vector
    """
    features = []
    
    try:
        # Extract efficiency metrics
        if "efficiency" in config["features"]:
            if isinstance(observation, dict) and 'efficiency' in observation:
                efficiency = min(max(observation['efficiency'], 0.0), 1.0)
                features.append(efficiency)
            else:
                # Default efficiency
                features.append(0.7)  # Optimistic default value
                
        # Extract cost optimization metrics
        if "cost_optimization" in config["features"]:
            if isinstance(observation, dict) and 'cost' in observation:
                cost_data = observation['cost']
                
                # Cost optimization potential (ratio of actual cost to optimal cost)
                optimization_potential = min(max(cost_data.get('potential', 0.5), 0.0), 1.0)
                
                # Cost trend (direction of recent cost changes)
                # Normalize to [0,1], 0 means cost increase, 1 means cost decrease
                cost_trend = min(max((cost_data.get('trend', 0) + 1.0) / 2.0, 0.0), 1.0)
                
                features.extend([optimization_potential, cost_trend])
            else:
                # Default cost metrics
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"Error extracting scheduling module features: {e}")
        # Return all-0.5 vector as fallback (middle values)
        expected_dim = 0
        if "efficiency" in config["features"]:
            expected_dim += 1
        if "cost_optimization" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def compute_cross_module_correlations(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """Calculate cross-module correlation features
    
    Args:
        observations: Dictionary of observations for each module
        config: Feature configuration
        
    Returns:
        Correlation feature vector
    """
    correlations = []
    
    try:
        # Time synchronization features
        if all(["generate" in observations, "trading" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # Get time from generation module (assuming first 24 elements of one-hot are time)
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) >= 24:
                gen_hour = np.argmax(gen_obs[:24])
                
                # Get time from trading module (assuming it's in a dictionary or at some position)
                trade_hour = None
                if isinstance(trade_obs, dict) and 'time' in trade_obs:
                    trade_hour = trade_obs['time'].hour if hasattr(trade_obs['time'], 'hour') else 0
                elif isinstance(trade_obs, np.ndarray) and len(trade_obs) > 0:
                    # Assume first element is related to time
                    trade_hour = int(trade_obs[0] * 24) if trade_obs[0] <= 1 else 0
                
                if trade_hour is not None:
                    # Calculate time difference metric, normalize to [0,1]
                    # 0 means completely out of sync, 1 means completely synchronized
                    time_diff = abs(gen_hour - trade_hour)
                    time_sync = 1.0 - min(time_diff / 12.0, 1.0)  # Max 12 hours difference considered completely out of sync
                    correlations.append(time_sync)
                else:
                    correlations.append(0.5)  # Default medium synchronization
            else:
                correlations.append(0.5)
        else:
            correlations.append(0.5)
            
        # Energy flow vector (assuming: generation-aggregation-trading-scheduling energy flow)
        if all(["generate" in observations, "trading" in observations]):
            # Simplified calculation: energy balance between generation and trading modules
            gen_energy = 0.5  # Default generation energy
            trade_energy = 0.5  # Default trading energy
            
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # Extract generation module's energy generation (assuming)
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                # Simplified: Use average of device states as energy generation metric
                gen_energy = min(max(np.mean(gen_obs[30:]) / 2.0, 0.0), 1.0)
            
            # Extract trading module's energy demand (assuming)
            if isinstance(trade_obs, dict) and 'demand' in trade_obs:
                trade_energy = min(max(trade_obs['demand'] / 100.0, 0.0), 1.0)
            
            # Calculate energy match (0=severe mismatch, 1=perfect match)
            energy_match = 1.0 - min(abs(gen_energy - trade_energy), 1.0)
            
            # Calculate energy flow direction (0=consumption>generation, 1=generation>consumption)
            energy_direction = 1.0 if gen_energy > trade_energy else 0.0
            
            correlations.extend([energy_match, energy_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # Value flow vector
        # Simplified: assume value flow is based on price and cost
        if all(["generate" in observations, "trading" in observations, "schedule" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            sched_obs = observations["schedule"]
            
            # Get generation cost (assuming)
            gen_cost = 0.5
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                # Assume index 24 is electricity price
                gen_cost = min(max(gen_obs[24], 0.0), 1.0)
            
            # Get trading price (assuming)
            trade_price = 0.5
            if isinstance(trade_obs, dict) and 'price' in trade_obs:
                trade_price = min(max(trade_obs['price'] / 100.0, 0.0), 1.0)
            
            # Get scheduling cost (assuming)
            sched_cost = 0.5
            if isinstance(sched_obs, dict) and 'cost' in sched_obs:
                if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                    sched_cost = min(max(sched_obs['cost']['value'] / 100.0, 0.0), 1.0)
            
            # Calculate value flow metrics
            value_efficiency = min(max(trade_price / (gen_cost + sched_cost + 0.01), 0.0), 1.0)
            value_direction = min(max((trade_price - gen_cost) / max(gen_cost, 0.01), 0.0), 1.0)
            
            correlations.extend([value_efficiency, value_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # State consistency metrics
        # Simplified: calculate consistency of state vectors across modules
        enabled_modules = [
            module for module, data in observations.items() 
            if data is not None and module in config and config[module].get("enabled", True)
        ]
        
        if len(enabled_modules) > 1:
            # Simple consistency measure: average cosine similarity of normalized state vectors
            consistency = 0.5  # Default medium consistency
            
            # Simplified calculation, actual consistency needs to be calculated based on specific module states
            correlations.append(consistency)
        else:
            correlations.append(0.5)
    
    except Exception as e:
        logger.error(f"Error calculating cross-module correlations: {e}")
        # Default correlation vector
        correlations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
    return np.array(correlations[:6], dtype=np.float32)  # Return at most 6 correlation features

def compute_global_metrics(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """Calculate global optimization metrics
    
    Args:
        observations: Dictionary of observations for each module
        config: Feature configuration
        
    Returns:
        Global metrics vector
    """
    metrics = []
    
    try:
        global_config = config.get("global", {})
        enabled_features = global_config.get("features", [])
        
        # System efficiency metrics
        if "efficiency" in enabled_features:
            # Calculate overall efficiency based on modules
            efficiency_values = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                # Assume calculation of generation efficiency (e.g., based on device states)
                gen_efficiency = 0.8  # Default value
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # Simplified: calculate efficiency using average of device states
                    gen_efficiency = min(max(np.mean(gen_obs[30:]), 0.0), 1.0)
                efficiency_values.append(gen_efficiency)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                # Assume calculation of trading efficiency
                trade_efficiency = 0.7  # Default value
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_efficiency = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                efficiency_values.append(trade_efficiency)
            
            if "schedule" in observations:
                sched_obs = observations["schedule"]
                # Assume calculation of scheduling efficiency
                sched_efficiency = 0.9  # Default value
                if isinstance(sched_obs, dict) and 'efficiency' in sched_obs:
                    sched_efficiency = min(max(sched_obs['efficiency'], 0.0), 1.0)
                efficiency_values.append(sched_efficiency)
            
            # Calculate overall efficiency
            if efficiency_values:
                system_efficiency = sum(efficiency_values) / len(efficiency_values)
                metrics.append(system_efficiency)
            else:
                metrics.append(0.8)  # Default higher efficiency
                
        # Economic metrics
        if "economic" in enabled_features:
            # Calculate economic metrics based on cost and price
            economic_score = 0.6  # Default slightly above medium
            
            # Simplified: use generation cost, trading price, and scheduling cost
            costs = []
            revenues = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                    # Assume index 24 is electricity price, used as cost metric
                    costs.append(gen_obs[24] * 100)  # Scaling assumption
            
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
                economic_score = min(max((profit_margin + 1.0) / 2.0, 0.0), 1.0)  # Normalize to [0,1]
            
            metrics.append(economic_score)
            
        # Reliability metrics
        if "reliability" in enabled_features:
            # Calculate system reliability metrics
            reliability_score = 0.75  # Default higher reliability
            
            # Simplified: based on device states and trading success rate
            reliability_factors = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # Assume device state stability reflects reliability
                    device_reliability = 1.0 - min(np.std(gen_obs[30:]) / 2.0, 1.0)
                    reliability_factors.append(device_reliability)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_reliability = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                    reliability_factors.append(trade_reliability)
            
            # Calculate overall reliability
            if reliability_factors:
                reliability_score = sum(reliability_factors) / len(reliability_factors)
            
            metrics.append(reliability_score)
            
        # Environmental metrics
        if "environmental" in enabled_features:
            # Calculate environmental impact metrics
            environmental_score = 0.7  # Default better
            
            # Simplified: based on proportion of renewable energy usage
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 25:
                    # Assume renewable energy preference can be inferred from user preferences
                    environmental_score = min(max(gen_obs[28], 0.0), 1.0)  # Assume index 28 is renewable energy preference
            
            metrics.append(environmental_score)
    
    except Exception as e:
        logger.error(f"Error calculating global metrics: {e}")
        # Default global metrics
        expected_dim = 0
        if "efficiency" in enabled_features:
            expected_dim += 1
        if "economic" in enabled_features:
            expected_dim += 1
        if "reliability" in enabled_features:
            expected_dim += 1
        if "environmental" in enabled_features:
            expected_dim += 1
        metrics = [0.7] * expected_dim  # Default better metrics
        
    return np.array(metrics, dtype=np.float32) 
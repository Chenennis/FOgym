"""特征提取功能"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

def extract_generate_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """从生成模块提取关键特征
    
    Args:
        observation: 原始观测向量
        config: 特征配置
        
    Returns:
        提取的特征向量
    """
    features = []
    
    try:
        # 提取时间特征（从one-hot压缩为时段分类）
        if "time" in config["features"]:
            if len(observation) >= 24:
                hour_onehot = observation[:24]
                hour = np.argmax(hour_onehot)
                # 将小时映射到时段（早晨、中午、晚上、深夜）
                time_period = hour // 6  # 0-5, 6-11, 12-17, 18-23
                features.append(time_period / 4.0)  # 归一化到[0,1]
            else:
                logger.warning("观测向量长度不足，无法提取时间特征")
                features.append(0.0)
        
        # 提取用户需求特征
        if "user_demand" in config["features"]:
            # 假设：用户偏好在索引25-28，我们使用它们计算总需求
            if len(observation) >= 29:
                # 从用户偏好计算基本需求（简化计算）
                preference_sum = sum(observation[25:29])
                normalized_demand = min(preference_sum / 2.0, 1.0)  # 归一化
                
                # 添加当前和预测需求
                features.append(normalized_demand)
                features.append(normalized_demand * 1.1)  # 简单预测，假设增加10%
            else:
                logger.warning("观测向量长度不足，无法提取用户需求特征")
                features.extend([0.0, 0.0])
        
        # 提取设备统计特征
        if "device_stats" in config["features"]:
            # 简化：截取30后的部分作为设备状态，计算平均值和其他统计量
            if len(observation) > 30:
                device_states = observation[30:]
                
                # 计算基本统计量
                mean_value = np.mean(device_states)
                max_value = np.max(device_states)
                min_value = np.min(device_states)
                std_value = np.std(device_states)
                median_value = np.median(device_states)
                
                # 归一化，确保结果在[0,1]范围内
                features.extend([
                    min(max(mean_value, 0.0), 1.0),
                    min(max(max_value/10.0, 0.0), 1.0),
                    min(max(min_value+0.5, 0.0), 1.0),
                    min(max(std_value/2.0, 0.0), 1.0),
                    min(max(median_value, 0.0), 1.0)
                ])
            else:
                logger.warning("观测向量长度不足，无法提取设备统计特征")
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    except Exception as e:
        logger.error(f"提取生成模块特征时出错: {e}")
        # 返回所有0的向量作为后备
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
    """从聚合模块提取关键特征
    
    Args:
        observation: 原始聚合信息
        config: 特征配置
        
    Returns:
        提取的特征向量
    """
    features = []
    
    try:
        # 聚合模块可能没有传统的observation，而是DFO/SFO系统
        # 下面是处理这种情况的简化代码
        
        # 提取能量边界信息
        if "energy_bounds" in config["features"]:
            # 假设observation是能量边界信息的某种表示
            if isinstance(observation, dict) and 'energy_min' in observation and 'energy_max' in observation:
                e_min = observation['energy_min']
                e_max = observation['energy_max']
                
                if isinstance(e_min, (list, np.ndarray)) and isinstance(e_max, (list, np.ndarray)):
                    # 计算统计量
                    min_e_min = min(e_min)
                    max_e_min = max(e_min)
                    min_e_max = min(e_max)
                    max_e_max = max(e_max)
                    
                    # 归一化
                    features.extend([
                        min(max((min_e_min + 100) / 200, 0.0), 1.0),
                        min(max((max_e_min + 100) / 200, 0.0), 1.0),
                        min(max((min_e_max) / 200, 0.0), 1.0),
                        min(max((max_e_max) / 200, 0.0), 1.0)
                    ])
                else:
                    features.extend([0.5, 0.5, 0.5, 0.5])  # 默认值
            else:
                # 如果没有能量边界信息，使用默认值
                features.extend([0.5, 0.5, 0.5, 0.5])
            
        # 提取灵活性指标
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
                # 默认灵活性指标
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"提取聚合模块特征时出错: {e}")
        # 返回所有0.5的向量作为后备（中间值）
        expected_dim = 0
        if "energy_bounds" in config["features"]:
            expected_dim += 4
        if "flexibility" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_trading_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """从交易模块提取关键特征
    
    Args:
        observation: 原始交易状态
        config: 特征配置
        
    Returns:
        提取的特征向量
    """
    features = []
    
    try:
        # 提取价格趋势特征
        if "price_trends" in config["features"]:
            if isinstance(observation, dict) and 'prices' in observation:
                prices = observation['prices']
                if len(prices) >= 3:
                    # 计算简单趋势指标
                    current_price = prices[-1]
                    prev_price = prices[-2]
                    earliest_price = prices[0]
                    
                    # 短期趋势（归一化到[-1,1]，再归一化到[0,1]）
                    short_trend = min(max((current_price - prev_price) / max(prev_price, 0.01), -1.0), 1.0)
                    short_trend = (short_trend + 1.0) / 2.0  # 归一化到[0,1]
                    
                    # 长期趋势
                    long_trend = min(max((current_price - earliest_price) / max(earliest_price, 0.01), -1.0), 1.0)
                    long_trend = (long_trend + 1.0) / 2.0  # 归一化到[0,1]
                    
                    # 价格波动性（标准差/均值）
                    volatility = min(np.std(prices) / max(np.mean(prices), 0.01), 1.0)
                    
                    features.extend([short_trend, long_trend, volatility])
                else:
                    features.extend([0.5, 0.5, 0.5])  # 默认中间值
            else:
                # 如果没有价格信息，使用默认值
                features.extend([0.5, 0.5, 0.5])
                
        # 提取交易统计特征
        if "trade_stats" in config["features"]:
            if isinstance(observation, dict) and 'trades' in observation:
                trades = observation['trades']
                
                # 计算成功率
                success_rate = trades.get('success_rate', 0.5)
                
                # 计算成交量
                volume = min(trades.get('volume', 50) / 100.0, 1.0)
                
                # 计算平均价格偏差（实际成交价与目标价格的差异）
                price_deviation = min(max((trades.get('price_deviation', 0) + 0.2) / 0.4, 0.0), 1.0)
                
                # 计算交易频率
                frequency = min(trades.get('frequency', 0.5), 1.0)
                
                features.extend([success_rate, volume, price_deviation, frequency])
            else:
                # 默认交易统计
                features.extend([0.5, 0.5, 0.5, 0.5])
    
    except Exception as e:
        logger.error(f"提取交易模块特征时出错: {e}")
        # 返回所有0.5的向量作为后备（中间值）
        expected_dim = 0
        if "price_trends" in config["features"]:
            expected_dim += 3
        if "trade_stats" in config["features"]:
            expected_dim += 4
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def extract_schedule_features(observation: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """从调度模块提取关键特征
    
    Args:
        observation: 原始调度状态
        config: 特征配置
        
    Returns:
        提取的特征向量
    """
    features = []
    
    try:
        # 提取效率指标
        if "efficiency" in config["features"]:
            if isinstance(observation, dict) and 'efficiency' in observation:
                efficiency = min(max(observation['efficiency'], 0.0), 1.0)
                features.append(efficiency)
            else:
                # 默认效率
                features.append(0.7)  # 乐观的默认值
                
        # 提取成本优化指标
        if "cost_optimization" in config["features"]:
            if isinstance(observation, dict) and 'cost' in observation:
                cost_data = observation['cost']
                
                # 成本优化潜力（实际成本与最优成本的比率）
                optimization_potential = min(max(cost_data.get('potential', 0.5), 0.0), 1.0)
                
                # 成本趋势（最近成本变化的方向）
                # 归一化到[0,1]，0表示成本增加，1表示成本减少
                cost_trend = min(max((cost_data.get('trend', 0) + 1.0) / 2.0, 0.0), 1.0)
                
                features.extend([optimization_potential, cost_trend])
            else:
                # 默认成本指标
                features.extend([0.5, 0.5])
    
    except Exception as e:
        logger.error(f"提取调度模块特征时出错: {e}")
        # 返回所有0.5的向量作为后备（中间值）
        expected_dim = 0
        if "efficiency" in config["features"]:
            expected_dim += 1
        if "cost_optimization" in config["features"]:
            expected_dim += 2
        features = [0.5] * expected_dim
        
    return np.array(features, dtype=np.float32)

def compute_cross_module_correlations(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """计算模块间的相关性特征
    
    Args:
        observations: 各模块的观测字典
        config: 特征配置
        
    Returns:
        相关性特征向量
    """
    correlations = []
    
    try:
        # 时间同步特征
        if all(["generate" in observations, "trading" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # 从生成模块获取时间（假设是24维one-hot的前24个元素）
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) >= 24:
                gen_hour = np.argmax(gen_obs[:24])
                
                # 从交易模块获取时间（假设在字典或者数组的某个位置）
                trade_hour = None
                if isinstance(trade_obs, dict) and 'time' in trade_obs:
                    trade_hour = trade_obs['time'].hour if hasattr(trade_obs['time'], 'hour') else 0
                elif isinstance(trade_obs, np.ndarray) and len(trade_obs) > 0:
                    # 假设第一个元素与时间相关
                    trade_hour = int(trade_obs[0] * 24) if trade_obs[0] <= 1 else 0
                
                if trade_hour is not None:
                    # 计算时间差异指标，归一化到[0,1]
                    # 0表示完全不同步，1表示完全同步
                    time_diff = abs(gen_hour - trade_hour)
                    time_sync = 1.0 - min(time_diff / 12.0, 1.0)  # 最多差12小时视为完全不同步
                    correlations.append(time_sync)
                else:
                    correlations.append(0.5)  # 默认中等同步度
            else:
                correlations.append(0.5)
        else:
            correlations.append(0.5)
            
        # 能源流向量（假设：生成-聚合-交易-调度的能源流）
        if all(["generate" in observations, "trading" in observations]):
            # 简化计算：生成与交易模块之间的能源平衡
            gen_energy = 0.5  # 默认生成能源
            trade_energy = 0.5  # 默认交易能源
            
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            
            # 提取生成模块的能源生成（假设）
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                # 简化：使用设备状态的平均值作为能源生成指标
                gen_energy = min(max(np.mean(gen_obs[30:]) / 2.0, 0.0), 1.0)
            
            # 提取交易模块的能源需求（假设）
            if isinstance(trade_obs, dict) and 'demand' in trade_obs:
                trade_energy = min(max(trade_obs['demand'] / 100.0, 0.0), 1.0)
            
            # 计算能源匹配度（0=严重不匹配，1=完美匹配）
            energy_match = 1.0 - min(abs(gen_energy - trade_energy), 1.0)
            
            # 计算能源流方向（0=消耗>生成，1=生成>消耗）
            energy_direction = 1.0 if gen_energy > trade_energy else 0.0
            
            correlations.extend([energy_match, energy_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # 价值流向量
        # 简化：假设价值流是基于价格和成本的
        if all(["generate" in observations, "trading" in observations, "schedule" in observations]):
            gen_obs = observations["generate"]
            trade_obs = observations["trading"]
            sched_obs = observations["schedule"]
            
            # 获取生成成本（假设）
            gen_cost = 0.5
            if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                # 假设索引24是电价
                gen_cost = min(max(gen_obs[24], 0.0), 1.0)
            
            # 获取交易价格（假设）
            trade_price = 0.5
            if isinstance(trade_obs, dict) and 'price' in trade_obs:
                trade_price = min(max(trade_obs['price'] / 100.0, 0.0), 1.0)
            
            # 获取调度成本（假设）
            sched_cost = 0.5
            if isinstance(sched_obs, dict) and 'cost' in sched_obs:
                if isinstance(sched_obs['cost'], dict) and 'value' in sched_obs['cost']:
                    sched_cost = min(max(sched_obs['cost']['value'] / 100.0, 0.0), 1.0)
            
            # 计算价值流指标
            value_efficiency = min(max(trade_price / (gen_cost + sched_cost + 0.01), 0.0), 1.0)
            value_direction = min(max((trade_price - gen_cost) / max(gen_cost, 0.01), 0.0), 1.0)
            
            correlations.extend([value_efficiency, value_direction])
        else:
            correlations.extend([0.5, 0.5])
            
        # 状态一致性指标
        # 简化：计算各模块状态向量的一致性
        enabled_modules = [
            module for module, data in observations.items() 
            if data is not None and module in config and config[module].get("enabled", True)
        ]
        
        if len(enabled_modules) > 1:
            # 简单的一致性度量：各模块归一化后的状态向量夹角余弦的平均值
            consistency = 0.5  # 默认中等一致性
            
            # 这里简化计算，实际需要基于具体模块状态计算
            correlations.append(consistency)
        else:
            correlations.append(0.5)
    
    except Exception as e:
        logger.error(f"计算模块间相关性时出错: {e}")
        # 默认相关性向量
        correlations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
    return np.array(correlations[:6], dtype=np.float32)  # 最多返回6个相关性特征

def compute_global_metrics(observations: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """计算全局优化指标
    
    Args:
        observations: 各模块的观测字典
        config: 特征配置
        
    Returns:
        全局指标向量
    """
    metrics = []
    
    try:
        global_config = config.get("global", {})
        enabled_features = global_config.get("features", [])
        
        # 系统效率指标
        if "efficiency" in enabled_features:
            # 基于各模块计算综合效率
            efficiency_values = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                # 假设计算生成效率（例如，基于设备状态）
                gen_efficiency = 0.8  # 默认值
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # 简化：使用设备状态的平均值计算效率
                    gen_efficiency = min(max(np.mean(gen_obs[30:]), 0.0), 1.0)
                efficiency_values.append(gen_efficiency)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                # 假设计算交易效率
                trade_efficiency = 0.7  # 默认值
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_efficiency = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                efficiency_values.append(trade_efficiency)
            
            if "schedule" in observations:
                sched_obs = observations["schedule"]
                # 假设计算调度效率
                sched_efficiency = 0.9  # 默认值
                if isinstance(sched_obs, dict) and 'efficiency' in sched_obs:
                    sched_efficiency = min(max(sched_obs['efficiency'], 0.0), 1.0)
                efficiency_values.append(sched_efficiency)
            
            # 计算整体效率
            if efficiency_values:
                system_efficiency = sum(efficiency_values) / len(efficiency_values)
                metrics.append(system_efficiency)
            else:
                metrics.append(0.8)  # 默认较高效率
                
        # 经济性指标
        if "economic" in enabled_features:
            # 基于成本和价格计算经济指标
            economic_score = 0.6  # 默认中等偏上
            
            # 简化：使用生成成本、交易价格和调度成本
            costs = []
            revenues = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 24:
                    # 假设索引24是电价，用作成本指标
                    costs.append(gen_obs[24] * 100)  # 缩放假设
            
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
                economic_score = min(max((profit_margin + 1.0) / 2.0, 0.0), 1.0)  # 归一化到[0,1]
            
            metrics.append(economic_score)
            
        # 可靠性指标
        if "reliability" in enabled_features:
            # 计算系统可靠性指标
            reliability_score = 0.75  # 默认较高可靠性
            
            # 简化：基于设备状态和交易成功率
            reliability_factors = []
            
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 30:
                    # 假设设备状态稳定性反映可靠性
                    device_reliability = 1.0 - min(np.std(gen_obs[30:]) / 2.0, 1.0)
                    reliability_factors.append(device_reliability)
            
            if "trading" in observations:
                trade_obs = observations["trading"]
                if isinstance(trade_obs, dict) and 'trades' in trade_obs:
                    trade_reliability = min(max(trade_obs['trades'].get('success_rate', 0.7), 0.0), 1.0)
                    reliability_factors.append(trade_reliability)
            
            # 计算整体可靠性
            if reliability_factors:
                reliability_score = sum(reliability_factors) / len(reliability_factors)
            
            metrics.append(reliability_score)
            
        # 环保性指标
        if "environmental" in enabled_features:
            # 计算环境影响指标
            environmental_score = 0.7  # 默认较好
            
            # 简化：基于可再生能源使用比例
            if "generate" in observations:
                gen_obs = observations["generate"]
                if isinstance(gen_obs, np.ndarray) and len(gen_obs) > 25:
                    # 假设可从用户偏好的环保性推断
                    environmental_score = min(max(gen_obs[28], 0.0), 1.0)  # 假设索引28是环保性偏好
            
            metrics.append(environmental_score)
    
    except Exception as e:
        logger.error(f"计算全局指标时出错: {e}")
        # 默认全局指标
        expected_dim = 0
        if "efficiency" in enabled_features:
            expected_dim += 1
        if "economic" in enabled_features:
            expected_dim += 1
        if "reliability" in enabled_features:
            expected_dim += 1
        if "environmental" in enabled_features:
            expected_dim += 1
        metrics = [0.7] * expected_dim  # 默认较好的指标
        
    return np.array(metrics, dtype=np.float32) 
"""
动态观测质量管理模块
实现观测质量的动态变化机制，包括网络状况模拟、质量评估等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

class NetworkCondition(Enum):
    """网络状况枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class ObservationQualityMetrics:
    """观测质量指标"""
    accuracy: float = 1.0          # 准确度 [0,1]
    completeness: float = 1.0      # 完整性 [0,1]
    timeliness: float = 1.0        # 及时性 [0,1]
    reliability: float = 1.0       # 可靠性 [0,1]
    consistency: float = 1.0       # 一致性 [0,1]
    
    def overall_quality(self) -> float:
        """计算总体质量得分"""
        return float(np.mean([
            self.accuracy,
            self.completeness,
            self.timeliness,
            self.reliability,
            self.consistency
        ]))

class DynamicObservationQuality:
    """动态观测质量管理器"""
    
    def __init__(self):
        # 网络状况历史
        self.network_history: List[NetworkCondition] = []
        
        # Manager间的通信质量
        self.communication_quality: Dict[str, float] = {}
        
        # 观测质量历史
        self.quality_history: Dict[str, List[ObservationQualityMetrics]] = {}
        
        # 时间步数
        self.current_step = 0
        
        # 质量变化参数
        self.quality_params = {
            'network_volatility': 0.1,      # 网络状况波动性
            'degradation_rate': 0.02,       # 质量降级速率
            'recovery_rate': 0.05,          # 质量恢复速率
            'baseline_quality': 0.85,       # 基线质量
            'min_quality': 0.3,             # 最低质量
            'max_quality': 1.0,             # 最高质量
        }
    
    def update_network_condition(self) -> NetworkCondition:
        """
        更新网络状况
        基于马尔可夫链模拟网络状况变化
        """
        # 获取当前网络状况
        if len(self.network_history) == 0:
            current_condition = NetworkCondition.GOOD
        else:
            current_condition = self.network_history[-1]
        
        # 网络状况转移概率矩阵
        transition_probs = {
            NetworkCondition.EXCELLENT: {
                NetworkCondition.EXCELLENT: 0.7,
                NetworkCondition.GOOD: 0.25,
                NetworkCondition.FAIR: 0.05,
                NetworkCondition.POOR: 0.0,
                NetworkCondition.CRITICAL: 0.0
            },
            NetworkCondition.GOOD: {
                NetworkCondition.EXCELLENT: 0.1,
                NetworkCondition.GOOD: 0.6,
                NetworkCondition.FAIR: 0.25,
                NetworkCondition.POOR: 0.05,
                NetworkCondition.CRITICAL: 0.0
            },
            NetworkCondition.FAIR: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.2,
                NetworkCondition.FAIR: 0.5,
                NetworkCondition.POOR: 0.25,
                NetworkCondition.CRITICAL: 0.05
            },
            NetworkCondition.POOR: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.05,
                NetworkCondition.FAIR: 0.25,
                NetworkCondition.POOR: 0.6,
                NetworkCondition.CRITICAL: 0.1
            },
            NetworkCondition.CRITICAL: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.0,
                NetworkCondition.FAIR: 0.1,
                NetworkCondition.POOR: 0.4,
                NetworkCondition.CRITICAL: 0.5
            }
        }
        
        # 基于转移概率选择下一个状态
        probs = transition_probs[current_condition]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # 使用随机数选择状态
        random_value = np.random.random()
        cumulative_prob = 0.0
        next_condition = states[0]  # 默认值
        
        for state, prob in zip(states, probabilities):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                next_condition = state
                break
        self.network_history.append(next_condition)
        
        # 限制历史长度
        if len(self.network_history) > 100:
            self.network_history = self.network_history[-100:]
        
        return next_condition
    
    def calculate_communication_quality(self, manager_i: str, manager_j: str) -> float:
        """
        计算Manager间的通信质量
        基于距离、网络状况、负载等因素
        """
        # 获取当前网络状况
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        
        # 网络状况对通信质量的影响
        network_impact = {
            NetworkCondition.EXCELLENT: 1.0,
            NetworkCondition.GOOD: 0.9,
            NetworkCondition.FAIR: 0.7,
            NetworkCondition.POOR: 0.5,
            NetworkCondition.CRITICAL: 0.3
        }
        
        # 基础通信质量（模拟Manager间的地理距离和基础设施）
        manager_ids = sorted([manager_i, manager_j])
        manager_pair_key = f"{manager_ids[0]}_{manager_ids[1]}"
        
        if manager_pair_key not in self.communication_quality:
            # 初始化通信质量（基于Manager ID差异模拟距离）
            id_diff = abs(int(manager_i.split('_')[-1]) - int(manager_j.split('_')[-1]))
            distance_factor = max(0.5, 1.0 - id_diff * 0.1)  # 距离越远质量越低
            
            # 添加随机性
            random_factor = np.random.uniform(0.8, 1.2)
            
            base_quality = distance_factor * random_factor
            self.communication_quality[manager_pair_key] = np.clip(base_quality, 0.3, 1.0)
        
        # 获取基础质量
        base_quality = self.communication_quality[manager_pair_key]
        
        # 应用网络状况影响
        current_quality = base_quality * network_impact[current_network]
        
        # 添加时间变化（模拟网络拥塞）
        time_factor = 1.0 + 0.1 * math.sin(self.current_step * 0.1)  # 周期性变化
        
        final_quality = current_quality * time_factor
        
        return np.clip(final_quality, 0.1, 1.0)
    
    def calculate_observation_quality(self, manager_id: str, 
                                    other_manager_ids: List[str]) -> ObservationQualityMetrics:
        """
        计算Manager的观测质量指标
        """
        # 准确度：基于网络状况和噪声水平
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        network_accuracy = {
            NetworkCondition.EXCELLENT: 0.98,
            NetworkCondition.GOOD: 0.95,
            NetworkCondition.FAIR: 0.88,
            NetworkCondition.POOR: 0.75,
            NetworkCondition.CRITICAL: 0.60
        }
        accuracy = network_accuracy[current_network]
        
        # 完整性：基于与其他Manager的通信质量
        communication_qualities = []
        for other_id in other_manager_ids:
            if other_id != manager_id:
                comm_quality = self.calculate_communication_quality(manager_id, other_id)
                communication_qualities.append(comm_quality)
        
        if communication_qualities:
            completeness = np.mean(communication_qualities)
        else:
            completeness = 1.0
        
        # 及时性：基于网络延迟和系统负载
        timeliness = network_accuracy[current_network] * np.random.uniform(0.9, 1.0)
        
        # 可靠性：基于历史观测质量的一致性
        if manager_id in self.quality_history and len(self.quality_history[manager_id]) > 0:
            recent_qualities = [q.overall_quality() for q in self.quality_history[manager_id][-10:]]
            quality_variance = np.var(recent_qualities)
            reliability = max(0.5, 1.0 - quality_variance * 2)  # 方差越大可靠性越低
        else:
            reliability = 0.9
        
        # 一致性：基于观测值的时间一致性
        consistency = np.random.uniform(0.85, 0.98)  # 模拟数据一致性
        
        # 应用随机波动
        volatility = self.quality_params['network_volatility']
        accuracy *= np.random.uniform(1 - volatility, 1 + volatility)
        completeness *= np.random.uniform(1 - volatility, 1 + volatility)
        timeliness *= np.random.uniform(1 - volatility, 1 + volatility)
        reliability *= np.random.uniform(1 - volatility, 1 + volatility)
        consistency *= np.random.uniform(1 - volatility, 1 + volatility)
        
        # 确保在合理范围内
        quality_metrics = ObservationQualityMetrics(
            accuracy=np.clip(accuracy, 0.3, 1.0),
            completeness=np.clip(completeness, 0.3, 1.0),
            timeliness=np.clip(timeliness, 0.3, 1.0),
            reliability=np.clip(reliability, 0.3, 1.0),
            consistency=np.clip(consistency, 0.3, 1.0)
        )
        
        return quality_metrics
    
    def apply_quality_degradation(self, observation: np.ndarray, 
                                quality_metrics: ObservationQualityMetrics) -> np.ndarray:
        """
        根据质量指标对观测应用降级效果
        """
        degraded_obs = observation.copy()
        
        # 1. 准确度影响：添加噪声
        if quality_metrics.accuracy < 1.0:
            noise_std = (1.0 - quality_metrics.accuracy) * 0.2
            noise = np.random.normal(0, noise_std, size=observation.shape)
            degraded_obs += noise
        
        # 2. 完整性影响：随机置零部分观测
        if quality_metrics.completeness < 1.0:
            missing_prob = (1.0 - quality_metrics.completeness) * 0.3
            missing_mask = np.random.random(observation.shape) < missing_prob
            degraded_obs[missing_mask] = 0.0
        
        # 3. 及时性影响：使用历史观测值
        if quality_metrics.timeliness < 0.9:
            # 这里简化处理，在实际环境中会使用历史观测
            delay_factor = 1.0 - quality_metrics.timeliness
            degraded_obs *= (1.0 - delay_factor * 0.1)
        
        # 4. 可靠性影响：添加系统性偏差
        if quality_metrics.reliability < 0.9:
            bias = (1.0 - quality_metrics.reliability) * 0.1
            degraded_obs += bias
        
        # 5. 一致性影响：添加随机扰动
        if quality_metrics.consistency < 0.9:
            inconsistency = (1.0 - quality_metrics.consistency) * 0.15
            perturbation = np.random.uniform(-inconsistency, inconsistency, size=observation.shape)
            degraded_obs += perturbation
        
        return degraded_obs
    
    def update_quality_history(self, manager_id: str, quality_metrics: ObservationQualityMetrics):
        """更新观测质量历史"""
        if manager_id not in self.quality_history:
            self.quality_history[manager_id] = []
        
        self.quality_history[manager_id].append(quality_metrics)
        
        # 限制历史长度
        if len(self.quality_history[manager_id]) > 50:
            self.quality_history[manager_id] = self.quality_history[manager_id][-50:]
    
    def step(self):
        """执行一步更新"""
        self.current_step += 1
        self.update_network_condition()
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量报告"""
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        
        # 计算平均质量
        average_qualities = {}
        for manager_id, qualities in self.quality_history.items():
            if qualities:
                avg_quality = np.mean([q.overall_quality() for q in qualities[-10:]])
                average_qualities[manager_id] = avg_quality
        
        return {
            'current_network_condition': current_network.value,
            'network_history_length': len(self.network_history),
            'average_manager_qualities': average_qualities,
            'communication_pairs': len(self.communication_quality),
            'current_step': self.current_step,
            'quality_parameters': self.quality_params
        }
    
    def reset(self):
        """重置质量管理器"""
        self.network_history.clear()
        self.communication_quality.clear()
        self.quality_history.clear()
        self.current_step = 0 
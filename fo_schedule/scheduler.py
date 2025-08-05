import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import sys
import random
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlexScheduler")

# 添加项目根目录到系统路径以便导入原始模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from fo_aggregate.manager import Manager, User, Device
from fo_aggregate.aggregator import AggregatedFlexOffer
from fo_trading.pool import TradingPool, WeatherModel, DemandModel, Trade

# ========== 新增：FO分解算法数据结构 ==========

@dataclass
class DisaggregationRequest:
    """FO分解请求数据结构"""
    aggregated_result: Any  # 聚合结果（AggregatedResult或AggregatedFlexOffer）
    original_data: List[Dict]  # 原始数据列表
    total_energy: float  # 总能量
    time_step: int  # 时间步
    metadata: Dict[str, Any]  # 额外元数据
    
    def __post_init__(self):
        """验证输入数据"""
        # 检查原始数据列表
        if self.original_data is None:
            logger.warning("原始数据列表为None，初始化为空列表")
            self.original_data = []
        
        # 检查总能量
        if self.total_energy is None:
            logger.warning("总能量为None，设置为0")
            self.total_energy = 0.0
        elif self.total_energy < 0:
            logger.warning(f"总能量为负值({self.total_energy})，设置为0")
            self.total_energy = 0.0
        
        # 检查时间步
        if self.time_step is None:
            logger.warning("时间步为None，设置为0")
            self.time_step = 0
        elif self.time_step < 0:
            logger.warning(f"时间步为负值({self.time_step})，设置为0")
            self.time_step = 0
        
        # 确保元数据是字典
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class DisaggregationResult:
    """FO分解结果数据结构"""
    disaggregated_data: List[Dict]  # 分解后的数据列表
    algorithm_used: str  # 使用的算法名称
    allocation_ratios: List[float]  # 分配比例列表
    total_allocated_energy: float  # 总分配能量
    metadata: Dict[str, Any]  # 元数据
    
    def __post_init__(self):
        """验证结果数据"""
        if len(self.disaggregated_data) != len(self.allocation_ratios):
            raise ValueError("分解数据和分配比例数量不匹配")
        if self.total_allocated_energy < 0:
            raise ValueError("总分配能量不能为负数")

# ========== 新增：分解算法抽象基类 ==========

class DisaggregationAlgorithm(ABC):
    """FO分解算法抽象基类"""
    
    def __init__(self, algorithm_name: str):
        """
        初始化分解算法
        
        Args:
            algorithm_name: 算法名称
        """
        self.algorithm_name = algorithm_name
        self.total_requests = 0
        self.total_energy_processed = 0.0
        self.performance_metrics = {}
    
    @abstractmethod
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        执行分解操作的抽象方法
        
        Args:
            request: 分解请求
            
        Returns:
            DisaggregationResult: 分解结果
        """
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "name": self.algorithm_name,
            "total_requests": self.total_requests,
            "total_energy_processed": self.total_energy_processed,
            "performance_metrics": self.performance_metrics
        }
    
    def _validate_request(self, request: DisaggregationRequest) -> bool:
        """验证分解请求"""
        if not isinstance(request, DisaggregationRequest):
            raise ValueError("无效的分解请求类型")
        return True
    
    def _update_metrics(self, request: DisaggregationRequest, result: DisaggregationResult):
        """更新性能指标"""
        self.total_requests += 1
        self.total_energy_processed += request.total_energy
        
        # 计算分配效率
        efficiency = result.total_allocated_energy / request.total_energy if request.total_energy > 0 else 0
        if 'allocation_efficiency' not in self.performance_metrics:
            self.performance_metrics['allocation_efficiency'] = []
        self.performance_metrics['allocation_efficiency'].append(efficiency)

# ========== 新增：平均分解算法实现 ==========

class AverageDisaggregationAlgorithm(DisaggregationAlgorithm):
    """平均分解算法：E_i = E/N"""
    
    def __init__(self):
        super().__init__("average")
        logger.info("初始化平均分解算法")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        执行平均分解：将总能量平均分配给所有参与者
        
        Args:
            request: 分解请求
            
        Returns:
            DisaggregationResult: 分解结果
        """
        self._validate_request(request)
        
        logger.info(f"开始平均分解，原始数据数量: {len(request.original_data)}，总能量: {request.total_energy:.2f}")
        
        # 计算平均分配能量：E_i = E/N
        num_participants = len(request.original_data)
        
        # 检查参与者数量，避免除零错误
        if num_participants == 0:
            logger.error("无法执行平均分解：没有参与者")
            # 返回空结果
            return DisaggregationResult(
                disaggregated_data=[],
                algorithm_used=self.algorithm_name,
                allocation_ratios=[],
                total_allocated_energy=0.0,
                metadata={
                    'error': 'no_participants',
                    'time_step': request.time_step
                }
            )
            
        average_energy = request.total_energy / num_participants
        
        # 创建分解结果
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # 复制原始数据
            new_item = copy.deepcopy(item)
            
            # 分配平均能量
            new_item['allocated_energy'] = average_energy
            new_item['allocation_method'] = 'average'
            # 避免除零错误
            original_energy = item.get('energy', 0)
            new_item['allocation_ratio'] = average_energy / original_energy if original_energy > 0 else 1.0
            
            disaggregated_data.append(new_item)
            # 避免除零错误
            allocation_ratio = average_energy / request.total_energy if request.total_energy > 0 else 1.0 / num_participants
            allocation_ratios.append(allocation_ratio)
        
        # 创建结果对象
        result = DisaggregationResult(
            disaggregated_data=disaggregated_data,
            algorithm_used=self.algorithm_name,
            allocation_ratios=allocation_ratios,
            total_allocated_energy=request.total_energy,
            metadata={
                'average_energy_per_participant': average_energy,
                'num_participants': num_participants,
                'time_step': request.time_step
            }
        )
        
        # 更新性能指标
        self._update_metrics(request, result)
        
        logger.info(f"平均分解完成，每个参与者分配: {average_energy:.2f} kWh")
        return result

# ========== 新增：等比例分解算法实现 ==========

class ProportionalDisaggregationAlgorithm(DisaggregationAlgorithm):
    """等比例分解算法：E_i = (w_i/W) * E"""
    
    def __init__(self, weight_key: str = 'energy'):
        """
        初始化等比例分解算法
        
        Args:
            weight_key: 用于计算权重的键名，默认为'energy'
        """
        super().__init__("proportional")
        self.weight_key = weight_key
        logger.info(f"初始化等比例分解算法，权重键: {weight_key}")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        执行等比例分解：根据权重按比例分配能量
        
        Args:
            request: 分解请求
            
        Returns:
            DisaggregationResult: 分解结果
        """
        self._validate_request(request)
        
        logger.info(f"开始等比例分解，原始数据数量: {len(request.original_data)}，总能量: {request.total_energy:.2f}")
        
        # 计算总权重：W = Σw_i
        total_weight = sum(item.get(self.weight_key, 1.0) for item in request.original_data)
        
        if total_weight <= 0:
            logger.warning("总权重为0，回退到平均分配")
            # 如果总权重为0，回退到平均分配
            average_algo = AverageDisaggregationAlgorithm()
            return average_algo.disaggregate(request)
        
        # 创建分解结果
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # 复制原始数据
            new_item = copy.deepcopy(item)
            
            # 计算权重比例：w_i/W
            weight = item.get(self.weight_key, 1.0)
            weight_ratio = weight / total_weight
            
            # 按比例分配能量：E_i = (w_i/W) * E
            allocated_energy = weight_ratio * request.total_energy
            
            new_item['allocated_energy'] = allocated_energy
            new_item['allocation_method'] = 'proportional'
            new_item['weight_used'] = weight
            new_item['weight_ratio'] = weight_ratio
            new_item['allocation_ratio'] = allocated_energy / item.get('energy', 1.0) if item.get('energy', 0) > 0 else 1.0
            
            disaggregated_data.append(new_item)
            allocation_ratios.append(weight_ratio)
        
        # 创建结果对象
        result = DisaggregationResult(
            disaggregated_data=disaggregated_data,
            algorithm_used=self.algorithm_name,
            allocation_ratios=allocation_ratios,
            total_allocated_energy=request.total_energy,
            metadata={
                'total_weight': total_weight,
                'weight_key_used': self.weight_key,
                'time_step': request.time_step
            }
        )
        
        # 更新性能指标
        self._update_metrics(request, result)
        
        logger.info(f"等比例分解完成，总权重: {total_weight:.2f}")
        return result

# ========== 新增：分解算法工厂 ==========

class DisaggregationAlgorithmFactory:
    """FO分解算法工厂"""
    
    _algorithms = {}
    _initialized = False
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type, **kwargs):
        """
        注册分解算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
            **kwargs: 算法初始化参数
        """
        cls._algorithms[name] = {
            'class': algorithm_class,
            'kwargs': kwargs
        }
        logger.info(f"已注册FO分解算法: {name}")
    
    @classmethod
    def create_algorithm(cls, name: str, **override_kwargs) -> DisaggregationAlgorithm:
        """
        创建分解算法实例
        
        Args:
            name: 算法名称
            **override_kwargs: 覆盖默认参数
            
        Returns:
            DisaggregationAlgorithm: 算法实例
        """
        if name not in cls._algorithms:
            raise ValueError(f"未知的分解算法: {name}")
        
        algo_info = cls._algorithms[name]
        algo_class = algo_info['class']
        
        # 合并参数
        kwargs = algo_info['kwargs'].copy()
        kwargs.update(override_kwargs)
        
        return algo_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """获取可用算法列表"""
        return list(cls._algorithms.keys())
    
    @classmethod
    def initialize_default_algorithms(cls):
        """初始化默认算法"""
        if cls._initialized:
            return
        
        # 注册平均分解算法
        cls.register_algorithm("average", AverageDisaggregationAlgorithm)
        
        # 注册等比例分解算法
        cls.register_algorithm("proportional", ProportionalDisaggregationAlgorithm, weight_key='energy')
        cls.register_algorithm("equal_proportion", ProportionalDisaggregationAlgorithm, weight_key='energy')
        
        # 为了兼容性，注册原有的方法名
        cls.register_algorithm("equal", AverageDisaggregationAlgorithm)
        cls.register_algorithm("priority", ProportionalDisaggregationAlgorithm, weight_key='priority')
        
        cls._initialized = True
        logger.info("默认FO分解算法初始化完成")

# 初始化默认算法
DisaggregationAlgorithmFactory.initialize_default_algorithms()

# 定义一个简单的FlexOffer类作为过渡
class FlexOffer:
    """简化的FlexOffer类，用于兼容现有代码"""
    def __init__(self, resource_id=None, resource_type=None, location=None, 
                 time_horizon=24, time_interval=1, quantity=0, price=0, 
                 time_window=None, device_type=None, constraints=None):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.location = location
        self.time_horizon = time_horizon
        self.time_interval = time_interval
        self.quantity = quantity
        self.price = price
        self.time_window = time_window or (0, 24)
        self.device_type = device_type
        self.constraints = constraints or {}
        self.power_profile = np.zeros((time_horizon, 2))
        self.baseline_profile = np.zeros(time_horizon)
        self.reliability = 1.0
    
    def set_power_profile(self, profile):
        self.power_profile = profile
    
    def set_baseline_profile(self, profile):
        self.baseline_profile = profile
    
    def set_reliability(self, reliability):
        self.reliability = reliability

# 定义一个简单的FlexOfferManager类作为过渡
class FlexOfferManager:
    """简化的FlexOfferManager类，用于兼容现有代码"""
    def __init__(self, manager_id=None, location=None):
        self.manager_id = manager_id
        self.location = location
        self.offers = []
    
    def add_offer(self, offer):
        self.offers.append(offer)

class FlexOfferDisaggregator:
    """聚合FlexOffer分解器，将聚合的FlexOffer分解回原始FlexOffer"""
    
    def __init__(self, time_horizon: int = 24):
        """
        初始化分解器
        
        Args:
            time_horizon: 时间范围
        """
        self.time_horizon = time_horizon
    
    def disaggregate(self, 
                     aggregated_offer: FlexOffer, 
                     original_offers: List[FlexOffer]) -> List[FlexOffer]:
        """
        分解聚合的FlexOffer
        
        Args:
            aggregated_offer: 聚合的FlexOffer
            original_offers: 原始FlexOffer列表
            
        Returns:
            List[FlexOffer]: 分解后的FlexOffer列表
        """
        if not original_offers:
            raise ValueError("No original offers to disaggregate to")
        
        # 创建分解后的FlexOffer列表
        disaggregated_offers = []
        
        # 总功率和基线
        total_min_power = np.zeros(self.time_horizon)
        total_max_power = np.zeros(self.time_horizon)
        total_baseline = np.zeros(self.time_horizon)
        
        for fo in original_offers:
            total_min_power += fo.power_profile[:, 0]
            total_max_power += fo.power_profile[:, 1]
            total_baseline += fo.baseline_profile
        
        # 计算聚合FlexOffer的实际功率
        aggregated_power = aggregated_offer.baseline_profile
        
        # 分配功率
        for fo in original_offers:
            # 创建新的FlexOffer
            new_fo = FlexOffer(
                resource_id=fo.resource_id,
                resource_type=fo.resource_type,
                location=fo.location,
                time_horizon=fo.time_horizon,
                time_interval=fo.time_interval
            )
            
            # 复制功率范围
            new_fo.set_power_profile(fo.power_profile.copy())
            
            # 按原始贡献比例分配功率
            baseline_ratio = np.zeros(self.time_horizon)
            for t in range(self.time_horizon):
                if total_baseline[t] > 0:
                    baseline_ratio[t] = fo.baseline_profile[t] / total_baseline[t]
                else:
                    baseline_ratio[t] = 1.0 / len(original_offers)
            
            # 计算新的基线
            new_baseline = aggregated_power * baseline_ratio
            new_fo.set_baseline_profile(new_baseline)
            
            # 设置可靠性
            new_fo.set_reliability(fo.reliability)
            
            disaggregated_offers.append(new_fo)
        
        return disaggregated_offers


class UserScheduler:
    """用户调度器，根据用户需求分配能源"""
    
    def __init__(self, 
                 num_users: int = 20,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1):
        """
        初始化调度器
        
        Args:
            num_users: 用户数量
            time_horizon: 时间范围(小时)
            time_steps_per_hour: 每小时的时间步数
        """
        self.num_users = num_users
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        
        # 用户需求
        self.user_demands = np.zeros((num_users, self.total_steps))
        
        # 用户分配
        self.user_allocations = np.zeros((num_users, self.total_steps))
        
        # 用户能源来源
        self.user_sources = {}
        
        # 用户配置(可以指定用户优先级、偏好等)
        self.user_configs = [
            {'id': i, 'priority': random.uniform(0, 1), 'preferences': {}} 
            for i in range(num_users)
        ]
        
        logger.info(f"初始化用户调度器，用户数量: {num_users}，时间范围: {time_horizon}小时")
    
    def set_user_demands(self, demands: np.ndarray):
        """
        设置用户需求
        
        Args:
            demands: 用户需求，维度为[num_users, total_steps]
        """
        assert demands.shape == (self.num_users, self.total_steps), f"需求维度不匹配: {demands.shape} vs {(self.num_users, self.total_steps)}"
        self.user_demands = demands
        # 移除重复的日志输出，由上层ScheduleManager统一输出
    
    def get_user_demand(self, user_id: int, step: int) -> float:
        """
        获取特定用户在特定时间的需求
        
        Args:
            user_id: 用户ID
            step: 时间步
            
        Returns:
            float: A用户需求
        """
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(f"用户ID {user_id} 超出范围 [0, {self.num_users-1}]")
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"时间步 {step} 超出范围 [0, {self.total_steps-1}]")
        
        return self.user_demands[user_id, step]
    
    def schedule(self, 
                energy_resources: List[Dict], 
                step: int,
                method: str = 'priority') -> Dict[int, List[Dict]]:
        """
        根据用户需求分配能源资源
        
        Args:
            energy_resources: 能源资源列表，每个字典包含能源配置信息
            step: 当前时间步
            method: 分配方法，可选'priority'、'fairness'、'cost'
            
        Returns:
            Dict[int, List[Dict]]: 用户分配结果，键为用户ID，值为分配的资源列表
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"时间步 {step} 超出范围 [0, {self.total_steps-1}]")
        
        # 获取当前时间步的用户需求
        current_demands = self.user_demands[:, step].copy()
        
        # 按照选择的方法对用户进行排序
        if method == 'priority':
            # 按优先级排序（优先级高的用户先分配）
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: self.user_configs[i]['priority'],
                reverse=True
            )
        elif method == 'fairness':
            # 按照历史满足率排序（满足率低的用户先分配）
            satisfaction_rates = []
            for user_id in range(self.num_users):
                total_demand = np.sum(self.user_demands[user_id, :step+1])
                total_allocation = np.sum(self.user_allocations[user_id, :step+1])
                
                if total_demand > 0:
                    rate = total_allocation / total_demand
                else:
                    rate = 1.0
                
                satisfaction_rates.append(rate)
            
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: satisfaction_rates[i]
            )
        elif method == 'cost':
            # 按需求排序（需求大的用户先分配，假设可以获得规模效应）
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: current_demands[i],
                reverse=True
            )
        else:
            # 默认按用户ID排序
            sorted_user_indices = list(range(self.num_users))
        
        # 筛选有需求的用户
        sorted_user_indices = [i for i in sorted_user_indices if current_demands[i] > 0]
        
        # 统计可用能源总量
        total_available_energy = sum(item.get('allocated_energy', 0) for item in energy_resources)
        
        # 统计当前总需求
        total_demand = sum(current_demands)
        
        logger.info(f"时间步 {step}: 用户数量={len(sorted_user_indices)}, "
                   f"总需求={total_demand:.2f} kWh, 可用能源={total_available_energy:.2f} kWh")
        
        # 分配结果
        allocations = {user_id: [] for user_id in range(self.num_users)}
        
        # 优先分配能源资源
        for user_id in sorted_user_indices:
            user_demand = current_demands[user_id]
            
            # 如果用户没有需求，跳过
            if user_demand <= 0:
                continue
            
            # 为用户分配资源
            remaining_demand = user_demand
            
            for resource in energy_resources:
                # 检查资源是否还有能量
                available_energy = resource.get('allocated_energy', 0)
                if available_energy <= 0:
                    continue
                
                # 分配量 = min(用户剩余需求, 可用能量)
                allocation_amount = min(remaining_demand, available_energy)
                
                if allocation_amount > 0:
                    # 更新资源的可用能量
                    resource['allocated_energy'] -= allocation_amount
                    
                    # 更新用户剩余需求
                    remaining_demand -= allocation_amount
                    
                    # 记录分配结果
                    allocation = {
                        'resource_id': resource.get('resource_id', ''),
                        'energy_type': resource.get('energy_type', ''),
                        'amount': allocation_amount,
                        'price': resource.get('price', 0.0)
                    }
                    
                    allocations[user_id].append(allocation)
                    
                    # 更新用户分配记录
                    self.user_allocations[user_id, step] += allocation_amount
                
                # 如果用户需求已满足，结束循环
                if remaining_demand <= 0:
                    break
        
        # 更新用户能源来源记录
        self.user_sources[step] = allocations
        
        # 计算能源资源利用率
        total_allocated = sum(self.user_allocations[:, step])
        utilization_rate = total_allocated / total_available_energy if total_available_energy > 0 else 0.0
        
        logger.info(f"时间步 {step}: 分配完成, 总分配={total_allocated:.2f} kWh, "
                   f"资源利用率={utilization_rate*100:.2f}%")
        
        return allocations
    
    def get_user_satisfaction(self, step: int) -> np.ndarray:
        """
        获取用户满意度（需求满足率）
        
        Args:
            step: 时间步
            
        Returns:
            np.ndarray: 用户满意度，取值范围[0,1]
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"时间步 {step} 超出范围 [0, {self.total_steps-1}]")
        
        # 计算用户满意度
        satisfaction = np.zeros(self.num_users)
        for user_id in range(self.num_users):
            user_demand = self.user_demands[user_id, step]
            if user_demand > 0:
                satisfaction[user_id] = min(1.0, self.user_allocations[user_id, step] / user_demand)
            else:
                satisfaction[user_id] = 1.0  # 没有需求，默认满意
        
        return satisfaction
    
    def get_overall_satisfaction(self) -> float:
        """
        获取整体满意度（总需求满足率）
        
        Returns:
            float: 整体满意度，取值范围[0,1]
        """
        total_demand = np.sum(self.user_demands)
        total_allocation = np.sum(self.user_allocations)
        
        if total_demand > 0:
            return float(min(1.0, total_allocation / total_demand))
        else:
            return 1.0
    
    def visualize_allocation(self, step: Optional[int] = None, save_path: Optional[str] = None):
        """
        可视化用户能源分配情况
        
        Args:
            step: 时间步，如果为None则显示所有时间步的总分配
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(12, 6))
        
        if step is not None:
            if step < 0 or step >= self.total_steps:
                raise ValueError(f"时间步 {step} 超出范围 [0, {self.total_steps-1}]")
            
            # 显示特定时间步的分配
            demands = self.user_demands[:, step]
            allocations = self.user_allocations[:, step]
            
            # 计算满意度
            satisfaction = self.get_user_satisfaction(step)
            
            # 设置x轴和图表
            x = np.arange(self.num_users)
            width = 0.4
            
            # 绘制需求和分配
            plt.bar(x - width/2, demands, width, label='需求')
            plt.bar(x + width/2, allocations, width, label='分配')
            
            # 绘制满意度线
            plt.plot(x, satisfaction, 'r-', label='满意度')
            
            plt.xlabel('用户ID')
            plt.ylabel('能源 (kWh)')
            plt.title(f'时间步 {step} 的用户能源分配')
            plt.xticks(x)
            plt.legend()
            
        else:
            # 显示所有时间步的总分配
            total_demands = np.sum(self.user_demands, axis=1)
            total_allocations = np.sum(self.user_allocations, axis=1)
            
            # 计算整体满意度
            satisfaction = []
            for user_id in range(self.num_users):
                if total_demands[user_id] > 0:
                    satisfaction.append(min(1.0, total_allocations[user_id] / total_demands[user_id]))
                else:
                    satisfaction.append(1.0)
            
            # 设置x轴和图表
            x = np.arange(self.num_users)
            width = 0.4
            
            # 绘制总需求和分配
            plt.bar(x - width/2, total_demands, width, label='总需求')
            plt.bar(x + width/2, total_allocations, width, label='总分配')
            
            # 绘制满意度线
            plt.plot(x, satisfaction, 'r-', label='整体满意度')
            
            plt.xlabel('用户ID')
            plt.ylabel('能源 (kWh)')
            plt.title('所有时间的用户能源分配总量')
            plt.xticks(x)
            plt.legend()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def update_cumulative_demands(self, cumulative_demands, timestep):
        """更新累积需求状态"""
        try:
            if cumulative_demands.shape[0] == self.num_users:
                # 更新到当前时间步的累积需求
                if hasattr(self, 'cumulative_user_demands'):
                    # 更新已有的累积需求
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                else:
                    # 初始化累积需求
                    self.cumulative_user_demands = np.zeros((self.num_users, self.total_steps))
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                
                logger.debug(f"UserScheduler累积需求已更新到时间步 {timestep}")  # 改为DEBUG级别避免重复
            else:
                logger.warning(f"累积需求数据维度不匹配: 期望 {self.num_users} 用户，实际 {cumulative_demands.shape[0]} 用户")
        except Exception as e:
            logger.error(f"更新累积需求时出错: {e}")


class ScheduleManager:
    """调度管理器，协调能源资源的分解和用户调度"""
    
    def __init__(self, 
                 managers: List[Manager],
                 trading_pool: TradingPool,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1,
                 disaggregation_algorithm: str = 'proportional'):
        """
        初始化调度管理器
        
        Args:
            managers: Manager列表
            trading_pool: 交易池
            time_horizon: 时间范围(小时)
            time_steps_per_hour: 每小时的时间步数
            disaggregation_algorithm: 分解算法，可选'average'、'proportional'、'equal_proportion'
        """
        self.managers = managers
        self.trading_pool = trading_pool
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        self.disaggregation_algorithm = disaggregation_algorithm
        
        # 创建分解器（使用新的算法架构）
        self.disaggregator = AggregatedResultDisaggregator(
            time_horizon=time_horizon,
            default_algorithm=disaggregation_algorithm
        )
        
        # 创建用户调度器(每个管理器一个，根据实际用户数量)
        self.user_schedulers = {}
        for manager in managers:
            actual_users = len(manager.users)  # 使用实际用户数量
            scheduler = UserScheduler(
                num_users=actual_users,
                time_horizon=time_horizon,
                time_steps_per_hour=time_steps_per_hour
            )
            self.user_schedulers[manager.manager_id] = scheduler
            logger.info(f"为Manager {manager.manager_id} 创建用户调度器，用户数量: {actual_users}")
        
        # 用户需求数据
        self.user_demands = None
        
        # 满意度历史
        self.satisfaction_history = []
        
        # 交易历史缓存
        self.processed_trades = set()
        
        logger.info(f"初始化调度管理器，管理器数量: {len(managers)}，时间范围: {time_horizon}小时，分解算法: {disaggregation_algorithm}")
        logger.info(f"可用分解算法: {self.disaggregator.get_available_algorithms()}")
    
    def set_disaggregation_algorithm(self, algorithm_name: str):
        """
        设置分解算法
        
        Args:
            algorithm_name: 算法名称
        """
        self.disaggregation_algorithm = algorithm_name
        self.disaggregator.set_default_algorithm(algorithm_name)
        logger.info(f"已切换分解算法为: {algorithm_name}")
    
    def get_disaggregation_performance(self) -> Dict[str, Any]:
        """获取分解算法性能统计"""
        return self.disaggregator.get_performance_summary()
    
    def set_user_demands(self, demands: np.ndarray):
        """
        设置用户需求
        
        Args:
            demands: 用户需求，维度为 [actual_total_users, total_steps]
        """
        # 计算实际总用户数
        actual_total_users = sum(len(manager.users) for manager in self.managers)
        expected_shape = (actual_total_users, self.total_steps)
        
        logger.info(f"期望需求维度: {expected_shape}, 实际输入维度: {demands.shape}")
        
        if demands.shape != expected_shape:
            logger.warning(f"需求维度不匹配: {demands.shape} vs {expected_shape}，将尝试调整")
            
            # 如果用户数量不匹配，按比例调整或截断/填充
            if demands.shape[0] > actual_total_users:
                demands = demands[:actual_total_users, :]
                logger.info(f"截断需求数据至前 {actual_total_users} 个用户")
            elif demands.shape[0] < actual_total_users:
                padding = np.zeros((actual_total_users - demands.shape[0], self.total_steps))
                demands = np.vstack([demands, padding])
                logger.info(f"为缺失的 {actual_total_users - demands.shape[0]} 个用户填充零需求")
                
            # 如果时间步数不匹配，调整
            if demands.shape[1] > self.total_steps:
                demands = demands[:, :self.total_steps]
            elif demands.shape[1] < self.total_steps:
                padding = np.zeros((demands.shape[0], self.total_steps - demands.shape[1]))
                demands = np.hstack([demands, padding])
        
        self.user_demands = demands
        
        # 根据实际用户分布更新每个调度器的用户需求
        current_user_index = 0
        for i, manager in enumerate(self.managers):
            manager_users = len(manager.users)
            
            if current_user_index < demands.shape[0]:
                end_user_index = min(current_user_index + manager_users, demands.shape[0])
                actual_assigned_users = end_user_index - current_user_index
                
                # 获取这个Manager的用户需求
                manager_demands = demands[current_user_index:end_user_index]
                
                # 如果需求数据不足，用零填充
                if manager_demands.shape[0] < manager_users:
                    padding_users = manager_users - manager_demands.shape[0]
                    padding = np.zeros((padding_users, self.total_steps))
                    manager_demands = np.vstack([manager_demands, padding])
                    logger.info(f"Manager {manager.manager_id}: 实际有 {manager_users} 个用户，分配了 {actual_assigned_users} 个用户需求，填充 {padding_users} 个零需求")
                
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    scheduler.set_user_demands(manager_demands)
                    avg_demand = np.mean(manager_demands)
                    total_demand = np.sum(manager_demands)
                    logger.info(f"为Manager {manager.manager_id} 设置 {manager_users} 个用户的需求（用户索引 {current_user_index}-{end_user_index-1}），平均需求: {avg_demand:.2f} kWh，总需求: {total_demand:.2f} kWh")
                
                current_user_index = end_user_index
            else:
                # 如果没有更多需求数据，为剩余Manager设置零需求
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    zero_demands = np.zeros((manager_users, self.total_steps))
                    scheduler.set_user_demands(zero_demands)
                    logger.warning(f"为Manager {manager.manager_id} 的 {manager_users} 个用户设置零需求（数据不足）")
        
        logger.info(f"设置用户需求完成，总需求量: {np.sum(demands):.2f} kWh")
    
    def process_trades(self, step: int) -> Dict:
        """
        处理交易和调度
        
        Args:
            step: 当前时间步
            
        Returns:
            Dict: 处理结果
        """
        # 验证步骤范围
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"时间步 {step} 超出范围 [0, {self.total_steps-1}]")
        
        # 获取当前天气数据
        current_weather = self.trading_pool.weather_model.get_current_weather()
        
        # 获取当前交易历史
        trade_history = self.trading_pool.trade_history
        
        # 按买家分组
        trades_by_buyer = {}
        
        # 获取新交易
        new_trades = []
        for trade in trade_history:
            # 跳过已处理的交易
            if trade.trade_id in self.processed_trades:
                continue
            
            # 只处理已完成的交易
            if trade.status != "completed":
                continue
            
            # 添加到新交易列表
            new_trades.append(trade)
            self.processed_trades.add(trade.trade_id)
            
            # 按买家ID分组
            buyer_id = trade.buyer_id
            if buyer_id not in trades_by_buyer:
                trades_by_buyer[buyer_id] = []
            
            trades_by_buyer[buyer_id].append(trade)
        
        logger.info(f"时间步 {step}: 处理新交易 {len(new_trades)} 个")
        
        # 为每个买家处理交易
        all_disaggregated = {}
        
        for buyer_id, trades in trades_by_buyer.items():
            # 找到买家对应的管理器
            buyer_manager = None
            for manager in self.managers:
                if manager.manager_id == buyer_id:
                    buyer_manager = manager
                    break
            
            if not buyer_manager:
                logger.warning(f"未找到买家管理器 {buyer_id}，跳过交易处理")
                continue
            
            # 处理每个交易
            buyer_resources = []
            
            for trade in trades:
                # 获取交易资源
                energy_type = trade.energy_type
                quantity = trade.quantity
                price = trade.price
                
                # 创建资源对象
                resource = {
                    'resource_id': trade.trade_id,
                    'energy_type': energy_type,
                    'allocated_energy': quantity,
                    'price': price,
                    'trade_time': trade.trade_time,
                    'seller_id': trade.seller_id
                }
                
                buyer_resources.append(resource)
            
            # 存储买家资源
            all_disaggregated[buyer_id] = buyer_resources
        
        # 为每个买家调度用户
        allocations = {}
        
        for buyer_id, resources in all_disaggregated.items():
            scheduler = self.user_schedulers.get(buyer_id)
            if scheduler and resources:
                # 进行用户调度
                allocations[buyer_id] = scheduler.schedule(
                    energy_resources=resources,
                    step=step,
                    method='priority'  # 可以根据需要选择不同的调度方法
                )
        
        # 计算满意度
        satisfaction = {}
        overall_satisfaction = 0.0
        
        for buyer_id, scheduler in self.user_schedulers.items():
            user_satisfaction = scheduler.get_user_satisfaction(step)
            satisfaction[buyer_id] = user_satisfaction
            overall_satisfaction += np.mean(user_satisfaction)
        
        if self.user_schedulers:
            overall_satisfaction /= len(self.user_schedulers)
        
        self.satisfaction_history.append(overall_satisfaction)
        
        # 返回结果
        return {
            'disaggregated_resources': all_disaggregated,
            'allocations': allocations,
            'satisfaction': satisfaction,
            'overall_satisfaction': overall_satisfaction
        }
    
    def get_satisfaction_history(self) -> List[float]:
        """
        获取满意度历史
        
        Returns:
            List[float]: 满意度历史
        """
        return self.satisfaction_history
    
    def get_overall_satisfaction(self) -> float:
        """
        获取整体满意度
        
        Returns:
            float: 整体满意度
        """
        all_satisfaction = 0.0
        for scheduler in self.user_schedulers.values():
            all_satisfaction += scheduler.get_overall_satisfaction()
        
        if self.user_schedulers:
            return all_satisfaction / len(self.user_schedulers)
        else:
            return 0.0
    
    def visualize_satisfaction(self, save_path: Optional[str] = None):
        """
        可视化满意度历史
        
        Args:
            save_path: 保存路径，如果为None则显示图表
        """
        if not self.satisfaction_history:
            logger.warning("没有满意度历史数据")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.satisfaction_history, 'b-')
        plt.xlabel('时间步')
        plt.ylabel('整体满意度')
        plt.title('用户满意度历史')
        plt.grid(True)
        
        # 添加整体平均满意度线
        avg_satisfaction = float(np.mean(self.satisfaction_history))
        plt.axhline(y=avg_satisfaction, color='r', linestyle='--', 
                   label=f'平均满意度: {avg_satisfaction:.2f}')
        
        plt.legend()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_report(self, output_directory: Optional[str] = None):
        """
        生成调度结果报告
        
        Args:
            output_directory: 输出目录，如果为None则使用当前目录
        """
        # 如果未指定输出目录，使用当前目录
        if output_directory is None:
            output_directory = '.'
        
        # 确保目录存在
        os.makedirs(output_directory, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 计算各种指标
        overall_satisfaction = self.get_overall_satisfaction()
        satisfaction_trend = self.satisfaction_history
        
        # 各管理器的用户满意度
        manager_satisfaction = {}
        for manager_id, scheduler in self.user_schedulers.items():
            manager_satisfaction[manager_id] = scheduler.get_overall_satisfaction()
        
        # 创建报告数据
        report_data = {
            'timestamp': timestamp,
            'overall_satisfaction': overall_satisfaction,
            'manager_satisfaction': manager_satisfaction,
            'satisfaction_history': self.satisfaction_history
        }
        
        # 保存报告数据
        report_path = os.path.join(output_directory, f'schedule_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2)
        
        # 生成满意度图表
        satisfaction_path = os.path.join(output_directory, f'satisfaction_{timestamp}.png')
        self.visualize_satisfaction(satisfaction_path)
        
        # 为每个管理器生成用户分配图表
        for manager_id, scheduler in self.user_schedulers.items():
            allocation_path = os.path.join(output_directory, f'allocation_{manager_id}_{timestamp}.png')
            scheduler.visualize_allocation(save_path=allocation_path)
        
        logger.info(f"报告已生成到目录: {output_directory}")
        return report_path

    def update_user_demands_for_timestep(self, cumulative_demands, timestep):
        """为指定时间步更新用户需求状态"""
        try:
            logger.info(f"更新时间步 {timestep} 的用户需求状态")
            
            # 确保需求数据维度正确  
            total_users = sum(len(manager.users) for manager in self.managers)
            if cumulative_demands.shape[0] != total_users:
                logger.warning(f"用户数量不匹配: 需求数据 {cumulative_demands.shape[0]}, 实际用户 {total_users}")
                return
            
            # 更新当前时间步的需求状态
            if hasattr(self, 'current_timestep_demands'):
                self.current_timestep_demands = cumulative_demands
            else:
                self.current_timestep_demands = cumulative_demands
            
            # 更新每个Manager的用户调度器状态
            current_user_index = 0
            for manager in self.managers:
                manager_users = len(manager.users)
                
                if current_user_index < cumulative_demands.shape[0]:
                    # 获取到当前时间步为止的累积需求
                    end_user_index = min(current_user_index + manager_users, cumulative_demands.shape[0])
                    manager_cumulative_demands = cumulative_demands[current_user_index:end_user_index, :timestep+1]
                    
                    scheduler = self.user_schedulers.get(manager.manager_id)
                    if scheduler and hasattr(scheduler, 'update_cumulative_demands'):
                        scheduler.update_cumulative_demands(manager_cumulative_demands, timestep)
                    
                    current_user_index = end_user_index
            
            logger.info(f"时间步 {timestep} 用户需求状态更新完成")
            
        except Exception as e:
            logger.error(f"更新用户需求状态时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())


class AggregatedResultDisaggregator:
    """聚合结果分解器，将AggregatedResult分解为原始能源配置（重构版本）"""
    
    def __init__(self, time_horizon: int = 24, default_algorithm: str = 'proportional'):
        """
        初始化分解器
        
        Args:
            time_horizon: 时间范围
            default_algorithm: 默认分解算法，可选'average'、'proportional'、'equal_proportion'
        """
        self.time_horizon = time_horizon
        self.default_algorithm = default_algorithm
        self.algorithm_cache = {}  # 缓存算法实例
        self.performance_history = []  # 性能历史记录
        
        # 验证默认算法是否存在
        available_algorithms = DisaggregationAlgorithmFactory.get_available_algorithms()
        if default_algorithm not in available_algorithms:
            logger.warning(f"默认算法 '{default_algorithm}' 不存在，使用 'proportional' 算法")
            self.default_algorithm = 'proportional'
        
        logger.info(f"初始化聚合结果分解器，时间范围: {time_horizon}小时，默认算法: {self.default_algorithm}")
        logger.info(f"可用算法: {available_algorithms}")
    
    def disaggregate(self, 
                     aggregated_result: Union[AggregatedFlexOffer, Any], 
                     original_data: List[Dict], 
                     weighting_method: Optional[str] = None,
                     time_step: int = 0) -> List[Dict]:
        """
        分解聚合结果（重构版本）
        
        Args:
            aggregated_result: 聚合结果对象
            original_data: 原始数据列表，每个字典包含能源配置信息
            weighting_method: 权重分配方法，可选'average'、'proportional'、'equal_proportion'等
            time_step: 当前时间步
            
        Returns:
            List[Dict]: 分解后的能源配置列表
        """
        if not original_data:
            logger.warning("没有原始数据进行分解")
            return []
        
        # 确定使用的算法
        algorithm_name = weighting_method or self.default_algorithm
        
        # 处理旧版本的算法名称映射
        algorithm_mapping = {
            'equal': 'average',
            'proportional': 'proportional',
            'priority': 'priority'
        }
        algorithm_name = algorithm_mapping.get(algorithm_name, algorithm_name)
        
        # 获取总能量
        total_energy = 0.0
        
        if hasattr(aggregated_result, 'total_energy'):
            # 直接的total_energy属性
            total_energy = getattr(aggregated_result, 'total_energy', 0.0)
        elif hasattr(aggregated_result, 'total_energy_max'):
            # AggregatedFlexOffer的total_energy_max属性
            total_energy = getattr(aggregated_result, 'total_energy_max', 0.0)
            logger.debug(f"从AggregatedFlexOffer获取总能量: {total_energy}")
        elif hasattr(aggregated_result, 'aggregated_fo'):
            # 尝试从aggregated_fo获取
            agg_fo = getattr(aggregated_result, 'aggregated_fo', None)
            if agg_fo and hasattr(agg_fo, 'total_energy_max'):
                total_energy = getattr(agg_fo, 'total_energy_max', 0.0)
                logger.debug(f"从aggregated_fo获取总能量: {total_energy}")
            elif agg_fo and hasattr(agg_fo, 'quantity'):
                total_energy = getattr(agg_fo, 'quantity', 0.0)
                logger.debug(f"从aggregated_fo.quantity获取总能量: {total_energy}")
        
        # 如果仍然无法获取总能量，从原始数据计算
        if total_energy <= 0:
            total_energy = sum(item.get('energy', 0) for item in original_data)
            logger.debug(f"从原始数据计算总能量: {total_energy}")
        
        # 检查总能量是否为0，如果是，可能需要特殊处理
        if total_energy <= 0:
            logger.warning(f"总能量为0或负值({total_energy})，无法进行有效分解")
            # 如果是average算法，可以返回全零分配
            if algorithm_name == 'average':
                logger.info("使用average算法，返回全零分配")
                return [dict(item, allocated_energy=0.0, allocation_method='average', allocation_ratio=0.0) 
                        for item in original_data]
            # 对于其他算法，返回空列表
            return []
        
        # 创建分解请求
        request = DisaggregationRequest(
            aggregated_result=aggregated_result,
            original_data=original_data,
            total_energy=total_energy,
            time_step=time_step,
            metadata={
                'time_horizon': self.time_horizon,
                'original_count': len(original_data)
            }
        )
        
        # 获取或创建算法实例
        try:
            algorithm = self._get_algorithm(algorithm_name)
        except Exception as e:
            logger.error(f"获取算法实例失败: {e}")
            # 回退到average算法
            if algorithm_name != 'average':
                logger.info("回退到average算法")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # 如果average也失败，返回空列表
                return []
        
        # 执行分解
        try:
            result = algorithm.disaggregate(request)
            
            # 记录性能
            self._record_performance(algorithm_name, request, result)
            
            logger.info(f"分解完成，使用算法: {algorithm_name}，"
                       f"原始数据: {len(original_data)}，"
                       f"分解结果: {len(result.disaggregated_data)}，"
                       f"总能量: {total_energy:.2f} → {result.total_allocated_energy:.2f}")
            
            return result.disaggregated_data
            
        except Exception as e:
            logger.error(f"分解失败，算法: {algorithm_name}，错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 回退到平均分配
            if algorithm_name != 'average':
                logger.info("回退到平均分配算法")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # 如果平均分配也失败，返回带有零能量分配的结果
                logger.info("平均分配也失败，返回零能量分配")
                return [dict(item, allocated_energy=0.0, allocation_method='fallback', allocation_ratio=0.0) 
                        for item in original_data]
    
    def _get_algorithm(self, algorithm_name: str) -> DisaggregationAlgorithm:
        """
        获取算法实例（带缓存）
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            DisaggregationAlgorithm: 算法实例
        """
        if algorithm_name not in self.algorithm_cache:
            try:
                self.algorithm_cache[algorithm_name] = DisaggregationAlgorithmFactory.create_algorithm(algorithm_name)
            except ValueError as e:
                logger.error(f"创建算法失败: {e}")
                # 回退到默认算法
                if algorithm_name != self.default_algorithm:
                    logger.info(f"回退到默认算法: {self.default_algorithm}")
                    return self._get_algorithm(self.default_algorithm)
                else:
                    raise
        
        return self.algorithm_cache[algorithm_name]
    
    def _record_performance(self, algorithm_name: str, request: DisaggregationRequest, result: DisaggregationResult):
        """记录性能指标"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm_name,
            'time_step': request.time_step,
            'original_count': len(request.original_data),
            'total_energy': request.total_energy,
            'allocated_energy': result.total_allocated_energy,
            'allocation_efficiency': result.total_allocated_energy / request.total_energy if request.total_energy > 0 else 0
        }
        self.performance_history.append(performance_record)
        
        # 保持历史记录不超过1000条
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        if not self.performance_history:
            return {"message": "没有性能记录"}
        
        # 按算法分组统计
        algorithm_stats = {}
        for record in self.performance_history:
            alg = record['algorithm']
            if alg not in algorithm_stats:
                algorithm_stats[alg] = {
                    'count': 0,
                    'total_energy': 0,
                    'total_allocated': 0,
                    'efficiency_sum': 0
                }
            
            stats = algorithm_stats[alg]
            stats['count'] += 1
            stats['total_energy'] += record['total_energy']
            stats['total_allocated'] += record['allocated_energy']
            stats['efficiency_sum'] += record['allocation_efficiency']
        
        # 计算平均值
        summary = {}
        for alg, stats in algorithm_stats.items():
            summary[alg] = {
                'usage_count': stats['count'],
                'average_efficiency': stats['efficiency_sum'] / stats['count'],
                'total_energy_processed': stats['total_energy'],
                'total_energy_allocated': stats['total_allocated']
            }
        
        return {
            'total_operations': len(self.performance_history),
            'algorithm_performance': summary,
            'default_algorithm': self.default_algorithm,
            'cached_algorithms': list(self.algorithm_cache.keys())
        }
    
    def get_available_algorithms(self) -> List[str]:
        """获取可用算法列表"""
        return DisaggregationAlgorithmFactory.get_available_algorithms()
    
    def set_default_algorithm(self, algorithm_name: str):
        """设置默认算法"""
        available = self.get_available_algorithms()
        if algorithm_name not in available:
            raise ValueError(f"算法 '{algorithm_name}' 不存在。可用算法: {available}")
        
        self.default_algorithm = algorithm_name
        logger.info(f"默认分解算法已设置为: {algorithm_name}")
    
    def clear_cache(self):
        """清除算法缓存"""
        self.algorithm_cache.clear()
        logger.info("算法缓存已清除") 
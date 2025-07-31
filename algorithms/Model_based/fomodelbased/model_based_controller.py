"""
基于物理模型的FlexOffer控制器

提供纯粹基于物理模型的FlexOffer生成和优化功能，
不使用强化学习概念，而是采用传统的模型预测控制方法。
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import sys

# 处理导入方式
try:
    # 尝试作为包的一部分导入
    from .config import ModelBasedConfig
except (ImportError, SystemError):
    # 直接运行脚本时的导入方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import ModelBasedConfig

logger = logging.getLogger(__name__)


class DeviceModel:
    """设备物理模型基类"""
    
    def __init__(self, device_id: str, params: Dict[str, Any]):
        self.device_id = device_id
        self.params = params
        self.state = self.get_initial_state()
        
    def get_initial_state(self) -> Dict[str, float]:
        """获取初始状态"""
        return {}
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        """预测下一个状态"""
        return state
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        """获取最优控制"""
        return 0.0
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        """生成能量轮廓和时间灵活性"""
        return [0.0] * len(prices), 0


class BatteryModel(DeviceModel):
    """电池设备模型"""
    
    def get_initial_state(self) -> Dict[str, float]:
        """获取初始状态"""
        return {
            'soc': self.params.get('initial_soc', 0.5),  # 初始荷电状态
            'charge': self.params.get('initial_charge', 5.0),  # 初始电量(kWh)
        }
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        """预测下一个状态
        
        Args:
            state: 当前状态，包含'soc'和'charge'
            control: 控制量(kW)，正为充电，负为放电
            
        Returns:
            新状态
        """
        capacity = self.params.get('capacity', 10.0)  # 电池容量(kWh)
        efficiency = self.params.get('efficiency', 0.95)  # 充放电效率
        
        # 计算充放电电量
        if control > 0:  # 充电
            energy_delta = control * efficiency
        else:  # 放电
            energy_delta = control / efficiency
        
        # 更新电量
        new_charge = state['charge'] + energy_delta
        new_charge = max(0.0, min(capacity, new_charge))  # 限制在容量范围内
        
        # 更新SOC
        new_soc = new_charge / capacity
        
        return {
            'soc': new_soc,
            'charge': new_charge
        }
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        """获取最优控制
        
        基于价格和当前状态，确定最优充放电策略
        价格低时充电，价格高时放电
        
        Args:
            price: 当前电价
            current_state: 当前状态
            
        Returns:
            最优控制量(kW)，正为充电，负为放电
        """
        # 获取参数
        p_min = self.params.get('p_min', -3.0)  # 最小功率(kW)
        p_max = self.params.get('p_max', 3.0)  # 最大功率(kW)
        min_soc = self.params.get('min_soc', 0.1)  # 最小SOC
        max_soc = self.params.get('max_soc', 0.9)  # 最大SOC
        current_soc = current_state.get('soc', 0.5)
        
        # 简单策略：价格低于阈值充电，高于阈值放电
        low_price_threshold = 0.08  # 低价阈值
        high_price_threshold = 0.15  # 高价阈值
        
        # 默认控制：0（不充不放）
        control = 0.0
        
        if price <= low_price_threshold and current_soc < max_soc:
            # 低价且SOC未达到上限，充电
            control = p_max
        elif price >= high_price_threshold and current_soc > min_soc:
            # 高价且SOC未达到下限，放电
            control = p_min
        
        return control
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        """生成能量轮廓和时间灵活性
        
        基于电价序列，模拟优化充放电策略，生成能量轮廓
        
        Args:
            prices: 电价序列
            
        Returns:
            (energy_profile, time_flexibility): 能量轮廓和时间灵活性
        """
        # 初始化
        time_horizon = len(prices)
        energy_profile = [0.0] * time_horizon
        state = self.get_initial_state()
        
        # 模拟充放电过程
        for t in range(time_horizon):
            # 根据当前价格和状态获取最优控制
            control = self.get_optimal_control(prices[t], state)
            
            # 记录能量轮廓（正为消耗，负为产生）
            energy_profile[t] = control
            
            # 更新状态
            state = self.predict_next_state(state, control)
        
        # 电池的时间灵活性通常较高
        time_flexibility = min(3, time_horizon // 8)  # 灵活性为时间范围的1/8，最大为3小时
        
        return energy_profile, time_flexibility


class HeatPumpModel(DeviceModel):
    """热泵设备模型"""
    
    def get_initial_state(self) -> Dict[str, float]:
        """获取初始状态"""
        return {
            'temperature': self.params.get('initial_temp', 20.0),  # 初始温度(°C)
        }
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        """预测下一个状态
        
        Args:
            state: 当前状态，包含'temperature'
            control: 控制量(kW)，热泵功率
            
        Returns:
            新状态
        """
        # 获取参数
        outdoor_temp = self.params.get('outdoor_temp', 5.0)  # 室外温度(°C)
        thermal_mass = self.params.get('thermal_mass', 5000.0)  # 热质量(J/°C)
        heat_transfer_coeff = self.params.get('heat_transfer_coeff', 100.0)  # 传热系数(W/°C)
        cop = 3.0  # 性能系数，热泵效率
        
        # 当前温度
        current_temp = state['temperature']
        
        # 计算热泵提供的热量
        heat_pump_heat = control * cop * 1000  # 转换为W
        
        # 计算热损失
        heat_loss = heat_transfer_coeff * (current_temp - outdoor_temp)
        
        # 计算温度变化
        temp_change = (heat_pump_heat - heat_loss) / thermal_mass
        
        # 更新温度
        new_temp = current_temp + temp_change
        
        return {
            'temperature': new_temp
        }
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        """获取最优控制
        
        基于价格和当前温度，确定最优热泵功率
        
        Args:
            price: 当前电价
            current_state: 当前状态
            
        Returns:
            最优控制量(kW)
        """
        # 获取参数
        target_temp = self.params.get('target_temp', 21.0)  # 目标温度(°C)
        min_temp = self.params.get('min_temp', 18.0)  # 最低温度(°C)
        max_temp = self.params.get('max_temp', 22.0)  # 最高温度(°C)
        max_power = self.params.get('max_power', 2.0)  # 最大功率(kW)
        
        # 当前温度
        current_temp = current_state.get('temperature', 20.0)
        
        # 简单策略：价格低于阈值预热，高于阈值减少加热
        low_price_threshold = 0.08  # 低价阈值
        high_price_threshold = 0.15  # 高价阈值
        
        # 默认控制：根据温差调节加热功率
        temp_diff = target_temp - current_temp
        control = max_power * (temp_diff / 3.0)  # 简单比例控制
        control = max(0.0, min(max_power, control))  # 限制在功率范围内
        
        # 根据价格和温度调整控制策略
        if price <= low_price_threshold and current_temp < max_temp:
            # 低价预热，加热至较高温度
            temp_diff = max_temp - current_temp
            control = max_power * (temp_diff / 3.0)
            control = max(0.0, min(max_power, control))
        elif price >= high_price_threshold and current_temp > min_temp:
            # 高价节能，降低加热功率
            control = 0.0
        
        return control
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        """生成能量轮廓和时间灵活性
        
        基于电价序列，模拟热泵运行，生成能量轮廓
        
        Args:
            prices: 电价序列
            
        Returns:
            (energy_profile, time_flexibility): 能量轮廓和时间灵活性
        """
        # 初始化
        time_horizon = len(prices)
        energy_profile = [0.0] * time_horizon
        state = self.get_initial_state()
        
        # 模拟热泵运行过程
        for t in range(time_horizon):
            # 根据当前价格和状态获取最优控制
            control = self.get_optimal_control(prices[t], state)
            
            # 记录能量轮廓（正为消耗，负为产生）
            energy_profile[t] = control
            
            # 更新状态
            state = self.predict_next_state(state, control)
        
        # 热泵的时间灵活性通常较低
        time_flexibility = min(2, time_horizon // 12)  # 灵活性为时间范围的1/12，最大为2小时
        
        return energy_profile, time_flexibility


class ModelBasedController:
    """基于模型的FlexOffer控制器"""
    
    def __init__(self, 
                manager_id: str,
                time_horizon: int = 24,
                time_step: float = 1.0,
                config: Optional[ModelBasedConfig] = None):
        self.manager_id = manager_id
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.config = config or ModelBasedConfig()
        
        self.device_models = {}  # device_id -> DeviceModel
        self.device_stats = {}   # device_id -> 统计信息
        
        self.current_timestep = 0
        logger.info(f"ModelBasedController初始化: manager_id={manager_id}, time_horizon={time_horizon}")
    
    def add_device_model(self, device_id: str, device_type: str, device_params: Dict[str, Any]):
        """添加设备模型"""
        if device_id in self.device_models:
            logger.warning(f"设备 {device_id} 已存在，将被覆盖")
        
        # 根据设备类型创建对应的模型
        if 'BATTERY' in device_type.upper():
            model = BatteryModel(device_id, device_params)
            logger.info(f"添加电池模型: {device_id}")
        elif 'HEAT' in device_type.upper() or 'PUMP' in device_type.upper():
            model = HeatPumpModel(device_id, device_params)
            logger.info(f"添加热泵模型: {device_id}")
        else:
            model = DeviceModel(device_id, device_params)
            logger.info(f"添加通用设备模型: {device_id}")
        
        self.device_models[device_id] = model
        
        # 初始化设备统计信息
        self.device_stats[device_id] = {
            'type': device_type,
            'params': device_params,
            'energy_consumed': 0.0,
            'energy_produced': 0.0
        }
    
    def generate_flex_offers(self, prices: List[float]) -> Dict[str, Dict[str, Any]]:
        """
        生成FlexOffer
        
        Args:
            prices: 电价序列
            
        Returns:
            device_id -> {
                'energy_profile': [...],
                'time_flexibility': int
            }
        """
        if len(prices) < self.time_horizon:
            # 如果价格序列不够长，扩展价格序列
            prices = prices + [prices[-1]] * (self.time_horizon - len(prices))
        
        fo_dict = {}
        
        for device_id, model in self.device_models.items():
            # 生成能量轮廓和时间灵活性
            energy_profile, time_flexibility = model.generate_energy_profile(prices[:self.time_horizon])
            
            # 更新设备统计信息
            consumed = sum(max(0, e) for e in energy_profile)
            produced = sum(abs(min(0, e)) for e in energy_profile)
            self.device_stats[device_id]['energy_consumed'] += consumed
            self.device_stats[device_id]['energy_produced'] += produced
            
            # 记录FlexOffer
            fo_dict[device_id] = {
                'energy_profile': energy_profile,
                'time_flexibility': time_flexibility
            }
        
        logger.info(f"生成了 {len(fo_dict)} 个FlexOffer")
        return fo_dict
    
    def calculate_reward(self, 
                        schedules: Dict[str, List[float]], 
                        revenue: float,
                        original_profiles: Dict[str, List[float]]) -> float:
        """
        计算奖励
        
        Args:
            schedules: 设备调度，device_id -> schedule
            revenue: 交易收益
            original_profiles: 原始能量轮廓，device_id -> profile
            
        Returns:
            奖励值
        """
        # 计算用户满意度 - 调度与原始需求的相似度
        satisfaction = 0.0
        profile_count = 0
        
        # 计算设备调度与原始需求的相似度
        for device_id, schedule in schedules.items():
            if device_id in original_profiles:
                original = original_profiles[device_id]
                
                # 确保长度一致
                min_len = min(len(schedule), len(original))
                
                if min_len > 0:
                    # 计算相对误差
                    schedule_np = np.array(schedule[:min_len])
                    original_np = np.array(original[:min_len])
                    
                    # 避免除零
                    total_energy = np.sum(np.abs(original_np))
                    if total_energy > 0:
                        error = np.sum(np.abs(schedule_np - original_np)) / total_energy
                        similarity = max(0, 1 - error)  # 转换为相似度
                    else:
                        similarity = 1.0  # 如果原始能量为0，认为完全满足
                    
                    satisfaction += similarity
                    profile_count += 1
        
        # 计算平均满意度
        avg_satisfaction = satisfaction / max(1, profile_count)
        
        # 归一化收益 (假设最大可能收益为设备数量 * 10)
        max_possible_revenue = len(schedules) * 10
        normalized_revenue = min(1.0, revenue / max(0.1, max_possible_revenue))
        
        # 综合奖励 = 满意度权重 * 满意度 + 收益权重 * 归一化收益
        satisfaction_weight = 0.7
        revenue_weight = 0.3
        
        # 基础奖励计算
        base_reward = satisfaction_weight * avg_satisfaction + revenue_weight * normalized_revenue
        
        # 放大奖励尺度（乘以36）
        reward = base_reward * 36.0
        
        logger.info(f"计算奖励: 满意度={avg_satisfaction:.4f}, 归一化收益={normalized_revenue:.4f}, 总奖励={reward:.4f}")
        return reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'manager_id': self.manager_id,
            'device_count': len(self.device_models),
            'current_timestep': self.current_timestep,
            'device_stats': self.device_stats
        }
    
    def reset(self):
        """重置控制器状态"""
        self.current_timestep = 0
        
        # 重置设备状态
        for model in self.device_models.values():
            model.state = model.get_initial_state()
        
        # 重置统计信息
        for device_id in self.device_stats:
            self.device_stats[device_id]['energy_consumed'] = 0.0
            self.device_stats[device_id]['energy_produced'] = 0.0
        
        logger.info("控制器已重置")
    
    def step(self, time_step: int = 1):
        """更新时间步"""
        self.current_timestep += time_step
        logger.debug(f"时间步更新: {self.current_timestep}") 
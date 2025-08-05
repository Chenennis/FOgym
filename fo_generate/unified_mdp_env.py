"""
统一MDP环境框架 - FlexOffer完整MDP实现

整合了设备级MDP和环境级MDP，提供完整的FlexOffer马尔可夫决策过程框架。

核心特性：
- 完整的马尔可夫性质保证
- 统一的设备接口
- 确定性状态转移
- 多目标奖励函数
- 与FlexOffer流程完全集成
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
import math
from abc import ABC, abstractmethod

from fo_generate.battery_model import BatteryModel, BatteryParameters
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior
from fo_generate.pv_model import PVModel, PVParameters
from fo_generate.dishwasher_model import DishwasherModel, DishwasherParameters, DishwasherUserBehavior
from fo_generate.dfo import DFOSystem, DFOSlice
from fo_generate.price_loader import PriceLoader

logger = logging.getLogger(__name__)

class DeviceType:
    """设备类型枚举"""
    BATTERY = "battery"
    HEAT_PUMP = "heat_pump"
    EV = "ev"
    PV = "pv"
    DISHWASHER = "dishwasher"

class EnvironmentDynamics:
    """环境动态管理 - 确保马尔可夫性质"""
    
    def __init__(self, price_data: pd.DataFrame = None, weather_data: pd.DataFrame = None, 
                 data_dir: str = "data"):
        self.price_data = price_data
        self.weather_data = weather_data
        
        # 初始化电价加载器
        self.price_loader = PriceLoader(data_dir)
        
        # 马尔可夫状态历史（仅保留计算趋势所需的最小历史）
        self.price_history = []
        self.weather_history = []
        
        # 环境参数
        self.price_volatility = 0.1
        self.weather_noise = 0.05

    def get_current_state(self, current_time: datetime) -> Dict[str, Any]:
        """获取当前环境状态（马尔可夫状态）"""
        # 获取当前价格和天气
        current_price = self._get_price_at_time(current_time)
        current_weather = self._get_weather_at_time(current_time)
        
        # 更新历史
        self.price_history.append(current_price)
        self.weather_history.append(current_weather)
        
        # 只保留最近3个时间步的历史（计算趋势所需）
        if len(self.price_history) > 3:
            self.price_history = self.price_history[-3:]
        if len(self.weather_history) > 3:
            self.weather_history = self.weather_history[-3:]
        
        # 计算趋势
        price_trend = self._get_price_trend()
        weather_trend = self._get_weather_trend()
        
        # 预测未来3小时的价格和天气
        future_prices = self._predict_future_prices(current_time)
        future_weather = self._predict_future_weather(current_time)
        
        return {
            'price': current_price,
            'price_trend': price_trend,
            'future_prices': future_prices,
            'temperature': current_weather['temperature'],
            'solar_irradiance': current_weather['solar_irradiance'],
            'weather_trend': weather_trend,
            'future_weather': future_weather
        }

    def _get_price_at_time(self, current_time: datetime) -> float:
        """获取指定时间的电价，优先使用丹麦电价数据"""
        try:
            # 首先尝试使用价格加载器获取丹麦电价数据
            current_price_info = self.price_loader.get_current_price(current_time)
            base_price = current_price_info['price']
            logger.debug(f"使用{current_price_info['source']}电价数据: {base_price:.4f} USD/kWh")
        except Exception as e:
            logger.warning(f"电价加载器获取价格失败: {e}，使用备选方案")
            
            # 备选方案1：使用传入的价格数据
            if self.price_data is not None:
                time_diffs = abs((pd.to_datetime(self.price_data['timestamp']) - current_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                base_price = self.price_data.loc[closest_idx, 'price']
            else:
                # 备选方案2：简化的价格模型
                hour = current_time.hour
                if 0 <= hour < 6:  # 夜间低价
                    base_price = 0.08
                elif 6 <= hour < 18:  # 白天
                    base_price = 0.15 + 0.05 * math.sin(math.pi * (hour - 6) / 12)
                else:  # 晚上峰值
                    base_price = 0.20
                
        # 添加随机波动
        price_noise = np.random.normal(0, self.price_volatility * 0.1)
        return max(0.01, base_price + price_noise)

    def _get_weather_at_time(self, current_time: datetime) -> Dict[str, float]:
        """获取指定时间的天气数据"""
        if self.weather_data is not None:
            time_diffs = abs((pd.to_datetime(self.weather_data['timestamp']) - current_time).dt.total_seconds())
            closest_idx = time_diffs.idxmin()
            return {
                'temperature': self.weather_data.loc[closest_idx, 'temperature'],
                'solar_irradiance': self.weather_data.loc[closest_idx, 'solar_irradiance']
            }
        else:
            # 简化的天气模型
            hour = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
            # 温度模型
            seasonal_temp = 20 + 10 * math.sin(2 * math.pi * day_of_year / 365)
            daily_variation = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
            temperature = seasonal_temp + daily_variation
            
            # 太阳辐照度模型
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                irradiance = 800 * solar_angle * max(0, math.sin(2 * math.pi * day_of_year / 365 + math.pi/2))
            else:
                irradiance = 0
                
            return {
                'temperature': temperature + np.random.normal(0, 1),
                'solar_irradiance': max(0, irradiance + np.random.normal(0, 50))
            }

    def _get_price_trend(self) -> float:
        """计算价格趋势"""
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] - self.price_history[0]) / max(self.price_history[0], 0.01)

    def _get_weather_trend(self) -> Dict[str, float]:
        """计算天气趋势"""
        if len(self.weather_history) < 2:
            return {'temperature_trend': 0.0, 'irradiance_trend': 0.0}
        
        temp_trend = (self.weather_history[-1]['temperature'] - self.weather_history[0]['temperature']) / max(abs(self.weather_history[0]['temperature']), 1.0)
        irr_trend = (self.weather_history[-1]['solar_irradiance'] - self.weather_history[0]['solar_irradiance']) / max(self.weather_history[0]['solar_irradiance'], 1.0)
        
        return {'temperature_trend': temp_trend, 'irradiance_trend': irr_trend}

    def _predict_future_prices(self, current_time: datetime) -> List[float]:
        """预测未来3小时的价格"""
        future_prices = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            future_price = self._get_price_at_time(future_time)
            future_prices.append(future_price)
        return future_prices

    def _predict_future_weather(self, current_time: datetime) -> List[Dict[str, float]]:
        """预测未来3小时的天气"""
        future_weather = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            weather = self._get_weather_at_time(future_time)
            future_weather.append(weather)
        return future_weather

class DeviceMDPInterface(ABC):
    """设备MDP接口"""
    
    @abstractmethod
    def get_state_features(self) -> np.ndarray:
        """获取设备状态特征"""
        pass
    
    @abstractmethod
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """设备状态转移"""
        pass
    
    @abstractmethod
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算设备奖励"""
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[float, float]:
        """获取动作边界"""
        pass
    
    @abstractmethod
    def reset_state(self):
        """重置设备状态"""
        pass

class DishwasherMDPDevice(DeviceMDPInterface):
    """洗碗机设备MDP实现"""
    
    def __init__(self, dishwasher_model: DishwasherModel):
        self.dishwasher = dishwasher_model
        
    def get_state_features(self) -> np.ndarray:
        """获取洗碗机状态特征 [是否部署, 是否运行, 是否完成, 当前步骤/总步骤, 紧急度, 剩余能量需求]"""
        is_deployed = 1.0 if self.dishwasher.is_deployed else 0.0
        is_running = 1.0 if self.dishwasher.is_running else 0.0
        is_completed = 1.0 if self.dishwasher.is_completed else 0.0
        
        # 进度（当前步骤/总步骤）
        progress = self.dishwasher.current_cycle_step / max(1, self.dishwasher.total_cycle_steps)
        
        # 紧急度
        urgency = self.dishwasher.calculate_urgency(datetime.now())
        
        # 剩余能量需求
        remaining_energy = self.dishwasher.params.total_energy - self.dishwasher.energy_consumed
        remaining_energy_norm = remaining_energy / max(1, self.dishwasher.params.total_energy)
        
        return np.array([is_deployed, is_running, is_completed, progress, urgency, remaining_energy_norm])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """洗碗机状态转移
        
        action: 0-1，表示是否启动洗碗机（仅在已部署但未运行时有效）
        """
        current_time = datetime.now()
        
        # 如果还未部署，随机模拟部署（在实际应用中由用户触发）
        if not self.dishwasher.is_deployed:
            # 模拟用户可能在某些时间部署洗碗机
            if np.random.random() < 0.1:  # 10%概率部署
                self.dishwasher.deploy(current_time)
                
        # 如果已部署但未运行，根据action决定是否启动
        start_success = False
        if self.dishwasher.is_deployed and not self.dishwasher.is_running and not self.dishwasher.is_completed:
            if action > 0.5:  # action > 0.5表示决定启动
                start_success = self.dishwasher.start_operation(current_time)
        
        # 如果正在运行，继续运行一个时间步
        power_consumed = 0.0
        operation_completed = False
        if self.dishwasher.is_running:
            # 洗碗机运行需要固定功率
            available_power = env_state.get('available_power', self.dishwasher.params.power_rating)
            power_consumed, operation_completed = self.dishwasher.step_operation(current_time, available_power)
        
        return {
            'is_deployed': self.dishwasher.is_deployed,
            'is_running': self.dishwasher.is_running,
            'is_completed': self.dishwasher.is_completed,
            'power_consumed': power_consumed,
            'operation_completed': operation_completed,
            'start_success': start_success,
            'current_cycle_step': self.dishwasher.current_cycle_step,
            'energy_consumed': self.dishwasher.energy_consumed
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算洗碗机奖励 - 修复版本，减少稀疏奖励"""
        reward = 0.0
        reward_components = {}
        
        # 🔧 修复1: 完成任务奖励保持高奖励
        if next_state['operation_completed']:
            completion_reward = 50.0  # 降低但仍然较高的完成奖励
            reward += completion_reward
            reward_components['completion_reward'] = completion_reward
        
        # 🔧 修复2: 增加运行进度奖励，鼓励启动和持续运行
        if next_state['is_running']:
            # 根据进度给予递增奖励
            progress = getattr(self.dishwasher, 'current_cycle_step', 0)
            total_steps = getattr(self.dishwasher, 'total_cycle_steps', 10)
            
            if total_steps > 0:
                progress_ratio = progress / total_steps
                progress_reward = 5.0 + progress_ratio * 10.0  # 5-15分的进度奖励
            else:
                progress_reward = 8.0  # 默认运行奖励
                
            reward += progress_reward
            reward_components['progress_reward'] = progress_reward
        
        # 🔧 修复3: 重新设计启动时机奖励，更宽容
        if next_state.get('start_success', False):
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'calculate_urgency'):
                urgency = self.dishwasher.calculate_urgency(current_time)
            else:
                urgency = 0.5  # 默认中等紧急度
            
            if urgency > 0.6:  # 较高紧急度时启动
                timing_reward = 15.0 * urgency  # 最高9分奖励
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            elif urgency > 0.3:  # 中等紧急度
                timing_reward = 5.0 * urgency  # 1.5-3分奖励
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            else:  # 低紧急度时启动，轻微惩罚
                timing_penalty = -2.0  # 减少惩罚
                reward += timing_penalty
                reward_components['timing_penalty'] = timing_penalty
        
        # 🔧 修复4: 重新设计能耗成本，不要过度惩罚
        power_consumed = next_state.get('power_consumed', 0.0)
        price = env_state.get('price', 0.15)
        
        if power_consumed > 0:
            # 能耗成本相对于收益的惩罚，而不是绝对惩罚
            energy_cost = power_consumed * price * 0.3  # 降低成本权重
            reward -= energy_cost
            reward_components['energy_cost'] = -energy_cost
        
        # 🔧 修复5: 重新设计等待时间惩罚，更宽容
        if (self.dishwasher.is_deployed and 
            not getattr(self.dishwasher, 'is_running', False) and 
            not getattr(self.dishwasher, 'is_completed', False)):
            
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'deployment_time') and self.dishwasher.deployment_time:
                wait_time = (current_time - self.dishwasher.deployment_time).total_seconds() / 3600
                max_delay = getattr(self.dishwasher.params, 'max_start_delay', 6.0)
                
                if wait_time > max_delay:
                    # 等待超时，重度惩罚但减少幅度
                    timeout_penalty = -20.0  # 从-50减少到-20
                    reward += timeout_penalty
                    reward_components['timeout_penalty'] = timeout_penalty
                elif wait_time > max_delay * 0.8:
                    # 接近超时，轻度惩罚
                    wait_penalty = -5.0 * (wait_time / max_delay)  # 减少惩罚
                    reward += wait_penalty
                    reward_components['wait_penalty'] = wait_penalty
        
        # 🔧 修复6: 添加部署奖励，鼓励参与
        if self.dishwasher.is_deployed:
            deployment_reward = 2.0  # 部署即有奖励
            reward += deployment_reward
            reward_components['deployment_reward'] = deployment_reward
        
        # 🔧 修复7: 添加基础参与奖励
        base_participation_reward = 1.0
        reward += base_participation_reward
        reward_components['participation_reward'] = base_participation_reward
        
        return reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """获取动作边界"""
        return 0.0, 1.0  # 0表示不启动，1表示启动
    
    def reset_state(self):
        """重置洗碗机状态"""
        self.dishwasher.is_deployed = False
        self.dishwasher.is_running = False
        self.dishwasher.is_completed = False
        self.dishwasher.current_cycle_step = 0
        self.dishwasher.deployment_time = None
        self.dishwasher.start_time = None
        self.dishwasher.completion_time = None
        self.dishwasher.energy_consumed = 0.0

class BatteryMDPDevice(DeviceMDPInterface):
    """电池设备MDP实现"""
    
    def __init__(self, battery_model: BatteryModel):
        self.battery = battery_model
        self.efficiency = battery_model.params.efficiency
        self.capacity = battery_model.params.capacity_kwh
    
    def get_state_features(self) -> np.ndarray:
        """获取电池状态特征 [SOC, 最大充电功率, 最大放电功率, 健康度]"""
        soc = self.battery.current_soc
        
        # 计算可用功率范围
        max_charge_energy = (self.battery.params.soc_max - soc) * self.capacity
        max_charge_power = min(self.battery.params.p_max, max_charge_energy / self.efficiency)
        
        max_discharge_energy = (soc - self.battery.params.soc_min) * self.capacity
        max_discharge_power = min(abs(self.battery.params.p_min), max_discharge_energy * self.efficiency)
        
        # 健康度（简化模型）
        health = max(0.8, 1.0 - soc * 0.1)  # 基于SOC的简化健康度
        
        return np.array([soc, max_charge_power, max_discharge_power, health])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """电池状态转移"""
        soc = self.battery.current_soc
        
        # 计算SOC变化
        if action > 0:  # 充电
            energy_change = action * self.efficiency
        else:  # 放电
            energy_change = action / self.efficiency
        
        new_soc = soc + energy_change / self.capacity
        new_soc = np.clip(new_soc, self.battery.params.soc_min, self.battery.params.soc_max)
        
        # 更新电池状态
        self.battery.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': action,
            'energy_change': energy_change,
            'efficiency_loss': abs(energy_change) * (1 - self.efficiency) if action != 0 else 0
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算电池奖励 - 增强学习信号版本"""
        reward_components = {}
        
        # 🔧 增强1: 重新设计经济性奖励，增大差异化
        price = env_state.get('price', 0.15)
        base_price = 0.15
        price_ratio = price / base_price
        
        # 电价时段分析 - 增大奖励差异
        if price < 0.10:  # 超低电价时段
            if action > 0:  # 充电
                economic_reward = abs(action) * 10.0 * (1.0 - price_ratio)  # 最高10分
            else:
                economic_reward = -abs(action) * 2.0  # 错失机会惩罚
        elif price < 0.12:  # 低电价时段
            if action > 0:  # 充电
                economic_reward = abs(action) * 5.0 * (1.0 - price_ratio)  # 最高5分
            else:
                economic_reward = 0.0
        elif price > 0.25:  # 超高电价时段
            if action < 0:  # 放电
                economic_reward = abs(action) * 15.0 * (price_ratio - 1.0)  # 最高15分
            else:
                economic_reward = -abs(action) * 5.0  # 高价充电重罚
        elif price > 0.18:  # 高电价时段
            if action < 0:  # 放电
                economic_reward = abs(action) * 8.0 * (price_ratio - 1.0)  # 最高8分
            else:
                economic_reward = -abs(action) * 2.0  # 高价充电轻罚
        else:  # 中等电价
            economic_reward = -abs(action) * 0.5  # 轻微惩罚
        
        reward_components['economic'] = economic_reward
        
        # 🔧 增强2: SOC管理奖励，创造更大差异
        soc = next_state.get('soc', 0.5)
        
        if 0.45 <= soc <= 0.75:  # 最佳SOC区间
            soc_reward = 8.0
        elif 0.35 <= soc <= 0.85:  # 良好SOC区间
            soc_reward = 4.0
        elif 0.25 <= soc <= 0.9:  # 可接受区间
            soc_reward = 1.0
        elif 0.15 <= soc <= 0.95:  # 边界区间
            soc_reward = -2.0
        else:  # 危险区间
            soc_reward = -10.0  # 重罚
            
        reward_components['soc_maintenance'] = soc_reward
        
        # 🔧 增强3: 连续决策奖励，鼓励合理的action sequence
        action_consistency_reward = 0.0
        if hasattr(self, 'prev_action'):
            prev_action = self.prev_action
            # 奖励合理的动作连续性
            if abs(action - prev_action) < 0.5:  # 平稳操作
                action_consistency_reward = 2.0
            elif abs(action - prev_action) > 2.0:  # 剧烈变化
                action_consistency_reward = -1.0
        
        self.prev_action = action
        reward_components['action_consistency'] = action_consistency_reward
        
        # 🔧 增强4: 状态改善奖励，鼓励positive state changes
        state_improvement_reward = 0.0
        if hasattr(self, 'prev_soc'):
            prev_soc = self.prev_soc
            soc_change = soc - prev_soc
            
            # 奖励向理想SOC区间移动
            ideal_soc = 0.6
            prev_distance = abs(prev_soc - ideal_soc)
            current_distance = abs(soc - ideal_soc)
            
            if current_distance < prev_distance:  # 向理想状态移动
                state_improvement_reward = 3.0 * (prev_distance - current_distance)
            else:  # 远离理想状态
                state_improvement_reward = -2.0 * (current_distance - prev_distance)
        
        self.prev_soc = soc
        reward_components['state_improvement'] = state_improvement_reward
        
        # 🔧 增强5: 任务完成奖励，基于时间进度
        hour = datetime.now().hour
        task_completion_reward = 0.0
        
        # 根据一天中的时间给予不同的任务完成奖励
        if 6 <= hour <= 9:  # 早高峰
            if 0.7 <= soc <= 0.9:  # 为白天做好准备
                task_completion_reward = 5.0
        elif 18 <= hour <= 22:  # 晚高峰
            if action < 0 and soc > 0.5:  # 放电支持负荷
                task_completion_reward = 6.0
        elif 22 <= hour or hour <= 6:  # 夜间
            if action > 0 and price < 0.12:  # 夜间低价充电
                task_completion_reward = 4.0
                
        reward_components['task_completion'] = task_completion_reward
        
        # 🔧 增强6: 重新平衡权重，增大总体奖励范围
        total_reward = (
            0.4 * economic_reward +           # 提高经济权重
            0.3 * soc_reward +               # SOC管理
            0.1 * action_consistency_reward + # 动作一致性
            0.1 * state_improvement_reward +  # 状态改善
            0.1 * task_completion_reward      # 任务完成
        )
        
        # 🔧 增强7: 去掉固定的基础奖励，让差异化更明显
        # 不再添加base_participation_reward，让好坏动作的差异更大
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """获取动作边界"""
        return self.battery.params.p_min, self.battery.params.p_max
    
    def reset_state(self):
        """重置电池状态"""
        self.battery.current_soc = self.battery.params.initial_soc

class HeatPumpMDPDevice(DeviceMDPInterface):
    """热泵设备MDP实现"""
    
    def __init__(self, heatpump_model: HeatPumpModel):
        self.heatpump = heatpump_model
        self.cop = heatpump_model.params.cop
    
    def get_state_features(self) -> np.ndarray:
        """获取热泵状态特征 [当前温度, 目标温度, 舒适度]"""
        current_temp = self.heatpump.current_temp
        target_temp = self._get_target_temperature()
        comfort_score = 1.0 - min(1.0, abs(current_temp - target_temp) / 3.0)
        
        return np.array([current_temp, target_temp, comfort_score])
    
    def _get_target_temperature(self) -> float:
        """获取目标温度（基于时间）"""
        hour = datetime.now().hour
        if 8 <= hour < 22:
            return self.heatpump.params.primary_target_temp
        else:
            return self.heatpump.params.secondary_target_temp
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """热泵状态转移"""
        current_temp = self.heatpump.current_temp
        outside_temp = env_state['temperature']
        
        # 计算热量输出
        heat_output = action * self.cop if action > 0 else 0
        
        # 热损失
        heat_loss = self.heatpump.params.heat_loss_coef * (current_temp - outside_temp)
        
        # 温度变化
        net_heat = heat_output - heat_loss
        temp_change = net_heat / (self.heatpump.params.room_volume * 1.2)
        
        new_temp = current_temp + temp_change
        new_temp = np.clip(new_temp, self.heatpump.params.temp_min, self.heatpump.params.temp_max)
        
        # 更新热泵状态
        self.heatpump.current_temp = new_temp
        
        return {
            'temperature': new_temp,
            'power': action,
            'heat_output': heat_output,
            'heat_loss': heat_loss
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算热泵奖励 - 修复版本，提供正向激励"""
        reward_components = {}
        
        # 🔧 修复1: 重新设计经济性奖励，鼓励高效使用
        price = env_state.get('price', 0.15)
        
        if action <= 0:  # 不使用热泵
            economic_reward = 0.1  # 小的基础奖励
        else:
            # 根据COP和电价计算效率
            heat_output = action * self.cop
            efficiency_ratio = heat_output / action if action > 0 else 0
            
            if price < 0.12:  # 低电价时段
                economic_reward = 1.0 - (action * price * 0.5)  # 鼓励使用
            elif price > 0.20:  # 高电价时段
                if efficiency_ratio > 3.5:  # 高效率使用
                    economic_reward = 0.5 - (action * price * 0.3)
                else:
                    economic_reward = -(action * price * 0.8)  # 惩罚低效率高价使用
            else:  # 中等电价
                economic_reward = 0.2 - (action * price * 0.4)
        
        reward_components['economic'] = economic_reward
        
        # 🔧 修复2: 重新设计舒适度奖励，更宽容的温度控制
        current_temp = next_state['temperature']
        target_temp = self._get_target_temperature()
        temp_diff = abs(current_temp - target_temp)
        
        if temp_diff <= 1.0:  # 很好的温度控制
            comfort_reward = 3.0 - temp_diff * 2.0  # 1.0-3.0分
        elif temp_diff <= 2.5:  # 可接受的温度控制
            comfort_reward = 2.0 - temp_diff * 0.5  # 0.75-1.75分
        elif temp_diff <= 4.0:  # 基本可接受
            comfort_reward = 1.0 - temp_diff * 0.2  # 0.2-1.0分
        else:  # 温度控制差
            comfort_reward = -temp_diff * 0.5  # 负分
            
        reward_components['comfort'] = comfort_reward
        
        # 🔧 修复3: 添加温度稳定性奖励
        # 检查温度变化（需要历史温度，这里简化处理）
        if hasattr(self.heatpump, 'prev_temp'):
            temp_change = abs(current_temp - self.heatpump.prev_temp)
            if temp_change <= 0.5:  # 温度稳定
                stability_reward = 1.0
            elif temp_change <= 1.5:  # 适度变化
                stability_reward = 0.5
            else:  # 温度波动大
                stability_reward = -0.5
        else:
            stability_reward = 0.0
            
        self.heatpump.prev_temp = current_temp  # 保存当前温度
        reward_components['stability'] = stability_reward
        
        # 🔧 修复4: 添加适时使用奖励
        hour = datetime.now().hour
        if 8 <= hour <= 22:  # 白天使用时段
            time_appropriateness = 1.0 if action > 0 else 0.0
        else:  # 夜间时段
            time_appropriateness = 0.5 if action > 0 else 0.2
            
        reward_components['time_appropriateness'] = time_appropriateness
        
        # 🔧 修复5: 重新平衡权重，确保正向激励
        total_reward = (
            0.2 * economic_reward +        # 降低经济权重
            0.5 * comfort_reward +         # 提高舒适度权重
            0.2 * stability_reward +       # 温度稳定性
            0.1 * time_appropriateness     # 使用时机
        )
        
        # 🔧 修复6: 添加基础参与奖励
        base_participation_reward = 0.2
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """获取动作边界"""
        return 0.0, self.heatpump.params.max_power
    
    def reset_state(self):
        """重置热泵状态"""
        self.heatpump.current_temp = self.heatpump.params.initial_temp

class EVMDPDevice(DeviceMDPInterface):
    """电动汽车设备MDP实现"""
    
    def __init__(self, ev_model: EVModel):
        self.ev = ev_model
        self.battery_capacity = ev_model.params.battery_capacity
    
    def get_state_features(self) -> np.ndarray:
        """获取EV状态特征 [SOC, 连接状态, 充电紧急度]"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # 充电紧急度（基于用户行为）
        if self.ev.user_behavior and is_connected:
            remaining_time = max(0, (self.ev.user_behavior.disconnection_time - datetime.now()).total_seconds() / 3600)
            soc_gap = max(0, self.ev.user_behavior.target_soc - soc)
            urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
        else:
            urgency = 0.0
        
        return np.array([soc, float(is_connected), urgency])
    
    def _is_connected(self) -> bool:
        """检查EV是否连接"""
        if not self.ev.user_behavior:
            return True
        now = datetime.now()
        return self.ev.user_behavior.connection_time <= now < self.ev.user_behavior.disconnection_time
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """EV状态转移"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # 只有连接时才能充电
        actual_power = action if is_connected and action > 0 else 0
        
        if actual_power > 0:
            energy_change = actual_power * self.ev.params.efficiency
            new_soc = soc + energy_change / self.battery_capacity
        else:
            energy_change = 0
            new_soc = soc
        
        new_soc = np.clip(new_soc, self.ev.params.soc_min, self.ev.params.soc_max)
        
        # 更新EV状态
        self.ev.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': actual_power,
            'connected': is_connected,
            'energy_added': energy_change
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算EV奖励 - 修复版本，提供更好的学习信号"""
        reward_components = {}
        
        # 🔧 修复1: 重新设计经济性奖励，鼓励智能充电
        power = next_state.get('power', 0.0)
        price = env_state.get('price', 0.15)
        
        if power <= 0:  # 不充电
            economic_reward = 0.1  # 小的基础奖励
        else:
            if price < 0.12:  # 低电价时段充电
                economic_reward = 2.0 - (power * price * 0.5)  # 鼓励低价充电
            elif price > 0.20:  # 高电价时段充电
                economic_reward = -(power * price * 0.8)  # 惩罚高价充电
            else:  # 中等电价
                economic_reward = 0.5 - (power * price * 0.6)
                
        reward_components['economic'] = economic_reward
        
        # 🔧 修复2: 重新设计充电完成奖励，提供渐进奖励
        current_soc = next_state.get('soc', 0.0)
        
        if self.ev.user_behavior:
            target_soc = self.ev.user_behavior.target_soc
            min_required_soc = getattr(self.ev.user_behavior, 'min_required_soc', 0.6)
            
            if current_soc >= target_soc:
                completion_reward = 5.0  # 达到目标SOC高奖励
            elif current_soc >= min_required_soc:
                # 达到最低要求SOC后的渐进奖励
                progress = (current_soc - min_required_soc) / (target_soc - min_required_soc)
                completion_reward = 2.0 + progress * 3.0  # 2-5分渐进奖励
            else:
                # 向最低要求SOC努力的奖励
                progress = current_soc / min_required_soc
                completion_reward = progress * 2.0  # 0-2分
        else:
            # 默认目标SOC为0.8
            if current_soc >= 0.8:
                completion_reward = 3.0
            elif current_soc >= 0.6:
                completion_reward = 1.0 + (current_soc - 0.6) / 0.2 * 2.0
            else:
                completion_reward = current_soc / 0.6
                
        reward_components['completion'] = completion_reward
        
        # 🔧 修复3: 重新设计连接性奖励，更合理
        is_connected = next_state.get('connected', False)
        
        if not is_connected:
            if action > 0:
                connection_reward = -2.0  # 尝试给断开的车充电，惩罚
            else:
                connection_reward = 0.0  # 车未连接且不充电，正常
        else:
            # 车已连接
            if action > 0:
                connection_reward = 1.0  # 连接且充电，奖励
            else:
                connection_reward = 0.2  # 连接但不充电，小奖励
                
        reward_components['connection'] = connection_reward
        
        # 🔧 修复4: 添加充电紧急度奖励
        urgency_reward = 0.0
        if is_connected and hasattr(self.ev, 'user_behavior') and self.ev.user_behavior:
            try:
                from datetime import datetime
                now = datetime.now()
                remaining_time = (self.ev.user_behavior.disconnection_time - now).total_seconds() / 3600
                soc_gap = max(0, self.ev.user_behavior.target_soc - current_soc)
                
                if remaining_time > 0 and soc_gap > 0:
                    urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
                    if urgency > 0.7 and action > 0:  # 高紧急度且充电
                        urgency_reward = 2.0 * urgency
                    elif urgency < 0.3 and action <= 0:  # 低紧急度且不充电
                        urgency_reward = 0.5
            except:
                urgency_reward = 0.0
                
        reward_components['urgency'] = urgency_reward
        
        # 🔧 修复5: 重新平衡权重
        total_reward = (
            0.2 * economic_reward +     # 降低经济权重
            0.5 * completion_reward +   # 提高完成奖励权重
            0.2 * connection_reward +   # 连接奖励
            0.1 * urgency_reward        # 紧急度奖励
        )
        
        # 🔧 修复6: 添加基础参与奖励
        base_participation_reward = 0.3
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """获取动作边界"""
        return 0.0, self.ev.params.max_charging_power
    
    def reset_state(self):
        """重置EV状态"""
        self.ev.current_soc = self.ev.params.initial_soc

class PVMDPDevice(DeviceMDPInterface):
    """光伏设备MDP实现（只读设备）"""
    
    def __init__(self, pv_model: PVModel):
        self.pv = pv_model
    
    def get_state_features(self) -> np.ndarray:
        """获取PV状态特征 [当前发电功率, 预测发电功率]"""
        # PV是只读设备，这里返回状态信息
        current_power = 0.0  # 简化实现
        forecast_power = 0.0
        return np.array([current_power, forecast_power])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """PV状态转移（PV不受控制）"""
        # 计算实际发电功率（基于太阳辐照度）
        irradiance = env_state['solar_irradiance']
        max_power = self.pv.params.max_power
        efficiency = self.pv.params.efficiency
        
        # 简化的发电模型
        actual_power = max_power * efficiency * (irradiance / 1000.0) if irradiance > 0 else 0
        
        return {
            'power': actual_power,
            'irradiance': irradiance,
            'efficiency': efficiency
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """计算PV奖励（发电收益）"""
        power_generated = next_state['power']
        price = env_state['price']
        
        # PV发电收益
        generation_reward = power_generated * price
        
        return generation_reward, {'generation': generation_reward}
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """PV没有动作空间"""
        return 0.0, 0.0
    
    def reset_state(self):
        """重置PV状态"""
        pass

class FlexOfferEnv(gym.Env):
    """统一的FlexOffer MDP环境"""
    
    def __init__(
        self,
        devices: Dict[str, Dict],
        time_horizon: int = 24,
        time_step: float = 1.0,
        start_time: datetime = None,
        price_data: pd.DataFrame = None,
        user_preferences: Dict[str, float] = None,
        weather_data: pd.DataFrame = None,
        data_dir: str = "data",
    ):
        """
        初始化统一的FlexOffer环境
        
        Args:
            devices: 设备配置字典
            time_horizon: 时间范围
            time_step: 时间步长
            start_time: 开始时间
            price_data: 价格数据
            user_preferences: 用户偏好
            weather_data: 天气数据
        """
        super().__init__()
        
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time if start_time else datetime.now()
        self.current_time = self.start_time
        self.current_step = 0
        
        # 初始化环境动态，传递data_dir参数
        self.env_dynamics = EnvironmentDynamics(price_data, weather_data, data_dir)
        
        # 初始化用户偏好
        self.user_preferences = {
            "economic": 0.25,
            "comfort": 0.25,
            "self_sufficient": 0.25,
            "environmental": 0.25
        }
        if user_preferences:
            self.user_preferences.update(user_preferences)
            # 归一化
            total = sum(self.user_preferences.values())
            self.user_preferences = {k: v/total for k, v in self.user_preferences.items()}
        
        # 初始化设备MDP
        self.device_mdps = {}
        self.device_ids = []
        self.device_types = {}
        
        for device_id, config in devices.items():
            device_type = config['type']
            device_model = self._create_device_model(device_type, config['params'])
            device_mdp = self._create_device_mdp(device_type, device_model)
            
            self.device_mdps[device_id] = device_mdp
            self.device_ids.append(device_id)
            self.device_types[device_id] = device_type
        
        # 马尔可夫历史状态
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # 定义观测和动作空间
        self._setup_spaces()
    
    def _create_device_model(self, device_type: str, params):
        """创建设备模型"""
        if device_type == DeviceType.BATTERY:
            return BatteryModel(params)
        elif device_type == DeviceType.HEAT_PUMP:
            return HeatPumpModel(params)
        elif device_type == DeviceType.EV:
            return EVModel(params)
        elif device_type == DeviceType.PV:
            return PVModel(params)
        elif device_type == DeviceType.DISHWASHER:
            return DishwasherModel(params)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def _create_device_mdp(self, device_type: str, device_model) -> DeviceMDPInterface:
        """创建设备MDP"""
        if device_type == DeviceType.BATTERY:
            return BatteryMDPDevice(device_model)
        elif device_type == DeviceType.HEAT_PUMP:
            return HeatPumpMDPDevice(device_model)
        elif device_type == DeviceType.EV:
            return EVMDPDevice(device_model)
        elif device_type == DeviceType.PV:
            return PVMDPDevice(device_model)
        elif device_type == DeviceType.DISHWASHER:
            return DishwasherMDPDevice(device_model)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 计算状态空间维度
        # 通用状态: 时间(4) + 环境(5) + 马尔可夫历史(设备数+3) = 12+设备数
        # 设备状态: 每设备的特征维度之和
        env_state_dim = 4 + 5 + len(self.device_ids) + 3  # 环境和马尔可夫状态
        device_state_dim = sum(len(mdp.get_state_features()) for mdp in self.device_mdps.values())
        total_state_dim = env_state_dim + device_state_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_state_dim,), dtype=np.float32
        )
        
        # 动作空间：每个可控设备一个连续动作
        controllable_devices = [
            device_id for device_id in self.device_ids 
            if self.device_types[device_id] != DeviceType.PV
        ]
        
        action_bounds = []
        for device_id in controllable_devices:
            low, high = self.device_mdps[device_id].get_action_bounds()
            action_bounds.append([low, high])
        
        if action_bounds:
            action_bounds = np.array(action_bounds)
            self.action_space = spaces.Box(
                low=action_bounds[:, 0], high=action_bounds[:, 1], dtype=np.float32
            )
        else:
            # 如果没有可控设备，创建一个虚拟动作空间
            self.action_space = spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_time = self.start_time
        self.current_step = 0
        
        # 重置马尔可夫历史
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # 重置环境动态
        self.env_dynamics.price_history = []
        self.env_dynamics.weather_history = []
        
        # 重置所有设备
        for device_mdp in self.device_mdps.values():
            device_mdp.reset_state()
        
        # 获取初始观测
        observation = self._get_observation()
        info = {'time': self.current_time, 'step': self.current_step}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """执行一步"""
        # 获取当前环境状态
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # 映射动作到设备
        device_actions = self._map_actions_to_devices(action)
        
        # 执行设备状态转移
        device_next_states = {}
        total_reward = 0.0
        all_reward_components = {}
        total_cost = 0.0
        
        for device_id, device_action in device_actions.items():
            device_mdp = self.device_mdps[device_id]
            
            # 状态转移
            next_state = device_mdp.transition_state(device_action, env_state)
            device_next_states[device_id] = next_state
            
            # 计算奖励
            device_reward, reward_components = device_mdp.calculate_reward(
                device_action, next_state, env_state
            )
            
            total_reward += device_reward
            all_reward_components[device_id] = reward_components
            
            # 累积成本
            if 'power' in next_state:
                cost = next_state['power'] * env_state['price'] * self.time_step
                total_cost += cost
        
        # 应用用户偏好权重
        weighted_reward = self._apply_user_preferences(total_reward, all_reward_components)
        
        # 更新马尔可夫历史
        self.markov_history['prev_actions'] = np.array(list(device_actions.values()))
        self.markov_history['prev_reward'] = weighted_reward
        self.markov_history['cumulative_cost'] += total_cost
        self.markov_history['cumulative_energy'] += sum(abs(a) for a in device_actions.values()) * self.time_step
        
        # 更新时间
        self.current_time += timedelta(hours=self.time_step)
        self.current_step += 1
        
        # 检查终止条件
        done = self.current_step >= self.time_horizon
        
        # 获取下一个观测
        next_observation = self._get_observation()
        
        # 构建信息字典
        info = {
            'time': self.current_time,
            'step': self.current_step,
            'device_states': device_next_states,
            'reward_components': all_reward_components,
            'total_cost': total_cost,
            'env_state': env_state
        }
        
        return next_observation, weighted_reward, done, False, info
    
    def _map_actions_to_devices(self, action: np.ndarray) -> Dict[str, float]:
        """映射动作到设备"""
        device_actions = {}
        action_idx = 0
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            
            if device_type == DeviceType.PV:
                # PV设备不可控
                device_actions[device_id] = 0.0
            else:
                # 可控设备
                if action_idx < len(action):
                    device_actions[device_id] = float(action[action_idx])
                    action_idx += 1
                else:
                    device_actions[device_id] = 0.0
        
        return device_actions
    
    def _apply_user_preferences(self, base_reward: float, reward_components: Dict) -> float:
        """应用用户偏好权重"""
        # 这里可以根据reward_components中的不同组件应用用户偏好
        # 简化实现：直接返回基础奖励
        return base_reward
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测状态"""
        # 时间特征
        hour = self.current_time.hour
        time_features = np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if self.current_time.weekday() < 5 else 0.0,
            self.current_step / self.time_horizon
        ])
        
        # 环境特征
        env_state = self.env_dynamics.get_current_state(self.current_time)
        env_features = np.array([
            env_state['price'],
            env_state['price_trend'],
            env_state['temperature'],
            env_state['solar_irradiance'],
            env_state['weather_trend']['temperature_trend']
        ])
        
        # 马尔可夫历史特征
        markov_features = np.concatenate([
            self.markov_history['prev_actions'],
            [self.markov_history['prev_reward']],
            [self.markov_history['cumulative_cost']],
            [self.markov_history['cumulative_energy']]
        ])
        
        # 设备状态特征
        device_features = []
        for device_id in self.device_ids:
            device_state = self.device_mdps[device_id].get_state_features()
            device_features.append(device_state)
        
        device_features = np.concatenate(device_features)
        
        # 合并所有特征
        full_observation = np.concatenate([
            time_features,
            env_features,
            markov_features,
            device_features
        ])
        
        return full_observation.astype(np.float32)
    
    def generate_dfo(self) -> Dict[str, DFOSystem]:
        """生成DFO系统（与FlexOffer流程集成）"""
        dfo_systems = {}
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            device_mdp = self.device_mdps[device_id]
            
            if device_type != DeviceType.PV:  # 只为可控设备生成DFO
                dfo = DFOSystem(self.time_horizon)
                
                for t in range(self.time_horizon):
                    # 获取动作边界
                    p_min, p_max = device_mdp.get_action_bounds()
                    
                    # 创建时间片
                    dfo_slice = DFOSlice(
                        time_step=t,
                        energy_min=p_min * self.time_step,
                        energy_max=p_max * self.time_step,
                        constraints=[]
                    )
                    
                    dfo.add_slice(dfo_slice)
                
                dfo_systems[device_id] = dfo
        
        return dfo_systems

# 向后兼容的别名
FlexOfferEnvMDP = FlexOfferEnv 
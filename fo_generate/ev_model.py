from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class EVParameters:
    """电动汽车参数"""
    ev_id: str           # 电动汽车ID
    battery_capacity: float  # 电池容量 (kWh)
    soc_min: float        # 最小荷电状态
    soc_max: float        # 最大荷电状态
    max_charging_power: float  # 最大充电功率 (kW)
    efficiency: float     # 充电效率
    initial_soc: float    # 初始荷电状态
    fast_charge_capable: bool  # 是否支持快充
    behavior: Optional['EVUserBehavior'] = None  # 用户行为

@dataclass
class EVUserBehavior:
    """电动汽车用户行为模型"""
    ev_id: str           # 电动汽车ID
    connection_time: datetime  # 连接时刻（EV接入充电桩的时间）
    disconnection_time: datetime  # 断开时刻（EV离开充电桩的时间）
    next_departure_time: datetime  # 第二天使用时间（需要保证在此之前充到足够电量）
    target_soc: float     # 目标荷电状态
    min_required_soc: float  # 最小所需荷电状态（不能影响用户使用）
    fast_charge_preferred: bool  # 是否偏好快充
    location: str         # 充电位置
    priority: int         # 优先级 (1-5，5最高)
    charge_flexibility: float = 0.8  # 充电灵活性（0-1，1表示可以完全断断续续充电）

class EVModel:
    """电动汽车模型类"""
    def __init__(self, params: EVParameters, user_behavior: Optional[EVUserBehavior] = None):
        self.params = params
        self.user_behavior = user_behavior
        self.current_soc = params.initial_soc
        self.is_connected = False  # 是否已连接充电桩
        self.connection_start_time = None  # 连接开始时间
        self.last_charge_time = None  # 上次充电时间
        self.total_charged_energy = 0.0  # 总充电能量
        
    def connect(self, current_time: datetime):
        """连接充电桩"""
        if not self.is_connected:
            self.is_connected = True
            self.connection_start_time = current_time
            
    def disconnect(self, current_time: datetime):
        """断开充电桩连接"""
        if self.is_connected:
            self.is_connected = False
            self.connection_start_time = None
    
    def is_available_for_charging(self, current_time: datetime) -> bool:
        """检查是否可以充电（是否在连接时间段内）"""
        if not self.user_behavior:
            return self.is_connected
            
        # 检查是否在连接时间范围内
        in_connection_period = (self.user_behavior.connection_time <= current_time < self.user_behavior.disconnection_time)
        
        return in_connection_period

    def update_soc(self, power: float, time_step: float = 1.0, current_time: Optional[datetime] = None) -> float:
        """更新荷电状态"""
        # 只有在连接时间段内才能充电
        if current_time and not self.is_available_for_charging(current_time):
            return self.current_soc
            
        # 只处理充电，电动汽车通常不放电回电网 (V2G模式除外)
        if power > 0:
            energy = power * time_step * self.params.efficiency
            self.current_soc += energy / self.params.battery_capacity
            self.total_charged_energy += energy
            if current_time:
                self.last_charge_time = current_time
            
        # 确保SOC在合理范围内
        self.current_soc = np.clip(self.current_soc, self.params.soc_min, self.params.soc_max)
        
        return self.current_soc
        
    def get_available_power(self, current_time: datetime) -> Tuple[float, float]:
        """获取可用功率范围"""
        # 如果没有用户行为信息，使用基本参数
        if not self.user_behavior:
            # 只能充电，不能放电
            p_min = 0
            # 最大充电功率
            p_max = self.params.max_charging_power
            
            # 考虑当前SOC约束
            remaining_capacity = (self.params.soc_max - self.current_soc) * self.params.battery_capacity
            p_max = min(p_max, remaining_capacity / self.params.efficiency)
            
            return p_min, p_max
            
        # 检查车辆是否在连接时间段内
        if not self.is_available_for_charging(current_time):
            return 0, 0  # 不在连接时间段，无法充电
            
        # 计算到第二天使用前的剩余时间（小时）
        time_to_next_use = (self.user_behavior.next_departure_time - current_time).total_seconds() / 3600
        time_in_connection = (self.user_behavior.disconnection_time - current_time).total_seconds() / 3600
        
        # 使用较小的时间作为充电时间约束
        available_charging_time = min(time_to_next_use, time_in_connection)
        
        if available_charging_time <= 0:
            return 0, 0
        
        # 计算需要充电的能量
        target_energy = (self.user_behavior.target_soc - self.current_soc) * self.params.battery_capacity
        min_required_energy = (self.user_behavior.min_required_soc - self.current_soc) * self.params.battery_capacity
        
        # 计算最大充电功率
        remaining_capacity = (self.params.soc_max - self.current_soc) * self.params.battery_capacity
        p_max = min(self.params.max_charging_power, remaining_capacity / self.params.efficiency)
        
        # 根据用户偏好和时间约束调整充电功率
        if self.user_behavior.fast_charge_preferred and self.params.fast_charge_capable:
            # 快充模式：尽可能快速充电
            p_max = self.params.max_charging_power
        else:
            # 根据充电灵活性调整最大功率
            if self.user_behavior.charge_flexibility > 0.5:
                # 高灵活性：可以断断续续充电，降低峰值功率
                p_max = min(p_max, self.params.max_charging_power * 0.7)
        
        # 计算最小充电功率
        p_min = 0  # EV可以断断续续充电，所以最小功率可以为0
        
        # 如果SOC低于最小要求且时间紧迫，需要设置最小充电功率
        if min_required_energy > 0 and available_charging_time < 8:  # 如果剩余时间少于8小时
            min_power_needed = min_required_energy / (available_charging_time * self.params.efficiency)
            p_min = min(min_power_needed, p_max)
        
        # 如果已经达到目标SOC，不需要充电
        if self.current_soc >= self.user_behavior.target_soc:
            p_max = 0
            
        return max(0, p_min), max(0, p_max)
        
    def generate_dfo(self, 
                     start_time=None, 
                     time_horizon: Optional[int] = None, 
                     time_step: float = 1.0) -> DFOSystem:
        """生成DFO系统
        
        Args:
            start_time: 可选，起始时间，如果为None，使用当前时间。
                        如果是整数且time_horizon为None，则作为time_horizon使用
            time_horizon: 时间范围，如果为None且第一个参数为整数，则使用第一个参数
            time_step: 时间步长，默认为1小时
            
        Returns:
            DFO系统对象
        """
        # 兼容旧的调用方式，如果第一个参数是整数且第二个参数为None
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # 如果start_time为None，使用当前时间
        current_time = start_time if start_time is not None and not isinstance(start_time, int) else datetime.now()
        
        # 确保time_horizon有值
        if time_horizon is None:
            time_horizon = 24  # 默认值为24小时
        
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # 计算能量边界
            energy_min, energy_max = self.get_available_power(current_time)
            
            # 创建约束
            constraints = []
            
            # 添加SOC约束
            soc_constraint = np.array([1.0, -1.0])  # SOC >= min, SOC <= max
            constraints.append((soc_constraint, self.params.soc_max - self.current_soc))
            constraints.append((-soc_constraint, self.current_soc - self.params.soc_min))
            
            # 如果有用户行为信息，添加目标SOC约束
            if self.user_behavior and t == time_horizon - 1:
                target_constraint = np.array([1.0])  # SOC >= target_soc
                constraints.append((target_constraint, self.user_behavior.target_soc - self.current_soc))
            
            # 创建时间片
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
            # 更新时间
            current_time += timedelta(hours=time_step)
            
            # 模拟一个时间步后的状态变化
            avg_power = (energy_min + energy_max) / 2
            self.update_soc(avg_power, time_step)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, behavior_file: Optional[str] = None, ev_id: Optional[str] = None) -> 'EVModel':
        """从CSV文件创建电动汽车模型"""
        # 读取参数文件
        params_df = pd.read_csv(params_file, comment='#')
        
        # 如果指定了ev_id，查找对应数据；否则使用第一行
        if ev_id:
            ev_data = params_df[params_df['ev_id'] == ev_id]
            if ev_data.empty:
                raise ValueError(f"EV ID {ev_id} not found in {params_file}")
            ev_data = ev_data.iloc[0]
        else:
            ev_data = params_df.iloc[0]
            ev_id = ev_data['ev_id']
        
        # 创建参数对象
        params = EVParameters(
            ev_id=ev_id,
            battery_capacity=float(ev_data['battery_capacity']),
            soc_min=float(ev_data['soc_min']),
            soc_max=float(ev_data['soc_max']),
            max_charging_power=float(ev_data['max_charging_power']),
            efficiency=float(ev_data['efficiency']),
            initial_soc=float(ev_data['initial_soc']),
            fast_charge_capable=ev_data['fast_charge_capable'] == 'True'
        )
        
        # 如果提供了行为文件，读取用户行为
        user_behavior = None
        if behavior_file:
            behavior_df = pd.read_csv(behavior_file, comment='#')
            behavior_data = behavior_df[behavior_df['ev_id'] == ev_id]
            
            if not behavior_data.empty:
                behavior_data = behavior_data.iloc[0]
                connection_time = datetime.strptime(behavior_data['arrival_time'], '%Y-%m-%d %H:%M:%S')
                disconnection_time = datetime.strptime(behavior_data['departure_time'], '%Y-%m-%d %H:%M:%S')
                user_behavior = EVUserBehavior(
                    ev_id=ev_id,
                    connection_time=connection_time,
                    disconnection_time=disconnection_time,
                    next_departure_time=disconnection_time,  # 使用断开时间作为下次出发时间
                    target_soc=float(behavior_data['target_soc']),
                    fast_charge_preferred=behavior_data['fast_charge_preferred'] == 'True',
                    min_required_soc=float(behavior_data['min_required_soc']),
                    location=behavior_data['location'],
                    priority=int(behavior_data['priority'])
                )
        
        return cls(params, user_behavior)
        
    @classmethod
    def get_all_ev_ids(cls, params_file: str) -> List[str]:
        """获取CSV文件中所有的电动汽车ID"""
        df = pd.read_csv(params_file, comment='#')
        return df['ev_id'].tolist() 
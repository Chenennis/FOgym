from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class BatteryParameters:
    """电池参数"""
    battery_id: str   # 电池ID
    soc_min: float    # 最小荷电状态
    soc_max: float    # 最大荷电状态
    p_min: float      # 最小功率
    p_max: float      # 最大功率
    efficiency: float # 效率
    initial_soc: float # 初始荷电状态
    battery_type: str  # 电池类型
    capacity_kwh: float # 容量

@dataclass
class BatteryScheduleParams:
    """电池调度参数"""
    battery_id: str     # 电池ID
    time_horizon: int   # 时间范围
    start_time: datetime # 开始时间
    end_time: datetime   # 结束时间
    schedule_type: str   # 调度类型
    priority: int        # 优先级
    available_period: str # 可用时段
    target_soc: float    # 目标SOC
    location: str        # 位置

class BatteryModel:
    """电池模型类"""
    def __init__(self, params: BatteryParameters, schedule_params: Optional[BatteryScheduleParams] = None):
        self.params = params
        self.schedule_params = schedule_params
        self.current_soc = params.initial_soc
        
    def update_soc(self, power: float, time_step: float = 1.0) -> float:
        """更新荷电状态"""
        if power > 0:  # 充电
            self.current_soc += power * time_step * self.params.efficiency
        else:  # 放电
            self.current_soc += power * time_step / self.params.efficiency
        return self.current_soc
        
    def get_available_power(self) -> Tuple[float, float]:
        """获取可用功率范围"""
        # 基于当前SOC计算可用功率
        max_charge = (self.params.soc_max - self.current_soc) / self.params.efficiency
        max_discharge = (self.current_soc - self.params.soc_min) * self.params.efficiency
        
        p_min = max(self.params.p_min, -max_discharge)
        p_max = min(self.params.p_max, max_charge)
        
        return p_min, p_max
        
    def generate_dfo(self, time_horizon: int) -> DFOSystem:
        """生成DFO系统"""
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # 计算能量边界
            p_min, p_max = self.get_available_power()
            energy_min = p_min
            energy_max = p_max
            
            # 创建约束
            constraints = []
            # 添加SOC约束
            soc_constraint = np.array([1.0, -1.0])  # SOC >= min, SOC <= max
            constraints.append((soc_constraint, self.params.soc_max - self.current_soc))
            constraints.append((-soc_constraint, self.current_soc - self.params.soc_min))
            
            # 创建时间片
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
            # 更新SOC（假设按照可用功率的中间值进行）
            # 这里仅用于模拟，实际调度时应该使用实际功率
            avg_power = (energy_min + energy_max) / 2
            self.update_soc(avg_power)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, schedule_file: str, battery_id: str) -> 'BatteryModel':
        """从CSV文件创建电池模型"""
        # 读取参数文件
        params_df = pd.read_csv(params_file, comment='#')
        
        # 查找对应battery_id的行
        battery_data = params_df[params_df['battery_id'] == battery_id]
        if battery_data.empty:
            raise ValueError(f"Battery ID {battery_id} not found in {params_file}")
        
        battery_data = battery_data.iloc[0]
        
        # 创建参数对象
        params = BatteryParameters(
            battery_id=battery_id,
            soc_min=float(battery_data['soc_min']),
            soc_max=float(battery_data['soc_max']),
            p_min=float(battery_data['p_min']),
            p_max=float(battery_data['p_max']),
            efficiency=float(battery_data['efficiency']),
            initial_soc=float(battery_data['initial_soc']),
            battery_type=battery_data['battery_type'],
            capacity_kwh=float(battery_data['capacity_kwh'])
        )
        
        # 读取调度文件
        schedule_df = pd.read_csv(schedule_file, comment='#')
        
        # 查找对应battery_id的行
        schedule_data = schedule_df[schedule_df['battery_id'] == battery_id]
        if schedule_data.empty:
            return cls(params)
        
        schedule_data = schedule_data.iloc[0]
        
        # 创建调度参数对象
        schedule_params = BatteryScheduleParams(
            battery_id=battery_id,
            time_horizon=int(schedule_data['time_horizon']),
            start_time=datetime.strptime(schedule_data['start_time'], '%Y-%m-%d %H:%M:%S'),
            end_time=datetime.strptime(schedule_data['end_time'], '%Y-%m-%d %H:%M:%S'),
            schedule_type=schedule_data['调度类型'],
            priority=int(schedule_data['优先级']),
            available_period=schedule_data['可用时段'],
            target_soc=float(schedule_data['所需SOC']),
            location=schedule_data['位置']
        )
        
        return cls(params, schedule_params)
        
    @classmethod
    def get_all_battery_ids(cls, params_file: str) -> List[str]:
        """获取CSV文件中所有的电池ID"""
        df = pd.read_csv(params_file, comment='#')
        return df['battery_id'].tolist() 
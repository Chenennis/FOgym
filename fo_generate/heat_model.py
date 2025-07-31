from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class HeatPumpParameters:
    """热泵参数"""
    room_id: str         # 房间ID
    room_area: float      # 房间面积
    room_volume: float    # 房间体积
    temp_min: float      # 最小温度
    temp_max: float      # 最大温度
    initial_temp: float  # 初始温度
    cop: float          # 性能系数
    heat_loss_coef: float  # 热损失系数
    primary_use_period: str  # 主要使用时段
    secondary_use_period: str  # 次要使用时段
    primary_target_temp: float  # 主要时段目标温度
    secondary_target_temp: float  # 次要时段目标温度
    max_power: float     # 最大功率

class HeatPumpModel:
    """热泵模型类"""
    def __init__(self, params: HeatPumpParameters):
        self.params = params
        self.current_temp = params.initial_temp
        
    def calculate_heat_required(self, target_temp: float) -> float:
        """计算达到目标温度所需的热量"""
        temp_diff = target_temp - self.current_temp
        heat_required = self.params.room_volume * temp_diff
        return heat_required
        
    def update_temperature(self, heat_energy: float, time_step: float = 1.0) -> float:
        """更新温度"""
        # 考虑热损失
        heat_loss = self.params.heat_loss_coef * (self.current_temp - self.params.temp_min)
        net_heat = heat_energy - heat_loss * time_step
        
        # 更新温度
        temp_change = net_heat / (self.params.room_volume)
        self.current_temp += temp_change
        
        return self.current_temp
        
    def get_available_heat(self) -> Tuple[float, float]:
        """获取可用热量范围"""
        # 计算达到最大温度所需的热量
        max_heat = self.calculate_heat_required(self.params.temp_max)
        # 计算达到最小温度所需的热量（负值表示需要冷却）
        min_heat = self.calculate_heat_required(self.params.temp_min)
        
        return min_heat, max_heat
        
    def generate_dfo(self, time_horizon: int) -> DFOSystem:
        """生成DFO系统"""
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # 计算热量边界
            heat_min, heat_max = self.get_available_heat()
            
            # 转换为电能（考虑COP）
            energy_min = heat_min / self.params.cop
            energy_max = heat_max / self.params.cop
            
            # 创建约束
            constraints = []
            # 添加温度约束
            temp_constraint = np.array([1.0, -1.0])  # T >= min, T <= max
            constraints.append((temp_constraint, self.params.temp_max - self.current_temp))
            constraints.append((-temp_constraint, self.current_temp - self.params.temp_min))
            
            # 创建时间片
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
        return dfo

    @classmethod
    def from_csv(cls, csv_file: str, room_id: str) -> 'HeatPumpModel':
        """从CSV文件创建热泵模型"""
        # 读取CSV文件
        df = pd.read_csv(csv_file, comment='#')
        
        # 查找对应room_id的行
        room_data = df[df['room_id'] == room_id]
        if room_data.empty:
            raise ValueError(f"Room ID {room_id} not found in {csv_file}")
        
        room_data = room_data.iloc[0]
        
        # 创建参数对象
        params = HeatPumpParameters(
            room_id=room_id,
            room_area=float(room_data['room_area']),
            room_volume=float(room_data['room_volume']),
            temp_min=float(room_data['temp_min']),
            temp_max=float(room_data['temp_max']),
            initial_temp=float(room_data['initial_temp']),
            cop=float(room_data['cop']),
            heat_loss_coef=float(room_data['heat_loss_coef']),
            primary_use_period=room_data['主要使用时段'],
            secondary_use_period=room_data['次要使用时段'],
            primary_target_temp=float(room_data['主要时段目标温度']),
            secondary_target_temp=float(room_data['次要时段目标温度']),
            max_power=float(room_data['最大功率'])
        )
        
        return cls(params)
        
    @classmethod
    def get_all_room_ids(cls, csv_file: str) -> List[str]:
        """获取CSV文件中所有的房间ID"""
        df = pd.read_csv(csv_file, comment='#')
        return df['room_id'].tolist() 
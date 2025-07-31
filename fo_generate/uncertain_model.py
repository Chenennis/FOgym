from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Dict
import numpy as np
import math
import pandas as pd
from scipy import stats
from fo_generate.dfo import DFOSystem, DFOSlice
from fo_generate.sfo import SFOSystem, SFOSlice

@dataclass
class UncertainParameters:
    """不确定性参数"""
    time_step: str           # 时间步
    probability_threshold: float  # 概率阈值 P_0
    default_energy: float    # 默认能量值 d_t
    energy_range: np.ndarray  # 能量范围
    probability_function: Callable[[float], float]  # 概率函数 f_t
    time_availability: float  # 时间可用性 P_t
    energy_type: str         # 能源类型
    min_value: float         # 最小可能值
    max_value: float         # 最大可能值

class UncertainModel:
    """不确定性模型类"""
    def __init__(self, params_list: List[UncertainParameters]):
        self.params_list = params_list
        
    def calculate_probability(self, energy: float, time_step_idx: int) -> float:
        """计算给定能量的概率"""
        return self.params_list[time_step_idx].probability_function(energy)
        
    def find_energy_bounds(self, time_step_idx: int, p_r_t: float) -> Tuple[float, float]:
        """查找满足概率阈值的能量边界"""
        params = self.params_list[time_step_idx]
        
        # 初始化边界为默认值
        energy_min = params.default_energy
        energy_max = params.default_energy
        
        # 遍历能量范围
        for energy in params.energy_range:
            prob = self.calculate_probability(energy, time_step_idx)
            if prob >= p_r_t:
                energy_min = min(energy_min, energy)
                energy_max = max(energy_max, energy)
                
        return energy_min, energy_max
        
    def generate_sfo(self, time_horizon: Optional[int] = None) -> SFOSystem:
        """生成SFO系统"""
        # 如果未指定时间范围，使用params_list的长度
        if time_horizon is None:
            time_horizon = len(self.params_list)
        else:
            time_horizon = min(time_horizon, len(self.params_list))
            
        # 创建SFO系统
        sfo = SFOSystem(time_horizon)
        
        # 检查每个时间步的时间可用性
        total_time_availability = min([params.time_availability for params in self.params_list[:time_horizon]])
        total_probability_threshold = max([params.probability_threshold for params in self.params_list[:time_horizon]])
        
        # Step 1: 检查时间可用性
        if total_time_availability < total_probability_threshold:
            # 如果时间可用性不足，返回默认值
            for t in range(time_horizon):
                slice = SFOSlice(
                    time_step=t,
                    energy_min=self.params_list[t].default_energy,
                    energy_max=self.params_list[t].default_energy
                )
                sfo.add_slice(slice)
            return sfo
            
        # Step 2: 计算剩余能量层面的置信要求
        p_r = total_probability_threshold / total_time_availability
        
        # Step 3: 均匀分配能量置信度
        p_r_t = math.pow(p_r, 1/time_horizon)
        
        # Step 4: 计算每个时间点的能量范围
        for t in range(time_horizon):
            e_min, e_max = self.find_energy_bounds(t, p_r_t)
            slice = SFOSlice(
                time_step=t,
                energy_min=e_min,
                energy_max=e_max
            )
            sfo.add_slice(slice)
            
        # Step 5: 返回SFO
        return sfo
        
    def generate_dfo(self, time_horizon: Optional[int] = None) -> DFOSystem:
        """将SFO转换为DFO系统"""
        sfo = self.generate_sfo(time_horizon)
        return sfo.to_dfo()
        
    @classmethod
    def from_csv(cls, csv_file: str, energy_type: str = None) -> 'UncertainModel':
        """从CSV文件创建不确定性模型"""
        # 读取CSV文件
        df = pd.read_csv(csv_file, comment='#')
        
        # 如果指定了能源类型，则进行过滤
        if energy_type:
            df = df[df['energy_type'] == energy_type]
        
        params_list = []
        
        for _, row in df.iterrows():
            # 解析概率分布参数
            prob_type = row['probability_type']
            params_str = row['parameters']
            
            # 创建能量范围
            min_value = float(row['min_value'])
            max_value = float(row['max_value'])
            energy_range = np.arange(min_value, max_value, 0.1)
            
            # 根据概率分布类型创建概率函数
            if prob_type == 'normal':
                # 解析正态分布参数
                params_dict = dict(param.split('=') for param in params_str.split(';'))
                mean = float(params_dict['mean'])
                std = float(params_dict['std'])
                
                # 创建概率函数
                def prob_func(energy, mean=mean, std=std):
                    if std == 0:  # 处理标准差为0的情况
                        return 1.0 if energy == mean else 0.0
                    return stats.norm.pdf(energy, mean, std) / stats.norm.pdf(mean, mean, std)
            else:
                # 默认为均匀分布
                def prob_func(energy, min_val=min_value, max_val=max_value):
                    return 1.0 if min_val <= energy <= max_val else 0.0
            
            # 创建参数对象
            params = UncertainParameters(
                time_step=row['time_step'],
                probability_threshold=float(row['confidence']),
                default_energy=float(row['default_value']),
                energy_range=energy_range,
                probability_function=prob_func,
                time_availability=0.98,  # 默认时间可用性为0.98
                energy_type=row['energy_type'],
                min_value=min_value,
                max_value=max_value
            )
            
            params_list.append(params)
        
        return cls(params_list)
        
    @classmethod
    def get_energy_types(cls, csv_file: str) -> List[str]:
        """获取CSV文件中所有的能源类型"""
        df = pd.read_csv(csv_file, comment='#')
        return df['energy_type'].unique().tolist() 
from dataclasses import dataclass
from typing import List, Tuple, Dict, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class SFOSlice:
    """表示单个时间片的SFO"""
    time_step: int
    energy_min: float
    energy_max: float

class SFOSystem:
    """统一的SFO系统类"""
    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon
        self.slices: List[SFOSlice] = []
        
    def add_slice(self, slice: SFOSlice):
        """添加时间片"""
        self.slices.append(slice)
        
    def get_energy_bounds(self, time_step: int) -> Tuple[float, float]:
        """获取指定时间步的能量边界"""
        return self.slices[time_step].energy_min, self.slices[time_step].energy_max
        
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'time_horizon': self.time_horizon,
            'e_min': [s.energy_min for s in self.slices],
            'e_max': [s.energy_max for s in self.slices]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'SFOSystem':
        """从字典创建SFO系统"""
        system = cls(data['time_horizon'])
        for t in range(len(data['e_min'])):
            slice = SFOSlice(
                time_step=t,
                energy_min=data['e_min'][t],
                energy_max=data['e_max'][t]
            )
            system.add_slice(slice)
        return system
        
    def to_dfo(self) -> Any:
        """将SFO转换为DFO"""
        # 导入放在函数内部避免循环导入
        from fo_generate.dfo import DFOSystem, DFOSlice
        
        dfo = DFOSystem(self.time_horizon)
        
        for s in self.slices:
            # 创建时间片
            dfo_slice = DFOSlice(
                time_step=s.time_step,
                energy_min=s.energy_min,
                energy_max=s.energy_max,
                constraints=[]  # SFO没有约束条件
            )
            dfo.add_slice(dfo_slice)
            
        return dfo 
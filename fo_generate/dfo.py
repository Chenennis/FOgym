from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np

@dataclass
class DFOSlice:
    """表示单个时间片的DFO - 包含FlexOffer的核心属性"""
    time_step: int
    energy_min: float
    energy_max: float
    constraints: List[Tuple[np.ndarray, float]]  # (A, b) where Ax <= b
    power_min: float = 0.0  # Minimum power (kW)
    power_max: float = 0.0  # Maximum power (kW)
    start_time: Optional[datetime] = None  # Start time of time window
    end_time: Optional[datetime] = None    # End time of time window
    flexibility_factor: float = 0.5  # Flexibility factor [0, 1]
    device_type: str = "unknown"  # Device type
    device_id: str = ""    # Device ID
    
    def get_duration_hours(self) -> float:
        """get duration of time window (hours)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600.0
        return 1.0  # default 1 hour
    
    def get_energy_range(self) -> float:
        """get energy range"""
        return self.energy_max - self.energy_min
    
    def get_power_range(self) -> float:
        """get power range"""
        return self.power_max - self.power_min

class DFOSystem:
    def __init__(self, time_horizon: int, device_id: str = "", device_type: str = "unknown"):
        self.time_horizon = time_horizon
        self.device_id = device_id
        self.device_type = device_type
        self.slices: List[DFOSlice] = []
        
    def add_slice(self, slice: DFOSlice):
        """add time slice"""
        self.slices.append(slice)
        
    def get_energy_bounds(self, time_step: int) -> Tuple[float, float]:
        """get energy bounds for a given time step"""
        if time_step < len(self.slices):
            return self.slices[time_step].energy_min, self.slices[time_step].energy_max
        return 0.0, 0.0
        
    def get_power_bounds(self, time_step: int) -> Tuple[float, float]:
        """get power bounds for a given time step"""
        if time_step < len(self.slices):
            return self.slices[time_step].power_min, self.slices[time_step].power_max
        return 0.0, 0.0
        
    def get_constraints(self, time_step: int) -> List[Tuple[np.ndarray, float]]:
        """get constraints for a given time step"""
        if time_step < len(self.slices):
            return self.slices[time_step].constraints
        return []
    
    def get_time_window(self, time_step: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """get time window for a given time step"""
        if time_step < len(self.slices):
            return self.slices[time_step].start_time, self.slices[time_step].end_time
        return None, None
    
    def get_total_energy(self) -> Tuple[float, float]:
        """get total energy range"""
        total_min = sum(s.energy_min for s in self.slices)
        total_max = sum(s.energy_max for s in self.slices)
        return total_min, total_max
        
    def to_dict(self) -> dict:
        """convert to dictionary format"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'time_horizon': self.time_horizon,
            'slices': [
                {
                    'time_step': s.time_step,
                    'energy_min': s.energy_min,
                    'energy_max': s.energy_max,
                    'power_min': s.power_min,
                    'power_max': s.power_max,
                    'start_time': s.start_time.isoformat() if s.start_time else None,
                    'end_time': s.end_time.isoformat() if s.end_time else None,
                    'flexibility_factor': s.flexibility_factor,
                    'device_type': s.device_type,
                    'device_id': s.device_id,
                    'constraints': [(a.tolist(), b) for a, b in s.constraints]
                }
                for s in self.slices
            ]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'DFOSystem':
        system = cls(
            data['time_horizon'], 
            data.get('device_id', ''), 
            data.get('device_type', 'unknown')
        )
        
        # restore time slices
        for slice_data in data['slices']:
            start_time = None
            end_time = None
            if slice_data.get('start_time'):
                start_time = datetime.fromisoformat(slice_data['start_time'])
            if slice_data.get('end_time'):
                end_time = datetime.fromisoformat(slice_data['end_time'])
                
            slice = DFOSlice(
                time_step=slice_data['time_step'],
                energy_min=slice_data['energy_min'],
                energy_max=slice_data['energy_max'],
                constraints=[(np.array(a), b) for a, b in slice_data['constraints']],
                power_min=slice_data.get('power_min', 0.0),
                power_max=slice_data.get('power_max', 0.0),
                start_time=start_time,
                end_time=end_time,
                flexibility_factor=slice_data.get('flexibility_factor', 0.5),
                device_type=slice_data.get('device_type', 'unknown'),
                device_id=slice_data.get('device_id', '')
            )
            system.add_slice(slice)
        
        return system 
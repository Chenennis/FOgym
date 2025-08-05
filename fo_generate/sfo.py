from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from datetime import datetime
import numpy as np

if False:
    from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class SFOSlice:
    """Represents a single time slice of SFO"""
    time_step: int
    energy_min: float
    energy_max: float

class SFOSystem:
    """SFO System class"""
    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon
        self.slices: List[SFOSlice] = []
        
    def add_slice(self, slice: SFOSlice):
        """Add time slice"""
        self.slices.append(slice)
        
    def get_energy_bounds(self, time_step: int) -> Tuple[float, float]:
        """Get energy bounds for specified time step"""
        if time_step < len(self.slices):
            return self.slices[time_step].energy_min, self.slices[time_step].energy_max
        return 0.0, 0.0
        
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'time_horizon': self.time_horizon,
            'slices': [{'time_step': s.time_step, 'energy_min': s.energy_min, 'energy_max': s.energy_max} for s in self.slices]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'SFOSystem':
        """Create SFO system from dictionary"""
        system = cls(data['time_horizon'])
        for slice_data in data['slices']:
            slice = SFOSlice(
                time_step=slice_data['time_step'],
                energy_min=slice_data['energy_min'],
                energy_max=slice_data['energy_max']
            )
            system.add_slice(slice)
        return system
        
    def to_dfo(self) -> Any:
        """Convert to DFO format"""
        # Import here to avoid circular imports
        from fo_generate.dfo import DFOSystem, DFOSlice
        
        dfo = DFOSystem(self.time_horizon)
        for s in self.slices:
            dfo_slice = DFOSlice(
                time_step=s.time_step,
                energy_min=s.energy_min,
                energy_max=s.energy_max,
                constraints=[]  # No constraints in simple conversion
            )
            dfo.add_slice(dfo_slice)
        return dfo 
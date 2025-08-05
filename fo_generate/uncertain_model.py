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
    """Uncertainty parameters"""
    time_step: str           # Time step
    probability_threshold: float  # Probability threshold P_0
    default_energy: float    # Default energy value d_t
    energy_range: np.ndarray  # Energy range
    probability_function: Callable[[float], float]  # Probability function f_t
    time_availability: float  # Time availability P_t
    energy_type: str         # Energy type
    min_value: float         # Minimum possible value
    max_value: float         # Maximum possible value

class UncertainModel:
    """Uncertainty model class"""
    def __init__(self, params_list: List[UncertainParameters]):
        self.params_list = params_list
        
    def calculate_probability(self, energy: float, time_step_idx: int) -> float:
        """Calculate probability for given energy"""
        return self.params_list[time_step_idx].probability_function(energy)
        
    def find_energy_bounds(self, time_step_idx: int, p_r_t: float) -> Tuple[float, float]:
        """Find energy bounds that satisfy probability threshold"""
        params = self.params_list[time_step_idx]
        
        # Initialize bounds to default values
        energy_min = params.default_energy
        energy_max = params.default_energy
        
        # Iterate through energy range
        for energy in params.energy_range:
            prob = self.calculate_probability(energy, time_step_idx)
            if prob >= p_r_t:
                energy_min = min(energy_min, energy)
                energy_max = max(energy_max, energy)
                
        return energy_min, energy_max
        
    def generate_sfo(self, time_horizon: Optional[int] = None) -> SFOSystem:
        """Generate SFO system"""
        # If time horizon not specified, use length of params_list
        if time_horizon is None:
            time_horizon = len(self.params_list)
        else:
            time_horizon = min(time_horizon, len(self.params_list))
            
        # Create SFO system
        sfo = SFOSystem(time_horizon)
        
        # Check time availability for each time step
        total_time_availability = min([params.time_availability for params in self.params_list[:time_horizon]])
        total_probability_threshold = max([params.probability_threshold for params in self.params_list[:time_horizon]])
        
        # Step 1: Check time availability
        if total_time_availability < total_probability_threshold:
            # If time availability insufficient, return default values
            for t in range(time_horizon):
                slice = SFOSlice(
                    time_step=t,
                    energy_min=self.params_list[t].default_energy,
                    energy_max=self.params_list[t].default_energy
                )
                sfo.add_slice(slice)
            return sfo
            
        # Step 2: Calculate remaining energy level confidence requirement
        p_r = total_probability_threshold / total_time_availability
        
        # Step 3: Distribute energy confidence uniformly
        p_r_t = math.pow(p_r, 1/time_horizon)
        
        # Step 4: Calculate energy range for each time point
        for t in range(time_horizon):
            e_min, e_max = self.find_energy_bounds(t, p_r_t)
            slice = SFOSlice(
                time_step=t,
                energy_min=e_min,
                energy_max=e_max
            )
            sfo.add_slice(slice)
            
        # Step 5: Return SFO
        return sfo
        
    def generate_dfo(self, time_horizon: Optional[int] = None) -> DFOSystem:
        """Convert SFO to DFO system"""
        sfo = self.generate_sfo(time_horizon)
        return sfo.to_dfo()
        
    @classmethod
    def from_csv(cls, csv_file: str, energy_type: str = None) -> 'UncertainModel':
        """Create uncertainty model from CSV file"""
        # Read CSV file
        df = pd.read_csv(csv_file, comment='#')
        
        # Filter by energy type if specified
        if energy_type:
            df = df[df['energy_type'] == energy_type]
        
        params_list = []
        
        for _, row in df.iterrows():
            # Parse probability distribution parameters
            prob_type = row['probability_type']
            params_str = row['parameters']
            
            # Create energy range
            min_value = float(row['min_value'])
            max_value = float(row['max_value'])
            energy_range = np.arange(min_value, max_value, 0.1)
            
            # Create probability function based on distribution type
            if prob_type == 'normal':
                # Parse normal distribution parameters
                params_dict = dict(param.split('=') for param in params_str.split(';'))
                mean = float(params_dict['mean'])
                std = float(params_dict['std'])
                
                # Create probability function
                def prob_func(energy, mean=mean, std=std):
                    if std == 0:  # Handle case where standard deviation is 0
                        return 1.0 if energy == mean else 0.0
                    return stats.norm.pdf(energy, mean, std) / stats.norm.pdf(mean, mean, std)
            else:
                # Default to uniform distribution
                def prob_func(energy, min_val=min_value, max_val=max_value):
                    return 1.0 if min_val <= energy <= max_val else 0.0
            
            # Create parameter object
            params = UncertainParameters(
                time_step=row['time_step'],
                probability_threshold=float(row['confidence']),
                default_energy=float(row['default_value']),
                energy_range=energy_range,
                probability_function=prob_func,
                time_availability=0.98,  # Default time availability is 0.98
                energy_type=row['energy_type'],
                min_value=min_value,
                max_value=max_value
            )
            
            params_list.append(params)
        
        return cls(params_list)
        
    @classmethod
    def get_energy_types(cls, csv_file: str) -> List[str]:
        """Get all energy types from CSV file"""
        df = pd.read_csv(csv_file, comment='#')
        return df['energy_type'].unique().tolist() 
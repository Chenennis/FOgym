from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class BatteryParameters:
    """Battery parameters"""
    battery_id: str   # Battery ID
    soc_min: float    # Minimum state of charge
    soc_max: float    # Maximum state of charge
    p_min: float      # Minimum power
    p_max: float      # Maximum power
    efficiency: float # Efficiency
    initial_soc: float # Initial state of charge
    battery_type: str  # Battery type
    capacity_kwh: float # Capacity

@dataclass
class BatteryScheduleParams:
    """Battery scheduling parameters"""
    battery_id: str     # Battery ID
    time_horizon: int   # Time horizon
    start_time: datetime # Start time
    end_time: datetime   # End time
    schedule_type: str   # Schedule type
    priority: int        # Priority
    available_period: str # Available period
    target_soc: float    # Target SOC
    location: str        # Location

class BatteryModel:
    """Battery model class"""
    def __init__(self, params: BatteryParameters, schedule_params: Optional[BatteryScheduleParams] = None):
        self.params = params
        self.schedule_params = schedule_params
        self.current_soc = params.initial_soc
        
    def update_soc(self, power: float, time_step: float = 1.0) -> float:
        """Update state of charge"""
        if power > 0:  # Charging
            self.current_soc += power * time_step * self.params.efficiency
        else:  # Discharging
            self.current_soc += power * time_step / self.params.efficiency
        return self.current_soc
        
    def get_available_power(self) -> Tuple[float, float]:
        """Get available power range"""
        # Calculate available power based on current SOC
        max_charge = (self.params.soc_max - self.current_soc) / self.params.efficiency
        max_discharge = (self.current_soc - self.params.soc_min) * self.params.efficiency
        
        p_min = max(self.params.p_min, -max_discharge)
        p_max = min(self.params.p_max, max_charge)
        
        return p_min, p_max
        
    def generate_dfo(self, time_horizon: int) -> DFOSystem:
        """Generate DFO system"""
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # Calculate energy boundaries
            p_min, p_max = self.get_available_power()
            energy_min = p_min
            energy_max = p_max
            
            # Create constraints
            constraints = []
            # Add SOC constraints
            soc_constraint = np.array([1.0, -1.0])  # SOC >= min, SOC <= max
            constraints.append((soc_constraint, self.params.soc_max - self.current_soc))
            constraints.append((-soc_constraint, self.current_soc - self.params.soc_min))
            
            # Create time slice
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
            # Update SOC (assuming using the middle value of available power)
            # This is only for simulation, actual scheduling should use actual power
            avg_power = (energy_min + energy_max) / 2
            self.update_soc(avg_power)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, schedule_file: str, battery_id: str) -> 'BatteryModel':
        """Create battery model from CSV file"""
        # Read parameter file
        params_df = pd.read_csv(params_file, comment='#')
        
        # Find row with corresponding battery_id
        battery_data = params_df[params_df['battery_id'] == battery_id]
        if battery_data.empty:
            raise ValueError(f"Battery ID {battery_id} not found in {params_file}")
        
        battery_data = battery_data.iloc[0]
        
        # Create parameter object
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
        
        # Read schedule file
        schedule_df = pd.read_csv(schedule_file, comment='#')
        
        # Find row with corresponding battery_id
        schedule_data = schedule_df[schedule_df['battery_id'] == battery_id]
        if schedule_data.empty:
            return cls(params)
        
        schedule_data = schedule_data.iloc[0]
        
        # Create schedule parameter object
        schedule_params = BatteryScheduleParams(
            battery_id=battery_id,
            time_horizon=int(schedule_data['time_horizon']),
            start_time=datetime.strptime(schedule_data['start_time'], '%Y-%m-%d %H:%M:%S'),
            end_time=datetime.strptime(schedule_data['end_time'], '%Y-%m-%d %H:%M:%S'),
            schedule_type=schedule_data['schedule_type'],
            priority=int(schedule_data['priority']),
            available_period=schedule_data['available_period'],
            target_soc=float(schedule_data['target_soc']),
            location=schedule_data['location']
        )
        
        return cls(params, schedule_params)
        
    @classmethod
    def get_all_battery_ids(cls, params_file: str) -> List[str]:
        """Get all battery IDs from CSV file"""
        df = pd.read_csv(params_file, comment='#')
        return df['battery_id'].tolist() 
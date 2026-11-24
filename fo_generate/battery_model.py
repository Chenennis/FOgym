from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class BatteryParameters:
    """battery parameters"""
    battery_id: str   # battery id
    soc_min: float    # minimum state of charge
    soc_max: float    # maximum state of charge
    p_min: float      # minimum power
    p_max: float      # maximum power
    efficiency: float # efficiency
    initial_soc: float # initial state of charge
    battery_type: str  # battery type
    capacity_kwh: float # capacity

@dataclass
class BatteryScheduleParams:
    """battery schedule parameters"""
    battery_id: str     # battery id
    time_horizon: int   # time horizon
    start_time: datetime # start time
    end_time: datetime   # end time
    schedule_type: str   # schedule type
    priority: int        # priority
    available_period: str # available period
    target_soc: float    # target state of charge
    location: str        # location

class BatteryModel:
    """battery model class"""
    def __init__(self, params: BatteryParameters, schedule_params: Optional[BatteryScheduleParams] = None):
        self.params = params
        self.schedule_params = schedule_params
        self.current_soc = params.initial_soc
        
    def _next_soc(self, soc: float, power: float, time_step: float) -> float:
        """Return fractional SoC after a power command; positive power charges."""
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.params.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not 0 < self.params.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")

        energy_kwh = power * time_step
        if power >= 0:
            delta_soc = energy_kwh * self.params.efficiency / self.params.capacity_kwh
        else:
            delta_soc = energy_kwh / (self.params.efficiency * self.params.capacity_kwh)

        return float(np.clip(soc + delta_soc, self.params.soc_min, self.params.soc_max))

    def update_soc(self, power: float, time_step: float = 1.0) -> float:
        """Apply a charge/discharge command and update the real battery SoC."""
        self.current_soc = self._next_soc(self.current_soc, power, time_step)
        return self.current_soc

    def get_available_power(self, time_step: float = 1.0, soc: Optional[float] = None) -> Tuple[float, float]:
        """Return feasible power limits (kW) for the specified interval."""
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.params.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not 0 < self.params.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")

        current_soc = self.current_soc if soc is None else soc
        charge_headroom_kwh = max(0.0, self.params.soc_max - current_soc) * self.params.capacity_kwh
        discharge_headroom_kwh = max(0.0, current_soc - self.params.soc_min) * self.params.capacity_kwh

        max_charge_kw = charge_headroom_kwh / (self.params.efficiency * time_step)
        max_discharge_kw = discharge_headroom_kwh * self.params.efficiency / time_step

        p_min = max(self.params.p_min, -max_discharge_kw)
        p_max = min(self.params.p_max, max_charge_kw)
        return p_min, p_max

    def generate_dfo(self, time_horizon: int) -> DFOSystem:
        """Generate a candidate offer without modifying the real battery SoC.

        The midpoint trajectory is an internal planning approximation, not an
        executed dispatch. Coupled SoC feasibility across all slices must still
        be checked by downstream scheduling.
        """
        dfo = DFOSystem(time_horizon)
        simulated_soc = self.current_soc

        for t in range(time_horizon):
            p_min, p_max = self.get_available_power(time_step=1.0, soc=simulated_soc)
            energy_min = p_min  # kW * 1 hour = kWh
            energy_max = p_max

            constraints = []
            soc_constraint = np.array([1.0, -1.0])
            constraints.append((soc_constraint, self.params.soc_max - simulated_soc))
            constraints.append((-soc_constraint, simulated_soc - self.params.soc_min))

            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)

            avg_power = (energy_min + energy_max) / 2.0
            simulated_soc = self._next_soc(simulated_soc, avg_power, time_step=1.0)

        return dfo

    @classmethod
    def from_csv(cls, params_file: str, schedule_file: str, battery_id: str) -> 'BatteryModel':
        """create battery model from CSV file"""
        # read parameters file
        params_df = pd.read_csv(params_file, comment='#')
        
        # find the row corresponding to battery_id
        battery_data = params_df[params_df['battery_id'] == battery_id]
        if battery_data.empty:
            raise ValueError(f"Battery ID {battery_id} not found in {params_file}")
        
        battery_data = battery_data.iloc[0]
        
        # create parameter object
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
        
        # read schedule file
        schedule_df = pd.read_csv(schedule_file, comment='#')
        
        # find the row corresponding to battery_id
        schedule_data = schedule_df[schedule_df['battery_id'] == battery_id]
        if schedule_data.empty:
            return cls(params)
        
        schedule_data = schedule_data.iloc[0]
        
        # create schedule parameter object
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
        """get all battery IDs from CSV file"""
        df = pd.read_csv(params_file, comment='#')
        return df['battery_id'].tolist() 
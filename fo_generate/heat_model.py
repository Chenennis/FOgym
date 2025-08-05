from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class HeatPumpParameters:
    """heat pump parameters"""
    room_id: str         # room ID
    room_area: float      # room area
    room_volume: float    # room volume
    temp_min: float      # minimum temperature
    temp_max: float      # maximum temperature
    initial_temp: float  # initial temperature
    cop: float          # performance coefficient
    heat_loss_coef: float  # heat loss coefficient
    primary_use_period: str  # primary use period
    secondary_use_period: str  # secondary use period
    primary_target_temp: float  # primary target temperature
    secondary_target_temp: float  # secondary target temperature
    max_power: float     # maximum power

class HeatPumpModel:
    """heat pump model class"""
    def __init__(self, params: HeatPumpParameters):
        self.params = params
        self.current_temp = params.initial_temp
        
    def calculate_heat_required(self, target_temp: float) -> float:
        """calculate heat required to reach target temperature"""
        temp_diff = target_temp - self.current_temp
        heat_required = self.params.room_volume * temp_diff
        return heat_required
        
    def update_temperature(self, heat_energy: float, time_step: float = 1.0) -> float:
        """update temperature"""
        # consider heat loss
        heat_loss = self.params.heat_loss_coef * (self.current_temp - self.params.temp_min)
        net_heat = heat_energy - heat_loss * time_step
        
        # update temperature
        temp_change = net_heat / (self.params.room_volume)
        self.current_temp += temp_change
        
        return self.current_temp
        
    def get_available_heat(self) -> Tuple[float, float]:
        """get available heat range"""
        # calculate heat required to reach maximum temperature
        max_heat = self.calculate_heat_required(self.params.temp_max)
        # calculate heat required to reach minimum temperature (negative value means cooling)
        min_heat = self.calculate_heat_required(self.params.temp_min)
        
        return min_heat, max_heat
        
    def generate_dfo(self, time_horizon: int) -> DFOSystem:
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # calculate heat boundary
            heat_min, heat_max = self.get_available_heat()
            
            # convert to energy (consider COP)
            energy_min = heat_min / self.params.cop
            energy_max = heat_max / self.params.cop
            
            # create constraints
            constraints = []
            # add temperature constraint
            temp_constraint = np.array([1.0, -1.0])  # T >= min, T <= max
            constraints.append((temp_constraint, self.params.temp_max - self.current_temp))
            constraints.append((-temp_constraint, self.current_temp - self.params.temp_min))
            
            # create time slice
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
        """create heat pump model from CSV file"""
        # read CSV file
        df = pd.read_csv(csv_file, comment='#')
        
        # find corresponding row for room_id
        room_data = df[df['room_id'] == room_id]
        if room_data.empty:
            raise ValueError(f"Room ID {room_id} not found in {csv_file}")
        
        room_data = room_data.iloc[0]
        
        # create parameters object
        params = HeatPumpParameters(
            room_id=room_id,
            room_area=float(room_data['room_area']),
            room_volume=float(room_data['room_volume']),
            temp_min=float(room_data['temp_min']),
            temp_max=float(room_data['temp_max']),
            initial_temp=float(room_data['initial_temp']),
            cop=float(room_data['cop']),
            heat_loss_coef=float(room_data['heat_loss_coef']),
            primary_use_period=room_data['primary_use_period'],
            secondary_use_period=room_data['secondary_use_period'],
            primary_target_temp=float(room_data['primary_target_temp']),
            secondary_target_temp=float(room_data['secondary_target_temp']),
            max_power=float(room_data['max_power'])
        )
        
        return cls(params)
        
    @classmethod
    def get_all_room_ids(cls, csv_file: str) -> List[str]:
        """get all room IDs from CSV file"""
        df = pd.read_csv(csv_file, comment='#')
        return df['room_id'].tolist() 
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class EVParameters:
    """EV parameters"""
    ev_id: str           # EV ID
    battery_capacity: float  # battery capacity (kWh)
    soc_min: float        # minimum SOC
    soc_max: float        # maximum SOC
    max_charging_power: float  # maximum charging power (kW)
    efficiency: float     # charging efficiency
    initial_soc: float    # initial SOC
    fast_charge_capable: bool  # whether supports fast charging
    behavior: Optional['EVUserBehavior'] = None  # user behavior

@dataclass
class EVUserBehavior:
    """EV user behavior model"""
    ev_id: str           # EV ID
    connection_time: datetime  # connection time (EV connects to charging station)
    disconnection_time: datetime  # disconnection time (EV leaves charging station)
    next_departure_time: datetime  # next departure time (need to charge enough before this time)
    target_soc: float     # target SOC
    min_required_soc: float  # minimum required SOC (cannot affect user usage)
    fast_charge_preferred: bool  # whether prefers fast charging
    location: str         # charging location
    priority: int         # priority (1-5, 5 highest)
    charge_flexibility: float = 0.8  # charging flexibility (0-1, 1 means can charge intermittently)

class EVModel:
    """EV model class"""
    def __init__(self, params: EVParameters, user_behavior: Optional[EVUserBehavior] = None):
        self.params = params
        self.user_behavior = user_behavior
        self.current_soc = params.initial_soc
        self.is_connected = False  # whether connected to charging station
        self.connection_start_time = None  # connection start time
        self.last_charge_time = None  # last charging time
        self.total_charged_energy = 0.0  # total charged energy
        
    def connect(self, current_time: datetime):
        """connect to charging station"""
        if not self.is_connected:
            self.is_connected = True
            self.connection_start_time = current_time
            
    def disconnect(self, current_time: datetime):
        """disconnect from charging station"""
        if self.is_connected:
            self.is_connected = False
            self.connection_start_time = None
    
    def is_available_for_charging(self, current_time: datetime) -> bool:
        """check if can charge (whether in connection period)"""
        if not self.user_behavior:
            return self.is_connected
            
        # check if in connection period
        in_connection_period = (self.user_behavior.connection_time <= current_time < self.user_behavior.disconnection_time)
        
        return in_connection_period

    def update_soc(self, power: float, time_step: float = 1.0, current_time: Optional[datetime] = None) -> float:
        """update SOC"""
        # can only charge in connection period
        if current_time and not self.is_available_for_charging(current_time):
            return self.current_soc
            
        # only charge, EV usually does not discharge to grid (V2G mode excepted)
        if power > 0:
            energy = power * time_step * self.params.efficiency
            self.current_soc += energy / self.params.battery_capacity
            self.total_charged_energy += energy
            if current_time:
                self.last_charge_time = current_time
            
        # ensure SOC is within reasonable range
        self.current_soc = np.clip(self.current_soc, self.params.soc_min, self.params.soc_max)
        
        return self.current_soc
        
    def get_available_power(self, current_time: datetime) -> Tuple[float, float]:
        """get available power range"""
        # if no user behavior information, use basic parameters
        if not self.user_behavior:
            # can only charge, cannot discharge
            p_min = 0
            # maximum charging power
            p_max = self.params.max_charging_power
            
            # consider current SOC constraint
            remaining_capacity = (self.params.soc_max - self.current_soc) * self.params.battery_capacity
            p_max = min(p_max, remaining_capacity / self.params.efficiency)
            
            return p_min, p_max
            
        # check if vehicle is in connection period
        if not self.is_available_for_charging(current_time):
            return 0, 0  # not in connection period, cannot charge
            
        # calculate remaining time to next use (hours)
        time_to_next_use = (self.user_behavior.next_departure_time - current_time).total_seconds() / 3600
        time_in_connection = (self.user_behavior.disconnection_time - current_time).total_seconds() / 3600
        
        # use smaller time as charging time constraint
        available_charging_time = min(time_to_next_use, time_in_connection)
        
        if available_charging_time <= 0:
            return 0, 0
        
        # calculate required energy to charge
        target_energy = (self.user_behavior.target_soc - self.current_soc) * self.params.battery_capacity
        min_required_energy = (self.user_behavior.min_required_soc - self.current_soc) * self.params.battery_capacity
        
        # calculate maximum charging power
        remaining_capacity = (self.params.soc_max - self.current_soc) * self.params.battery_capacity
        p_max = min(self.params.max_charging_power, remaining_capacity / self.params.efficiency)
        
        # adjust charging power based on user preference and time constraint
        if self.user_behavior.fast_charge_preferred and self.params.fast_charge_capable:
            # fast charging mode: charge as fast as possible
            p_max = self.params.max_charging_power
        else:
            # adjust maximum power based on charging flexibility
            if self.user_behavior.charge_flexibility > 0.5:
                # high flexibility: can charge intermittently, reduce peak power
                p_max = min(p_max, self.params.max_charging_power * 0.7)
        
        # calculate minimum charging power
        p_min = 0  # EV can charge intermittently, so minimum power can be 0
        
        # if SOC is below minimum requirement and time is urgent, set minimum charging power
        if min_required_energy > 0 and available_charging_time < 8:  # if remaining time is less than 8 hours
            min_power_needed = min_required_energy / (available_charging_time * self.params.efficiency)
            p_min = min(min_power_needed, p_max)
        
        # if already reached target SOC, no need to charge
        if self.current_soc >= self.user_behavior.target_soc:
            p_max = 0
            
        return max(0, p_min), max(0, p_max)
        
    def generate_dfo(self, 
                     start_time=None, 
                     time_horizon: Optional[int] = None, 
                     time_step: float = 1.0) -> DFOSystem:

        # compatible with old call, if first parameter is integer and second parameter is None
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # if start_time is None, use current time
        current_time = start_time if start_time is not None and not isinstance(start_time, int) else datetime.now()
        
        # ensure time_horizon has value
        if time_horizon is None:
            time_horizon = 24  # default value is 24 hours
        
        dfo = DFOSystem(time_horizon)
        
        for t in range(time_horizon):
            # calculate energy boundary
            energy_min, energy_max = self.get_available_power(current_time)
            
            # create constraints
            constraints = []
            
            # add SOC constraint
            soc_constraint = np.array([1.0, -1.0])  # SOC >= min, SOC <= max
            constraints.append((soc_constraint, self.params.soc_max - self.current_soc))
            constraints.append((-soc_constraint, self.current_soc - self.params.soc_min))
            
            # if there is user behavior information, add target SOC constraint
            if self.user_behavior and t == time_horizon - 1:
                target_constraint = np.array([1.0])  # SOC >= target_soc
                constraints.append((target_constraint, self.user_behavior.target_soc - self.current_soc))
            
            # create time slice
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
            # update time
            current_time += timedelta(hours=time_step)
            
            # simulate state change after one time step
            avg_power = (energy_min + energy_max) / 2
            self.update_soc(avg_power, time_step)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, behavior_file: Optional[str] = None, ev_id: Optional[str] = None) -> 'EVModel':
        """create EV model from CSV file"""
        # read parameters file
        params_df = pd.read_csv(params_file, comment='#')
        
        # if ev_id is specified, find corresponding data; otherwise use first row
        if ev_id:
            ev_data = params_df[params_df['ev_id'] == ev_id]
            if ev_data.empty:
                raise ValueError(f"EV ID {ev_id} not found in {params_file}")
            ev_data = ev_data.iloc[0]
        else:
            ev_data = params_df.iloc[0]
            ev_id = ev_data['ev_id']
        
        # create parameters object
        params = EVParameters(
            ev_id=ev_id,
            battery_capacity=float(ev_data['battery_capacity']),
            soc_min=float(ev_data['soc_min']),
            soc_max=float(ev_data['soc_max']),
            max_charging_power=float(ev_data['max_charging_power']),
            efficiency=float(ev_data['efficiency']),
            initial_soc=float(ev_data['initial_soc']),
            fast_charge_capable=ev_data['fast_charge_capable'] == 'True'
        )
        
        # if behavior file is provided, read user behavior
        user_behavior = None
        if behavior_file:
            behavior_df = pd.read_csv(behavior_file, comment='#')
            behavior_data = behavior_df[behavior_df['ev_id'] == ev_id]
            
            if not behavior_data.empty:
                behavior_data = behavior_data.iloc[0]
                connection_time = datetime.strptime(behavior_data['arrival_time'], '%Y-%m-%d %H:%M:%S')
                disconnection_time = datetime.strptime(behavior_data['departure_time'], '%Y-%m-%d %H:%M:%S')
                user_behavior = EVUserBehavior(
                    ev_id=ev_id,
                    connection_time=connection_time,
                    disconnection_time=disconnection_time,
                    next_departure_time=disconnection_time,  # use disconnection time as next departure time
                    target_soc=float(behavior_data['target_soc']),
                    fast_charge_preferred=behavior_data['fast_charge_preferred'] == 'True',
                    min_required_soc=float(behavior_data['min_required_soc']),
                    location=behavior_data['location'],
                    priority=int(behavior_data['priority'])
                )
        
        return cls(params, user_behavior)
        
    @classmethod
    def get_all_ev_ids(cls, params_file: str) -> List[str]:
        """get all EV IDs from CSV file"""
        df = pd.read_csv(params_file, comment='#')
        return df['ev_id'].tolist() 
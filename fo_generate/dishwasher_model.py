from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class DishwasherParameters:
    """dishwasher parameters"""
    dishwasher_id: str           # dishwasher ID
    total_energy: float          # total energy demand (kWh) fixed value
    power_rating: float          # rated power (kW)
    operation_hours: float       # running time (hours) usually 3-4 hours
    min_start_delay: float       # minimum start delay (hours) avoid immediate start
    max_start_delay: float       # maximum start delay (hours) avoid waiting too long
    efficiency: float            # energy efficiency
    can_interrupt: bool          # whether can interrupt (usually False)
    behavior: Optional['DishwasherUserBehavior'] = None  # user behavior
    
@dataclass
class DishwasherUserBehavior:
    """dishwasher user behavior model"""
    dishwasher_id: str           # dishwasher ID
    deployment_time: datetime    # deployment time (user presses start)
    preferred_start_time: Optional[datetime] = None  # preferred start time
    latest_completion_time: Optional[datetime] = None  # latest completion time
    priority: int = 3            # priority (1-5, 5 highest)
    user_tolerance: float = 2.0  # user tolerance for delay time (hours)

class DishwasherModel:
    """dishwasher model class"""
    def __init__(self, params: DishwasherParameters, user_behavior: Optional[DishwasherUserBehavior] = None):
        self.params = params
        self.user_behavior = user_behavior
        
        # dishwasher state
        self.is_deployed = False      # whether deployed (user pressed start)
        self.is_running = False       # whether running
        self.is_completed = False     # whether completed
        self.current_cycle_step = 0   # current running step
        self.total_cycle_steps = int(params.operation_hours)  # total running steps
        self.deployment_time = None   # actual deployment time
        self.start_time = None        # actual start time
        self.completion_time = None   # actual completion time
        self.energy_consumed = 0.0    # energy consumed
        
    def deploy(self, current_time: datetime):
        """deploy dishwasher (user pressed start)"""
        if not self.is_deployed:
            self.is_deployed = True
            self.deployment_time = current_time
            if self.user_behavior:
                self.user_behavior.deployment_time = current_time
            
    def can_start(self, current_time: datetime) -> bool:
        """check if can start"""
        if not self.is_deployed or self.is_running or self.is_completed:
            return False
            
        # check minimum delay
        if self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            if time_since_deployment < self.params.min_start_delay:
                return False
        
        # check maximum delay
        if self.user_behavior and self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            if time_since_deployment > self.params.max_start_delay:
                return True  # must start, cannot wait anymore
                
        return True
    
    def must_start(self, current_time: datetime) -> bool:
        """check if must start (cannot delay anymore)"""
        if not self.is_deployed or self.is_running or self.is_completed:
            return False
            
        if self.user_behavior and self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            
            # if exceeds maximum delay time, must start
            if time_since_deployment >= self.params.max_start_delay:
                return True
                
            # if there is latest completion time constraint
            if self.user_behavior.latest_completion_time:
                time_to_deadline = (self.user_behavior.latest_completion_time - current_time).total_seconds() / 3600
                if time_to_deadline <= self.params.operation_hours:
                    return True
                    
        return False
    
    def start_operation(self, current_time: datetime) -> bool:
        """start dishwasher operation"""
        if self.can_start(current_time) and not self.is_running:
            self.is_running = True
            self.start_time = current_time
            self.current_cycle_step = 0
            return True
        return False
    
    def step_operation(self, current_time: datetime, available_power: float) -> Tuple[float, bool]:
        """run one time step
        
        Returns:
            required_power: required power
            is_completed: whether completed
        """
        if not self.is_running or self.is_completed:
            return 0.0, self.is_completed
            
        # dishwasher needs fixed power to run
        required_power = self.params.power_rating
        
        # check if there is enough power
        if available_power >= required_power:
            # consume energy
            energy_step = required_power * 1.0  # assume 1 hour time step
            self.energy_consumed += energy_step
            self.current_cycle_step += 1
            
            # check if completed
            if self.current_cycle_step >= self.total_cycle_steps:
                self.is_completed = True
                self.is_running = False
                self.completion_time = current_time
                return required_power, True
            
            return required_power, False
        else:
            # power insufficient, dishwasher cannot run (this should be avoided)
            # in actual FlexOffer generation, it should ensure that there is enough continuous power when starting
            return required_power, False
    
    def get_required_power_profile(self, start_time: datetime) -> List[float]:
        """get power demand curve from given start time"""
        power_profile = []
        for i in range(self.total_cycle_steps):
            power_profile.append(self.params.power_rating)
        return power_profile
    
    def get_flexibility_window(self, current_time: datetime) -> Tuple[datetime, datetime]:
        """get flexibility time window"""
        if not self.is_deployed:
            return current_time, current_time
            
        earliest_start = self.deployment_time + timedelta(hours=self.params.min_start_delay)
        latest_start = self.deployment_time + timedelta(hours=self.params.max_start_delay)
        
        # consider latest completion time constraint
        if self.user_behavior and self.user_behavior.latest_completion_time:
            latest_by_completion = self.user_behavior.latest_completion_time - timedelta(hours=self.params.operation_hours)
            latest_start = min(latest_start, latest_by_completion)
        
        return max(earliest_start, current_time), latest_start
    
    def calculate_urgency(self, current_time: datetime) -> float:
        """calculate urgency (0-1, 1 most urgent)"""
        if not self.is_deployed or self.is_completed:
            return 0.0
            
        if self.is_running:
            return 1.0  # running, most urgent
            
        earliest_start, latest_start = self.get_flexibility_window(current_time)
        
        if current_time >= latest_start:
            return 1.0  # must start immediately
            
        total_window = (latest_start - earliest_start).total_seconds() / 3600
        elapsed_time = (current_time - earliest_start).total_seconds() / 3600
        
        if total_window <= 0:
            return 1.0
            
        urgency = max(0.0, elapsed_time / total_window)
        return min(1.0, urgency)
    
    def generate_dfo(self, 
                     start_time=None, 
                     time_horizon: int = None, 
                     time_step: float = 1.0) -> DFOSystem:

        # compatible with old call
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # if start_time is None, use current time
        current_time = start_time if start_time is not None and not isinstance(start_time, int) else datetime.now()
        
        # ensure time_horizon has value
        if time_horizon is None:
            time_horizon = 24  # default value is 24 hours
        
        dfo = DFOSystem(time_horizon)
        
        # if not deployed, all time steps have 0 power demand
        if not self.is_deployed:
            for t in range(time_horizon):
                slice = DFOSlice(
                    time_step=t,
                    energy_min=0.0,
                    energy_max=0.0,
                    constraints=[]
                )
                dfo.add_slice(slice)
            return dfo
        
        # if completed, all time steps have 0 power demand
        if self.is_completed:
            for t in range(time_horizon):
                slice = DFOSlice(
                    time_step=t,
                    energy_min=0.0,
                    energy_max=0.0,
                    constraints=[]
                )
                dfo.add_slice(slice)
            return dfo
        
        # if running, must continuously provide power
        if self.is_running:
            remaining_steps = self.total_cycle_steps - self.current_cycle_step
            for t in range(time_horizon):
                if t < remaining_steps:
                    # must run
                    energy_required = self.params.power_rating
                    slice = DFOSlice(
                        time_step=t,
                        energy_min=energy_required,
                        energy_max=energy_required,
                        constraints=[]
                    )
                else:
                    # completed
                    slice = DFOSlice(
                        time_step=t,
                        energy_min=0.0,
                        energy_max=0.0,
                        constraints=[]
                    )
                dfo.add_slice(slice)
            return dfo
        
        # deployed but not running: generate flexibility offer
        earliest_start, latest_start = self.get_flexibility_window(current_time)
        
        # calculate time step corresponding to start window
        earliest_step = max(0, int((earliest_start - current_time).total_seconds() / 3600 / time_step))
        latest_step = min(time_horizon - self.total_cycle_steps, 
                         int((latest_start - current_time).total_seconds() / 3600 / time_step))
        
        for t in range(time_horizon):
            # check if this time step is possible start time
            can_start_at_t = earliest_step <= t <= latest_step
            
            if can_start_at_t:
                # this time step may start, check if there is enough continuous time to complete running
                remaining_time_steps = time_horizon - t
                if remaining_time_steps >= self.total_cycle_steps:
                    # enough time to complete running
                    energy_min = 0.0  # can choose not to start at this time step
                    energy_max = self.params.power_rating  # if start, need this power
                else:
                    # not enough time to complete running, cannot start at this time step
                    energy_min = 0.0
                    energy_max = 0.0
            else:
                # cannot start at this time step
                energy_min = 0.0
                energy_max = 0.0
            
            # if current time is already the time to start
            current_step_time = current_time + timedelta(hours=t * time_step)
            if self.must_start(current_step_time) and t == 0:
                # must start immediately
                energy_min = self.params.power_rating
                energy_max = self.params.power_rating
            
            # create constraints
            constraints = []
            
            # add running continuity constraint (if start, must run continuously)

            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, behavior_file: str = None, dishwasher_id: str = None) -> 'DishwasherModel':
        """create dishwasher model from CSV file"""
        # read parameters file
        params_df = pd.read_csv(params_file, comment='#')
        
        # if dishwasher_id is specified, find corresponding data; otherwise use first row
        if dishwasher_id:
            device_data = params_df[params_df['dishwasher_id'] == dishwasher_id]
            if device_data.empty:
                raise ValueError(f"Dishwasher ID {dishwasher_id} not found in {params_file}")
            device_data = device_data.iloc[0]
        else:
            device_data = params_df.iloc[0]
            dishwasher_id = device_data['dishwasher_id']
        
        # create parameters object
        params = DishwasherParameters(
            dishwasher_id=dishwasher_id,
            total_energy=float(device_data['total_energy']),
            power_rating=float(device_data['power_rating']),
            operation_hours=float(device_data['operation_hours']),
            min_start_delay=float(device_data['min_start_delay']),
            max_start_delay=float(device_data['max_start_delay']),
            efficiency=float(device_data['efficiency']),
            can_interrupt=device_data['can_interrupt'] == 'True'
        )
        
        # if behavior file is provided, read user behavior
        user_behavior = None
        if behavior_file:
            behavior_df = pd.read_csv(behavior_file, comment='#')
            behavior_data = behavior_df[behavior_df['dishwasher_id'] == dishwasher_id]
            
            if not behavior_data.empty:
                behavior_data = behavior_data.iloc[0]
                user_behavior = DishwasherUserBehavior(
                    dishwasher_id=dishwasher_id,
                    deployment_time=pd.to_datetime(behavior_data['deployment_time']),
                    preferred_start_time=pd.to_datetime(behavior_data['preferred_start_time']) if pd.notna(behavior_data['preferred_start_time']) else None,
                    latest_completion_time=pd.to_datetime(behavior_data['latest_completion_time']) if pd.notna(behavior_data['latest_completion_time']) else None,
                    priority=int(behavior_data['priority']),
                    user_tolerance=float(behavior_data['user_tolerance'])
                )
        
        return cls(params, user_behavior)

    @classmethod
    def get_all_dishwasher_ids(cls, params_file: str) -> List[str]:
        """get all dishwasher IDs from parameters file"""
        try:
            params_df = pd.read_csv(params_file, comment='#')
            return params_df['dishwasher_id'].tolist()
        except Exception as e:
            print(f"failed to read dishwasher parameters file: {e}")
            return []

    def get_status_summary(self) -> Dict:
        """get status summary"""
        return {
            'dishwasher_id': self.params.dishwasher_id,
            'is_deployed': self.is_deployed,
            'is_running': self.is_running,
            'is_completed': self.is_completed,
            'current_cycle_step': self.current_cycle_step,
            'total_cycle_steps': self.total_cycle_steps,
            'energy_consumed': self.energy_consumed,
            'total_energy_required': self.params.total_energy,
            'deployment_time': self.deployment_time.isoformat() if self.deployment_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'completion_time': self.completion_time.isoformat() if self.completion_time else None
        } 
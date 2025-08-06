import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import sys

# handle import method
try:
    # try to import as part of the package
    from .config import ModelBasedConfig
except (ImportError, SystemError):
    # import method when running script directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import ModelBasedConfig

logger = logging.getLogger(__name__)


class DeviceModel:
    """device physical model base class"""
    
    def __init__(self, device_id: str, params: Dict[str, Any]):
        self.device_id = device_id
        self.params = params
        self.state = self.get_initial_state()
        
    def get_initial_state(self) -> Dict[str, float]:
        """get initial state"""
        return {}
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        """predict next state"""
        return state
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        """get optimal control"""
        return 0.0
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        """generate energy profile and time flexibility"""
        return [0.0] * len(prices), 0


class BatteryModel(DeviceModel):
    """battery device model"""
    
    def get_initial_state(self) -> Dict[str, float]:
        """get initial state"""
        return {
            'soc': self.params.get('initial_soc', 0.5),  # initial state of charge
            'charge': self.params.get('initial_charge', 5.0),  # initial charge (kWh)
        }
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        capacity = self.params.get('capacity', 10.0)  # battery capacity (kWh)
        efficiency = self.params.get('efficiency', 0.95)  # charge/discharge efficiency
        
        # calculate charge/discharge energy
        if control > 0:  # charging
            energy_delta = control * efficiency
        else:  # discharging
            energy_delta = control / efficiency
        
        # update charge
        new_charge = state['charge'] + energy_delta
        new_charge = max(0.0, min(capacity, new_charge))  # limit within capacity range
        
        # update SOC
        new_soc = new_charge / capacity
        
        return {
            'soc': new_soc,
            'charge': new_charge
        }
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        # get parameters
        p_min = self.params.get('p_min', -3.0)  # minimum power (kW)
        p_max = self.params.get('p_max', 3.0)  # maximum power (kW)
        min_soc = self.params.get('min_soc', 0.1)  # minimum SOC
        max_soc = self.params.get('max_soc', 0.9)  # maximum SOC
        current_soc = current_state.get('soc', 0.5)
        
        low_price_threshold = 0.08  # low price threshold
        high_price_threshold = 0.15  # high price threshold
        
        # default control: 0 (no charge/discharge)
        control = 0.0
        
        if price <= low_price_threshold and current_soc < max_soc:
            # charge when price is low and SOC is not at maximum
            control = p_max
        elif price >= high_price_threshold and current_soc > min_soc:
            # discharge when price is high and SOC is not at minimum
            control = p_min
        
        return control
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        # initialize
        time_horizon = len(prices)
        energy_profile = [0.0] * time_horizon
        state = self.get_initial_state()
        
        # simulate charge/discharge process
        for t in range(time_horizon):
            # get optimal control based on current price and state
            control = self.get_optimal_control(prices[t], state)
            
            # record energy profile (positive for consumption, negative for production)
            energy_profile[t] = control
            
            # update state
            state = self.predict_next_state(state, control)
        
        # time flexibility of battery is usually high
        time_flexibility = min(3, time_horizon // 8)  # flexibility is 1/8 of time range, maximum 3 hours
        
        return energy_profile, time_flexibility


class HeatPumpModel(DeviceModel):
    """heat pump device model"""
    
    def get_initial_state(self) -> Dict[str, float]:
        """get initial state"""
        return {
            'temperature': self.params.get('initial_temp', 20.0),  # initial temperature (°C)
        }
    
    def predict_next_state(self, state: Dict[str, float], control: float) -> Dict[str, float]:
        # get parameters
        outdoor_temp = self.params.get('outdoor_temp', 5.0)  # outdoor temperature (°C)
        thermal_mass = self.params.get('thermal_mass', 5000.0)  # thermal mass (J/°C)
        heat_transfer_coeff = self.params.get('heat_transfer_coeff', 100.0)  # heat transfer coefficient (W/°C)
        cop = 3.0  # performance coefficient, heat pump efficiency
        
        # current temperature
        current_temp = state['temperature']
        
        # calculate heat pump heat
        heat_pump_heat = control * cop * 1000  # convert to W
        
        # calculate heat loss
        heat_loss = heat_transfer_coeff * (current_temp - outdoor_temp)
        
        # calculate temperature change
        temp_change = (heat_pump_heat - heat_loss) / thermal_mass
        
        # update temperature
        new_temp = current_temp + temp_change
        
        return {
            'temperature': new_temp
        }
    
    def get_optimal_control(self, price: float, current_state: Dict[str, float]) -> float:
        """get optimal control"""
        # get parameters
        target_temp = self.params.get('target_temp', 21.0)  # target temperature (°C)
        min_temp = self.params.get('min_temp', 18.0)  # minimum temperature (°C)
        max_temp = self.params.get('max_temp', 22.0)  # maximum temperature (°C)
        max_power = self.params.get('max_power', 2.0)  # maximum power (kW)
        
        # current temperature
        current_temp = current_state.get('temperature', 20.0)
        
        low_price_threshold = 0.08  # low price threshold
        high_price_threshold = 0.15  # high price threshold

        temp_diff = target_temp - current_temp
        control = max_power * (temp_diff / 3.0) 
        control = max(0.0, min(max_power, control))  
        
        # adjust control strategy based on price and temperature
        if price <= low_price_threshold and current_temp < max_temp:
            temp_diff = max_temp - current_temp
            control = max_power * (temp_diff / 3.0)
            control = max(0.0, min(max_power, control))
        elif price >= high_price_threshold and current_temp > min_temp:
            control = 0.0
        
        return control
    
    def generate_energy_profile(self, prices: List[float]) -> Tuple[List[float], int]:
        time_horizon = len(prices)
        energy_profile = [0.0] * time_horizon
        state = self.get_initial_state()
        
        for t in range(time_horizon):
            control = self.get_optimal_control(prices[t], state)
            
            energy_profile[t] = control
            
            state = self.predict_next_state(state, control)
        
        time_flexibility = min(2, time_horizon // 12)  
        
        return energy_profile, time_flexibility


class ModelBasedController:
    """model-based FlexOffer controller"""
    
    def __init__(self, 
                manager_id: str,
                time_horizon: int = 24,
                time_step: float = 1.0,
                config: Optional[ModelBasedConfig] = None):
        self.manager_id = manager_id
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.config = config or ModelBasedConfig()
        
        self.device_models = {}  # device_id -> DeviceModel
        self.device_stats = {}   # device_id -> statistics
        
        self.current_timestep = 0
        logger.info(f"ModelBasedController initialized: manager_id={manager_id}, time_horizon={time_horizon}")
    
    def add_device_model(self, device_id: str, device_type: str, device_params: Dict[str, Any]):
        """add device model"""
        if device_id in self.device_models:
            logger.warning(f"device {device_id} already exists, will be overwritten")
        
        if 'BATTERY' in device_type.upper():
            model = BatteryModel(device_id, device_params)
            logger.info(f"add battery model: {device_id}")
        elif 'HEAT' in device_type.upper() or 'PUMP' in device_type.upper():
            model = HeatPumpModel(device_id, device_params)
            logger.info(f"add heat pump model: {device_id}")
        else:
            model = DeviceModel(device_id, device_params)
            logger.info(f"add generic device model: {device_id}")
        
        self.device_models[device_id] = model
        
        self.device_stats[device_id] = {
            'type': device_type,
            'params': device_params,
            'energy_consumed': 0.0,
            'energy_produced': 0.0
        }
    
    def generate_flex_offers(self, prices: List[float]) -> Dict[str, Dict[str, Any]]:

        if len(prices) < self.time_horizon:
            prices = prices + [prices[-1]] * (self.time_horizon - len(prices))
        
        fo_dict = {}
        
        for device_id, model in self.device_models.items():
            energy_profile, time_flexibility = model.generate_energy_profile(prices[:self.time_horizon])
            
            consumed = sum(max(0, e) for e in energy_profile)
            produced = sum(abs(min(0, e)) for e in energy_profile)
            self.device_stats[device_id]['energy_consumed'] += consumed
            self.device_stats[device_id]['energy_produced'] += produced
            
            fo_dict[device_id] = {
                'energy_profile': energy_profile,
                'time_flexibility': time_flexibility
            }
        
        logger.info(f"generated {len(fo_dict)} FlexOffers")
        return fo_dict
    
    def calculate_reward(self, 
                        schedules: Dict[str, List[float]], 
                        revenue: float,
                        original_profiles: Dict[str, List[float]]) -> float:
        satisfaction = 0.0
        profile_count = 0
        
        for device_id, schedule in schedules.items():
            if device_id in original_profiles:
                original = original_profiles[device_id]
                
                min_len = min(len(schedule), len(original))
                
                if min_len > 0:
                    schedule_np = np.array(schedule[:min_len])
                    original_np = np.array(original[:min_len])
                    
                    total_energy = np.sum(np.abs(original_np))
                    if total_energy > 0:
                        error = np.sum(np.abs(schedule_np - original_np)) / total_energy
                        similarity = max(0, 1 - error)  
                    else:
                        similarity = 1.0  
                    
                    satisfaction += similarity
                    profile_count += 1
        
        avg_satisfaction = satisfaction / max(1, profile_count)
        
        max_possible_revenue = len(schedules) * 10
        normalized_revenue = min(1.0, revenue / max(0.1, max_possible_revenue))
        
        satisfaction_weight = 0.7
        revenue_weight = 0.3
        
        base_reward = satisfaction_weight * avg_satisfaction + revenue_weight * normalized_revenue
        
        reward = base_reward * 36.0
        
        logger.info(f"calculate reward: satisfaction={avg_satisfaction:.4f}, normalized revenue={normalized_revenue:.4f}, total reward={reward:.4f}")
        return reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """get statistics"""
        return {
            'manager_id': self.manager_id,
            'device_count': len(self.device_models),
            'current_timestep': self.current_timestep,
            'device_stats': self.device_stats
        }
    
    def reset(self):
        """reset controller state"""
        self.current_timestep = 0
        
        # reset device state
        for model in self.device_models.values():
            model.state = model.get_initial_state()
        
        # reset statistics
        for device_id in self.device_stats:
            self.device_stats[device_id]['energy_consumed'] = 0.0
            self.device_stats[device_id]['energy_produced'] = 0.0
        
        logger.info("controller reset")
    
    def step(self, time_step: int = 1):
        """update time step"""
        self.current_timestep += time_step
        logger.debug(f"time step updated: {self.current_timestep}") 
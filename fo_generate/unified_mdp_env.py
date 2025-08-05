import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
import math
from abc import ABC, abstractmethod

from fo_generate.battery_model import BatteryModel, BatteryParameters
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior
from fo_generate.pv_model import PVModel, PVParameters
from fo_generate.dishwasher_model import DishwasherModel, DishwasherParameters, DishwasherUserBehavior
from fo_generate.dfo import DFOSystem, DFOSlice
from fo_generate.price_loader import PriceLoader

logger = logging.getLogger(__name__)

class DeviceType:
    """device type enumeration"""
    BATTERY = "battery"
    HEAT_PUMP = "heat_pump"
    EV = "ev"
    PV = "pv"
    DISHWASHER = "dishwasher"

class EnvironmentDynamics:
    
    def __init__(self, price_data: pd.DataFrame = None, weather_data: pd.DataFrame = None, 
                 data_dir: str = "data"):
        self.price_data = price_data
        self.weather_data = weather_data
        
        # initialize price loader
        self.price_loader = PriceLoader(data_dir)
        
        self.price_history = []
        self.weather_history = []
        
        # environment parameters
        self.price_volatility = 0.1
        self.weather_noise = 0.05

    def get_current_state(self, current_time: datetime) -> Dict[str, Any]:

        # get current price and weather
        current_price = self._get_price_at_time(current_time)
        current_weather = self._get_weather_at_time(current_time)
        
        # update history
        self.price_history.append(current_price)
        self.weather_history.append(current_weather)
        
        # only keep recent 3 time steps for trend calculation
        if len(self.price_history) > 3:
            self.price_history = self.price_history[-3:]
        if len(self.weather_history) > 3:
            self.weather_history = self.weather_history[-3:]
        
        # calculate trend
        price_trend = self._get_price_trend()
        weather_trend = self._get_weather_trend()
        
        # predict future 3 hours price and weather
        future_prices = self._predict_future_prices(current_time)
        future_weather = self._predict_future_weather(current_time)
        
        return {
            'price': current_price,
            'price_trend': price_trend,
            'future_prices': future_prices,
            'temperature': current_weather['temperature'],
            'solar_irradiance': current_weather['solar_irradiance'],
            'weather_trend': weather_trend,
            'future_weather': future_weather
        }

    def _get_price_at_time(self, current_time: datetime) -> float:
        """get price at specified time, use danish price data if available"""
        try:
            current_price_info = self.price_loader.get_current_price(current_time)
            base_price = current_price_info['price']
            logger.debug(f"use {current_price_info['source']} price data: {base_price:.4f} USD/kWh")
        except Exception as e:
            logger.warning(f"price loader get price failed: {e}, use alternative")
            
            # alternative 1: use input price data
            if self.price_data is not None:
                time_diffs = abs((pd.to_datetime(self.price_data['timestamp']) - current_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                base_price = self.price_data.loc[closest_idx, 'price']
            else:
                # alternative 2: simplified price model
                hour = current_time.hour
                if 0 <= hour < 6:  # night low price
                    base_price = 0.08
                elif 6 <= hour < 18:  # day high price
                    base_price = 0.15 + 0.05 * math.sin(math.pi * (hour - 6) / 12)
                else:  # night peak price
                    base_price = 0.20
                
        # add random fluctuation
        price_noise = np.random.normal(0, self.price_volatility * 0.1)
        return max(0.01, base_price + price_noise)

    def _get_weather_at_time(self, current_time: datetime) -> Dict[str, float]:
        """get weather data at specified time"""
        if self.weather_data is not None:
            time_diffs = abs((pd.to_datetime(self.weather_data['timestamp']) - current_time).dt.total_seconds())
            closest_idx = time_diffs.idxmin()
            return {
                'temperature': self.weather_data.loc[closest_idx, 'temperature'],
                'solar_irradiance': self.weather_data.loc[closest_idx, 'solar_irradiance']
            }
        else:
            # weather model
            hour = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
            # temperature model
            seasonal_temp = 20 + 10 * math.sin(2 * math.pi * day_of_year / 365)
            daily_variation = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
            temperature = seasonal_temp + daily_variation
            
            # solar irradiance model
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                irradiance = 800 * solar_angle * max(0, math.sin(2 * math.pi * day_of_year / 365 + math.pi/2))
            else:
                irradiance = 0
                
            return {
                'temperature': temperature + np.random.normal(0, 1),
                'solar_irradiance': max(0, irradiance + np.random.normal(0, 50))
            }

    def _get_price_trend(self) -> float:
        """calculate price trend"""
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] - self.price_history[0]) / max(self.price_history[0], 0.01)

    def _get_weather_trend(self) -> Dict[str, float]:
        """calculate weather trend"""
        if len(self.weather_history) < 2:
            return {'temperature_trend': 0.0, 'irradiance_trend': 0.0}
        
        temp_trend = (self.weather_history[-1]['temperature'] - self.weather_history[0]['temperature']) / max(abs(self.weather_history[0]['temperature']), 1.0)
        irr_trend = (self.weather_history[-1]['solar_irradiance'] - self.weather_history[0]['solar_irradiance']) / max(self.weather_history[0]['solar_irradiance'], 1.0)
        
        return {'temperature_trend': temp_trend, 'irradiance_trend': irr_trend}

    def _predict_future_prices(self, current_time: datetime) -> List[float]:
        """predict future 3 hours price"""
        future_prices = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            future_price = self._get_price_at_time(future_time)
            future_prices.append(future_price)
        return future_prices

    def _predict_future_weather(self, current_time: datetime) -> List[Dict[str, float]]:
        """predict future 3 hours weather"""
        future_weather = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            weather = self._get_weather_at_time(future_time)
            future_weather.append(weather)
        return future_weather

class DeviceMDPInterface(ABC):
    """device MDP interface"""
    
    @abstractmethod
    def get_state_features(self) -> np.ndarray:
        """get device state features"""
        pass
    
    @abstractmethod
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """device state transition"""
        pass
    
    @abstractmethod
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate device reward"""
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[float, float]:
        """get action bounds"""
        pass
    
    @abstractmethod
    def reset_state(self):
        """reset device state"""
        pass

class DishwasherMDPDevice(DeviceMDPInterface):
    
    def __init__(self, dishwasher_model: DishwasherModel):
        self.dishwasher = dishwasher_model
        
    def get_state_features(self) -> np.ndarray:
        """get dishwasher state features [is deployed, is running, is completed, current step/total steps, urgency, remaining energy demand]"""
        is_deployed = 1.0 if self.dishwasher.is_deployed else 0.0
        is_running = 1.0 if self.dishwasher.is_running else 0.0
        is_completed = 1.0 if self.dishwasher.is_completed else 0.0
        
        # progress (current step/total steps)
        progress = self.dishwasher.current_cycle_step / max(1, self.dishwasher.total_cycle_steps)
        
        # urgency
        urgency = self.dishwasher.calculate_urgency(datetime.now())
        
        # remaining energy demand
        remaining_energy = self.dishwasher.params.total_energy - self.dishwasher.energy_consumed
        remaining_energy_norm = remaining_energy / max(1, self.dishwasher.params.total_energy)
        
        return np.array([is_deployed, is_running, is_completed, progress, urgency, remaining_energy_norm])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """dishwasher state transition
        
        action: 0-1, whether to start dishwasher (only valid when deployed but not running)
        """
        current_time = datetime.now()
        
        # if not deployed, randomly simulate deployment (in actual application triggered by user)
        if not self.dishwasher.is_deployed:
            # simulate user may deploy dishwasher at some time
            if np.random.random() < 0.1:  # 10% probability to deploy
                self.dishwasher.deploy(current_time)
                
        # if deployed but not running, decide whether to start based on action
        start_success = False
        if self.dishwasher.is_deployed and not self.dishwasher.is_running and not self.dishwasher.is_completed:
            if action > 0.5:  # action > 0.5 means decide to start
                start_success = self.dishwasher.start_operation(current_time)
        
        # if running, continue running one time step
        power_consumed = 0.0
        operation_completed = False
        if self.dishwasher.is_running:
            # dishwasher running needs fixed power
            available_power = env_state.get('available_power', self.dishwasher.params.power_rating)
            power_consumed, operation_completed = self.dishwasher.step_operation(current_time, available_power)
        
        return {
            'is_deployed': self.dishwasher.is_deployed,
            'is_running': self.dishwasher.is_running,
            'is_completed': self.dishwasher.is_completed,
            'power_consumed': power_consumed,
            'operation_completed': operation_completed,
            'start_success': start_success,
            'current_cycle_step': self.dishwasher.current_cycle_step,
            'energy_consumed': self.dishwasher.energy_consumed
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate dishwasher reward - fixed version, reduce sparse reward"""
        reward = 0.0
        reward_components = {}
        
        if next_state['operation_completed']:
            completion_reward = 50.0  # reduce but still high reward
            reward += completion_reward
            reward_components['completion_reward'] = completion_reward
        
        # add progress reward, encourage start and continue running
        if next_state['is_running']:
            # give increasing reward based on progress
            progress = getattr(self.dishwasher, 'current_cycle_step', 0)
            total_steps = getattr(self.dishwasher, 'total_cycle_steps', 10)
            
            if total_steps > 0:
                progress_ratio = progress / total_steps
                progress_reward = 5.0 + progress_ratio * 10.0  
            else:
                progress_reward = 8.0  
                
            reward += progress_reward
            reward_components['progress_reward'] = progress_reward
        
        # re-design start timing reward, more tolerant
        if next_state.get('start_success', False):
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'calculate_urgency'):
                urgency = self.dishwasher.calculate_urgency(current_time)
            else:
                urgency = 0.5  # default medium urgency
            
            if urgency > 0.6:  # high urgency
                timing_reward = 15.0 * urgency  # highest 9 points reward
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            elif urgency > 0.3:  # medium urgency
                timing_reward = 5.0 * urgency  # 1.5-3 points reward
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            else:  # low urgency, slight penalty
                timing_penalty = -2.0  # reduce penalty
                reward += timing_penalty
                reward_components['timing_penalty'] = timing_penalty
        
        # re-design energy cost, not over-penalize
        power_consumed = next_state.get('power_consumed', 0.0)
        price = env_state.get('price', 0.15)
        
        if power_consumed > 0:
            # energy cost relative to reward, not absolute penalty
            energy_cost = power_consumed * price * 0.3  # reduce cost weight
            reward -= energy_cost
            reward_components['energy_cost'] = -energy_cost
        
        # re-design waiting time penalty, more tolerant
        if (self.dishwasher.is_deployed and 
            not getattr(self.dishwasher, 'is_running', False) and 
            not getattr(self.dishwasher, 'is_completed', False)):
            
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'deployment_time') and self.dishwasher.deployment_time:
                wait_time = (current_time - self.dishwasher.deployment_time).total_seconds() / 3600
                max_delay = getattr(self.dishwasher.params, 'max_start_delay', 6.0)
                
                if wait_time > max_delay:
                    # wait timeout, heavy penalty but reduce
                    timeout_penalty = -20.0  
                    reward += timeout_penalty
                    reward_components['timeout_penalty'] = timeout_penalty
                elif wait_time > max_delay * 0.8:
                    # close to timeout, light penalty
                    wait_penalty = -5.0 * (wait_time / max_delay)  # reduce penalty
                    reward += wait_penalty
                    reward_components['wait_penalty'] = wait_penalty
        
        # add deployment reward, encourage participation
        if self.dishwasher.is_deployed:
            deployment_reward = 2.0  # deployment has reward
            reward += deployment_reward
            reward_components['deployment_reward'] = deployment_reward
        
        # add base participation reward
        base_participation_reward = 1.0
        reward += base_participation_reward
        reward_components['participation_reward'] = base_participation_reward
        
        return reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """get action bounds"""
        return 0.0, 1.0  # 0 means not start, 1 means start
    
    def reset_state(self):
        """reset dishwasher state"""
        self.dishwasher.is_deployed = False
        self.dishwasher.is_running = False
        self.dishwasher.is_completed = False
        self.dishwasher.current_cycle_step = 0
        self.dishwasher.deployment_time = None
        self.dishwasher.start_time = None
        self.dishwasher.completion_time = None
        self.dishwasher.energy_consumed = 0.0

class BatteryMDPDevice(DeviceMDPInterface):
    
    def __init__(self, battery_model: BatteryModel):
        self.battery = battery_model
        self.efficiency = battery_model.params.efficiency
        self.capacity = battery_model.params.capacity_kwh
    
    def get_state_features(self) -> np.ndarray:
        """get battery state features [SOC, max charge power, max discharge power, health]"""
        soc = self.battery.current_soc
        
        # calculate available power range
        max_charge_energy = (self.battery.params.soc_max - soc) * self.capacity
        max_charge_power = min(self.battery.params.p_max, max_charge_energy / self.efficiency)
        
        max_discharge_energy = (soc - self.battery.params.soc_min) * self.capacity
        max_discharge_power = min(abs(self.battery.params.p_min), max_discharge_energy * self.efficiency)
        
        # health
        health = max(0.8, 1.0 - soc * 0.1)  # simplified health based on SOC
        
        return np.array([soc, max_charge_power, max_discharge_power, health])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """battery state transition"""
        soc = self.battery.current_soc
        
        # calculate SOC change
        if action > 0:  # charge
            energy_change = action * self.efficiency
        else:  # discharge
            energy_change = action / self.efficiency
        
        new_soc = soc + energy_change / self.capacity
        new_soc = np.clip(new_soc, self.battery.params.soc_min, self.battery.params.soc_max)
        
        # update battery state
        self.battery.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': action,
            'energy_change': energy_change,
            'efficiency_loss': abs(energy_change) * (1 - self.efficiency) if action != 0 else 0
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate battery reward - enhanced learning signal version"""
        reward_components = {}
        
        # re-design economic reward, increase differentiation
        price = env_state.get('price', 0.15)
        base_price = 0.15
        price_ratio = price / base_price
        
        # price time analysis - increase reward difference
        if price < 0.10:  # super low price
            if action > 0:  # charge
                economic_reward = abs(action) * 10.0 * (1.0 - price_ratio)  # highest 10 points
            else:
                economic_reward = -abs(action) * 2.0  # miss opportunity penalty
        elif price < 0.12:  # low price
            if action > 0:  # charge
                economic_reward = abs(action) * 5.0 * (1.0 - price_ratio)  # highest 5 points
            else:
                economic_reward = 0.0
        elif price > 0.25:  # super high price
            if action < 0:  # discharge
                economic_reward = abs(action) * 15.0 * (price_ratio - 1.0)  # highest 15 points
            else:
                economic_reward = -abs(action) * 5.0  # high price charge heavy penalty
        elif price > 0.18:  # high price
            if action < 0:  # discharge
                economic_reward = abs(action) * 8.0 * (price_ratio - 1.0)  # highest 8 points
            else:
                economic_reward = -abs(action) * 2.0  # high price charge light penalty
        else:  # medium price
            economic_reward = -abs(action) * 0.5  # slight penalty
        
        reward_components['economic'] = economic_reward
        
        # SOC management reward, create bigger difference
        soc = next_state.get('soc', 0.5)
        
        if 0.45 <= soc <= 0.75:  # best SOC interval
            soc_reward = 8.0
        elif 0.35 <= soc <= 0.85:  # good SOC interval
            soc_reward = 4.0
        elif 0.25 <= soc <= 0.9:  # acceptable interval
            soc_reward = 1.0
        elif 0.15 <= soc <= 0.95:  # boundary interval
            soc_reward = -2.0
        else:  # dangerous interval
            soc_reward = -10.0  # heavy penalty
            
        reward_components['soc_maintenance'] = soc_reward
        
        # continuous decision reward, encourage reasonable action sequence
        action_consistency_reward = 0.0
        if hasattr(self, 'prev_action'):
            prev_action = self.prev_action
            # reward reasonable action continuity
            if abs(action - prev_action) < 0.5:  # smooth operation
                action_consistency_reward = 2.0
            elif abs(action - prev_action) > 2.0:  
                action_consistency_reward = -1.0
        
        self.prev_action = action
        reward_components['action_consistency'] = action_consistency_reward
        
        # state improvement reward, encourage positive state changes
        state_improvement_reward = 0.0
        if hasattr(self, 'prev_soc'):
            prev_soc = self.prev_soc
            soc_change = soc - prev_soc
            
            # reward to move to ideal SOC interval
            ideal_soc = 0.6
            prev_distance = abs(prev_soc - ideal_soc)
            current_distance = abs(soc - ideal_soc)
            
            if current_distance < prev_distance:  # move to ideal state
                state_improvement_reward = 3.0 * (prev_distance - current_distance)
            else:  # move away from ideal state
                state_improvement_reward = -2.0 * (current_distance - prev_distance)
        
        self.prev_soc = soc
        reward_components['state_improvement'] = state_improvement_reward
        
        # task completion reward, based on time progress
        hour = datetime.now().hour
        task_completion_reward = 0.0
        
        # give different task completion rewards based on time
        if 6 <= hour <= 9:  
            if 0.7 <= soc <= 0.9:  
                task_completion_reward = 5.0
        elif 18 <= hour <= 22:  
            if action < 0 and soc > 0.5:  
                task_completion_reward = 6.0
        elif 22 <= hour or hour <= 6:  
            if action > 0 and price < 0.12:  
                task_completion_reward = 4.0
                
        reward_components['task_completion'] = task_completion_reward
        
        # re-balance weights, increase overall reward range
        total_reward = (
            0.4 * economic_reward +           # increase economic weight
            0.3 * soc_reward +               # SOC management
            0.1 * action_consistency_reward + # action consistency
            0.1 * state_improvement_reward +  # state improvement
            0.1 * task_completion_reward      # task completion
        )
        
        # remove fixed base reward, make differentiation more obvious
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """get action bounds"""
        return self.battery.params.p_min, self.battery.params.p_max
    
    def reset_state(self):
        """reset battery state"""
        self.battery.current_soc = self.battery.params.initial_soc

class HeatPumpMDPDevice(DeviceMDPInterface):
    """heat pump device implementation"""
    
    def __init__(self, heatpump_model: HeatPumpModel):
        self.heatpump = heatpump_model
        self.cop = heatpump_model.params.cop
    
    def get_state_features(self) -> np.ndarray:
        """get heat pump state features [current temperature, target temperature, comfort]"""
        current_temp = self.heatpump.current_temp
        target_temp = self._get_target_temperature()
        comfort_score = 1.0 - min(1.0, abs(current_temp - target_temp) / 3.0)
        
        return np.array([current_temp, target_temp, comfort_score])
    
    def _get_target_temperature(self) -> float:
        """get target temperature (based on time)"""
        hour = datetime.now().hour
        if 8 <= hour < 22:
            return self.heatpump.params.primary_target_temp
        else:
            return self.heatpump.params.secondary_target_temp
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """heat pump state transition"""
        current_temp = self.heatpump.current_temp
        outside_temp = env_state['temperature']
        
        # calculate heat output
        heat_output = action * self.cop if action > 0 else 0
        
        # heat loss
        heat_loss = self.heatpump.params.heat_loss_coef * (current_temp - outside_temp)
        
        # temperature change
        net_heat = heat_output - heat_loss
        temp_change = net_heat / (self.heatpump.params.room_volume * 1.2)
        
        new_temp = current_temp + temp_change
        new_temp = np.clip(new_temp, self.heatpump.params.temp_min, self.heatpump.params.temp_max)
        
        # update heat pump state
        self.heatpump.current_temp = new_temp
        
        return {
            'temperature': new_temp,
            'power': action,
            'heat_output': heat_output,
            'heat_loss': heat_loss
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        reward_components = {}
        
        # re-design economic reward, encourage efficient use
        price = env_state.get('price', 0.15)
        
        if action <= 0:  # not use heat pump
            economic_reward = 0.1  # small base reward
        else:
            # calculate efficiency based on COP and price
            heat_output = action * self.cop
            efficiency_ratio = heat_output / action if action > 0 else 0
            
            if price < 0.12:  # low price
                economic_reward = 1.0 - (action * price * 0.5)  # encourage use
            elif price > 0.20:  # high price
                if efficiency_ratio > 3.5:  # high efficiency use
                    economic_reward = 0.5 - (action * price * 0.3)
                else:
                    economic_reward = -(action * price * 0.8)  # punish low efficiency high price use
            else:  # medium price
                economic_reward = 0.2 - (action * price * 0.4)
        
        reward_components['economic'] = economic_reward
        
        # re-design comfort reward, more tolerant temperature control
        current_temp = next_state['temperature']
        target_temp = self._get_target_temperature()
        temp_diff = abs(current_temp - target_temp)
        
        if temp_diff <= 1.0:  # good temperature control
            comfort_reward = 3.0 - temp_diff * 2.0  # 1.0-3.0 points
        elif temp_diff <= 2.5:  # acceptable temperature control
            comfort_reward = 2.0 - temp_diff * 0.5  # 0.75-1.75 points
        elif temp_diff <= 4.0:  # acceptable
            comfort_reward = 1.0 - temp_diff * 0.2  # 0.2-1.0 points
        else:  # bad temperature control
            comfort_reward = -temp_diff * 0.5  # negative points
            
        reward_components['comfort'] = comfort_reward
        
        # add temperature stability reward
        # check temperature change (need history temperature)
        if hasattr(self.heatpump, 'prev_temp'):
            temp_change = abs(current_temp - self.heatpump.prev_temp)
            if temp_change <= 0.5:  # temperature stable
                stability_reward = 1.0
            elif temp_change <= 1.5:  # moderate change
                stability_reward = 0.5
            else:  # large temperature fluctuation
                stability_reward = -0.5
        else:
            stability_reward = 0.0
            
        self.heatpump.prev_temp = current_temp  # save current temperature
        reward_components['stability'] = stability_reward
        
        # add time appropriateness reward
        hour = datetime.now().hour
        if 8 <= hour <= 22:  # daytime
            time_appropriateness = 1.0 if action > 0 else 0.0
        else:  # nighttime
            time_appropriateness = 0.5 if action > 0 else 0.2
            
        reward_components['time_appropriateness'] = time_appropriateness
        
        # re-balance weights, ensure positive incentive
        total_reward = (
            0.2 * economic_reward +        # decrease economic weight
            0.5 * comfort_reward +         # increase comfort weight
            0.2 * stability_reward +       # temperature stability
            0.1 * time_appropriateness     # time appropriateness
        )
        
        # add base participation reward
        base_participation_reward = 0.2
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """get action bounds"""
        return 0.0, self.heatpump.params.max_power
    
    def reset_state(self):
        """reset heat pump state"""
        self.heatpump.current_temp = self.heatpump.params.initial_temp

class EVMDPDevice(DeviceMDPInterface):
    """EV device MDP implementation"""
    
    def __init__(self, ev_model: EVModel):
        self.ev = ev_model
        self.battery_capacity = ev_model.params.battery_capacity
    
    def get_state_features(self) -> np.ndarray:
        """get EV state features [SOC, connection state, charging urgency]"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # charging urgency (based on user behavior)
        if self.ev.user_behavior and is_connected:
            remaining_time = max(0, (self.ev.user_behavior.disconnection_time - datetime.now()).total_seconds() / 3600)
            soc_gap = max(0, self.ev.user_behavior.target_soc - soc)
            urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
        else:
            urgency = 0.0
        
        return np.array([soc, float(is_connected), urgency])
    
    def _is_connected(self) -> bool:
        """check if EV is connected"""
        if not self.ev.user_behavior:
            return True
        now = datetime.now()
        return self.ev.user_behavior.connection_time <= now < self.ev.user_behavior.disconnection_time
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """EV state transition"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # only charge when connected
        actual_power = action if is_connected and action > 0 else 0
        
        if actual_power > 0:
            energy_change = actual_power * self.ev.params.efficiency
            new_soc = soc + energy_change / self.battery_capacity
        else:
            energy_change = 0
            new_soc = soc
        
        new_soc = np.clip(new_soc, self.ev.params.soc_min, self.ev.params.soc_max)
        
        # update EV state
        self.ev.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': actual_power,
            'connected': is_connected,
            'energy_added': energy_change
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate EV reward - provide better learning signal"""
        reward_components = {}
        
        # re-design economic reward, encourage smart charging
        power = next_state.get('power', 0.0)
        price = env_state.get('price', 0.15)
        
        if power <= 0:  # not charge
            economic_reward = 0.1  # small base reward
        else:
            if price < 0.12:  # low price
                economic_reward = 2.0 - (power * price * 0.5)  # encourage low price charge
            elif price > 0.20:  # high price
                economic_reward = -(power * price * 0.8)  # punish high price charge
            else:  # medium price
                economic_reward = 0.5 - (power * price * 0.6)
                
        reward_components['economic'] = economic_reward
        
        # re-design charging completion reward, provide progressive reward
        current_soc = next_state.get('soc', 0.0)
        
        if self.ev.user_behavior:
            target_soc = self.ev.user_behavior.target_soc
            min_required_soc = getattr(self.ev.user_behavior, 'min_required_soc', 0.6)
            
            if current_soc >= target_soc:
                completion_reward = 5.0  # high reward for reaching target SOC
            elif current_soc >= min_required_soc:
                # progressive reward after reaching minimum required SOC
                progress = (current_soc - min_required_soc) / (target_soc - min_required_soc)
                completion_reward = 2.0 + progress * 3.0  # 2-5 points progressive reward
            else:
                # reward for trying to reach minimum required SOC
                progress = current_soc / min_required_soc
                completion_reward = progress * 2.0  # 0-2 points
        else:
            # default target SOC is 0.8
            if current_soc >= 0.8:
                completion_reward = 3.0
            elif current_soc >= 0.6:
                completion_reward = 1.0 + (current_soc - 0.6) / 0.2 * 2.0
            else:
                completion_reward = current_soc / 0.6
                
        reward_components['completion'] = completion_reward
        
        # re-design connection reward, more reasonable
        is_connected = next_state.get('connected', False)
        
        if not is_connected:
            if action > 0:
                connection_reward = -2.0  
            else:
                connection_reward = 0.0  
        else:
            # car is connected
            if action > 0:
                connection_reward = 1.0  # connected and charging, reward
            else:
                connection_reward = 0.2  # connected but no charge, small reward
                
        reward_components['connection'] = connection_reward
        
        # add charging urgency reward
        urgency_reward = 0.0
        if is_connected and hasattr(self.ev, 'user_behavior') and self.ev.user_behavior:
            try:
                from datetime import datetime
                now = datetime.now()
                remaining_time = (self.ev.user_behavior.disconnection_time - now).total_seconds() / 3600
                soc_gap = max(0, self.ev.user_behavior.target_soc - current_soc)
                
                if remaining_time > 0 and soc_gap > 0:
                    urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
                    if urgency > 0.7 and action > 0:  # high urgency and charge
                        urgency_reward = 2.0 * urgency
                    elif urgency < 0.3 and action <= 0:  # low urgency and not charge
                        urgency_reward = 0.5
            except:
                urgency_reward = 0.0
                
        reward_components['urgency'] = urgency_reward
        
        # re-balance weights
        total_reward = (
            0.2 * economic_reward +     # decrease economic weight
            0.5 * completion_reward +   # increase completion reward weight
            0.2 * connection_reward +   # connection reward
            0.1 * urgency_reward        # urgency reward
        )
        
        # add base participation reward
        base_participation_reward = 0.3
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """get action bounds"""
        return 0.0, self.ev.params.max_charging_power
    
    def reset_state(self):
        """reset EV state"""
        self.ev.current_soc = self.ev.params.initial_soc

class PVMDPDevice(DeviceMDPInterface):
    """PV device MDP implementation (read-only device)"""
    
    def __init__(self, pv_model: PVModel):
        self.pv = pv_model
    
    def get_state_features(self) -> np.ndarray:
        """get PV state features [current power, forecast power]"""
        # PV is read-only device, here return state information
        current_power = 0.0  # simplified implementation
        forecast_power = 0.0
        return np.array([current_power, forecast_power])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """PV state transition (PV is read-only device)"""
        # calculate actual power (based on solar irradiance)
        irradiance = env_state['solar_irradiance']
        max_power = self.pv.params.max_power
        efficiency = self.pv.params.efficiency
        
        # power model
        actual_power = max_power * efficiency * (irradiance / 1000.0) if irradiance > 0 else 0
        
        return {
            'power': actual_power,
            'irradiance': irradiance,
            'efficiency': efficiency
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate PV reward (generation reward)"""
        power_generated = next_state['power']
        price = env_state['price']
        
        # PV generation reward
        generation_reward = power_generated * price
        
        return generation_reward, {'generation': generation_reward}
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """PV has no action space"""
        return 0.0, 0.0
    
    def reset_state(self):
        """reset PV state"""
        pass

class FlexOfferEnv(gym.Env):
    """unified FlexOffer MDP environment"""
    
    def __init__(
        self,
        devices: Dict[str, Dict],
        time_horizon: int = 24,
        time_step: float = 1.0,
        start_time: datetime = None,
        price_data: pd.DataFrame = None,
        user_preferences: Dict[str, float] = None,
        weather_data: pd.DataFrame = None,
        data_dir: str = "data",
    ):
        """
        initialize unified FlexOffer environment
        
        Args:
            devices: device configuration dictionary
            time_horizon: time range
            time_step: time step
            start_time: start time
            price_data: price data
            user_preferences: user preferences
            weather_data: weather data
        """
        super().__init__()
        
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time if start_time else datetime.now()
        self.current_time = self.start_time
        self.current_step = 0
        
        # initialize environment dynamics, pass data_dir parameter
        self.env_dynamics = EnvironmentDynamics(price_data, weather_data, data_dir)
        
        # initialize user preferences
        self.user_preferences = {
            "economic": 0.25,
            "comfort": 0.25,
            "self_sufficient": 0.25,
            "environmental": 0.25
        }
        if user_preferences:
            self.user_preferences.update(user_preferences)
            # normalize
            total = sum(self.user_preferences.values())
            self.user_preferences = {k: v/total for k, v in self.user_preferences.items()}
        
        # initialize device MDP
        self.device_mdps = {}
        self.device_ids = []
        self.device_types = {}
        
        for device_id, config in devices.items():
            device_type = config['type']
            device_model = self._create_device_model(device_type, config['params'])
            device_mdp = self._create_device_mdp(device_type, device_model)
            
            self.device_mdps[device_id] = device_mdp
            self.device_ids.append(device_id)
            self.device_types[device_id] = device_type
        
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # define observation and action space
        self._setup_spaces()
    
    def _create_device_model(self, device_type: str, params):
        """create device model"""
        if device_type == DeviceType.BATTERY:
            return BatteryModel(params)
        elif device_type == DeviceType.HEAT_PUMP:
            return HeatPumpModel(params)
        elif device_type == DeviceType.EV:
            return EVModel(params)
        elif device_type == DeviceType.PV:
            return PVModel(params)
        elif device_type == DeviceType.DISHWASHER:
            return DishwasherModel(params)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def _create_device_mdp(self, device_type: str, device_model) -> DeviceMDPInterface:
        """create device MDP"""
        if device_type == DeviceType.BATTERY:
            return BatteryMDPDevice(device_model)
        elif device_type == DeviceType.HEAT_PUMP:
            return HeatPumpMDPDevice(device_model)
        elif device_type == DeviceType.EV:
            return EVMDPDevice(device_model)
        elif device_type == DeviceType.PV:
            return PVMDPDevice(device_model)
        elif device_type == DeviceType.DISHWASHER:
            return DishwasherMDPDevice(device_model)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def _setup_spaces(self):
        """set observation and action space"""
        # calculate state space dimension
        # general state: time(4) + environment(5) + markov history(device number+3) = 12+device number
        # device state: feature dimension of each device
        env_state_dim = 4 + 5 + len(self.device_ids) + 3  
        device_state_dim = sum(len(mdp.get_state_features()) for mdp in self.device_mdps.values())
        total_state_dim = env_state_dim + device_state_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_state_dim,), dtype=np.float32
        )
        
        # action space: one continuous action for each controllable device
        controllable_devices = [
            device_id for device_id in self.device_ids 
            if self.device_types[device_id] != DeviceType.PV
        ]
        
        action_bounds = []
        for device_id in controllable_devices:
            low, high = self.device_mdps[device_id].get_action_bounds()
            action_bounds.append([low, high])
        
        if action_bounds:
            action_bounds = np.array(action_bounds)
            self.action_space = spaces.Box(
                low=action_bounds[:, 0], high=action_bounds[:, 1], dtype=np.float32
            )
        else:
            # if there is no controllable device, create a virtual action space
            self.action_space = spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """reset environment"""
        super().reset(seed=seed)
        
        self.current_time = self.start_time
        self.current_step = 0
        
        # reset markov history
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # reset environment dynamics
        self.env_dynamics.price_history = []
        self.env_dynamics.weather_history = []
        
        # reset all devices
        for device_mdp in self.device_mdps.values():
            device_mdp.reset_state()
        
        # get initial observation
        observation = self._get_observation()
        info = {'time': self.current_time, 'step': self.current_step}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """execute one step"""
        # get current environment state
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # map action to devices
        device_actions = self._map_actions_to_devices(action)
        
        # execute device state transition
        device_next_states = {}
        total_reward = 0.0
        all_reward_components = {}
        total_cost = 0.0
        
        for device_id, device_action in device_actions.items():
            device_mdp = self.device_mdps[device_id]
            
            # state transition
            next_state = device_mdp.transition_state(device_action, env_state)
            device_next_states[device_id] = next_state
            
            # calculate reward
            device_reward, reward_components = device_mdp.calculate_reward(
                device_action, next_state, env_state
            )
            
            total_reward += device_reward
            all_reward_components[device_id] = reward_components
            
            # accumulate cost
            if 'power' in next_state:
                cost = next_state['power'] * env_state['price'] * self.time_step
                total_cost += cost
        
        # apply user preferences weight
        weighted_reward = self._apply_user_preferences(total_reward, all_reward_components)
        
        # update markov history
        self.markov_history['prev_actions'] = np.array(list(device_actions.values()))
        self.markov_history['prev_reward'] = weighted_reward
        self.markov_history['cumulative_cost'] += total_cost
        self.markov_history['cumulative_energy'] += sum(abs(a) for a in device_actions.values()) * self.time_step
        
        # update time
        self.current_time += timedelta(hours=self.time_step)
        self.current_step += 1
        
        # check termination condition
        done = self.current_step >= self.time_horizon
        
        # get next observation
        next_observation = self._get_observation()
        
        # build information dictionary
        info = {
            'time': self.current_time,
            'step': self.current_step,
            'device_states': device_next_states,
            'reward_components': all_reward_components,
            'total_cost': total_cost,
            'env_state': env_state
        }
        
        return next_observation, weighted_reward, done, False, info
    
    def _map_actions_to_devices(self, action: np.ndarray) -> Dict[str, float]:
        """map action to devices"""
        device_actions = {}
        action_idx = 0
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            
            if device_type == DeviceType.PV:
                # PV device is read-only
                device_actions[device_id] = 0.0
            else:
                # controllable device
                if action_idx < len(action):
                    device_actions[device_id] = float(action[action_idx])
                    action_idx += 1
                else:
                    device_actions[device_id] = 0.0
        
        return device_actions
    
    def _apply_user_preferences(self, base_reward: float, reward_components: Dict) -> float:
        """apply user preferences weight"""
        # here can apply user preferences based on reward_components
        return base_reward
    
    def _get_observation(self) -> np.ndarray:
        """get current observation state"""
        # time feature
        hour = self.current_time.hour
        time_features = np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if self.current_time.weekday() < 5 else 0.0,
            self.current_step / self.time_horizon
        ])
        
        # environment feature
        env_state = self.env_dynamics.get_current_state(self.current_time)
        env_features = np.array([
            env_state['price'],
            env_state['price_trend'],
            env_state['temperature'],
            env_state['solar_irradiance'],
            env_state['weather_trend']['temperature_trend']
        ])
        
        # markov history feature
        markov_features = np.concatenate([
            self.markov_history['prev_actions'],
            [self.markov_history['prev_reward']],
            [self.markov_history['cumulative_cost']],
            [self.markov_history['cumulative_energy']]
        ])
        
        # device state feature
        device_features = []
        for device_id in self.device_ids:
            device_state = self.device_mdps[device_id].get_state_features()
            device_features.append(device_state)
        
        device_features = np.concatenate(device_features)
        
        # merge all features
        full_observation = np.concatenate([
            time_features,
            env_features,
            markov_features,
            device_features
        ])
        
        return full_observation.astype(np.float32)
    
    def generate_dfo(self) -> Dict[str, DFOSystem]:
        dfo_systems = {}
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            device_mdp = self.device_mdps[device_id]
            
            if device_type != DeviceType.PV:  
                dfo = DFOSystem(self.time_horizon)
                
                for t in range(self.time_horizon):
                    # get action bounds
                    p_min, p_max = device_mdp.get_action_bounds()
                    
                    # create time slice
                    dfo_slice = DFOSlice(
                        time_step=t,
                        energy_min=p_min * self.time_step,
                        energy_max=p_max * self.time_step,
                        constraints=[]
                    )
                    
                    dfo.add_slice(dfo_slice)
                
                dfo_systems[device_id] = dfo
        
        return dfo_systems

# backward compatible alias
FlexOfferEnvMDP = FlexOfferEnv 
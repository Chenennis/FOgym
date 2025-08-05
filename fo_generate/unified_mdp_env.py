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
    """Device type enumeration"""
    BATTERY = "battery"
    HEAT_PUMP = "heat_pump"
    EV = "ev"
    PV = "pv"
    DISHWASHER = "dishwasher"

class EnvironmentDynamics:
    """Environment dynamics management - ensures Markov property"""
    
    def __init__(self, price_data: pd.DataFrame = None, weather_data: pd.DataFrame = None, 
                 data_dir: str = "data"):
        self.data_dir = data_dir
        self.price_data = price_data
        self.weather_data = weather_data
        self.price_loader = PriceLoader(data_dir)
        
        # Cache for current state
        self._current_state = None
        self._last_update_time = None
        
        # Initialize data if not provided
        if self.price_data is None:
            self.price_data = self.price_loader.get_price_data(
                datetime.now().replace(minute=0, second=0, microsecond=0), 
                168  # One week of data
            )
        
        if self.weather_data is None:
            from fo_generate.data_loader import DataLoader
            data_loader = DataLoader(data_dir)
            self.weather_data = data_loader.load_weather_data()

    def get_current_state(self, current_time: datetime) -> Dict[str, Any]:
        """Get current environment state"""
        # If state is already cached for this time, return it
        if (self._current_state is not None and 
            self._last_update_time is not None and 
            self._last_update_time == current_time):
            return self._current_state
        
        # Get current price and weather
        current_price = self._get_price_at_time(current_time)
        current_weather = self._get_weather_at_time(current_time)
        
        # Get price trend (increasing/decreasing)
        price_trend = self._get_price_trend()
        
        # Get weather trend
        weather_trend = self._get_weather_trend()
        
        # Predict future prices and weather
        future_prices = self._predict_future_prices(current_time)
        future_weather = self._predict_future_weather(current_time)
        
        # Create state dictionary
        state = {
            'current_time': current_time,
            'current_price': current_price,
            'current_weather': current_weather,
            'price_trend': price_trend,
            'weather_trend': weather_trend,
            'future_prices': future_prices,
            'future_weather': future_weather
        }
        
        # Cache state
        self._current_state = state
        self._last_update_time = current_time
        
        return state

    def _get_price_at_time(self, current_time: datetime) -> float:
        """Get price at specified time"""
        if self.price_data is None or len(self.price_data) == 0:
            # Default price if no data available
            return 0.15
        
        # Find closest time in price data
        closest_idx = None
        min_diff = float('inf')
        
        for idx, row in self.price_data.iterrows():
            time_diff = abs((row['timestamp'] - current_time).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_idx = idx
        
        if closest_idx is not None:
            return self.price_data.iloc[closest_idx]['price']
        
        # If no match found, use current hour's typical price
        try:
            return self.price_loader.get_current_price(current_time)['price']
        except:
            # Fallback to default price
            return 0.15

    def _get_weather_at_time(self, current_time: datetime) -> Dict[str, float]:
        """Get weather at specified time"""
        if self.weather_data is None or len(self.weather_data) == 0:
            # Default weather if no data available
            return {
                'temperature': 20.0,
                'solar_irradiance': 0.0 if current_time.hour < 6 or current_time.hour > 18 else 500.0,
                'wind_speed': 5.0
            }
        
        # Find closest time in weather data
        closest_idx = None
        min_diff = float('inf')
        
        for idx, row in self.weather_data.iterrows():
            time_diff = abs((row['timestamp'] - current_time).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_idx = idx
        
        if closest_idx is not None:
            row = self.weather_data.iloc[closest_idx]
            return {
                'temperature': row['temperature'],
                'solar_irradiance': row['solar_irradiance'],
                'wind_speed': row['wind_speed']
            }
        
        # Fallback to default weather model
            hour = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
        # Simple seasonal temperature model
        base_temp = 10 + 15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        daily_variation = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
        temperature = base_temp + daily_variation
            
        # Solar irradiance model
        if 6 <= hour <= 18:
            solar_angle = math.sin(math.pi * (hour - 6) / 12)
            seasonal_factor = max(0.2, math.sin(2 * math.pi * (day_of_year - 80) / 365))
            solar_irradiance = 800 * solar_angle * seasonal_factor
        else:
            solar_irradiance = 0.0
        
        # Wind speed model
        wind_speed = 5.0 + 2.0 * math.sin(2 * math.pi * day_of_year / 365)
                
        return {
        'temperature': temperature,
        'solar_irradiance': solar_irradiance,
        'wind_speed': wind_speed
        }

    def _get_price_trend(self) -> float:
        """Get price trend (positive: increasing, negative: decreasing)"""
        # Use cached price data to calculate trend
        if self.price_data is None or len(self.price_data) < 3:
            return 0.0  # No trend if insufficient data
        
        # Get last 3 prices
        prices = self.price_data['price'].tail(3).values
        
        # Calculate trend using simple linear regression slope
        x = np.array([0, 1, 2])
        y = prices
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize trend to [-1, 1] range
        max_slope = 0.05  # Maximum expected slope
        normalized_trend = max(-1.0, min(1.0, slope / max_slope))
        
        return normalized_trend

    def _get_weather_trend(self) -> Dict[str, float]:
        """Get weather trend for each component"""
        # Use cached weather data to calculate trend
        if self.weather_data is None or len(self.weather_data) < 3:
            return {'temperature': 0.0, 'solar_irradiance': 0.0, 'wind_speed': 0.0}
        
        # Get last 3 weather records
        temps = self.weather_data['temperature'].tail(3).values
        solar = self.weather_data['solar_irradiance'].tail(3).values
        wind = self.weather_data['wind_speed'].tail(3).values
        
        # Calculate trends using simple linear regression slopes
        x = np.array([0, 1, 2])
        temp_slope, _ = np.polyfit(x, temps, 1)
        solar_slope, _ = np.polyfit(x, solar, 1)
        wind_slope, _ = np.polyfit(x, wind, 1)
        
        # Normalize trends to [-1, 1] range
        max_temp_slope = 2.0  # Maximum expected temperature change per hour
        max_solar_slope = 100.0  # Maximum expected solar irradiance change per hour
        max_wind_slope = 1.0  # Maximum expected wind speed change per hour
        
        normalized_temp_trend = max(-1.0, min(1.0, temp_slope / max_temp_slope))
        normalized_solar_trend = max(-1.0, min(1.0, solar_slope / max_solar_slope))
        normalized_wind_trend = max(-1.0, min(1.0, wind_slope / max_wind_slope))
        
        return {
            'temperature': normalized_temp_trend,
            'solar_irradiance': normalized_solar_trend,
            'wind_speed': normalized_wind_trend
        }

    def _predict_future_prices(self, current_time: datetime) -> List[float]:
        """Predict prices for the next 3 hours"""
        future_prices = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            future_price = self._get_price_at_time(future_time)
            future_prices.append(future_price)
        return future_prices

    def _predict_future_weather(self, current_time: datetime) -> List[Dict[str, float]]:
        """Predict weather for the next 3 hours"""
        future_weather = []
        for h in range(1, 4):
            future_time = current_time + timedelta(hours=h)
            weather = self._get_weather_at_time(future_time)
            future_weather.append(weather)
        return future_weather

class DeviceMDPInterface(ABC):
    """Device MDP interface"""
    
    @abstractmethod
    def get_state_features(self) -> np.ndarray:
        """Get device state features"""
        pass
    
    @abstractmethod
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """Device state transition"""
        pass
    
    @abstractmethod
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """Calculate device reward"""
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[float, float]:
        """Get action bounds"""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset device state"""
        pass

class DishwasherMDPDevice(DeviceMDPInterface):
    """Dishwasher device MDP implementation"""
    
    def __init__(self, dishwasher_model: DishwasherModel):
        self.dishwasher = dishwasher_model
        
    def get_state_features(self) -> np.ndarray:
        """Get dishwasher state features [is_deployed, is_running, is_completed, current_step/total_steps, urgency, remaining_energy_demand]"""
        is_deployed = 1.0 if self.dishwasher.is_deployed else 0.0
        is_running = 1.0 if self.dishwasher.is_running else 0.0
        is_completed = 1.0 if self.dishwasher.is_completed else 0.0
        
        # Progress (current step/total steps)
        progress = self.dishwasher.current_cycle_step / max(1, self.dishwasher.total_cycle_steps)
        
        # Urgency
        urgency = self.dishwasher.calculate_urgency(datetime.now())
        
        # Remaining energy demand
        remaining_energy = self.dishwasher.params.total_energy - self.dishwasher.energy_consumed
        remaining_energy_norm = remaining_energy / max(1, self.dishwasher.params.total_energy)
        
        return np.array([is_deployed, is_running, is_completed, progress, urgency, remaining_energy_norm])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """Dishwasher state transition
        
        action: 0-1, indicates whether to start the dishwasher (only effective when deployed but not running)
        """
        current_time = datetime.now()
        
        # If not deployed yet, randomly simulate deployment (triggered by user in actual application)
        if not self.dishwasher.is_deployed:
            # Simulate user possibly deploying dishwasher at certain times
            if np.random.random() < 0.1:  # 10% chance of deployment
                self.dishwasher.deploy(current_time)
                
        # If deployed but not running, decide whether to start based on action
        start_success = False
        if self.dishwasher.is_deployed and not self.dishwasher.is_running and not self.dishwasher.is_completed:
            if action > 0.5:  # action > 0.5 means decision to start
                start_success = self.dishwasher.start_operation(current_time)
        
        # If running, continue running for one time step
        power_consumed = 0.0
        operation_completed = False
        if self.dishwasher.is_running:
            # Dishwasher requires fixed power to run
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
        """Calculate dishwasher reward - fixed version, reduce sparse rewards"""
        reward = 0.0
        reward_components = {}
        
        # 🔧 Fix 1: Keep high reward for task completion
        if next_state['operation_completed']:
            completion_reward = 50.0  # Reduced but still high completion reward
            reward += completion_reward
            reward_components['completion_reward'] = completion_reward
        
        # 🔧 Fix 2: Add running progress reward, encourage starting and continuous operation
        if next_state['is_running']:
            # Give increasing reward based on progress
            progress = getattr(self.dishwasher, 'current_cycle_step', 0)
            total_steps = getattr(self.dishwasher, 'total_cycle_steps', 10)
            
            if total_steps > 0:
                progress_ratio = progress / total_steps
                progress_reward = 5.0 + progress_ratio * 10.0  # 5-15 points progress reward
            else:
                progress_reward = 8.0  # Default running reward
                
            reward += progress_reward
            reward_components['progress_reward'] = progress_reward
        
        # 🔧 Fix 3: Redesign start timing reward, more lenient
        if next_state.get('start_success', False):
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'calculate_urgency'):
                urgency = self.dishwasher.calculate_urgency(current_time)
            else:
                urgency = 0.5  # Default medium urgency
            
            if urgency > 0.6:  # Starting at high urgency
                timing_reward = 15.0 * urgency  # Up to 9 points reward
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            elif urgency > 0.3:  # Medium urgency
                timing_reward = 5.0 * urgency  # 1.5-3 points reward
                reward += timing_reward
                reward_components['timing_reward'] = timing_reward
            else:  # Starting at low urgency, slight penalty
                timing_penalty = -2.0  # Reduced penalty
                reward += timing_penalty
                reward_components['timing_penalty'] = timing_penalty
        
        # 🔧 Fix 4: Redesign energy cost, don't over-penalize
        power_consumed = next_state.get('power_consumed', 0.0)
        price = env_state.get('price', 0.15)
        
        if power_consumed > 0:
            # Energy cost penalty relative to rewards, not absolute
            energy_cost = power_consumed * price * 0.3  # Reduced cost weight
            reward -= energy_cost
            reward_components['energy_cost'] = -energy_cost
        
        # 🔧 Fix 5: Redesign waiting time penalty, more lenient
        if (self.dishwasher.is_deployed and 
            not getattr(self.dishwasher, 'is_running', False) and 
            not getattr(self.dishwasher, 'is_completed', False)):
            
            current_time = datetime.now()
            if hasattr(self.dishwasher, 'deployment_time') and self.dishwasher.deployment_time:
                wait_time = (current_time - self.dishwasher.deployment_time).total_seconds() / 3600
                max_delay = getattr(self.dishwasher.params, 'max_start_delay', 6.0)
                
                if wait_time > max_delay:
                    # Timeout, heavy penalty but reduced magnitude
                    timeout_penalty = -20.0  # Reduced from -50 to -20
                    reward += timeout_penalty
                    reward_components['timeout_penalty'] = timeout_penalty
                elif wait_time > max_delay * 0.8:
                    # Near timeout, light penalty
                    wait_penalty = -5.0 * (wait_time / max_delay)  # Reduced penalty
                    reward += wait_penalty
                    reward_components['wait_penalty'] = wait_penalty
        
        # 🔧 Fix 6: Add deployment reward, encourage participation
        if self.dishwasher.is_deployed:
            deployment_reward = 2.0  # Reward just for deployment
            reward += deployment_reward
            reward_components['deployment_reward'] = deployment_reward
        
        # 🔧 Fix 7: Add base participation reward
        base_participation_reward = 1.0
        reward += base_participation_reward
        reward_components['participation_reward'] = base_participation_reward
        
        return reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """Get action bounds"""
        return 0.0, 1.0  # 0 means don't start, 1 means start
    
    def reset_state(self):
        """Reset dishwasher state"""
        self.dishwasher.is_deployed = False
        self.dishwasher.is_running = False
        self.dishwasher.is_completed = False
        self.dishwasher.current_cycle_step = 0
        self.dishwasher.deployment_time = None
        self.dishwasher.start_time = None
        self.dishwasher.completion_time = None
        self.dishwasher.energy_consumed = 0.0

class BatteryMDPDevice(DeviceMDPInterface):
    """Battery device MDP implementation"""
    
    def __init__(self, battery_model: BatteryModel):
        self.battery = battery_model
        self.efficiency = battery_model.params.efficiency
        self.capacity = battery_model.params.capacity_kwh
    
    def get_state_features(self) -> np.ndarray:
        """Get battery state features [SOC, max_charging_power, max_discharging_power, health]"""
        soc = self.battery.current_soc
        
        # Calculate available power range
        max_charge_energy = (self.battery.params.soc_max - soc) * self.capacity
        max_charge_power = min(self.battery.params.p_max, max_charge_energy / self.efficiency)
        
        max_discharge_energy = (soc - self.battery.params.soc_min) * self.capacity
        max_discharge_power = min(abs(self.battery.params.p_min), max_discharge_energy * self.efficiency)
        
        # Health (simplified model)
        health = max(0.8, 1.0 - soc * 0.1)  # Simplified health based on SOC
        
        return np.array([soc, max_charge_power, max_discharge_power, health])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """Battery state transition"""
        soc = self.battery.current_soc
        
        # Calculate SOC change
        if action > 0:  # Charging
            energy_change = action * self.efficiency
        else:  # Discharging
            energy_change = action / self.efficiency
        
        new_soc = soc + energy_change / self.capacity
        new_soc = np.clip(new_soc, self.battery.params.soc_min, self.battery.params.soc_max)
        
        # Update battery state
        self.battery.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': action,
            'energy_change': energy_change,
            'efficiency_loss': abs(energy_change) * (1 - self.efficiency) if action != 0 else 0
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """Calculate battery reward - enhanced learning signal version"""
        reward_components = {}
        
        # 🔧 Enhanced 1: Redesign economic reward, increase differentiation
        price = env_state.get('price', 0.15)
        base_price = 0.15
        price_ratio = price / base_price
        
        # Price period analysis - increase reward difference
        if price < 0.10:  # Ultra-low price period
            if action > 0:  # Charging
                economic_reward = abs(action) * 10.0 * (1.0 - price_ratio)  # Up to 10 points
            else:
                economic_reward = -abs(action) * 2.0  # Penalty for missed opportunity
        elif price < 0.12:  # Low price period
            if action > 0:  # Charging
                economic_reward = abs(action) * 5.0 * (1.0 - price_ratio)  # Up to 5 points
            else:
                economic_reward = 0.0
        elif price > 0.25:  # Ultra-high price period
            if action < 0:  # Discharging
                economic_reward = abs(action) * 15.0 * (price_ratio - 1.0)  # Up to 15 points
            else:
                economic_reward = -abs(action) * 5.0  # Heavy penalty for high-priced charging
        elif price > 0.18:  # High price period
            if action < 0:  # Discharging
                economic_reward = abs(action) * 8.0 * (price_ratio - 1.0)  # Up to 8 points
            else:
                economic_reward = -abs(action) * 2.0  # Light penalty for high-priced charging
        else:  # Medium price
            economic_reward = -abs(action) * 0.5  # Mild penalty
        
        reward_components['economic'] = economic_reward
        
        # 🔧 Enhanced 2: SOC management reward, create greater difference
        soc = next_state.get('soc', 0.5)
        
        if 0.45 <= soc <= 0.75:  # Optimal SOC range
            soc_reward = 8.0
        elif 0.35 <= soc <= 0.85:  # Good SOC range
            soc_reward = 4.0
        elif 0.25 <= soc <= 0.9:  # Acceptable range
            soc_reward = 1.0
        elif 0.15 <= soc <= 0.95:  # Boundary range
            soc_reward = -2.0
        else:  # Dangerous range
            soc_reward = -10.0  # Heavy penalty
            
        reward_components['soc_maintenance'] = soc_reward
        
        # 🔧 Enhanced 3: Continuous decision reward, encourage reasonable action sequence
        action_consistency_reward = 0.0
        if hasattr(self, 'prev_action'):
            prev_action = self.prev_action
            # Reward reasonable action continuity
            if abs(action - prev_action) < 0.5:  # Smooth operation
                action_consistency_reward = 2.0
            elif abs(action - prev_action) > 2.0:  # Severe change
                action_consistency_reward = -1.0
        
        self.prev_action = action
        reward_components['action_consistency'] = action_consistency_reward
        
        # 🔧 Enhanced 4: State improvement reward, encourage positive state changes
        state_improvement_reward = 0.0
        if hasattr(self, 'prev_soc'):
            prev_soc = self.prev_soc
            soc_change = soc - prev_soc
            
            # Reward moving towards ideal SOC range
            ideal_soc = 0.6
            prev_distance = abs(prev_soc - ideal_soc)
            current_distance = abs(soc - ideal_soc)
            
            if current_distance < prev_distance:  # Moving towards ideal state
                state_improvement_reward = 3.0 * (prev_distance - current_distance)
            else:  # Moving away from ideal state
                state_improvement_reward = -2.0 * (current_distance - prev_distance)
        
        self.prev_soc = soc
        reward_components['state_improvement'] = state_improvement_reward
        
        # 🔧 Enhanced 5: Task completion reward, based on time progress
        hour = datetime.now().hour
        task_completion_reward = 0.0
        
        # Give different task completion rewards based on time of day
        if 6 <= hour <= 9:  # Early morning peak
            if 0.7 <= soc <= 0.9:  # Prepare for daytime
                task_completion_reward = 5.0
        elif 18 <= hour <= 22:  # Evening peak
            if action < 0 and soc > 0.5:  # Discharging to support load
                task_completion_reward = 6.0
        elif 22 <= hour or hour <= 6:  # Night
            if action > 0 and price < 0.12:  # Nighttime low-price charging
                task_completion_reward = 4.0
                
        reward_components['task_completion'] = task_completion_reward
        
        # 🔧 Enhanced 6: Rebalance weights, increase overall reward range
        total_reward = (
            0.4 * economic_reward +           # Increase economic weight
            0.3 * soc_reward +               # SOC management
            0.1 * action_consistency_reward + # Action consistency
            0.1 * state_improvement_reward +  # State improvement
            0.1 * task_completion_reward      # Task completion
        )
        
        # 🔧 Enhanced 7: Remove fixed base reward, let differentiation be more obvious
        # No longer add base_participation_reward, let good/bad actions have greater difference
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """Get action bounds"""
        return self.battery.params.p_min, self.battery.params.p_max
    
    def reset_state(self):
        """Reset battery state"""
        self.battery.current_soc = self.battery.params.initial_soc

class HeatPumpMDPDevice(DeviceMDPInterface):
    """Heat pump device MDP implementation"""
    
    def __init__(self, heatpump_model: HeatPumpModel):
        self.heatpump = heatpump_model
        self.cop = heatpump_model.params.cop
    
    def get_state_features(self) -> np.ndarray:
        """Get heat pump state features [current_temp, target_temp, comfort_score]"""
        current_temp = self.heatpump.current_temp
        target_temp = self._get_target_temperature()
        comfort_score = 1.0 - min(1.0, abs(current_temp - target_temp) / 3.0)
        
        return np.array([current_temp, target_temp, comfort_score])
    
    def _get_target_temperature(self) -> float:
        """Get target temperature (based on time)"""
        hour = datetime.now().hour
        if 8 <= hour < 22:
            return self.heatpump.params.primary_target_temp
        else:
            return self.heatpump.params.secondary_target_temp
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """Heat pump state transition"""
        current_temp = self.heatpump.current_temp
        outside_temp = env_state['temperature']
        
        # Calculate heat output
        heat_output = action * self.cop if action > 0 else 0
        
        # Heat loss
        heat_loss = self.heatpump.params.heat_loss_coef * (current_temp - outside_temp)
        
        # Temperature change
        net_heat = heat_output - heat_loss
        temp_change = net_heat / (self.heatpump.params.room_volume * 1.2)
        
        new_temp = current_temp + temp_change
        new_temp = np.clip(new_temp, self.heatpump.params.temp_min, self.heatpump.params.temp_max)
        
        # Update heat pump state
        self.heatpump.current_temp = new_temp
        
        return {
            'temperature': new_temp,
            'power': action,
            'heat_output': heat_output,
            'heat_loss': heat_loss
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """Calculate heat pump reward - fixed version, provide positive incentive"""
        reward_components = {}
        
        # 🔧 Fix 1: Redesign economic reward, encourage efficient use
        price = env_state.get('price', 0.15)
        
        if action <= 0:  # Not using heat pump
            economic_reward = 0.1  # Small base reward
        else:
            # Calculate efficiency based on COP and price
            heat_output = action * self.cop
            efficiency_ratio = heat_output / action if action > 0 else 0
            
            if price < 0.12:  # Low price period
                economic_reward = 1.0 - (action * price * 0.5)  # Encourage use
            elif price > 0.20:  # High price period
                if efficiency_ratio > 3.5:  # Efficient use
                    economic_reward = 0.5 - (action * price * 0.3)
                else:
                    economic_reward = -(action * price * 0.8)  # Penalize low-efficiency high-priced use
            else:  # Medium price
                economic_reward = 0.2 - (action * price * 0.4)
        
        reward_components['economic'] = economic_reward
        
        # 🔧 Fix 2: Redesign comfort reward, more lenient temperature control
        current_temp = next_state['temperature']
        target_temp = self._get_target_temperature()
        temp_diff = abs(current_temp - target_temp)
        
        if temp_diff <= 1.0:  # Excellent temperature control
            comfort_reward = 3.0 - temp_diff * 2.0  # 1.0-3.0 points
        elif temp_diff <= 2.5:  # Acceptable temperature control
            comfort_reward = 2.0 - temp_diff * 0.5  # 0.75-1.75 points
        elif temp_diff <= 4.0:  # Basically acceptable
            comfort_reward = 1.0 - temp_diff * 0.2  # 0.2-1.0 points
        else:  # Poor temperature control
            comfort_reward = -temp_diff * 0.5  # Negative points
            
        reward_components['comfort'] = comfort_reward
        
        # 🔧 Fix 3: Add temperature stability reward
        # Check temperature change (requires historical temperature, simplified here)
        if hasattr(self.heatpump, 'prev_temp'):
            temp_change = abs(current_temp - self.heatpump.prev_temp)
            if temp_change <= 0.5:  # Temperature stable
                stability_reward = 1.0
            elif temp_change <= 1.5:  # Moderate change
                stability_reward = 0.5
            else:  # Large temperature fluctuation
                stability_reward = -0.5
        else:
            stability_reward = 0.0
            
        self.heatpump.prev_temp = current_temp  # Save current temperature
        reward_components['stability'] = stability_reward
        
        # 🔧 Fix 4: Add timely use reward
        hour = datetime.now().hour
        if 8 <= hour <= 22:  # Daytime use period
            time_appropriateness = 1.0 if action > 0 else 0.0
        else:  # Nighttime period
            time_appropriateness = 0.5 if action > 0 else 0.2
            
        reward_components['time_appropriateness'] = time_appropriateness
        
        # 🔧 Fix 5: Rebalance weights, ensure positive incentive
        total_reward = (
            0.2 * economic_reward +        # Decrease economic weight
            0.5 * comfort_reward +         # Increase comfort weight
            0.2 * stability_reward +       # Temperature stability
            0.1 * time_appropriateness     # Use timing
        )
        
        # 🔧 Fix 6: Add base participation reward
        base_participation_reward = 0.2
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """Get action bounds"""
        return 0.0, self.heatpump.params.max_power
    
    def reset_state(self):
        """Reset heat pump state"""
        self.heatpump.current_temp = self.heatpump.params.initial_temp

class EVMDPDevice(DeviceMDPInterface):
    """Electric vehicle device MDP implementation"""
    
    def __init__(self, ev_model: EVModel):
        self.ev = ev_model
        self.battery_capacity = ev_model.params.battery_capacity
    
    def get_state_features(self) -> np.ndarray:
        """Get EV state features [SOC, connection_status, charging_urgency]"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # Charging urgency (based on user behavior)
        if self.ev.user_behavior and is_connected:
            remaining_time = max(0, (self.ev.user_behavior.disconnection_time - datetime.now()).total_seconds() / 3600)
            soc_gap = max(0, self.ev.user_behavior.target_soc - soc)
            urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
        else:
            urgency = 0.0
        
        return np.array([soc, float(is_connected), urgency])
    
    def _is_connected(self) -> bool:
        """Check if EV is connected"""
        if not self.ev.user_behavior:
            return True
        now = datetime.now()
        return self.ev.user_behavior.connection_time <= now < self.ev.user_behavior.disconnection_time
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """EV state transition"""
        soc = self.ev.current_soc
        is_connected = self._is_connected()
        
        # Only charge if connected
        actual_power = action if is_connected and action > 0 else 0
        
        if actual_power > 0:
            energy_change = actual_power * self.ev.params.efficiency
            new_soc = soc + energy_change / self.battery_capacity
        else:
            energy_change = 0
            new_soc = soc
        
        new_soc = np.clip(new_soc, self.ev.params.soc_min, self.ev.params.soc_max)
        
        # Update EV state
        self.ev.current_soc = new_soc
        
        return {
            'soc': new_soc,
            'power': actual_power,
            'connected': is_connected,
            'energy_added': energy_change
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """Calculate EV reward - fixed version, provide better learning signal"""
        reward_components = {}
        
        # 🔧 Fix 1: Redesign economic reward, encourage smart charging
        power = next_state.get('power', 0.0)
        price = env_state.get('price', 0.15)
        
        if power <= 0:  # Not charging
            economic_reward = 0.1  # Small base reward
        else:
            if price < 0.12:  # Low-price period charging
                economic_reward = 2.0 - (power * price * 0.5)  # Encourage low-price charging
            elif price > 0.20:  # High-price period charging
                economic_reward = -(power * price * 0.8)  # Penalize high-price charging
            else:  # Medium price
                economic_reward = 0.5 - (power * price * 0.6)
                
        reward_components['economic'] = economic_reward
        
        # 🔧 Fix 2: Redesign charging completion reward, provide progressive reward
        current_soc = next_state.get('soc', 0.0)
        
        if self.ev.user_behavior:
            target_soc = self.ev.user_behavior.target_soc
            min_required_soc = getattr(self.ev.user_behavior, 'min_required_soc', 0.6)
            
            if current_soc >= target_soc:
                completion_reward = 5.0  # High reward for reaching target SOC
            elif current_soc >= min_required_soc:
                # Progressive reward after reaching minimum required SOC
                progress = (current_soc - min_required_soc) / (target_soc - min_required_soc)
                completion_reward = 2.0 + progress * 3.0  # 2-5 points progressive reward
            else:
                # Reward for trying to reach minimum required SOC
                progress = current_soc / min_required_soc
                completion_reward = progress * 2.0  # 0-2 points
        else:
            # Default target SOC is 0.8
            if current_soc >= 0.8:
                completion_reward = 3.0
            elif current_soc >= 0.6:
                completion_reward = 1.0 + (current_soc - 0.6) / 0.2 * 2.0
            else:
                completion_reward = current_soc / 0.6
                
        reward_components['completion'] = completion_reward
        
        # 🔧 Fix 3: Redesign connection reward, more reasonable
        is_connected = next_state.get('connected', False)
        
        if not is_connected:
            if action > 0:
                connection_reward = -2.0  # Penalize trying to charge a disconnected car
            else:
                connection_reward = 0.0  # Car not connected and not charging, normal
        else:
            # Car is connected
            if action > 0:
                connection_reward = 1.0  # Connected and charging, reward
            else:
                connection_reward = 0.2  # Connected but not charging, small reward
                
        reward_components['connection'] = connection_reward
        
        # 🔧 Fix 4: Add charging urgency reward
        urgency_reward = 0.0
        if is_connected and hasattr(self.ev, 'user_behavior') and self.ev.user_behavior:
            try:
                from datetime import datetime
                now = datetime.now()
                remaining_time = (self.ev.user_behavior.disconnection_time - now).total_seconds() / 3600
                soc_gap = max(0, self.ev.user_behavior.target_soc - current_soc)
                
                if remaining_time > 0 and soc_gap > 0:
                    urgency = min(1.0, soc_gap / max(remaining_time, 0.1))
                    if urgency > 0.7 and action > 0:  # High urgency and charging
                        urgency_reward = 2.0 * urgency
                    elif urgency < 0.3 and action <= 0:  # Low urgency and not charging
                        urgency_reward = 0.5
            except:
                urgency_reward = 0.0
                
        reward_components['urgency'] = urgency_reward
        
        # 🔧 Fix 5: Rebalance weights
        total_reward = (
            0.2 * economic_reward +     # Decrease economic weight
            0.5 * completion_reward +   # Increase completion reward weight
            0.2 * connection_reward +   # Connection reward
            0.1 * urgency_reward        # Urgency reward
        )
        
        # 🔧 Fix 6: Add base participation reward
        base_participation_reward = 0.3
        total_reward += base_participation_reward
        
        return total_reward, reward_components
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """Get action bounds"""
        return 0.0, self.ev.params.max_charging_power
    
    def reset_state(self):
        """Reset EV state"""
        self.ev.current_soc = self.ev.params.initial_soc

class PVMDPDevice(DeviceMDPInterface):
    """PV device MDP implementation (read-only device)"""
    
    def __init__(self, pv_model: PVModel):
        self.pv = pv_model
    
    def get_state_features(self) -> np.ndarray:
        """Get PV state features [current_power, forecast_power]"""
        # PV is a read-only device, return state information here
        current_power = 0.0  # Simplified implementation
        forecast_power = 0.0
        return np.array([current_power, forecast_power])
    
    def transition_state(self, action: float, env_state: Dict) -> Dict[str, Any]:
        """PV state transition (PV is not controllable)"""
        # Calculate actual power generation (based on solar irradiance)
        irradiance = env_state['solar_irradiance']
        max_power = self.pv.params.max_power
        efficiency = self.pv.params.efficiency
        
        # Simplified power generation model
        actual_power = max_power * efficiency * (irradiance / 1000.0) if irradiance > 0 else 0
        
        return {
            'power': actual_power,
            'irradiance': irradiance,
            'efficiency': efficiency
        }
    
    def calculate_reward(self, action: float, next_state: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """Calculate PV reward (generation revenue)"""
        power_generated = next_state['power']
        price = env_state['price']
        
        # PV generation revenue
        generation_reward = power_generated * price
        
        return generation_reward, {'generation': generation_reward}
    
    def get_action_bounds(self) -> Tuple[float, float]:
        """PV has no action space"""
        return 0.0, 0.0
    
    def reset_state(self):
        """Reset PV state"""
        pass

class FlexOfferEnv(gym.Env):
    """Unified FlexOffer MDP environment"""
    
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
        Initialize unified FlexOffer environment
        
        Args:
            devices: Device configuration dictionary
            time_horizon: Time range
            time_step: Time step
            start_time: Start time
            price_data: Price data
            user_preferences: User preferences
            weather_data: Weather data
        """
        super().__init__()
        
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time if start_time else datetime.now()
        self.current_time = self.start_time
        self.current_step = 0
        
        # Initialize environment dynamics, pass data_dir parameter
        self.env_dynamics = EnvironmentDynamics(price_data, weather_data, data_dir)
        
        # Initialize user preferences
        self.user_preferences = {
            "economic": 0.25,
            "comfort": 0.25,
            "self_sufficient": 0.25,
            "environmental": 0.25
        }
        if user_preferences:
            self.user_preferences.update(user_preferences)
            # Normalize
            total = sum(self.user_preferences.values())
            self.user_preferences = {k: v/total for k, v in self.user_preferences.items()}
        
        # Initialize device MDPs
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
        
        # Markov history state
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # Define observation and action spaces
        self._setup_spaces()
    
    def _create_device_model(self, device_type: str, params):
        """Create device model"""
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
        """Create device MDP"""
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
        """Set observation and action spaces"""
        # Calculate state space dimension
        # General state: time(4) + environment(5) + Markov history(device_count+3) = 12+device_count
        # Device state: Sum of feature dimensions for each device
        env_state_dim = 4 + 5 + len(self.device_ids) + 3  # Environment and Markov state
        device_state_dim = sum(len(mdp.get_state_features()) for mdp in self.device_mdps.values())
        total_state_dim = env_state_dim + device_state_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_state_dim,), dtype=np.float32
        )
        
        # Action space: one continuous action for each controllable device
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
            # If no controllable devices, create a virtual action space
            self.action_space = spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_time = self.start_time
        self.current_step = 0
        
        # Reset Markov history
        self.markov_history = {
            'prev_actions': np.zeros(len(self.device_ids)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0
        }
        
        # Reset environment dynamics
        self.env_dynamics.price_history = []
        self.env_dynamics.weather_history = []
        
        # Reset all devices
        for device_mdp in self.device_mdps.values():
            device_mdp.reset_state()
        
        # Get initial observation
        observation = self._get_observation()
        info = {'time': self.current_time, 'step': self.current_step}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Execute one step"""
        # Get current environment state
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # Map actions to devices
        device_actions = self._map_actions_to_devices(action)
        
        # Execute device state transition
        device_next_states = {}
        total_reward = 0.0
        all_reward_components = {}
        total_cost = 0.0
        
        for device_id, device_action in device_actions.items():
            device_mdp = self.device_mdps[device_id]
            
            # State transition
            next_state = device_mdp.transition_state(device_action, env_state)
            device_next_states[device_id] = next_state
            
            # Calculate reward
            device_reward, reward_components = device_mdp.calculate_reward(
                device_action, next_state, env_state
            )
            
            total_reward += device_reward
            all_reward_components[device_id] = reward_components
            
            # Accumulate cost
            if 'power' in next_state:
                cost = next_state['power'] * env_state['price'] * self.time_step
                total_cost += cost
        
        # Apply user preference weights
        weighted_reward = self._apply_user_preferences(total_reward, all_reward_components)
        
        # Update Markov history
        self.markov_history['prev_actions'] = np.array(list(device_actions.values()))
        self.markov_history['prev_reward'] = weighted_reward
        self.markov_history['cumulative_cost'] += total_cost
        self.markov_history['cumulative_energy'] += sum(abs(a) for a in device_actions.values()) * self.time_step
        
        # Update time
        self.current_time += timedelta(hours=self.time_step)
        self.current_step += 1
        
        # Check termination conditions
        done = self.current_step >= self.time_horizon
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Build info dictionary
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
        """Map actions to devices"""
        device_actions = {}
        action_idx = 0
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            
            if device_type == DeviceType.PV:
                # PV device is not controllable
                device_actions[device_id] = 0.0
            else:
                # Controllable devices
                if action_idx < len(action):
                    device_actions[device_id] = float(action[action_idx])
                    action_idx += 1
                else:
                    device_actions[device_id] = 0.0
        
        return device_actions
    
    def _apply_user_preferences(self, base_reward: float, reward_components: Dict) -> float:
        """Apply user preference weights"""
        # Here, user preferences can be applied based on different components in reward_components
        # Simplified implementation: return base reward directly
        return base_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # Time features
        hour = self.current_time.hour
        time_features = np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if self.current_time.weekday() < 5 else 0.0,
            self.current_step / self.time_horizon
        ])
        
        # Environment features
        env_state = self.env_dynamics.get_current_state(self.current_time)
        env_features = np.array([
            env_state['price'],
            env_state['price_trend'],
            env_state['temperature'],
            env_state['solar_irradiance'],
            env_state['weather_trend']['temperature_trend']
        ])
        
        # Markov history features
        markov_features = np.concatenate([
            self.markov_history['prev_actions'],
            [self.markov_history['prev_reward']],
            [self.markov_history['cumulative_cost']],
            [self.markov_history['cumulative_energy']]
        ])
        
        # Device state features
        device_features = []
        for device_id in self.device_ids:
            device_state = self.device_mdps[device_id].get_state_features()
            device_features.append(device_state)
        
        device_features = np.concatenate(device_features)
        
        # Combine all features
        full_observation = np.concatenate([
            time_features,
            env_features,
            markov_features,
            device_features
        ])
        
        return full_observation.astype(np.float32)
    
    def generate_dfo(self) -> Dict[str, DFOSystem]:
        """Generate DFO systems (integrated with FlexOffer process)"""
        dfo_systems = {}
        
        for device_id in self.device_ids:
            device_type = self.device_types[device_id]
            device_mdp = self.device_mdps[device_id]
            
            if device_type != DeviceType.PV:  # Only generate DFO for controllable devices
                dfo = DFOSystem(self.time_horizon)
                
                for t in range(self.time_horizon):
                    # Get action bounds
                    p_min, p_max = device_mdp.get_action_bounds()
                    
                    # Create time slice
                    dfo_slice = DFOSlice(
                        time_step=t,
                        energy_min=p_min * self.time_step,
                        energy_max=p_max * self.time_step,
                        constraints=[]
                    )
                    
                    dfo.add_slice(dfo_slice)
                
                dfo_systems[device_id] = dfo
        
        return dfo_systems

# Backward compatible alias
FlexOfferEnvMDP = FlexOfferEnv 
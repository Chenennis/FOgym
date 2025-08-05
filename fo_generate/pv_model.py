from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class PVParameters:
    """PV parameters"""
    pv_id: str           # PV ID
    max_power: float     # maximum output power
    efficiency: float    # efficiency
    area: float          # area
    location: str        # location
    tilt_angle: float    # tilt angle
    azimuth_angle: float # azimuth angle
    weather_dependent: bool  # whether depends on weather
    forecast_accuracy: float = 0.8  # forecast accuracy, default 80%
    

class PVModel:
    """PV model class"""
    def __init__(self, params: PVParameters):

        self.params = params
        
        # store forecast data
        self.forecast_data = None
        # store actual power history
        self.power_history = []
        # track PV stability
        self.stability_violations = 0
            
    def set_forecast_data(self, forecast_data: List[float]):

        self.forecast_data = forecast_data
        
    def predict_generation(self, 
                           time: datetime, 
                           weather_data: Optional[Dict] = None,
                           duration: float = 1.0,
                           use_forecast: bool = False) -> float:

        # if there is forecast data and need to use forecast data
        if use_forecast and self.forecast_data is not None:
            # calculate index of current time point in forecast data
            hour_diff = int((time - datetime.now()).total_seconds() / 3600)
            if 0 <= hour_diff < len(self.forecast_data):
                # add forecast error (±20%)
                accuracy = self.params.forecast_accuracy
                error = np.random.uniform(1 - (1 - accuracy), 1 + (1 - accuracy))
                return self.forecast_data[hour_diff] * error * duration
        
        # if there is no forecast data or not use forecast data, use model to calculate
        if not weather_data or not self.params.weather_dependent:
            # estimate solar radiation intensity based on time, here use simplified sunrise and sunset model
            hour = time.hour + time.minute / 60.0
            
            # day production, night not production
            if 6 <= hour <= 18:
                solar_intensity = np.sin(np.pi * (hour - 6) / 12)
                power = self.params.max_power * solar_intensity * self.params.efficiency
            else:
                power = 0.0
        else:
            # use weather data for more accurate prediction
            solar_intensity = weather_data.get('solar_radiation', 0)
            cloud_coverage = weather_data.get('cloud_coverage', 0)
            temperature = weather_data.get('temperature', 25)
            
            # consider cloud coverage
            solar_intensity *= (1 - 0.7 * cloud_coverage)
            
            # consider temperature effect on efficiency (temperature rise 1℃, efficiency drop about 0.4%)
            temp_efficiency = self.params.efficiency * (1 - 0.004 * max(0, temperature - 25))
            
            # consider angle factor
            angle_factor = np.cos(np.radians(self.params.tilt_angle))
            
            power = self.params.area * solar_intensity * temp_efficiency * angle_factor
            power = min(power, self.params.max_power)
        
        # add random fluctuation (±10%)
        power *= np.random.uniform(0.9, 1.1)
        
        # calculate total generation in duration
        energy = power * duration
        
        return energy
        
    def get_available_power(self, time: datetime, weather_data: Optional[Dict] = None) -> float:

        # use forecast data or model to calculate base generation
        base_generation = self.predict_generation(time, weather_data, use_forecast=True)
        
        # PV model now can only provide generation power, no storage ability
        return base_generation
    
    def calculate_stability_metrics(self, forecast_window: int = 12) -> Dict:

        if len(self.power_history) < 2 or self.forecast_data is None:
            return {
                "stability_score": 1.0,
                "forecast_deviation": 0.0,
                "storage_adequacy": 0.0  
            }
            
        # calculate history power fluctuation
        power_std = np.std(self.power_history[-forecast_window:]) if len(self.power_history) >= forecast_window else np.std(self.power_history)
        power_mean = np.mean(self.power_history[-forecast_window:]) if len(self.power_history) >= forecast_window else np.mean(self.power_history)
        power_volatility = power_std / (power_mean + 1e-6)  # avoid division by zero
        
        # calculate forecast deviation
        forecast_horizon = min(forecast_window, len(self.forecast_data))
        actual = self.power_history[-forecast_horizon:] if len(self.power_history) >= forecast_horizon else self.power_history
        forecast = self.forecast_data[:len(actual)]
        forecast_deviation = np.mean(np.abs(np.array(actual) - np.array(forecast[:len(actual)])) / (np.array(forecast[:len(actual)]) + 1e-6))
        
        # since there is no storage system, no storage adequacy
        storage_adequacy = 0.0
            
        # comprehensive stability score - adjust weights
        stability_score = 1.0 - (0.6 * power_volatility + 0.4 * forecast_deviation)
        stability_score = max(0, min(1, stability_score))
        
        return {
            "stability_score": stability_score,
            "power_volatility": power_volatility,
            "forecast_deviation": forecast_deviation,
            "storage_adequacy": storage_adequacy
        }
        
    def generate_dfo(self, start_time=None, time_horizon: int = None) -> DFOSystem:

        # compatible with old call, if first parameter is integer and second parameter is None
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # if start_time is None, use current time
        current_time = start_time if start_time is not None else datetime.now()
        
        # ensure time_horizon has value
        if time_horizon is None:
            time_horizon = 12  # default value
        
        dfo = DFOSystem(time_horizon)
        
        # ensure there is forecast data
        if self.forecast_data is None and time_horizon > 0:
            # if there is no forecast data, generate simulated forecast data
            self.forecast_data = []
            for t in range(min(12, time_horizon)):
                forecast_time = current_time + pd.Timedelta(hours=t)
                self.forecast_data.append(self.predict_generation(forecast_time))
        
        for t in range(time_horizon):
            # predict generation at current time
            forecast_time = current_time + pd.Timedelta(hours=t)
            
            # calculate energy boundary - now only return generation, minimum power is 0
            energy_max = self.predict_generation(forecast_time)
            energy_min = 0  # PV can only produce energy, not consume energy
            
            # create time slice
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=[]  # no SOC constraint
            )
            dfo.add_slice(slice)
            
            # simulate actual generation (add random fluctuation)
            actual_generation = self.predict_generation(forecast_time, use_forecast=False)
            self.power_history.append(actual_generation)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, pv_id: str = None) -> 'PVModel':

        # read parameter file
        params_df = pd.read_csv(params_file, comment='#')
        
        # if pv_id is specified, find corresponding data; otherwise use first row
        if pv_id:
            pv_data = params_df[params_df['pv_id'] == pv_id]
            if pv_data.empty:
                raise ValueError(f"PV ID {pv_id} not found in {params_file}")
            pv_data = pv_data.iloc[0]
        else:
            pv_data = params_df.iloc[0]
            pv_id = pv_data['pv_id']
        
        # create parameter object
        params = PVParameters(
            pv_id=pv_id,
            max_power=float(pv_data['max_power']),
            efficiency=float(pv_data['efficiency']),
            area=float(pv_data['area']),
            location=pv_data['location'],
            tilt_angle=float(pv_data['tilt_angle']),
            azimuth_angle=float(pv_data['azimuth_angle']),
            weather_dependent=pv_data['weather_dependent'] == 'True',
            forecast_accuracy=float(pv_data.get('forecast_accuracy', 0.8))
        )
        
        return cls(params)
    
    @classmethod
    def from_csv_with_forecast(cls, params_file: str, forecast_file: str = None, 
                               pv_id: str = None) -> 'PVModel':

        model = cls.from_csv(params_file, pv_id)
        
        # if forecast file is provided, load forecast data
        if forecast_file and os.path.exists(forecast_file):
            forecast_df = pd.read_csv(forecast_file)
            if pv_id in forecast_df.columns:
                # assume first column is time, then is forecast value of each PV ID
                model.forecast_data = forecast_df[pv_id].tolist()
                
        return model
        
    @classmethod
    def get_all_pv_ids(cls, params_file: str) -> List[str]:

        df = pd.read_csv(params_file, comment='#')
        return df['pv_id'].tolist() 
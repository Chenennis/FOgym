"""
Data loader - Load external data files or generate default data

Supports loading:
- Weather data (Danish characteristics)
- Price data (Danish electricity market)
- PV forecast data
- Working day data
- Manager and User configuration data
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
from .price_loader import PriceLoader

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_dir()
        # Initialize price loader
        self.price_loader = PriceLoader(data_dir)
        
        # 🔧 Add price cache mechanism
        self._cached_daily_prices = None  # Cache for 24-hour price data of a day
        self._cache_day_type = None  # Cached date type (weekday/weekend)
        self._cache_source = None  # Cache data source
        logger.info("DataLoader initialized, price cache mechanism enabled")
    
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def load_weather_data(self, filename: str = "weather_data.csv", 
                         start_time: datetime = None, 
                         hours: int = 168) -> pd.DataFrame:
        """Load weather data, generate Danish weather data if file doesn't exist"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"Successfully loaded weather data: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load weather data: {e}, using default generation")
        
        # Generate Danish weather data
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        return self._generate_danish_weather(start_time, hours)
    
    def load_price_data(self, filename: str = "price_data.csv",
                       start_time: datetime = None,
                       hours: int = 168) -> pd.DataFrame:
        """Load price data, prioritize reading Danish price data from grid_price.csv, use cache to avoid repeated reading"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Determine date type
        day_type = 'weekday' if start_time.weekday() < 5 else 'weekend'
        
        # 🔧 Check cache: if already have 24-hour data of the same date type, use cache directly
        if (self._cached_daily_prices is not None and 
            self._cache_day_type == day_type and 
            len(self._cached_daily_prices) == 24):
            
            logger.info(f"Using cached price data: {day_type}, data source: {self._cache_source}")
            return self._generate_price_data_from_cache(start_time, hours)
        
        # 🔧 First load or cache invalid: try to load from grid_price.csv and cache
        try:
            grid_price_file = os.path.join(self.data_dir, "grid_price.csv")
            if os.path.exists(grid_price_file):
                # Load and cache 24-hour price data for a day
                self._load_and_cache_daily_prices(grid_price_file, day_type)
                logger.info(f"Loaded and cached price data from grid_price.csv: {day_type}, 24-hour data cached")
                return self._generate_price_data_from_cache(start_time, hours)
        except Exception as e:
            logger.warning(f"Failed to load price data from grid_price.csv: {e}")
        
        # 🔧 Alternative 1: Use PriceLoader (maintain compatibility)
        try:
            price_data = self.price_loader.get_price_data(start_time, hours)
            logger.info(f"Using PriceLoader to get price data: {len(price_data)} records")
            return price_data
        except Exception as e:
            logger.warning(f"Failed to load price data using PriceLoader: {e}, trying to load traditional file")
        
        # 🔧 Alternative 2: Try to load traditional price_data.csv file
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"Successfully loaded alternative price data: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load alternative price data: {e}, using default generation")
        
        # 🔧 Last alternative: Generate Danish price data and cache
        logger.info("Using generated Danish price data")
        generated_data = self._generate_and_cache_daily_prices(day_type)
        return self._generate_price_data_from_cache(start_time, hours)
    
    def _load_and_cache_daily_prices(self, grid_price_file: str, day_type: str):
        """Load and cache 24-hour price data from grid_price.csv"""
        try:
            # Read grid_price.csv
            grid_data = pd.read_csv(grid_price_file)
            grid_data['timestamp'] = pd.to_datetime(grid_data['timestamp'])
            
            # Filter data of specified date type
            day_type_data = grid_data[grid_data['day_type'] == day_type]
            
            if len(day_type_data) == 0:
                logger.warning(f"No {day_type} data found in grid_price.csv")
                raise ValueError(f"No {day_type} data found in grid_price.csv")
            
            # Extract 24-hour data (0-23 hours)
            cached_prices = {}
            for hour in range(24):
                hour_data = day_type_data[day_type_data['hour'] == hour]
                if len(hour_data) > 0:
                    # Use the latest matching data
                    price_row = hour_data.iloc[-1]
                    cached_prices[hour] = {
                        'price_usd_kwh': price_row['price_usd_kwh'],
                        'price_dkk_kwh': price_row['price_dkk_kwh'],
                        'price_level': price_row['price_level'],
                        'hour': hour,
                        'day_type': day_type
                    }
                else:
                    # If data for a certain hour is missing, use predicted price
                    predicted_price = self._predict_price_for_hour(hour, day_type)
                    cached_prices[hour] = {
                        'price_usd_kwh': predicted_price,
                        'price_dkk_kwh': predicted_price * 7.0,
                        'price_level': self._get_price_level(predicted_price),
                        'hour': hour,
                        'day_type': day_type
                    }
            
            # Cache data
            self._cached_daily_prices = cached_prices
            self._cache_day_type = day_type
            self._cache_source = 'grid_data'
            
            logger.info(f"Successfully cached 24-hour price data for {day_type}, data completeness: {len(cached_prices)}/24")
            
        except Exception as e:
            logger.error(f"Failed to cache price data: {e}")
            raise
    
    def _generate_and_cache_daily_prices(self, day_type: str):
        """Generate and cache 24-hour price data for a day"""
        cached_prices = {}
        for hour in range(24):
            predicted_price = self._predict_price_for_hour(hour, day_type)
            cached_prices[hour] = {
                'price_usd_kwh': predicted_price,
                'price_dkk_kwh': predicted_price * 7.0,
                'price_level': self._get_price_level(predicted_price),
                'hour': hour,
                'day_type': day_type
            }
        
        # Cache data
        self._cached_daily_prices = cached_prices
        self._cache_day_type = day_type
        self._cache_source = 'predicted'
        
        logger.info(f"Successfully generated and cached 24-hour price data for {day_type}")
    
    def _generate_price_data_from_cache(self, start_time: datetime, hours: int) -> pd.DataFrame:
        """Generate price data for specified time range from cached 24-hour data"""
        if self._cached_daily_prices is None:
            raise ValueError("Price cache is empty")
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        result_data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            
            # Get price data for corresponding hour from cache
            if hour in self._cached_daily_prices:
                cached_hour_data = self._cached_daily_prices[hour]
                result_data.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_type': cached_hour_data['day_type'],
                    'price': cached_hour_data['price_usd_kwh'],
                    'price_dkk': cached_hour_data['price_dkk_kwh'],
                    'price_level': cached_hour_data['price_level'],
                    'source': self._cache_source
                })
            else:
                # Theoretically shouldn't happen, but add insurance
                logger.warning(f"Missing data for hour {hour} in cache, using default price")
                result_data.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_type': self._cache_day_type,
                    'price': 0.15,
                    'price_dkk': 1.05,
                    'price_level': 'medium',
                    'source': 'default'
                })
        
        result_df = pd.DataFrame(result_data)
        logger.debug(f"Generated price data from cache: {len(result_df)} records, time range {start_time} to {timestamps[-1]}")
        return result_df
    
    def _predict_price_for_hour(self, hour: int, day_type: str) -> float:
        """Predict price for a specific hour based on Danish price pattern (consistent with PriceLoader)"""
        base_price = 0.12  # Base price USD/kWh
        
        if day_type == 'weekday':
            # Workday price pattern
            if 0 <= hour <= 5:
                price_multiplier = np.random.uniform(0.7, 0.95)
            elif 6 <= hour <= 9:
                if hour == 6:
                    price_multiplier = np.random.uniform(1.3, 1.4)
                elif hour in [7, 8]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:  # hour == 9
                    price_multiplier = np.random.uniform(1.7, 1.9)
            elif 10 <= hour <= 16:
                price_multiplier = np.random.uniform(1.0, 1.2)
            elif 17 <= hour <= 21:
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(2.1, 2.3)
                else:
                    price_multiplier = np.random.uniform(1.9, 2.1)
            else:  # 22-23
                price_multiplier = np.random.uniform(1.1, 1.4)
        else:
            # Weekend price pattern
            if 0 <= hour <= 5:
                price_multiplier = np.random.uniform(0.6, 0.9)
            elif 6 <= hour <= 9:
                price_multiplier = np.random.uniform(1.0, 1.35)
            elif 10 <= hour <= 16:
                price_multiplier = np.random.uniform(1.2, 1.4)
            elif 17 <= hour <= 21:
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:
                    price_multiplier = np.random.uniform(1.6, 1.8)
            else:  # 22-23
                price_multiplier = np.random.uniform(1.0, 1.3)
        
        # Add random fluctuation
        noise = np.random.normal(0, 0.02)
        predicted_price = base_price * price_multiplier * (1 + noise)
        
        # Ensure price is within a reasonable range
        return max(0.08, min(0.35, predicted_price))
    
    def _get_price_level(self, price: float) -> str:
        """Determine price level based on price"""
        if price < 0.12:
            return 'low'
        elif price < 0.16:
            return 'medium'
        elif price < 0.20:
            return 'high'
        else:
            return 'peak'
    
    def load_pv_forecast_data(self, filename: str = "pv_forecast.csv",
                             start_time: datetime = None,
                             hours: int = 168) -> pd.DataFrame:
        """Load PV forecast data"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"Successfully loaded PV forecast data: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load PV forecast data: {e}, using default generation")
        
        # Generate PV forecast data
        return self._generate_pv_forecast(start_time, hours)
    
    def load_calendar_data(self, filename: str = "calendar_data.csv") -> pd.DataFrame:
        """Load working day data"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['date'] = pd.to_datetime(data['date'])
                logger.info(f"Successfully loaded working day data: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load working day data: {e}, using default generation")
        
        # Generate working day data
        return self._generate_calendar_data()
    
    def load_manager_config(self, filename: str = "manager_config.csv") -> pd.DataFrame:
        """Load Manager configuration data"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"Successfully loaded Manager configuration: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load Manager configuration: {e}, using default configuration")
        
        # Generate default Manager configuration
        return self._generate_default_manager_config()
    
    def load_user_config(self, filename: str = "user_config.csv") -> pd.DataFrame:
        """Load user configuration data"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"Successfully loaded user configuration: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load user configuration: {e}, using default configuration")
        
        # Generate default user configuration
        return self._generate_default_user_config()
    
    def load_device_config(self, filename: str = "device_config.csv") -> pd.DataFrame:
        """Load device configuration data"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"Successfully loaded device configuration: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load device configuration: {e}, using default configuration")
        
        # Generate default device configuration
        return self._generate_default_device_config()
    
    def _generate_danish_weather(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """Generate Danish weather data"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        weather_data = []
        
        for ts in timestamps:
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour
            
            # Danish seasonal temperature model
            seasonal_temp = 8 + 12 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            daily_variation = 4 * math.sin(2 * math.pi * (hour - 6) / 24)
            temperature = seasonal_temp + daily_variation + np.random.normal(0, 2)
            
            # Solar irradiance model (considering Danish latitude)
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                seasonal_factor = max(0.1, math.sin(2 * math.pi * (day_of_year - 80) / 365))
                irradiance = 600 * solar_angle * seasonal_factor
                irradiance = max(0, irradiance + np.random.normal(0, 50))
            else:
                irradiance = 0
            
            # Danish wind speed model
            wind_speed = 8 + 4 * math.sin(2 * math.pi * day_of_year / 365) + np.random.normal(0, 2)
            wind_speed = max(2, wind_speed)
            
            weather_data.append({
                'timestamp': ts,
                'temperature': round(temperature, 1),
                'solar_irradiance': round(irradiance, 1),
                'wind_speed': round(wind_speed, 1)
            })
        
        logger.info(f"Generated Danish weather data: {hours} hours")
        return pd.DataFrame(weather_data)
    
    def _generate_danish_prices(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """Generate Danish price data"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        price_data = []
        
        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.weekday() >= 5
            
            # Danish price model (DKK/kWh)
            if 0 <= hour < 6:  # Night low price
                base_price = 0.8
            elif 6 <= hour < 9:  # Morning peak
                base_price = 2.2
            elif 9 <= hour < 16:  # Daytime
                base_price = 1.5
            elif 16 <= hour < 20:  # Evening peak
                base_price = 2.5
            else:  # Evening
                base_price = 1.8
            
            # Weekend price adjustment
            if is_weekend:
                base_price *= 0.85
            
            # Add random fluctuation
            price = base_price + np.random.normal(0, 0.2)
            price = max(0.3, price)
            
            price_data.append({
                'timestamp': ts,
                'price': round(price, 3),
                'price_type': 'spot'
            })
        
        logger.info(f"Generated Danish price data: {hours} hours")
        return pd.DataFrame(price_data)
    
    def _generate_pv_forecast(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """Generate PV forecast data"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        pv_data = []
        
        for i, ts in enumerate(timestamps):
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour
            forecast_horizon = i + 1
            
            # 5kW system power forecast
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                seasonal_factor = max(0.2, math.sin(2 * math.pi * (day_of_year - 80) / 365))
                forecast_power = 5.0 * solar_angle * seasonal_factor
                
                # Add prediction uncertainty
                uncertainty = min(0.3, forecast_horizon * 0.02)
                forecast_power *= (1 + np.random.normal(0, uncertainty))
                forecast_power = max(0, forecast_power)
            else:
                forecast_power = 0
            
            # Confidence decreases with forecast time
            confidence = max(0.6, 0.95 - forecast_horizon * 0.015)
            
            pv_data.append({
                'timestamp': ts,
                'forecast_power': round(forecast_power, 2),
                'confidence': round(confidence, 3),
                'forecast_horizon': forecast_horizon
            })
        
        logger.info(f"Generated PV forecast data: {hours} hours")
        return pd.DataFrame(pv_data)
    
    def _generate_calendar_data(self) -> pd.DataFrame:
        """Generate working day data"""
        start_date = datetime.now().date()
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        calendar_data = []
        for date in dates:
            is_weekday = 1 if date.weekday() < 5 else 0
            holiday_type = "normal"  # Simplified implementation, no specific holidays
            
            calendar_data.append({
                'date': date,
                'is_weekday': is_weekday,
                'holiday_type': holiday_type
            })
        
        logger.info("Generated working day data: 365 days")
        return pd.DataFrame(calendar_data)
    
    def _generate_default_manager_config(self) -> pd.DataFrame:
        """Generate default Manager configuration"""
        managers = [
            {'manager_id': 'manager_1', 'location_x': 2.5, 'location_y': 3.2, 
             'coverage_area': 1.5, 'user_count': 6, 'district_type': 'residential'},
            {'manager_id': 'manager_2', 'location_x': 5.8, 'location_y': 7.1, 
             'coverage_area': 2.3, 'user_count': 10, 'district_type': 'mixed'},
            {'manager_id': 'manager_3', 'location_x': 8.2, 'location_y': 4.6, 
             'coverage_area': 1.8, 'user_count': 8, 'district_type': 'residential'},
            {'manager_id': 'manager_4', 'location_x': 11.5, 'location_y': 9.3, 
             'coverage_area': 3.1, 'user_count': 12, 'district_type': 'commercial'}
        ]
        
        logger.info("Generated default Manager configuration: 4 managers")
        return pd.DataFrame(managers)
    
    def _generate_default_user_config(self) -> pd.DataFrame:
        """Generate default user configuration"""
        manager_config = self._generate_default_manager_config()
        users = []
        
        for _, manager in manager_config.iterrows():
            manager_id = manager['manager_id']
            manager_x, manager_y = manager['location_x'], manager['location_y']
            user_count = manager['user_count']
            
            # Generate users around the Manager
            for i in range(user_count):
                user_id = f"user_{manager_id}_{i+1}"
                
                # User location is randomly distributed around the Manager
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, math.sqrt(manager['coverage_area'] / np.pi))
                user_x = manager_x + distance * math.cos(angle)
                user_y = manager_y + distance * math.sin(angle)
                
                # Random user type and preferences
                user_type = np.random.choice(['prosumer', 'consumer', 'producer'], 
                                           p=[0.4, 0.5, 0.1])
                
                # Generate normalized preferences
                prefs = np.random.dirichlet([1, 1, 1])  # Ensure sum to 1
                
                users.append({
                    'user_id': user_id,
                    'manager_id': manager_id,
                    'location_x': round(user_x, 2),
                    'location_y': round(user_y, 2),
                    'user_type': user_type,
                    'economic_pref': round(prefs[0], 3),
                    'comfort_pref': round(prefs[1], 3),
                    'environmental_pref': round(prefs[2], 3)
                })
        
        logger.info(f"Generated default user configuration: {len(users)} users")
        return pd.DataFrame(users)
    
    def _generate_default_device_config(self) -> pd.DataFrame:
        """Generate default device configuration"""
        user_config = self._generate_default_user_config()
        devices = []
        user_list = user_config['user_id'].tolist()
        
        
        battery_users = np.random.choice(user_list, size=24, replace=False)
        battery_users_set = set(battery_users)
        
        
        prosumer_producer_users = user_config[user_config['user_type'].isin(['prosumer', 'producer'])]['user_id'].tolist()
        if len(prosumer_producer_users) >= 8:
            pv_users = np.random.choice(prosumer_producer_users, size=8, replace=False)
        else:
            remaining_users = [u for u in user_list if u not in prosumer_producer_users]
            additional_users = np.random.choice(remaining_users, size=8-len(prosumer_producer_users), replace=False)
            pv_users = prosumer_producer_users + list(additional_users)
        pv_users_set = set(pv_users)
        
        
        ev_users = np.random.choice(user_list, size=14, replace=False)
        ev_users_set = set(ev_users)
        
        for _, user in user_config.iterrows():
            user_id = user['user_id']
            
            
            if user_id in pv_users_set:
                devices.append({
                    'device_id': f"pv_{user_id}",
                    'user_id': user_id,
                    'device_type': 'pv',
                    'capacity': 0.0,
                    'max_power': round(np.random.uniform(3, 8), 2),
                    'efficiency': round(np.random.uniform(0.15, 0.22), 3),
                    'initial_state': 0.0,
                    'param1': round(np.random.uniform(25.0, 35.0), 1),  # tilt_angle
                    'param2': round(np.random.uniform(160.0, 200.0), 1),  # azimuth_angle
                    'param3': round(np.random.uniform(15.0, 40.0), 1),  # area
                    'can_interrupt': 0,
                    'priority': 1
                })
            
            
            if user_id in battery_users_set:
                capacity = round(np.random.uniform(5, 15), 2)
                max_power = round(capacity * np.random.uniform(0.4, 0.6), 2)
                devices.append({
                    'device_id': f"battery_{user_id}",
                    'user_id': user_id,
                    'device_type': 'battery',
                    'capacity': capacity,
                    'max_power': max_power,
                    'efficiency': round(np.random.uniform(0.92, 0.98), 3),
                    'initial_state': round(np.random.uniform(0.3, 0.7), 3),
                    'param1': 0.1,  # soc_min
                    'param2': 0.9,  # soc_max
                    'param3': capacity * 1000,  # capacity_wh for compatibility
                    'can_interrupt': 1,
                    'priority': 3
                })
            
            
                devices.append({
                    'device_id': f"heatpump_{user_id}",
                    'user_id': user_id,
                    'device_type': 'heat_pump',
                    'capacity': 0.0,
                'max_power': round(np.random.uniform(2, 8), 2),
                'efficiency': round(np.random.uniform(3.0, 4.5), 2),  # COP
                'initial_state': round(np.random.uniform(19, 22), 1),
                    'param1': 18.0,  # temp_min
                    'param2': 26.0,  # temp_max
                'param3': round(np.random.uniform(0.05, 0.15), 3),  # heat_loss_coef
                'can_interrupt': 1,
                'priority': 4
                })
            
            if user_id in ev_users_set:
                capacity = round(np.random.uniform(40, 80), 2)
                max_power = round(np.random.uniform(3, 11), 2)
                devices.append({
                    'device_id': f"ev_{user_id}",
                    'user_id': user_id,
                    'device_type': 'ev',
                    'capacity': capacity,
                    'max_power': max_power,
                    'efficiency': round(np.random.uniform(0.85, 0.95), 3),
                    'initial_state': round(np.random.uniform(0.2, 0.8), 3),
                    'param1': 0.1,  # soc_min
                    'param2': 0.95,  # soc_max
                    'param3': round(np.random.uniform(18.0, 22.0), 1),  # departure_hour
                    'can_interrupt': 1,
                    'priority': 2
                })
            
            devices.append({
                'device_id': f"dishwasher_{user_id}",
                'user_id': user_id,
                'device_type': 'dishwasher',
                'capacity': round(np.random.uniform(2.5, 3.5), 2),  # Total energy demand (kWh)
                'max_power': round(np.random.uniform(1.8, 2.5), 2),  # Power (kW)
                'efficiency': round(np.random.uniform(0.85, 0.95), 3),
                'initial_state': 0.0,  # Initial state: not deployed
                'param1': round(np.random.uniform(3.0, 4.0), 2),  # Runtime (hours)
                'param2': round(np.random.uniform(0.5, 1.0), 2),  # Minimum start delay (hours)
                'param3': round(np.random.uniform(6.0, 8.0), 2),  # Maximum start delay (hours)
                'can_interrupt': 0,
                'priority': np.random.randint(2, 5)  # Priority 2-4
            })
        
        # Count device types
        device_counts = {}
        for device in devices:
            device_type = device['device_type']
            device_counts[device_type] = device_counts.get(device_type, 0) + 1
        
        logger.info(f"Generated default device configuration: Total {len(devices)} devices")
        logger.info(f"Device distribution: {device_counts}")
        logger.info("New configuration: Dishwasher 36 (100%), Battery 24 (67%), Heat Pump 36 (100%), EV 14 (39%), PV 8 (22%)")
        return pd.DataFrame(devices) 
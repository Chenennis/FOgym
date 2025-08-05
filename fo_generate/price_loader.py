import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PriceLoader:
    """price loader - read from price file, if not exist, predict"""
    
    def __init__(self, data_dir: str = "data"):
        """
        initialize price loader
        
        Args:
            data_dir: data directory path
        """
        self.data_dir = data_dir
        self.grid_price_file = os.path.join(data_dir, "grid_price.csv")
        self.price_data = None
        
        # try to load grid price data
        self._load_grid_price_data()
    
    def _load_grid_price_data(self):
        """load grid price data"""
        if os.path.exists(self.grid_price_file):
            try:
                self.price_data = pd.read_csv(self.grid_price_file)
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                logger.info(f"successfully load grid price data: {self.grid_price_file}")
                logger.info(f"price data range: {self.price_data['timestamp'].min()} to {self.price_data['timestamp'].max()}")
                
                # verify data format
                required_columns = ['timestamp', 'hour', 'day_type', 'price_usd_kwh']
                missing_columns = [col for col in required_columns if col not in self.price_data.columns]
                if missing_columns:
                    logger.warning(f"price data missing columns: {missing_columns}")
                    self.price_data = None
                else:
                    logger.info(f"price data columns verified: {list(self.price_data.columns)}")
                    
            except Exception as e:
                logger.error(f"failed to load grid price data: {e}")
                self.price_data = None
        else:
            logger.info(f"grid price file does not exist: {self.grid_price_file}, use price prediction")
            self.price_data = None
    
    def get_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:

        if self.price_data is not None:
            return self._get_grid_price_data(start_time, time_horizon)
        else:
            return self._generate_predicted_price_data(start_time, time_horizon)
    
    def _get_grid_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:
        """get price data from grid price data"""
        # ensure price_data is not None (this method is only called when price_data exists)
        assert self.price_data is not None, "price_data should not be None"
        
        timestamps = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        result_data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_type = 'weekday' if timestamp.weekday() < 5 else 'weekend'
            
            # find matching price data
            matching_prices = self.price_data[
                (self.price_data['hour'] == hour) & 
                (self.price_data['day_type'] == day_type)
            ]
            
            if not matching_prices.empty:
                # use latest matching data
                price_row = matching_prices.iloc[-1]
                result_data.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_type': day_type,
                    'price': price_row['price_usd_kwh'],
                    'price_dkk': price_row['price_dkk_kwh'] if 'price_dkk_kwh' in price_row else price_row['price_usd_kwh'] * 7.0,
                    'price_level': price_row['price_level'] if 'price_level' in price_row else 'unknown',
                    'source': 'grid_data'
                })
            else:
                # if no matching data, use predicted price
                predicted_price = self._predict_price_for_hour(hour, day_type)
                result_data.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'day_type': day_type,
                    'price': predicted_price,
                    'price_dkk': predicted_price * 7.0,
                    'price_level': self._get_price_level(predicted_price),
                    'source': 'predicted'
                })
        
        result_df = pd.DataFrame(result_data)
        # change to DEBUG level, avoid duplicate log output (cache mechanism has suitable log)
        logger.debug(f"get price data: {len(result_df)} records, time range {start_time} to {timestamps[-1]}")
        return result_df
    
    def _generate_predicted_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:
        """generate predicted price data (based on grid price pattern)"""
        logger.info("use price prediction model to generate data")
        
        timestamps = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        result_data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_type = 'weekday' if timestamp.weekday() < 5 else 'weekend'
            
            predicted_price = self._predict_price_for_hour(hour, day_type)
            
            result_data.append({
                'timestamp': timestamp,
                'hour': hour,
                'day_type': day_type,
                'price': predicted_price,
                'price_dkk': predicted_price * 7.0,
                'price_level': self._get_price_level(predicted_price),
                'source': 'predicted'
            })
        
        result_df = pd.DataFrame(result_data)
        logger.info(f"generate predicted price data: {len(result_df)} records")
        return result_df
    
    def _predict_price_for_hour(self, hour: int, day_type: str) -> float:

        base_price = 0.12  # base price dkk/kWh
        
        if day_type == 'weekday':
            # weekday price pattern
            if 0 <= hour <= 5:
                # 0:00-5:00 lower
                price_multiplier = np.random.uniform(0.7, 0.95)
            elif 6 <= hour <= 9:
                # 6:00-9:00 rise to higher
                if hour == 6:
                    price_multiplier = np.random.uniform(1.3, 1.4)
                elif hour in [7, 8]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:  # hour == 9
                    price_multiplier = np.random.uniform(1.7, 1.9)
            elif 10 <= hour <= 16:
                # 10:00-16:00 valley
                price_multiplier = np.random.uniform(1.0, 1.2)
            elif 17 <= hour <= 21:
                # 17:00-21:00 peak
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(2.1, 2.3)  
                else:
                    price_multiplier = np.random.uniform(1.9, 2.1)
            else:  # 22-23
                # 22:00-23:00 decrease
                price_multiplier = np.random.uniform(1.1, 1.4)
        else:
            # weekend price pattern (overall lower)
            if 0 <= hour <= 5:
                # 0:00-5:00 lower
                price_multiplier = np.random.uniform(0.6, 0.9)
            elif 6 <= hour <= 9:
                # 6:00-9:00 slow rise
                price_multiplier = np.random.uniform(1.0, 1.35)
            elif 10 <= hour <= 16:
                # 10:00-16:00 medium low
                price_multiplier = np.random.uniform(1.2, 1.4)
            elif 17 <= hour <= 21:
                # 17:00-21:00 peak (but lower than weekday)
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:
                    price_multiplier = np.random.uniform(1.6, 1.8)
            else:  # 22-23
                # 22:00-23:00 decrease
                price_multiplier = np.random.uniform(1.0, 1.3)
        
        # add random fluctuation
        noise = np.random.normal(0, 0.02)  # 2% random fluctuation
        predicted_price = base_price * price_multiplier * (1 + noise)
        
        # ensure price is in reasonable range
        return max(0.08, min(0.35, predicted_price))
    
    def _get_price_level(self, price: float) -> str:
        """determine price level based on price"""
        if price < 0.12:
            return 'low'
        elif price < 0.16:
            return 'medium'
        elif price < 0.20:
            return 'high'
        else:
            return 'peak'
    
    def get_price_forecast(self, start_time: datetime, time_horizon: int, 
                          confidence_level: float = 0.9) -> Dict:

        price_data = self.get_price_data(start_time, time_horizon)
        
        if self.price_data is not None:
            # uncertainty based on historical data
            uncertainty = 0.05  # 5% uncertainty
        else:
            # higher uncertainty for predicted data
            uncertainty = 0.15  # 15% uncertainty
        
        # calculate confidence interval
        alpha = 1 - confidence_level
        z_score = 1.96  # for 95% confidence interval
        
        forecast_result = {
            'timestamps': price_data['timestamp'].tolist(),
            'mean_prices': price_data['price'].tolist(),
            'lower_bound': (price_data['price'] * (1 - uncertainty * z_score)).tolist(),
            'upper_bound': (price_data['price'] * (1 + uncertainty * z_score)).tolist(),
            'uncertainty': uncertainty,
            'confidence_level': confidence_level,
            'data_source': 'grid_data' if self.price_data is not None else 'prediction'
        }
        
        return forecast_result
    
    def get_current_price(self, current_time: datetime) -> Dict:

        price_data = self.get_price_data(current_time, 1)
        
        if not price_data.empty:
            current_price_info = price_data.iloc[0]
            return {
                'price': current_price_info['price'],
                'price_dkk': current_price_info.get('price_dkk', current_price_info['price'] * 7.0),
                'price_level': current_price_info.get('price_level', 'unknown'),
                'hour': current_price_info['hour'],
                'day_type': current_price_info['day_type'],
                'source': current_price_info.get('source', 'unknown')
            }
        else:
            # backup price
            return {
                'price': 0.15,
                'price_dkk': 1.05,
                'price_level': 'medium',
                'hour': current_time.hour,
                'day_type': 'weekday' if current_time.weekday() < 5 else 'weekend',
                'source': 'default'
            }
    
    def is_peak_hour(self, current_time: datetime) -> bool:
        current_price_info = self.get_current_price(current_time)
        return current_price_info['price_level'] in ['high', 'peak']
    
    def get_cheapest_hours(self, start_time: datetime, time_horizon: int, 
                          num_hours: int = 1) -> List[Dict]:

        price_data = self.get_price_data(start_time, time_horizon)
        
        # sort by price
        sorted_data = price_data.sort_values('price').head(num_hours)
        
        result = []
        for _, row in sorted_data.iterrows():
            result.append({
                'timestamp': row['timestamp'],
                'hour': row['hour'],
                'price': row['price'],
                'price_level': row.get('price_level', 'unknown'),
                'savings_percent': (price_data['price'].max() - row['price']) / price_data['price'].max() * 100
            })
        
        return result 
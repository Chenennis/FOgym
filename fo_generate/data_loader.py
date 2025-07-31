"""
数据加载器 - 加载外部数据文件或生成默认数据

支持加载：
- 天气数据（丹麦特色）
- 电价数据（丹麦电力市场）
- 光伏预测数据
- 工作日数据
- Manager和User配置数据
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
    """数据加载器类"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_dir()
        # 初始化电价加载器
        self.price_loader = PriceLoader(data_dir)
        
        # 🔧 添加电价缓存机制
        self._cached_daily_prices = None  # 缓存一天的24小时电价数据
        self._cache_day_type = None  # 缓存的日期类型（weekday/weekend）
        self._cache_source = None  # 缓存数据来源
        logger.info("DataLoader初始化完成，电价缓存机制已启用")
    
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")
    
    def load_weather_data(self, filename: str = "weather_data.csv", 
                         start_time: datetime = None, 
                         hours: int = 168) -> pd.DataFrame:
        """加载天气数据，如果文件不存在则生成丹麦天气数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"成功加载天气数据: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载天气数据失败: {e}，使用默认生成")
        
        # 生成丹麦天气数据
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        return self._generate_danish_weather(start_time, hours)
    
    def load_price_data(self, filename: str = "price_data.csv",
                       start_time: datetime = None,
                       hours: int = 168) -> pd.DataFrame:
        """加载电价数据，优先从grid_price.csv读取丹麦电价数据，使用缓存避免重复读取"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # 确定日期类型
        day_type = 'weekday' if start_time.weekday() < 5 else 'weekend'
        
        # 🔧 检查缓存：如果已有相同日期类型的24小时数据，直接使用缓存
        if (self._cached_daily_prices is not None and 
            self._cache_day_type == day_type and 
            len(self._cached_daily_prices) == 24):
            
            logger.info(f"使用缓存的电价数据: {day_type}, 数据源: {self._cache_source}")
            return self._generate_price_data_from_cache(start_time, hours)
        
        # 🔧 首次加载或缓存失效：尝试从grid_price.csv加载并缓存
        try:
            grid_price_file = os.path.join(self.data_dir, "grid_price.csv")
            if os.path.exists(grid_price_file):
                # 加载并缓存一天的24小时电价数据
                self._load_and_cache_daily_prices(grid_price_file, day_type)
                logger.info(f"从grid_price.csv加载并缓存电价数据: {day_type}, 24小时数据已缓存")
                return self._generate_price_data_from_cache(start_time, hours)
        except Exception as e:
            logger.warning(f"从grid_price.csv加载电价数据失败: {e}")
        
        # 🔧 备选方案1：使用PriceLoader（保持兼容性）
        try:
            price_data = self.price_loader.get_price_data(start_time, hours)
            logger.info(f"使用PriceLoader获取电价数据: {len(price_data)}条记录")
            return price_data
        except Exception as e:
            logger.warning(f"使用PriceLoader加载电价数据失败: {e}，尝试加载传统文件")
        
        # 🔧 备选方案2：尝试加载传统price_data.csv文件
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"成功加载备选电价数据: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载备选电价数据失败: {e}，使用默认生成")
        
        # 🔧 最后备选：生成丹麦电价数据并缓存
        logger.info("使用生成的丹麦电价数据")
        generated_data = self._generate_and_cache_daily_prices(day_type)
        return self._generate_price_data_from_cache(start_time, hours)
    
    def _load_and_cache_daily_prices(self, grid_price_file: str, day_type: str):
        """从grid_price.csv中加载并缓存一天的24小时电价数据"""
        try:
            # 读取grid_price.csv
            grid_data = pd.read_csv(grid_price_file)
            grid_data['timestamp'] = pd.to_datetime(grid_data['timestamp'])
            
            # 筛选指定日期类型的数据
            day_type_data = grid_data[grid_data['day_type'] == day_type]
            
            if len(day_type_data) == 0:
                logger.warning(f"grid_price.csv中没有找到{day_type}的数据")
                raise ValueError(f"No {day_type} data found in grid_price.csv")
            
            # 提取24小时的数据（0-23小时）
            cached_prices = {}
            for hour in range(24):
                hour_data = day_type_data[day_type_data['hour'] == hour]
                if len(hour_data) > 0:
                    # 使用最新的匹配数据
                    price_row = hour_data.iloc[-1]
                    cached_prices[hour] = {
                        'price_usd_kwh': price_row['price_usd_kwh'],
                        'price_dkk_kwh': price_row['price_dkk_kwh'],
                        'price_level': price_row['price_level'],
                        'hour': hour,
                        'day_type': day_type
                    }
                else:
                    # 如果某个小时的数据缺失，使用预测价格
                    predicted_price = self._predict_price_for_hour(hour, day_type)
                    cached_prices[hour] = {
                        'price_usd_kwh': predicted_price,
                        'price_dkk_kwh': predicted_price * 7.0,
                        'price_level': self._get_price_level(predicted_price),
                        'hour': hour,
                        'day_type': day_type
                    }
            
            # 缓存数据
            self._cached_daily_prices = cached_prices
            self._cache_day_type = day_type
            self._cache_source = 'grid_data'
            
            logger.info(f"成功缓存{day_type}的24小时电价数据，数据完整性: {len(cached_prices)}/24")
            
        except Exception as e:
            logger.error(f"缓存电价数据失败: {e}")
            raise
    
    def _generate_and_cache_daily_prices(self, day_type: str):
        """生成并缓存一天的24小时电价数据"""
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
        
        # 缓存数据
        self._cached_daily_prices = cached_prices
        self._cache_day_type = day_type
        self._cache_source = 'predicted'
        
        logger.info(f"成功生成并缓存{day_type}的24小时电价数据")
    
    def _generate_price_data_from_cache(self, start_time: datetime, hours: int) -> pd.DataFrame:
        """从缓存的24小时数据生成指定时间范围的电价数据"""
        if self._cached_daily_prices is None:
            raise ValueError("电价缓存为空")
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        result_data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            
            # 从缓存中获取对应小时的价格数据
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
                # 理论上不应该发生，但加个保险
                logger.warning(f"缓存中缺少小时{hour}的数据，使用默认价格")
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
        logger.debug(f"从缓存生成电价数据: {len(result_df)}条记录，时间范围 {start_time} 到 {timestamps[-1]}")
        return result_df
    
    def _predict_price_for_hour(self, hour: int, day_type: str) -> float:
        """基于丹麦电价模式预测指定小时的电价（与PriceLoader保持一致）"""
        base_price = 0.12  # 基础电价 USD/kWh
        
        if day_type == 'weekday':
            # 工作日电价模式
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
            # 休息日电价模式
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
        
        # 添加随机波动
        noise = np.random.normal(0, 0.02)
        predicted_price = base_price * price_multiplier * (1 + noise)
        
        # 确保价格在合理范围内
        return max(0.08, min(0.35, predicted_price))
    
    def _get_price_level(self, price: float) -> str:
        """根据价格确定价格水平"""
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
        """加载光伏预测数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info(f"成功加载光伏预测数据: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载光伏预测数据失败: {e}，使用默认生成")
        
        # 生成光伏预测数据
        return self._generate_pv_forecast(start_time, hours)
    
    def load_calendar_data(self, filename: str = "calendar_data.csv") -> pd.DataFrame:
        """加载工作日数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                data['date'] = pd.to_datetime(data['date'])
                logger.info(f"成功加载工作日数据: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载工作日数据失败: {e}，使用默认生成")
        
        # 生成工作日数据
        return self._generate_calendar_data()
    
    def load_manager_config(self, filename: str = "manager_config.csv") -> pd.DataFrame:
        """加载Manager配置数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"成功加载Manager配置: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载Manager配置失败: {e}，使用默认配置")
        
        # 生成默认Manager配置
        return self._generate_default_manager_config()
    
    def load_user_config(self, filename: str = "user_config.csv") -> pd.DataFrame:
        """加载用户配置数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"成功加载用户配置: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载用户配置失败: {e}，使用默认配置")
        
        # 生成默认用户配置
        return self._generate_default_user_config()
    
    def load_device_config(self, filename: str = "device_config.csv") -> pd.DataFrame:
        """加载设备配置数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath)
                logger.info(f"成功加载设备配置: {filepath}")
                return data
            except Exception as e:
                logger.warning(f"加载设备配置失败: {e}，使用默认配置")
        
        # 生成默认设备配置
        return self._generate_default_device_config()
    
    def _generate_danish_weather(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """生成丹麦天气数据"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        weather_data = []
        
        for ts in timestamps:
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour
            
            # 丹麦季节性温度模型
            seasonal_temp = 8 + 12 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            daily_variation = 4 * math.sin(2 * math.pi * (hour - 6) / 24)
            temperature = seasonal_temp + daily_variation + np.random.normal(0, 2)
            
            # 太阳辐照度模型（考虑丹麦纬度）
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                seasonal_factor = max(0.1, math.sin(2 * math.pi * (day_of_year - 80) / 365))
                irradiance = 600 * solar_angle * seasonal_factor
                irradiance = max(0, irradiance + np.random.normal(0, 50))
            else:
                irradiance = 0
            
            # 丹麦风速模型
            wind_speed = 8 + 4 * math.sin(2 * math.pi * day_of_year / 365) + np.random.normal(0, 2)
            wind_speed = max(2, wind_speed)
            
            weather_data.append({
                'timestamp': ts,
                'temperature': round(temperature, 1),
                'solar_irradiance': round(irradiance, 1),
                'wind_speed': round(wind_speed, 1)
            })
        
        logger.info(f"生成丹麦天气数据: {hours}小时")
        return pd.DataFrame(weather_data)
    
    def _generate_danish_prices(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """生成丹麦电价数据"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        price_data = []
        
        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.weekday() >= 5
            
            # 丹麦电价模型（DKK/kWh）
            if 0 <= hour < 6:  # 夜间低价
                base_price = 0.8
            elif 6 <= hour < 9:  # 早高峰
                base_price = 2.2
            elif 9 <= hour < 16:  # 白天
                base_price = 1.5
            elif 16 <= hour < 20:  # 晚高峰
                base_price = 2.5
            else:  # 晚间
                base_price = 1.8
            
            # 周末价格调整
            if is_weekend:
                base_price *= 0.85
            
            # 添加随机波动
            price = base_price + np.random.normal(0, 0.2)
            price = max(0.3, price)
            
            price_data.append({
                'timestamp': ts,
                'price': round(price, 3),
                'price_type': 'spot'
            })
        
        logger.info(f"生成丹麦电价数据: {hours}小时")
        return pd.DataFrame(price_data)
    
    def _generate_pv_forecast(self, start_time: datetime = None, hours: int = 168) -> pd.DataFrame:
        """生成光伏预测数据"""
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
        pv_data = []
        
        for i, ts in enumerate(timestamps):
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour
            forecast_horizon = i + 1
            
            # 5kW系统的发电预测
            if 6 <= hour <= 18:
                solar_angle = math.sin(math.pi * (hour - 6) / 12)
                seasonal_factor = max(0.2, math.sin(2 * math.pi * (day_of_year - 80) / 365))
                forecast_power = 5.0 * solar_angle * seasonal_factor
                
                # 添加预测不确定性
                uncertainty = min(0.3, forecast_horizon * 0.02)
                forecast_power *= (1 + np.random.normal(0, uncertainty))
                forecast_power = max(0, forecast_power)
            else:
                forecast_power = 0
            
            # 置信度随预测时间递减
            confidence = max(0.6, 0.95 - forecast_horizon * 0.015)
            
            pv_data.append({
                'timestamp': ts,
                'forecast_power': round(forecast_power, 2),
                'confidence': round(confidence, 3),
                'forecast_horizon': forecast_horizon
            })
        
        logger.info(f"生成光伏预测数据: {hours}小时")
        return pd.DataFrame(pv_data)
    
    def _generate_calendar_data(self) -> pd.DataFrame:
        """生成工作日数据"""
        start_date = datetime.now().date()
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        calendar_data = []
        for date in dates:
            is_weekday = 1 if date.weekday() < 5 else 0
            holiday_type = "normal"  # 简化实现，不包含具体节假日
            
            calendar_data.append({
                'date': date,
                'is_weekday': is_weekday,
                'holiday_type': holiday_type
            })
        
        logger.info("生成工作日数据: 365天")
        return pd.DataFrame(calendar_data)
    
    def _generate_default_manager_config(self) -> pd.DataFrame:
        """生成默认Manager配置"""
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
        
        logger.info("生成默认Manager配置: 4个Manager")
        return pd.DataFrame(managers)
    
    def _generate_default_user_config(self) -> pd.DataFrame:
        """生成默认用户配置"""
        manager_config = self._generate_default_manager_config()
        users = []
        
        for _, manager in manager_config.iterrows():
            manager_id = manager['manager_id']
            manager_x, manager_y = manager['location_x'], manager['location_y']
            user_count = manager['user_count']
            
            # 在Manager周围生成用户
            for i in range(user_count):
                user_id = f"user_{manager_id}_{i+1}"
                
                # 用户位置在Manager周围随机分布
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, math.sqrt(manager['coverage_area'] / np.pi))
                user_x = manager_x + distance * math.cos(angle)
                user_y = manager_y + distance * math.sin(angle)
                
                # 随机用户类型和偏好
                user_type = np.random.choice(['prosumer', 'consumer', 'producer'], 
                                           p=[0.4, 0.5, 0.1])
                
                # 生成归一化的偏好
                prefs = np.random.dirichlet([1, 1, 1])  # 确保和为1
                
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
        
        logger.info(f"生成默认用户配置: {len(users)}个用户")
        return pd.DataFrame(users)
    
    def _generate_default_device_config(self) -> pd.DataFrame:
        """生成默认设备配置 - 更新：电池24个，热泵36个（每用户都有）"""
        user_config = self._generate_default_user_config()
        devices = []
        user_list = user_config['user_id'].tolist()
        
        # 为电池选择24个用户（大约2/3的用户）
        battery_users = np.random.choice(user_list, size=24, replace=False)
        battery_users_set = set(battery_users)
        
        # 为光伏选择8个用户（prosumer和producer优先）
        prosumer_producer_users = user_config[user_config['user_type'].isin(['prosumer', 'producer'])]['user_id'].tolist()
        if len(prosumer_producer_users) >= 8:
            pv_users = np.random.choice(prosumer_producer_users, size=8, replace=False)
        else:
            remaining_users = [u for u in user_list if u not in prosumer_producer_users]
            additional_users = np.random.choice(remaining_users, size=8-len(prosumer_producer_users), replace=False)
            pv_users = prosumer_producer_users + list(additional_users)
        pv_users_set = set(pv_users)
        
        # 为EV选择14个用户
        ev_users = np.random.choice(user_list, size=14, replace=False)
        ev_users_set = set(ev_users)
        
        for _, user in user_config.iterrows():
            user_id = user['user_id']
            
            # 光伏系统（8个用户）
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
            
            # 电池（24个用户）
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
            
            # 热泵（每个用户都有 - 36个）
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
            
            # 电动汽车（14个用户）
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
            
            # 洗碗机（100%部署率，每个用户都有 - 36个）
            devices.append({
                'device_id': f"dishwasher_{user_id}",
                'user_id': user_id,
                'device_type': 'dishwasher',
                'capacity': round(np.random.uniform(2.5, 3.5), 2),  # 总能量需求 (kWh)
                'max_power': round(np.random.uniform(1.8, 2.5), 2),  # 功率 (kW)
                'efficiency': round(np.random.uniform(0.85, 0.95), 3),
                'initial_state': 0.0,  # 初始状态：未部署
                'param1': round(np.random.uniform(3.0, 4.0), 2),  # 运行时长 (hours)
                'param2': round(np.random.uniform(0.5, 1.0), 2),  # 最小启动延迟 (hours)
                'param3': round(np.random.uniform(6.0, 8.0), 2),  # 最大启动延迟 (hours)
                'can_interrupt': 0,
                'priority': np.random.randint(2, 5)  # 优先级2-4
            })
        
        # 统计设备数量
        device_counts = {}
        for device in devices:
            device_type = device['device_type']
            device_counts[device_type] = device_counts.get(device_type, 0) + 1
        
        logger.info(f"生成默认设备配置: 总计{len(devices)}个设备")
        logger.info(f"设备分布: {device_counts}")
        logger.info("新配置: 洗碗机36个(100%), 电池24个(67%), 热泵36个(100%), EV14个(39%), 光伏8个(22%)")
        return pd.DataFrame(devices) 
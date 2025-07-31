import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PriceLoader:
    """电价加载器 - 优先从丹麦电价文件读取，不存在时进行预测"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化电价加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.grid_price_file = os.path.join(data_dir, "grid_price.csv")
        self.price_data = None
        
        # 尝试加载丹麦电价数据
        self._load_grid_price_data()
    
    def _load_grid_price_data(self):
        """加载丹麦电网电价数据"""
        if os.path.exists(self.grid_price_file):
            try:
                self.price_data = pd.read_csv(self.grid_price_file)
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                logger.info(f"成功加载丹麦电价数据: {self.grid_price_file}")
                logger.info(f"电价数据范围: {self.price_data['timestamp'].min()} 到 {self.price_data['timestamp'].max()}")
                
                # 验证数据格式
                required_columns = ['timestamp', 'hour', 'day_type', 'price_usd_kwh']
                missing_columns = [col for col in required_columns if col not in self.price_data.columns]
                if missing_columns:
                    logger.warning(f"电价数据缺少列: {missing_columns}")
                    self.price_data = None
                else:
                    logger.info(f"电价数据列验证通过: {list(self.price_data.columns)}")
                    
            except Exception as e:
                logger.error(f"加载丹麦电价数据失败: {e}")
                self.price_data = None
        else:
            logger.info(f"丹麦电价文件不存在: {self.grid_price_file}，将使用电价预测")
            self.price_data = None
    
    def get_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:
        """
        获取指定时间范围的电价数据
        
        Args:
            start_time: 开始时间
            time_horizon: 时间范围（小时）
            
        Returns:
            包含电价数据的DataFrame
        """
        if self.price_data is not None:
            return self._get_grid_price_data(start_time, time_horizon)
        else:
            return self._generate_predicted_price_data(start_time, time_horizon)
    
    def _get_grid_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:
        """从丹麦电价数据中获取指定时间范围的价格"""
        # 确保price_data不为None（此方法只在price_data存在时调用）
        assert self.price_data is not None, "price_data不应为None"
        
        timestamps = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        result_data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_type = 'weekday' if timestamp.weekday() < 5 else 'weekend'
            
            # 查找匹配的电价数据
            matching_prices = self.price_data[
                (self.price_data['hour'] == hour) & 
                (self.price_data['day_type'] == day_type)
            ]
            
            if not matching_prices.empty:
                # 使用最新的匹配数据
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
                # 如果没有匹配数据，使用预测价格
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
        # 改为DEBUG级别，避免重复日志输出（缓存机制已有合适的日志）
        logger.debug(f"获取电价数据: {len(result_df)}条记录，时间范围 {start_time} 到 {timestamps[-1]}")
        return result_df
    
    def _generate_predicted_price_data(self, start_time: datetime, time_horizon: int) -> pd.DataFrame:
        """生成预测的电价数据（基于丹麦电价模式）"""
        logger.info("使用电价预测模型生成数据")
        
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
        logger.info(f"生成预测电价数据: {len(result_df)}条记录")
        return result_df
    
    def _predict_price_for_hour(self, hour: int, day_type: str) -> float:
        """
        基于丹麦电价模式预测指定小时的电价
        
        Args:
            hour: 小时 (0-23)
            day_type: 日期类型 ('weekday' 或 'weekend')
            
        Returns:
            预测的电价 (USD/kWh)
        """
        base_price = 0.12  # 基础电价 USD/kWh
        
        if day_type == 'weekday':
            # 工作日电价模式
            if 0 <= hour <= 5:
                # 0:00-5:00 较低
                price_multiplier = np.random.uniform(0.7, 0.95)
            elif 6 <= hour <= 9:
                # 6:00-9:00 上升到较高
                if hour == 6:
                    price_multiplier = np.random.uniform(1.3, 1.4)
                elif hour in [7, 8]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:  # hour == 9
                    price_multiplier = np.random.uniform(1.7, 1.9)
            elif 10 <= hour <= 16:
                # 10:00-16:00 电价低谷
                price_multiplier = np.random.uniform(1.0, 1.2)
            elif 17 <= hour <= 21:
                # 17:00-21:00 高峰
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(2.1, 2.3)  # 最高峰
                else:
                    price_multiplier = np.random.uniform(1.9, 2.1)
            else:  # 22-23
                # 22:00-23:00 下降
                price_multiplier = np.random.uniform(1.1, 1.4)
        else:
            # 休息日电价模式（总体较低）
            if 0 <= hour <= 5:
                # 0:00-5:00 较低
                price_multiplier = np.random.uniform(0.6, 0.9)
            elif 6 <= hour <= 9:
                # 6:00-9:00 缓慢上升
                price_multiplier = np.random.uniform(1.0, 1.35)
            elif 10 <= hour <= 16:
                # 10:00-16:00 中等偏低
                price_multiplier = np.random.uniform(1.2, 1.4)
            elif 17 <= hour <= 21:
                # 17:00-21:00 高峰（但比工作日低）
                if hour in [18, 19]:
                    price_multiplier = np.random.uniform(1.8, 2.0)
                else:
                    price_multiplier = np.random.uniform(1.6, 1.8)
            else:  # 22-23
                # 22:00-23:00 下降
                price_multiplier = np.random.uniform(1.0, 1.3)
        
        # 添加随机波动
        noise = np.random.normal(0, 0.02)  # 2%的随机波动
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
    
    def get_price_forecast(self, start_time: datetime, time_horizon: int, 
                          confidence_level: float = 0.9) -> Dict:
        """
        获取电价预测（包含不确定性信息）
        
        Args:
            start_time: 开始时间
            time_horizon: 时间范围（小时）
            confidence_level: 置信水平
            
        Returns:
            包含预测价格和不确定性信息的字典
        """
        price_data = self.get_price_data(start_time, time_horizon)
        
        if self.price_data is not None:
            # 基于历史数据的不确定性
            uncertainty = 0.05  # 5%的不确定性
        else:
            # 预测数据的不确定性更高
            uncertainty = 0.15  # 15%的不确定性
        
        # 计算置信区间
        alpha = 1 - confidence_level
        z_score = 1.96  # 对于95%置信区间
        
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
        """
        获取当前时间的电价信息
        
        Args:
            current_time: 当前时间
            
        Returns:
            当前电价信息字典
        """
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
            # 备用价格
            return {
                'price': 0.15,
                'price_dkk': 1.05,
                'price_level': 'medium',
                'hour': current_time.hour,
                'day_type': 'weekday' if current_time.weekday() < 5 else 'weekend',
                'source': 'default'
            }
    
    def is_peak_hour(self, current_time: datetime) -> bool:
        """判断当前时间是否为电价高峰期"""
        current_price_info = self.get_current_price(current_time)
        return current_price_info['price_level'] in ['high', 'peak']
    
    def get_cheapest_hours(self, start_time: datetime, time_horizon: int, 
                          num_hours: int = 1) -> List[Dict]:
        """
        获取指定时间范围内最便宜的几个小时
        
        Args:
            start_time: 开始时间
            time_horizon: 时间范围（小时）
            num_hours: 需要的小时数
            
        Returns:
            最便宜小时的信息列表
        """
        price_data = self.get_price_data(start_time, time_horizon)
        
        # 按价格排序
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
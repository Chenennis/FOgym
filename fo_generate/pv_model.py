from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class PVParameters:
    """光伏发电参数"""
    pv_id: str           # 光伏ID
    max_power: float     # 最大输出功率
    efficiency: float    # 效率
    area: float          # 面积
    location: str        # 位置
    tilt_angle: float    # 倾斜角度
    azimuth_angle: float # 方位角
    weather_dependent: bool  # 是否依赖天气
    forecast_accuracy: float = 0.8  # 预测准确率，默认为80%
    
# 注意：PVStorageParameters类已移除，由home battery提供存储功能

class PVModel:
    """光伏发电模型类"""
    def __init__(self, params: PVParameters):
        """
        初始化光伏发电模型
        
        Args:
            params: 光伏参数
        """
        self.params = params
        
        # 存储预测数据
        self.forecast_data = None
        # 存储实际产生的功率历史
        self.power_history = []
        # 跟踪PV稳定性
        self.stability_violations = 0
            
    def set_forecast_data(self, forecast_data: List[float]):
        """设置未来12小时的发电预测数据"""
        self.forecast_data = forecast_data
        
    def predict_generation(self, 
                           time: datetime, 
                           weather_data: Optional[Dict] = None,
                           duration: float = 1.0,
                           use_forecast: bool = False) -> float:
        """
        预测光伏发电量
        
        Args:
            time: 时间点
            weather_data: 天气数据
            duration: 持续时间(小时)
            use_forecast: 是否使用预测数据而非模型计算
        """
        # 如果有预测数据且需要使用预测数据
        if use_forecast and self.forecast_data is not None:
            # 计算当前时间点对应的预测数据索引
            hour_diff = int((time - datetime.now()).total_seconds() / 3600)
            if 0 <= hour_diff < len(self.forecast_data):
                # 添加预测误差 (±20%)
                accuracy = self.params.forecast_accuracy
                error = np.random.uniform(1 - (1 - accuracy), 1 + (1 - accuracy))
                return self.forecast_data[hour_diff] * error * duration
        
        # 如果没有预测数据或不使用预测数据，使用模型计算
        # 如果没有提供天气数据，使用简化模型
        if not weather_data or not self.params.weather_dependent:
            # 根据时间估计太阳辐射强度，这里使用简化的日出日落模型
            hour = time.hour + time.minute / 60.0
            
            # 白天生产，夜间不生产
            if 6 <= hour <= 18:
                # 简单的钟形曲线模拟日出到日落的太阳辐射变化
                solar_intensity = np.sin(np.pi * (hour - 6) / 12)
                power = self.params.max_power * solar_intensity * self.params.efficiency
            else:
                power = 0.0
        else:
            # 使用天气数据进行更精确的预测
            solar_intensity = weather_data.get('solar_radiation', 0)
            cloud_coverage = weather_data.get('cloud_coverage', 0)
            temperature = weather_data.get('temperature', 25)
            
            # 考虑云覆盖的影响
            solar_intensity *= (1 - 0.7 * cloud_coverage)
            
            # 考虑温度对效率的影响（温度每上升1℃，效率下降约0.4%）
            temp_efficiency = self.params.efficiency * (1 - 0.004 * max(0, temperature - 25))
            
            # 考虑角度因素
            angle_factor = np.cos(np.radians(self.params.tilt_angle))
            
            power = self.params.area * solar_intensity * temp_efficiency * angle_factor
            power = min(power, self.params.max_power)
        
        # 添加随机波动 (±10%)
        power *= np.random.uniform(0.9, 1.1)
        
        # 计算持续时间内的总发电量
        energy = power * duration
        
        return energy
        
    def get_available_power(self, time: datetime, weather_data: Optional[Dict] = None) -> float:
        """
        获取可用功率
        
        Args:
            time: 时间点
            weather_data: 天气数据
            
        Returns:
            当前时间点可用的发电功率
        """
        # 使用预测数据或者模型计算基础发电量
        base_generation = self.predict_generation(time, weather_data, use_forecast=True)
        
        # PV模型现在只能提供发电功率，不再有储能能力
        return base_generation
    
    def calculate_stability_metrics(self, forecast_window: int = 12) -> Dict:
        """
        计算PV稳定性指标
        
        Args:
            forecast_window: 预测窗口大小（小时）
            
        Returns:
            稳定性指标字典
        """
        if len(self.power_history) < 2 or self.forecast_data is None:
            return {
                "stability_score": 1.0,
                "forecast_deviation": 0.0,
                "storage_adequacy": 0.0  # 不再有存储能力
            }
            
        # 计算历史功率波动
        power_std = np.std(self.power_history[-forecast_window:]) if len(self.power_history) >= forecast_window else np.std(self.power_history)
        power_mean = np.mean(self.power_history[-forecast_window:]) if len(self.power_history) >= forecast_window else np.mean(self.power_history)
        power_volatility = power_std / (power_mean + 1e-6)  # 避免除以零
        
        # 计算预测偏差
        forecast_horizon = min(forecast_window, len(self.forecast_data))
        actual = self.power_history[-forecast_horizon:] if len(self.power_history) >= forecast_horizon else self.power_history
        forecast = self.forecast_data[:len(actual)]
        forecast_deviation = np.mean(np.abs(np.array(actual) - np.array(forecast[:len(actual)])) / (np.array(forecast[:len(actual)]) + 1e-6))
        
        # 由于没有存储系统，不再计算存储充足度
        storage_adequacy = 0.0
            
        # 综合稳定性评分 - 调整权重
        stability_score = 1.0 - (0.6 * power_volatility + 0.4 * forecast_deviation)
        stability_score = max(0, min(1, stability_score))
        
        return {
            "stability_score": stability_score,
            "power_volatility": power_volatility,
            "forecast_deviation": forecast_deviation,
            "storage_adequacy": storage_adequacy
        }
        
    def generate_dfo(self, start_time=None, time_horizon: int = None) -> DFOSystem:
        """
        生成DFO系统
        
        Args:
            start_time: 可选，起始时间，如果为None，使用当前时间
            time_horizon: 时间范围，如果为None且第一个参数为整数，则使用第一个参数作为time_horizon
            
        Returns:
            DFO系统对象
        """
        # 兼容旧的调用方式，如果第一个参数是整数且第二个参数为None
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # 如果start_time为None，使用当前时间
        current_time = start_time if start_time is not None else datetime.now()
        
        # 确保time_horizon有值
        if time_horizon is None:
            time_horizon = 12  # 默认值
        
        dfo = DFOSystem(time_horizon)
        
        # 确保有预测数据
        if self.forecast_data is None and time_horizon > 0:
            # 如果没有预测数据，生成模拟预测数据
            self.forecast_data = []
            for t in range(min(12, time_horizon)):
                forecast_time = current_time + pd.Timedelta(hours=t)
                self.forecast_data.append(self.predict_generation(forecast_time))
        
        for t in range(time_horizon):
            # 预测当前时间的发电量
            forecast_time = current_time + pd.Timedelta(hours=t)
            
            # 计算能量边界 - 现在只返回发电量，最小功率为0
            energy_max = self.predict_generation(forecast_time)
            energy_min = 0  # PV只能产生能量，不消耗能量
            
            # 创建时间片
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=[]  # 不再有SOC约束
            )
            dfo.add_slice(slice)
            
            # 模拟实际发电量（加入随机波动）
            actual_generation = self.predict_generation(forecast_time, use_forecast=False)
            self.power_history.append(actual_generation)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, pv_id: str = None) -> 'PVModel':
        """
        从CSV文件创建光伏模型
        
        Args:
            params_file: 参数文件路径
            pv_id: 光伏ID，如果为None则使用文件中的第一个
            
        Returns:
            PVModel对象
        """
        # 读取参数文件
        params_df = pd.read_csv(params_file, comment='#')
        
        # 如果指定了pv_id，查找对应数据；否则使用第一行
        if pv_id:
            pv_data = params_df[params_df['pv_id'] == pv_id]
            if pv_data.empty:
                raise ValueError(f"PV ID {pv_id} not found in {params_file}")
            pv_data = pv_data.iloc[0]
        else:
            pv_data = params_df.iloc[0]
            pv_id = pv_data['pv_id']
        
        # 创建参数对象
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
        """从CSV文件创建光伏模型并加载预测数据"""
        model = cls.from_csv(params_file, pv_id)
        
        # 如果提供了预测文件，载入预测数据
        if forecast_file and os.path.exists(forecast_file):
            forecast_df = pd.read_csv(forecast_file)
            if pv_id in forecast_df.columns:
                # 假设第一列是时间，后面是对应各PV ID的预测值
                model.forecast_data = forecast_df[pv_id].tolist()
                
        return model
        
    @classmethod
    def get_all_pv_ids(cls, params_file: str) -> List[str]:
        """获取CSV文件中所有的光伏ID"""
        df = pd.read_csv(params_file, comment='#')
        return df['pv_id'].tolist() 
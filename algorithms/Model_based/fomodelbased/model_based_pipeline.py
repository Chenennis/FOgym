"""
基于物理模型的FlexOffer Pipeline

提供纯粹基于物理模型的FlexOffer生成、聚合、交易和分解流程，
不使用强化学习概念，而是采用传统的模型预测控制方法。
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import sys
import time # Added for random_factor in bidding

# 处理导入方式
try:
    # 尝试作为包的一部分导入
    from .config import PipelineConfig, ModelBasedConfig
    from .model_based_controller import ModelBasedController, DeviceModel, BatteryModel, HeatPumpModel
except (ImportError, SystemError):
    # 直接运行脚本时的导入方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import PipelineConfig, ModelBasedConfig
    from model_based_controller import ModelBasedController, DeviceModel, BatteryModel, HeatPumpModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelBasedPipeline')


class FlexOffer:
    """FlexOffer类，表示一个灵活报价"""
    def __init__(
        self,
        device_id: str,
        device_type: str,
        energy_profile: List[float],
        time_flexibility: int = 0,
        manager_id: str = None
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.energy_profile = energy_profile
        self.time_flexibility = time_flexibility
        self.manager_id = manager_id
        self.time_horizon = len(energy_profile)
    
    @classmethod
    def from_dict(cls, fo_dict: Dict) -> "FlexOffer":
        """从字典创建FlexOffer"""
        return cls(
            device_id=fo_dict.get('device_id'),
            device_type=fo_dict.get('device_type'),
            energy_profile=fo_dict.get('energy_profile', [0.0]),
            time_flexibility=fo_dict.get('time_flexibility', 0),
            manager_id=fo_dict.get('manager_id')
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'energy_profile': self.energy_profile,
            'time_flexibility': self.time_flexibility,
            'manager_id': self.manager_id,
            'time_horizon': self.time_horizon
        }


class Manager:
    """Manager类，管理多个设备"""
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.devices = {}  # device_id -> device_config
    
    def add_device(self, device_id: str, device_config: Dict):
        """添加设备"""
        self.devices[device_id] = device_config
    
    def get_device_ids(self) -> List[str]:
        """获取所有设备ID"""
        return list(self.devices.keys()) 


class ModelBasedPipeline:
    """基于物理模型的FlexOffer Pipeline"""
    
    def __init__(self, config: PipelineConfig):
        """初始化Pipeline"""
        self.config = config
        self.time_horizon = config.time_horizon
        self.time_step = config.time_step
        self.aggregation_method = config.aggregation_method
        self.trading_method = config.trading_method
        self.disaggregation_method = config.disaggregation_method
        
        # 设置随机种子（如果提供）
        if config.seed is not None:
            np.random.seed(config.seed)
            import random
            random.seed(config.seed)
            logger.info(f"已设置随机种子: {config.seed}")
        
        # 创建结果目录
        os.makedirs(config.results_dir, exist_ok=True)
        
        # 生成实验ID
        self.experiment_id = self._generate_experiment_id()
        
        # 加载设备配置和价格数据
        self._load_device_config()
        self._load_price_data()
        
        # 初始化Manager和模型控制器
        self._setup_managers()
        self._setup_model_controllers()
        
        # 初始化结果存储
        self.results = {
            'manager_rewards': {},
            'timestep_details': [],
            'total_rewards': [],
            'aggregated_fo_count': [],  # 每个时间步的聚合FO数量
            'traded_fo_count': [],      # 每个时间步的交易FO数量
            'traded_fo_value': [],      # 每个时间步的交易FO总价值
            'disaggregate_count': []    # 每个时间步的分解FO数量
        }
        
        logger.info(f"ModelBasedPipeline初始化完成，实验ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """生成实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        managers_count = self.config.num_managers
        users_count = sum(self.config.users_per_manager)
        
        # 如果设置了种子，将其包含在ID中
        seed_str = f"_seed{self.config.seed}" if self.config.seed is not None else ""
        
        return f"MODELBASED_m{managers_count}_u{users_count}{seed_str}_{timestamp}" 

    def _load_device_config(self):
        """加载设备配置"""
        try:
            self.device_config = pd.read_csv(self.config.device_config_file)
            logger.info(f"加载了 {len(self.device_config)} 个设备配置")
        except Exception as e:
            logger.warning(f"无法加载设备配置: {e}，使用默认设备配置")
            # 创建默认设备配置
            self.device_config = self._create_default_device_config()
    
    def _create_default_device_config(self) -> pd.DataFrame:
        """创建默认设备配置"""
        # 创建默认设备配置
        device_data = []
        
        # 总用户数
        total_users = sum(self.config.users_per_manager)
        
        # 为每个用户创建设备
        for user_idx in range(total_users):
            # 电池设备
            device_data.append({
                'device_id': f"battery_{user_idx}_0",
                'user_id': f"user_{user_idx}",
                'device_type': 'BATTERY',
                'capacity_kwh': 10.0,
                'initial_soc': 0.5,
                'soc_min': 0.1,
                'soc_max': 0.9,
                'p_min': -3.0,
                'p_max': 3.0,
                'efficiency': 0.95
            })
            
            # 热泵设备
            device_data.append({
                'device_id': f"heat_pump_{user_idx}_0",
                'user_id': f"user_{user_idx}",
                'device_type': 'HEAT_PUMP',
                'initial_temp': 20.0,
                'temp_min': 18.0,
                'temp_max': 22.0,
                'target_temp': 21.0,
                'max_power': 2.0
            })
            
            # 电动车设备 (每隔一个用户添加)
            if user_idx % 2 == 0:
                device_data.append({
                    'device_id': f"ev_{user_idx}_0",
                    'user_id': f"user_{user_idx}",
                    'device_type': 'EV',
                    'capacity_kwh': 40.0,
                    'initial_soc': 0.3,
                    'soc_min': 0.1,
                    'soc_max': 0.9,
                    'p_min': 0.0,
                    'p_max': 7.0,
                    'efficiency': 0.9
                })
        
        return pd.DataFrame(device_data)
    
    def _load_price_data(self):
        """加载电价数据"""
        try:
            self.price_data = pd.read_csv(self.config.price_data_file)
            logger.info(f"加载了 {len(self.price_data)} 行电价数据")
        except Exception as e:
            logger.warning(f"无法加载电价数据: {e}，使用默认电价")
            # 创建默认电价数据
            self.price_data = self._create_default_price_data()
    
    def _create_default_price_data(self) -> pd.DataFrame:
        """创建默认电价数据"""
        # 创建24小时电价数据
        hours = list(range(24))
        
        # 创建波动电价 - 白天高峰，夜间低谷
        prices = []
        for hour in hours:
            if 0 <= hour < 6:  # 深夜
                prices.append(0.05)  # 低谷电价
            elif 6 <= hour < 9:  # 早高峰
                prices.append(0.15)  # 高峰电价
            elif 9 <= hour < 17:  # 白天
                prices.append(0.12)  # 平峰电价
            elif 17 <= hour < 21:  # 晚高峰
                prices.append(0.18)  # 高峰电价
            else:  # 晚上
                prices.append(0.08)  # 低谷电价
        
        return pd.DataFrame({
            'hour': hours,
            'price': prices
        }) 

    def _setup_managers(self):
        """设置Manager"""
        self.managers = {}
        
        # 创建Manager并分配设备
        for manager_idx in range(self.config.num_managers):
            manager_id = f"manager_{manager_idx+1}"
            self.managers[manager_id] = Manager(manager_id)
            
            # 根据实际数据格式分配设备
            # 设备配置中用户ID格式为 "user_manager_X_Y"
            manager_filter = f"user_manager_{manager_idx+1}"
            manager_devices = self.device_config[self.device_config['user_id'].str.startswith(manager_filter)]
            
            # 记录该manager下的用户
            unique_users = manager_devices['user_id'].unique()
            
            # 添加每个设备到Manager
            for _, device_config in manager_devices.iterrows():
                device_id = device_config['device_id']
                self.managers[manager_id].add_device(device_id, device_config.to_dict())
            
            logger.info(f"创建Manager {manager_id}，管理 {len(self.managers[manager_id].devices)} 个设备，共 {len(unique_users)} 个用户")
        
        logger.info(f"共创建 {len(self.managers)} 个Manager")
    
    def _setup_model_controllers(self):
        """设置模型控制器"""
        self.model_controllers = {}
        
        # 为每个Manager创建一个控制器
        for manager_id, manager in self.managers.items():
            # 创建控制器
            controller = ModelBasedController(
                manager_id=manager_id,
                time_horizon=self.time_horizon,
                time_step=self.time_step,
                config=self.config.model_config
            )
            
            # 为控制器添加设备模型
            for device_id, device_config in manager.devices.items():
                device_type = str(device_config['device_type'])
                # 转换设备参数格式
                device_params = self._convert_device_config(device_config)
                # 添加到控制器
                controller.add_device_model(device_id, device_type, device_params)
            
            self.model_controllers[manager_id] = controller
            logger.info(f"为Manager {manager_id}创建了模型控制器，配置了 {len(manager.devices)} 个设备")
    
    def _convert_device_config(self, device_config: Dict) -> Dict:
        """将设备配置转换为设备模型参数"""
        device_params = {}
        device_type = str(device_config['device_type'])
        
        if 'BATTERY' in device_type.upper():
            device_params = {
                'capacity': device_config.get('capacity_kwh', 10.0),
                'initial_soc': device_config.get('initial_soc', 0.5),
                'min_soc': device_config.get('soc_min', 0.1),
                'max_soc': device_config.get('soc_max', 0.9),
                'p_min': device_config.get('p_min', -3.0),
                'p_max': device_config.get('p_max', 3.0),
                'efficiency': device_config.get('efficiency', 0.95),
                'initial_charge': device_config.get('initial_soc', 0.5) * device_config.get('capacity_kwh', 10.0)
            }
        elif 'HEAT' in device_type.upper() or 'PUMP' in device_type.upper():
            device_params = {
                'initial_temp': device_config.get('initial_temp', 20.0),
                'min_temp': device_config.get('temp_min', 18.0),
                'max_temp': device_config.get('temp_max', 22.0),
                'target_temp': device_config.get('target_temp', 21.0),
                'max_power': device_config.get('max_power', 2.0),
                'outdoor_temp': 5.0,  # 默认室外温度
                'thermal_mass': 5000.0,  # 默认热质量
                'heat_transfer_coeff': 100.0  # 默认传热系数
            }
        elif 'EV' in device_type.upper():
            device_params = {
                'capacity': device_config.get('capacity_kwh', 40.0),
                'initial_soc': device_config.get('initial_soc', 0.3),
                'min_soc': device_config.get('soc_min', 0.1),
                'max_soc': device_config.get('soc_max', 0.9),
                'p_min': device_config.get('p_min', 0.0),
                'p_max': device_config.get('p_max', 7.0),
                'efficiency': device_config.get('efficiency', 0.9),
                'initial_charge': device_config.get('initial_soc', 0.3) * device_config.get('capacity_kwh', 40.0)
            }
        else:
            # 对于其他设备类型，保留原始配置
            device_params = device_config.copy()
        
        return device_params 

    def generate_flexoffers(self, timestep: int) -> Dict[str, Dict[str, Any]]:
        """生成FlexOffer"""
        fo_systems = {}  # manager_id -> {device_id: fo_object}
        
        # 获取当前时间段的价格
        prices = self._get_prices_for_horizon(timestep)
        
        # 每个Manager生成FlexOffer
        for manager_id, manager in self.managers.items():
            fo_systems[manager_id] = {}
            
            if manager_id in self.model_controllers:
                controller = self.model_controllers[manager_id]
                
                # 使用基于模型的控制器生成FlexOffers
                fo_dict = controller.generate_flex_offers(prices)
                
                # 转换为FlexOffer对象
                for device_id, fo_data in fo_dict.items():
                    # 检查设备是否在manager的设备列表中
                    if device_id in manager.devices:
                        device_config = manager.devices[device_id]
                        device_type = str(device_config['device_type'])
                        
                        # 从控制器中获取能量轮廓和时间灵活性
                        energy_profile = fo_data.get('energy_profile', [0.0] * self.time_horizon)
                        time_flexibility = fo_data.get('time_flexibility', 1)
                        
                        # 创建FlexOffer对象
                        fo = FlexOffer(
                            device_id=device_id,
                            device_type=device_type,
                            energy_profile=energy_profile,
                            time_flexibility=time_flexibility,
                            manager_id=manager_id
                        )
                        
                        fo_systems[manager_id][device_id] = fo
        
        total_fo_count = sum(len(devices) for devices in fo_systems.values())
        logger.info(f"为时间步 {timestep} 生成了 {total_fo_count} 个FlexOffer")
        
        return fo_systems
    
    def _get_prices_for_horizon(self, start_timestep):
        """获取从start_timestep开始的time_horizon小时的电价"""
        prices = []
        
        for t in range(self.time_horizon):
            hour = (start_timestep + t) % 24
            # 使用正确的价格列名 'price_usd_kwh'
            hour_price = self.price_data[self.price_data['hour'] == hour]['price_usd_kwh'].values
            
            if len(hour_price) > 0:
                prices.append(hour_price[0])
            else:
                # 默认电价
                if 0 <= hour < 6:
                    prices.append(0.05)  # 夜间低谷
                elif 17 <= hour < 21:
                    prices.append(0.18)  # 晚高峰
                else:
                    prices.append(0.12)  # 其他时段
        
        return prices
    
    def aggregate_flexoffers(self, fo_systems, timestep):
        """聚合FlexOffer"""
        aggregated_results = {}
        
        # 对每个Manager的FlexOffers进行聚合
        for manager_id, devices in fo_systems.items():
            if not devices:
                continue
                
            # 收集该Manager的所有FO
            flexoffers = list(devices.values())
            
            # 根据聚合方法选择聚合算法
            if self.aggregation_method == "LP":
                # 线性规划聚合方法
                aggregated_fos = self._aggregate_lp(flexoffers, manager_id)
            else:
                # 动态规划聚合方法
                aggregated_fos = self._aggregate_dp(flexoffers, manager_id)
            
            # 保存聚合结果
            aggregated_results[manager_id] = {
                'aggregated_fos': aggregated_fos,  # 注意：现在是列表而不是单个对象
                'original_fos': flexoffers,
                'timestep': timestep
            }
        
        # 计算聚合FO总数量
        total_aggregated_fos = sum(len(result.get('aggregated_fos', [])) for result in aggregated_results.values())
        logger.info(f"时间步 {timestep} 聚合完成: 总计 {total_aggregated_fos} 个聚合FO")
        return aggregated_results 

    def _aggregate_lp(self, flexoffers, manager_id):
        """使用线性规划进行聚合 (Longest Profile)
        
        根据FlexOffer_Pipeline_Structure.md:
        1. 找出具有最大profile size的FlexOffers
        2. 选择具有最高时间灵活性的FlexOffer作为初始FlexOffer
        3. 添加所有其他FlexOffers到处理集
        4. 迭代执行二元聚合，添加能改善RMSE和CV的FlexOffers
        
        修改：添加能量上限(100KWh)限制，如果超过则创建新的聚合FO
        """
        # 如果没有FO，返回空列表
        if not flexoffers:
            return []
        
        # 能量上限(KWh)
        ENERGY_LIMIT = 100.0
        
        # 计算每个FlexOffer的profile size (非零能量的时间片数量)
        profile_sizes = {}
        for i, fo in enumerate(flexoffers):
            # 计算非零能量的时间片数量
            non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
            profile_sizes[i] = non_zero_count
        
        # 复制一份flexoffers用于处理
        remaining_fos = list(flexoffers)
        
        # 存储最终的聚合结果
        aggregated_fos = []
        
        # 处理剩余的FlexOffers，直到处理完毕
        while remaining_fos:
            # 找出具有最大profile size的FlexOffers
            profile_sizes = {}
            for i, fo in enumerate(remaining_fos):
                non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
                profile_sizes[i] = non_zero_count
                
            max_size = max(profile_sizes.values(), default=0)
            if max_size == 0:
                break  # 没有有效的FO了
                
            max_size_fos = [i for i, size in profile_sizes.items() if size == max_size]
            
            # 从最大profile size的FlexOffers中选择时间灵活性最高的作为初始FlexOffer
            initial_fo_idx = max(max_size_fos, key=lambda i: remaining_fos[i].time_flexibility)
            initial_fo = remaining_fos[initial_fo_idx]
            
            # 计算聚合的能量轮廓
            time_horizon = initial_fo.time_horizon
            agg_profile = np.array(initial_fo.energy_profile)
            
            # 计算初始的RMSE和CV
            rmse = 0.0  # 初始RMSE为0
            cv = np.std(agg_profile) / (np.mean(abs(agg_profile)) + 1e-10)  # 变异系数
            
            # 从remaining_fos中移除已选择的FO
            selected_fos = [initial_fo]
            remaining_fos.pop(initial_fo_idx)
            
            # 计算当前能量总量
            current_energy = sum(abs(e) for e in agg_profile)
            
            # 迭代添加其他FlexOffers，直到达到能量上限或处理完所有FO
            i = 0
            while i < len(remaining_fos):
                fo = remaining_fos[i]
                
                # 计算当前FO的能量总量
                fo_energy = sum(abs(e) for e in fo.energy_profile)
                
                # 检查是否超过能量上限
                if current_energy + fo_energy > ENERGY_LIMIT:
                    i += 1  # 尝试下一个FO
                    continue
                
                # 计算二元聚合
                temp_profile = agg_profile + np.array(fo.energy_profile)
                
                # 计算新的RMSE和CV
                mean_profile = np.mean(abs(temp_profile))
                new_cv = np.std(temp_profile) / (mean_profile + 1e-10)
                
                # 计算与原始FOs的均方误差
                new_rmse = 0.0
                for orig_fo in selected_fos + [fo]:
                    # 根据能量占比计算目标轮廓
                    orig_weight = np.sum(abs(np.array(orig_fo.energy_profile))) / (np.sum(abs(temp_profile)) + 1e-10)
                    target_profile = temp_profile * orig_weight
                    
                    # 计算与原始轮廓的均方误差
                    error = np.mean((target_profile - np.array(orig_fo.energy_profile)) ** 2)
                    new_rmse += error
                
                # 如果新的聚合改善了RMSE和CV，则接受它
                if new_rmse <= rmse or new_cv < cv:
                    agg_profile = temp_profile
                    rmse = new_rmse
                    cv = new_cv
                    selected_fos.append(fo)
                    current_energy += fo_energy
                    remaining_fos.pop(i)  # 移除已添加的FO
                else:
                    i += 1  # 尝试下一个FO
            
            # 计算时间灵活性 - 使用参与聚合的FOs的加权平均值
            total_energy = sum(abs(np.sum(fo.energy_profile)) for fo in selected_fos)
            if total_energy > 0:
                time_flexibility = sum(fo.time_flexibility * abs(np.sum(fo.energy_profile)) 
                                    for fo in selected_fos) / total_energy
            else:
                time_flexibility = 0
            
            # 创建聚合FlexOffer
            aggregated_fo = FlexOffer(
                device_id=f"aggregated_{manager_id}_{len(aggregated_fos)}",
                device_type="AGGREGATED",
                energy_profile=agg_profile.tolist(),
                time_flexibility=int(time_flexibility),
                manager_id=manager_id
            )
            
            # 添加到聚合结果列表
            aggregated_fos.append(aggregated_fo)
            
            logger.info(f"创建聚合FO: {aggregated_fo.device_id}，包含 {len(selected_fos)} 个FOs，总能量: {current_energy:.2f}kWh")
        
        logger.info(f"Manager {manager_id}: 创建了 {len(aggregated_fos)} 个聚合FO")
        
        # 如果没有聚合结果，返回空列表
        if not aggregated_fos:
            return []
            
        return aggregated_fos
    
    def _aggregate_dp(self, flexoffers, manager_id):
        """使用动态轮廓进行聚合 (Dynamic Profile)
        
        根据FlexOffer_Pipeline_Structure.md:
        1. 计算profile size的上限
        2. 过滤掉超出上限的FlexOffer
        3. 选择具有最长profile和最高时间灵活性的FlexOffer
        4. 迭代执行二元聚合
        
        修改：添加能量上限(100KWh)限制，如果超过则创建新的聚合FO
        """
        # 如果没有FO，返回空列表
        if not flexoffers:
            return []
        
        # 能量上限(KWh)
        ENERGY_LIMIT = 100.0
            
        # 计算每个FlexOffer的profile size
        profile_sizes = {}
        for i, fo in enumerate(flexoffers):
            non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
            profile_sizes[i] = non_zero_count
        
        # 计算分位数
        sizes = list(profile_sizes.values())
        q1 = np.percentile(sizes, 25) if sizes else 0
        q3 = np.percentile(sizes, 75) if sizes else 0
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        # 过滤掉异常值
        filtered_fos = [flexoffers[i] for i, size in profile_sizes.items() if size <= upper_fence]
        
        # 如果过滤后没有FO，使用原始列表
        if not filtered_fos:
            filtered_fos = list(flexoffers)
            
        # 复制一份filtered_fos用于处理
        remaining_fos = list(filtered_fos)
        
        # 存储最终的聚合结果
        aggregated_fos = []
        
        # 处理剩余的FlexOffers，直到处理完毕
        while remaining_fos:
            # 重新计算profile sizes
            profile_sizes = {}
            for i, fo in enumerate(remaining_fos):
                non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
                profile_sizes[i] = non_zero_count
                
            if not profile_sizes:
                break  # 没有有效的FO了
                
            # 选择profile size最大的FO
            max_size = max(profile_sizes.values(), default=0)
            if max_size == 0:
                break
                
            max_size_fos = [i for i, size in profile_sizes.items() if size == max_size]
            
            # 从最大profile的FO中选择时间灵活性最高的
            initial_fo_idx = max(max_size_fos, key=lambda i: remaining_fos[i].time_flexibility)
            initial_fo = remaining_fos[initial_fo_idx]
            
            # 初始化聚合轮廓
            agg_profile = np.array(initial_fo.energy_profile)
            
            # 计算初始的RMSE和CV
            rmse = 0.0
            cv = np.std(agg_profile) / (np.mean(abs(agg_profile)) + 1e-10)
            
            # 从remaining_fos中移除已选择的FO
            selected_fos = [initial_fo]
            remaining_fos.pop(initial_fo_idx)
            
            # 计算当前能量总量
            current_energy = sum(abs(e) for e in agg_profile)
            
            # 迭代添加其他FlexOffers，直到达到能量上限或处理完所有FO
            i = 0
            while i < len(remaining_fos):
                fo = remaining_fos[i]
                
                # 计算当前FO的能量总量
                fo_energy = sum(abs(e) for e in fo.energy_profile)
                
                # 检查是否超过能量上限
                if current_energy + fo_energy > ENERGY_LIMIT:
                    i += 1  # 尝试下一个FO
                    continue
                
                # 计算二元聚合
                temp_profile = agg_profile + np.array(fo.energy_profile)
                
                # 计算新的CV和RMSE
                mean_profile = np.mean(abs(temp_profile))
                new_cv = np.std(temp_profile) / (mean_profile + 1e-10)
                
                # 计算与原始FO的均方误差
                new_rmse = 0.0
                for orig_fo in selected_fos + [fo]:
                    orig_weight = np.sum(abs(np.array(orig_fo.energy_profile))) / (np.sum(abs(temp_profile)) + 1e-10)
                    target_profile = temp_profile * orig_weight
                    
                    error = np.mean((target_profile - np.array(orig_fo.energy_profile)) ** 2)
                    new_rmse += error
                
                # 如果新的聚合改善了指标，接受它
                if new_rmse <= rmse or new_cv < cv:
                    agg_profile = temp_profile
                    rmse = new_rmse
                    cv = new_cv
                    selected_fos.append(fo)
                    current_energy += fo_energy
                    remaining_fos.pop(i)  # 移除已添加的FO
                else:
                    i += 1  # 尝试下一个FO
            
            # 计算时间灵活性
            total_energy = sum(abs(np.sum(fo.energy_profile)) for fo in selected_fos)
            if total_energy > 0:
                time_flexibility = sum(fo.time_flexibility * abs(np.sum(fo.energy_profile))
                                    for fo in selected_fos) / total_energy
            else:
                time_flexibility = 0
            
            # 创建聚合FlexOffer
            aggregated_fo = FlexOffer(
                device_id=f"aggregated_{manager_id}_{len(aggregated_fos)}",
                device_type="AGGREGATED",
                energy_profile=agg_profile.tolist(),
                time_flexibility=int(time_flexibility),
                manager_id=manager_id
            )
            
            # 添加到聚合结果列表
            aggregated_fos.append(aggregated_fo)
            
            logger.info(f"创建聚合FO: {aggregated_fo.device_id}，包含 {len(selected_fos)} 个FOs，总能量: {current_energy:.2f}kWh")
        
        logger.info(f"Manager {manager_id}: 创建了 {len(aggregated_fos)} 个聚合FO")
        
        # 如果没有聚合结果，返回空列表
        if not aggregated_fos:
            return []
            
        return aggregated_fos
    
    def trade_flexoffers(self, aggregated_results, timestep):
        """交易FlexOffers"""
        trading_results = {}
        prices = self._get_prices_for_horizon(timestep)
        
        # 跟踪交易的FO ID
        traded_fo_ids = []
        
        # 对每个Manager的聚合FOs进行交易
        for manager_id, agg_data in aggregated_results.items():
            aggregated_fos = agg_data.get('aggregated_fos', [])
            
            # 跳过没有聚合FO的Manager
            if not aggregated_fos:
                continue
            
            # 对每个聚合FO进行交易
            for aggregated_fo in aggregated_fos:
                # 根据交易方法选择交易算法
                if self.trading_method == "bidding":
                    # 投标交易方法
                    schedule, revenue = self._trade_bidding(aggregated_fo, prices)
                else:
                    # 市场出清方法
                    schedule, revenue = self._trade_market_clearing(aggregated_fo, prices)
                
                # 保存交易结果（使用aggregated_fo.device_id作为键）
                fo_id = aggregated_fo.device_id
                trading_results[fo_id] = {
                    'schedule': schedule,
                    'revenue': revenue,
                    'original_fo': aggregated_fo,
                    'manager_id': manager_id,
                    'timestep': timestep
                }
                
                # 记录已交易的FO ID
                traded_fo_ids.append(fo_id)
        
        logger.info(f"时间步 {timestep} 交易完成: 总计 {len(traded_fo_ids)} 个交易FO")
        return trading_results
    
    def _trade_bidding(self, flexoffer, prices):
        """投标交易方法 (Bidding Algorithm)
        
        根据FlexOffer_Pipeline_Structure.md:
        - 生成买卖投标
        - 投标价格计算: base_price × (1 ± market_adj ± random_factor ± bias)
        - 基于时间灵活性寻找最佳调度，最大化收益
        """
        # 获取能量轮廓和时间灵活性
        energy_profile = np.array(flexoffer.energy_profile)
        time_flexibility = flexoffer.time_flexibility
        time_horizon = len(energy_profile)
        
        # 计算能量消耗（正为消耗，负为产生）
        energy_consumption = np.where(energy_profile > 0, energy_profile, 0)
        energy_production = np.where(energy_profile < 0, -energy_profile, 0)
        
        # 投标价格计算参数
        market_adj = 0.05  # 市场调整因子
        random_factor = 0.015  # 随机因子范围
        bias = 0.02  # 偏好因子
        
        # 计算买卖投标
        bid_volumes = energy_consumption  # 购买电量
        ask_volumes = energy_production   # 卖出电量
        
        # 计算投标价格
        # 注意：不再重新设置随机种子，使用全局设置的种子
        random_values = np.random.uniform(-random_factor, random_factor, time_horizon)
        
        # 买入价格（愿意支付的最高价格）- 高于市场价
        bid_prices = np.array(prices) * (1 + market_adj + random_values + bias)
        # 卖出价格（愿意接受的最低价格）- 低于市场价
        ask_prices = np.array(prices) * (1 - market_adj + random_values - bias)
        
        # 根据时间灵活性，寻找最佳调度（最大化收益）
        max_revenue = float('-inf')
        best_schedule = energy_profile.copy()
        
        # 考虑所有可能的时间偏移
        for shift in range(time_flexibility + 1):
            # 计算当前偏移的能量轮廓
            shifted_consumption = np.roll(energy_consumption, shift)
            shifted_production = np.roll(energy_production, shift)
            
            # 计算收益 = 卖出收入 - 购买成本
            sell_income = sum(shifted_production * prices)
            buy_cost = sum(shifted_consumption * prices)
            trade_factor = 4.0
            revenue = trade_factor * (buy_cost-sell_income)
            
            if revenue > max_revenue:
                max_revenue = revenue
                best_schedule = np.roll(energy_profile, shift)
        
        return best_schedule.tolist(), max_revenue
    
    def _trade_market_clearing(self, flexoffer, prices):
        """市场出清方法 (Market Clearing Algorithm)
        
        根据FlexOffer_Pipeline_Structure.md:
        - 确定出清价格、出清数量
        - 基于供需平衡和最大社会福利匹配投标
        - 优化社会福利: 消费者盈余 + 生产者盈余
        """
        # 获取能量轮廓和时间灵活性
        energy_profile = np.array(flexoffer.energy_profile)
        time_flexibility = flexoffer.time_flexibility
        time_horizon = len(energy_profile)
        
        # 社会福利优化参数
        consumer_surplus_weight = 0.5  # 消费者盈余权重
        producer_surplus_weight = 0.5  # 生产者盈余权重
        
        # 根据时间灵活性，寻找最佳调度（尽量将用电安排在价格低的时间，发电安排在价格高的时间）
        sorted_price_indices = np.argsort(prices)  # 价格从低到高排序的小时索引
        
        # 将时间段按价格分为三类：低、中、高
        low_price_indices = sorted_price_indices[:time_horizon//3]  # 低价时段
        high_price_indices = sorted_price_indices[-time_horizon//3:]  # 高价时段
        
        # 创建调度副本
        schedule = energy_profile.copy()
        
        # 计算原始消费者和生产者盈余
        consumer_surplus = 0
        producer_surplus = 0
        
        # 尝试优化调度以最大化社会福利
        consumption_volume = sum(energy_consumption for energy_consumption in energy_profile if energy_consumption > 0)
        production_volume = sum(-energy_production for energy_production in energy_profile if energy_production < 0)
        
        # 创建优化后的调度
        optimized_schedule = np.zeros(time_horizon)
        
        # 1. 将生产（负值）放在高价时段
        remaining_production = production_volume
        for idx in reversed(high_price_indices):  # 从最高价格开始
            if remaining_production <= 0:
                break
            
            # 计算在这个时间步可以放置的最大生产量
            max_production_at_step = min(remaining_production, 10.0)  # 假设每时间步最大10单位
            optimized_schedule[idx] = -max_production_at_step  # 负值表示生产
            remaining_production -= max_production_at_step
        
        # 2. 将消费（正值）放在低价时段
        remaining_consumption = consumption_volume
        for idx in low_price_indices:  # 从最低价格开始
            if remaining_consumption <= 0:
                break
            
            # 计算在这个时间步可以放置的最大消费量
            max_consumption_at_step = min(remaining_consumption, 10.0)  # 假设每时间步最大10单位
            optimized_schedule[idx] = max_consumption_at_step  # 正值表示消费
            remaining_consumption -= max_consumption_at_step
        
        # 计算新的社会福利
        new_consumer_surplus = sum(max(0, 0.2 - prices[i]) * optimized_schedule[i] 
                              for i in range(time_horizon) if optimized_schedule[i] > 0)
        new_producer_surplus = sum(max(0, prices[i] - 0.05) * (-optimized_schedule[i]) 
                              for i in range(time_horizon) if optimized_schedule[i] < 0)
        new_social_welfare = consumer_surplus_weight * new_consumer_surplus + producer_surplus_weight * new_producer_surplus
        
        # 如果新的社会福利更好，就使用优化后的调度
        original_consumer_surplus = sum(max(0, 0.2 - prices[i]) * energy_profile[i] 
                                  for i in range(time_horizon) if energy_profile[i] > 0)
        original_producer_surplus = sum(max(0, prices[i] - 0.05) * (-energy_profile[i]) 
                                  for i in range(time_horizon) if energy_profile[i] < 0)
        original_social_welfare = consumer_surplus_weight * original_consumer_surplus + producer_surplus_weight * original_producer_surplus
        
        if new_social_welfare > original_social_welfare:
            schedule = optimized_schedule
            revenue = new_consumer_surplus + new_producer_surplus
        else:
            revenue = original_consumer_surplus + original_producer_surplus
        
        return schedule.tolist(), revenue 

    def disaggregate_schedules(self, trading_results, aggregated_results):
        """分解调度计划"""
        disaggregated_results = {}
        
        # 按manager_id组织交易结果
        trading_by_manager = {}
        for fo_id, trade_data in trading_results.items():
            manager_id = trade_data.get('manager_id')
            if manager_id not in trading_by_manager:
                trading_by_manager[manager_id] = []
            trading_by_manager[manager_id].append(trade_data)
        
        # 对每个Manager进行分解
        for manager_id, trade_data_list in trading_by_manager.items():
            # 获取该Manager的原始FOs
            if manager_id not in aggregated_results:
                continue
                
            agg_data = aggregated_results[manager_id]
            original_fos = agg_data.get('original_fos', [])
            
            if not original_fos:
                continue
            
            # 存储该Manager的所有设备调度
            all_device_schedules = {}
            total_revenue = 0.0
            
            # 对每个交易结果进行分解
            for trade_data in trade_data_list:
                schedule = trade_data.get('schedule', [])
                revenue = trade_data.get('revenue', 0.0)
                aggregated_fo = trade_data.get('original_fo')
                
                if not aggregated_fo or not schedule:
                    continue
                
                # 根据分解方法选择分解算法
                if self.disaggregation_method == "proportional":
                    # 比例分解
                    device_schedules = self._disaggregate_proportional(schedule, aggregated_fo, original_fos)
                else:
                    # 平均分解
                    device_schedules = self._disaggregate_average(schedule, aggregated_fo, original_fos)
                
                # 合并到该Manager的所有设备调度中
                for device_id, device_data in device_schedules.items():
                    # 如果已存在该设备的调度，合并调度（取平均值）
                    if device_id in all_device_schedules:
                        existing_data = all_device_schedules[device_id]
                        existing_schedule = existing_data.get('schedule', [])
                        
                        # 确保长度一致
                        min_len = min(len(existing_schedule), len(device_data.get('schedule', [])))
                        
                        if min_len > 0:
                            # 计算合并后的调度（取平均值）
                            merged_schedule = []
                            for i in range(min_len):
                                merged_schedule.append((existing_schedule[i] + device_data.get('schedule', [])[i]) / 2)
                            
                            # 更新调度
                            existing_data['schedule'] = merged_schedule
                    else:
                        # 如果是新设备，直接添加
                        all_device_schedules[device_id] = device_data
                
                # 累加收益
                total_revenue += revenue
            
            # 保存该Manager的分解结果
            disaggregated_results[manager_id] = {
                'device_schedules': all_device_schedules,
                'revenue': total_revenue
            }
        
        return disaggregated_results
    
    def _disaggregate_proportional(self, schedule, aggregated_fo, original_fos):
        """按比例分解调度 (Proportional Disaggregation)
        
        根据FlexOffer_Pipeline_Structure.md:
        基于加权贡献分配能源，权重可以是能源需求、设备容量或优先级
        
        E_i = (w_i/W) × E （w_i是设备权重，W是总权重）
        """
        device_schedules = {}
        
        # 如果没有原始FO，返回空结果
        if not original_fos:
            return device_schedules
            
        time_horizon = len(schedule)
        schedule_array = np.array(schedule)
        
        # 获取原始聚合轮廓和总能量需求
        total_energy_needs = {}  # 每个时间步的总能量需求
        for t in range(time_horizon):
            # 分别计算消费和生产
            consumption = sum(max(0, fo.energy_profile[t]) for fo in original_fos)
            production = sum(abs(min(0, fo.energy_profile[t])) for fo in original_fos)
            total_energy_needs[t] = {'consumption': consumption, 'production': production}
        
        # 计算分解后的设备调度
        for fo in original_fos:
            device_id = fo.device_id
            orig_profile = np.array(fo.energy_profile)
            
            # 计算设备在每个时间步的占比
            device_schedule = np.zeros(time_horizon)
            for t in range(time_horizon):
                # 如果是消费（正值）
                if schedule_array[t] > 0:
                    if total_energy_needs[t]['consumption'] > 0:
                        weight = max(0, orig_profile[t]) / total_energy_needs[t]['consumption']
                        device_schedule[t] = schedule_array[t] * weight
                # 如果是生产（负值）
                elif schedule_array[t] < 0:
                    if total_energy_needs[t]['production'] > 0:
                        weight = abs(min(0, orig_profile[t])) / total_energy_needs[t]['production']
                        device_schedule[t] = schedule_array[t] * weight
            
            # 存储设备调度
            device_schedules[device_id] = {
                'schedule': device_schedule.tolist(),
                'original_fo': fo.to_dict()
            }
        
        return device_schedules
    
    def _disaggregate_average(self, schedule, aggregated_fo, original_fos):
        """按平均分解调度 (Average Disaggregation)
        
        根据FlexOffer_Pipeline_Structure.md:
        在所有参与者之间平均分配总能量，不考虑个体差异
        
        E_i = E/N （E是总能量，N是设备数量）
        """
        device_schedules = {}
        
        # 如果没有原始FO，返回空结果
        if not original_fos:
            return device_schedules
            
        time_horizon = len(schedule)
        schedule_array = np.array(schedule)
        
        # 根据设备类型分组
        device_types = {}
        for fo in original_fos:
            device_type = fo.device_type
            if device_type not in device_types:
                device_types[device_type] = []
            device_types[device_type].append(fo)
        
        # 对每个时间步，对每种设备类型进行平均分配
        for device_type, fos in device_types.items():
            num_devices = len(fos)
            for t in range(time_horizon):
                # 分别处理消费和生产
                if schedule_array[t] > 0:  # 消费
                    # 计算该类型设备在此时间步的消费总量
                    type_consumption = sum(max(0, fo.energy_profile[t]) for fo in fos)
                    if type_consumption > 0:
                        # 该类型设备占总消费的比例
                        type_ratio = type_consumption / sum(max(0, fo.energy_profile[t]) for fo in original_fos)
                        # 分配给该类型的能量
                        type_energy = schedule_array[t] * type_ratio
                        # 平均分配给每个设备
                        device_energy = type_energy / num_devices
                    else:
                        device_energy = 0
                elif schedule_array[t] < 0:  # 生产
                    # 计算该类型设备在此时间步的生产总量
                    type_production = sum(abs(min(0, fo.energy_profile[t])) for fo in fos)
                    if type_production > 0:
                        # 该类型设备占总生产的比例
                        type_ratio = type_production / sum(abs(min(0, fo.energy_profile[t])) for fo in original_fos)
                        # 分配给该类型的能量
                        type_energy = schedule_array[t] * type_ratio
                        # 平均分配给每个设备
                        device_energy = type_energy / num_devices
                    else:
                        device_energy = 0
                else:
                    device_energy = 0
                
                # 更新每个设备的调度
                for fo in fos:
                    device_id = fo.device_id
                    if device_id not in device_schedules:
                        device_schedules[device_id] = {
                            'schedule': [0.0] * time_horizon,
                            'original_fo': fo.to_dict()
                        }
                    device_schedules[device_id]['schedule'][t] = device_energy
        
        return device_schedules
    
    def calculate_rewards(self, disaggregated_results):
        """计算奖励"""
        rewards = {}
        
        for manager_id, disagg_data in disaggregated_results.items():
            device_schedules = disagg_data.get('device_schedules', {})
            revenue = disagg_data.get('revenue', 0.0)
            
            if not device_schedules:
                rewards[manager_id] = 0.0
                continue
            
            # 获取原始能量轮廓
            original_profiles = {}
            for device_id, device_data in device_schedules.items():
                original_fo = device_data.get('original_fo', {})
                original_profiles[device_id] = original_fo.get('energy_profile', [0.0])
            
            # 获取调度
            schedules = {}
            for device_id, device_data in device_schedules.items():
                schedules[device_id] = device_data.get('schedule', [0.0])
            
            # 使用模型控制器计算奖励
            if manager_id in self.model_controllers:
                controller = self.model_controllers[manager_id]
                reward = controller.calculate_reward(schedules, revenue, original_profiles)
                rewards[manager_id] = reward
            else:
                # 如果没有对应的控制器，使用默认奖励计算
                reward = self._calculate_default_reward(schedules, revenue, original_profiles)
                rewards[manager_id] = reward
        
        return rewards
    
    def _calculate_default_reward(self, schedules, revenue, original_profiles):
        """默认奖励计算"""
        # 简单的奖励计算：平衡用户满意度和收益
        satisfaction = 0.0
        profile_count = 0
        
        # 计算满意度 - 调度与原始需求的接近程度
        for device_id, schedule in schedules.items():
            if device_id in original_profiles:
                original = original_profiles[device_id]
                # 确保长度一致
                min_len = min(len(schedule), len(original))
                
                if min_len > 0:
                    # 计算相对误差
                    schedule_np = np.array(schedule[:min_len])
                    original_np = np.array(original[:min_len])
                    
                    # 避免除零
                    total_energy = np.sum(np.abs(original_np))
                    if total_energy > 0:
                        error = np.sum(np.abs(schedule_np - original_np)) / total_energy
                        similarity = max(0, 1 - error)  # 转换为相似度
                    else:
                        similarity = 1.0  # 如果原始能量为0，认为完全满足
                    
                    satisfaction += similarity
                    profile_count += 1
        
        # 计算平均满意度
        avg_satisfaction = satisfaction / max(1, profile_count)
        
        # 归一化收益 (假设最大可能收益为设备数量 * 10)
        max_possible_revenue = len(schedules) * 10
        normalized_revenue = min(1.0, revenue / max(0.1, max_possible_revenue))
        
        # 综合奖励 = 满意度权重 * 满意度 + 收益权重 * 归一化收益
        satisfaction_weight = 0.7
        revenue_weight = 0.3
        
        # 计算最终奖励（包含放大因子）
        reward = (satisfaction_weight * avg_satisfaction + revenue_weight * normalized_revenue) * 36.0
        
        return reward 

    def run(self, num_timesteps=1):
        """运行pipeline"""
        logger.info(f"开始运行ModelBased Pipeline，实验ID: {self.experiment_id}, 总时间步数: {num_timesteps}")
        
        # 初始化结果
        total_rewards = []
        
        for timestep in range(num_timesteps):
            logger.info(f"==== 时间步 {timestep} ====")
            
            # 步骤1：生成FlexOffers
            logger.info(f"第1步: 生成FlexOffers...")
            fo_systems = self.generate_flexoffers(timestep)
            
            # 步骤2：聚合FlexOffers
            logger.info(f"第2步: 聚合FlexOffers...")
            aggregated_results = self.aggregate_flexoffers(fo_systems, timestep)
            
            # 收集聚合FO数量
            aggregated_fo_count = sum(len(result.get('aggregated_fos', [])) for result in aggregated_results.values())
            self.results['aggregated_fo_count'].append(aggregated_fo_count)
            logger.info(f"聚合FO数量: {aggregated_fo_count}（能量上限100KWh）")
            
            # 步骤3：交易FlexOffers
            logger.info(f"第3步: 交易FlexOffers...")
            trading_results = self.trade_flexoffers(aggregated_results, timestep)
            
            # 收集交易FO数量和总价值
            traded_fo_count = len(trading_results)
            traded_fo_value = sum(result.get('revenue', 0.0) for result in trading_results.values())
            self.results['traded_fo_count'].append(traded_fo_count)
            self.results['traded_fo_value'].append(traded_fo_value)
            logger.info(f"交易FO数量: {traded_fo_count}，交易FO总价值: {traded_fo_value:.4f}")
            
            # 步骤4：分解调度
            logger.info(f"第4步: 分解调度...")
            disaggregated_results = self.disaggregate_schedules(trading_results, aggregated_results)
            
            # 收集分解FO数量 - 统计所有设备的数量，而不是manager的数量
            disaggregate_count = sum(len(result.get('device_schedules', {})) 
                                    for result in disaggregated_results.values())
            self.results['disaggregate_count'].append(disaggregate_count)
            logger.info(f"分解FO数量: {disaggregate_count}（分解到各个设备）")
            
            # 步骤5：计算奖励
            logger.info(f"第5步: 计算奖励...")
            rewards = self.calculate_rewards(disaggregated_results)
            
            # 汇总时间步结果
            timestep_reward = sum(rewards.values())
            total_rewards.append(timestep_reward)
            
            # 保存时间步结果
            self.results['timestep_details'].append({
                'timestep': timestep,
                'manager_rewards': rewards,
                'total_reward': timestep_reward,
                'aggregation_method': self.aggregation_method,
                'trading_method': self.trading_method,
                'disaggregation_method': self.disaggregation_method,
                'aggregated_fo_count': aggregated_fo_count,
                'traded_fo_count': traded_fo_count,
                'traded_fo_value': traded_fo_value,
                'disaggregate_count': disaggregate_count
            })
            
            # 更新Manager奖励
            for manager_id, reward in rewards.items():
                if manager_id not in self.results['manager_rewards']:
                    self.results['manager_rewards'][manager_id] = []
                self.results['manager_rewards'][manager_id].append(reward)
            
            logger.info(f"时间步 {timestep} 完成，总奖励: {timestep_reward:.4f}")
        
        # 保存最终结果
        self.results['total_rewards'] = total_rewards
        
        # 计算总奖励
        final_reward = sum(total_rewards)
        logger.info(f"Pipeline运行完成, 总奖励: {final_reward:.4f}")
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """保存结果"""
        # 创建结果目录
        result_dir = os.path.join(self.config.results_dir, self.experiment_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存时间步详情
        timestep_df = pd.DataFrame(self.results['timestep_details'])
        timestep_file = os.path.join(result_dir, "timestep_details.csv")
        timestep_df.to_csv(timestep_file, index=False)
        logger.info(f"时间步详情已保存到: {timestep_file}")
        
        # 保存Manager奖励
        manager_rewards = {}
        for manager_id, rewards in self.results['manager_rewards'].items():
            manager_rewards[f"manager_{manager_id}_rewards"] = rewards
        
        manager_df = pd.DataFrame(manager_rewards)
        manager_file = os.path.join(result_dir, "manager_rewards.csv")
        manager_df.to_csv(manager_file, index=False)
        logger.info(f"Manager奖励已保存到: {manager_file}")
        
        # 保存总奖励和交易指标
        metrics_df = pd.DataFrame({
            'timestep': list(range(len(self.results['total_rewards']))),
            'reward': self.results['total_rewards'],
            'aggregated_fo_count': self.results['aggregated_fo_count'],
            'traded_fo_count': self.results['traded_fo_count'],
            'traded_fo_value': self.results['traded_fo_value'],
            'disaggregate_count': self.results['disaggregate_count']
        })
        metrics_file = os.path.join(result_dir, "metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"指标数据已保存到: {metrics_file}")
        
        # 保存配置
        config_file = os.path.join(result_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"配置已保存到: {config_file}")
        
        # 保存统计信息
        stats = self._get_statistics()
        stats_file = os.path.join(result_dir, "statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"统计信息已保存到: {stats_file}")
        
        return result_dir
    
    def _get_statistics(self):
        """获取统计信息"""
        stats = {
            'experiment_id': self.experiment_id,
            'num_managers': len(self.managers),
            'num_devices': sum(len(manager.devices) for manager in self.managers.values()),
            'aggregation_method': self.aggregation_method,
            'trading_method': self.trading_method,
            'disaggregation_method': self.disaggregation_method,
            'total_reward': sum(self.results['total_rewards']),
            'manager_rewards': {
                manager_id: sum(rewards) 
                for manager_id, rewards in self.results['manager_rewards'].items()
            },
            # 添加新的统计指标
            'average_aggregated_fo_count': sum(self.results['aggregated_fo_count']) / max(1, len(self.results['aggregated_fo_count'])),
            'average_traded_fo_count': sum(self.results['traded_fo_count']) / max(1, len(self.results['traded_fo_count'])),
            'average_traded_fo_value': sum(self.results['traded_fo_value']) / max(1, len(self.results['traded_fo_value'])),
            'average_disaggregate_count': sum(self.results['disaggregate_count']) / max(1, len(self.results['disaggregate_count'])),
            'total_traded_fo_value': sum(self.results['traded_fo_value']),
            'aggregated_fo_count_per_timestep': self.results['aggregated_fo_count'],
            'traded_fo_count_per_timestep': self.results['traded_fo_count'],
            'traded_fo_value_per_timestep': self.results['traded_fo_value'],
            'disaggregate_count_per_timestep': self.results['disaggregate_count']
        }
        
        # 添加设备类型统计
        device_types = {}
        for manager in self.managers.values():
            for device_config in manager.devices.values():
                device_type = str(device_config['device_type'])
                if device_type not in device_types:
                    device_types[device_type] = 0
                device_types[device_type] += 1
        
        stats['device_types'] = device_types
        
        return stats 


def run_pipeline(config_path=None, num_timesteps=24, aggregation_method=None, trading_method=None, disaggregation_method=None, save_results=True, seed=None):
    """运行ModelBased Pipeline
    
    参数:
        config_path: 配置文件路径
        num_timesteps: 运行的时间步数
        aggregation_method: 聚合方法 ("LP" 或 "DP")
        trading_method: 交易方法 ("bidding" 或 "market-clearing")
        disaggregation_method: 分解方法 ("proportional" 或 "average")
        save_results: 是否保存结果
        seed: 随机种子，用于保证实验可重复性
    """
    # 加载配置
    try:
        from .config import load_config
    except (ImportError, SystemError):
        from config import load_config
        
    config = load_config(config_path)
    
    # 应用命令行参数覆盖配置
    if aggregation_method:
        config.aggregation_method = aggregation_method
    if trading_method:
        config.trading_method = trading_method
    if disaggregation_method:
        config.disaggregation_method = disaggregation_method
    
    # 设置随机种子
    if seed is not None:
        config.seed = seed
        print(f"使用随机种子: {seed}")
    
    # 打印选择的算法
    print(f"使用算法组合: {config.aggregation_method} + {config.trading_method} + {config.disaggregation_method}")
    
    # 创建并运行pipeline
    pipeline = ModelBasedPipeline(config)
    results = pipeline.run(num_timesteps)
    
    return results


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行ModelBased FlexOffer Pipeline")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--timesteps", type=int, default=24, help="运行的时间步数")
    parser.add_argument("--aggregation", type=str, default="LP", choices=["LP", "DP"], help="聚合方法")
    parser.add_argument("--trading", type=str, default="bidding", choices=["bidding", "market-clearing"], help="交易方法")
    parser.add_argument("--disaggregation", type=str, default="proportional", choices=["proportional", "average"], help="分解方法")
    parser.add_argument("--managers", type=int, default=4, help="Manager数量")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="每个Manager的用户数，逗号分隔")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于保证实验可重复性")
    
    args = parser.parse_args()
    
    # 导入依赖
    try:
        from .config import load_config, PipelineConfig
    except (ImportError, SystemError):
        from config import load_config, PipelineConfig
    
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()
    
    # 应用命令行参数
    config.aggregation_method = args.aggregation
    config.trading_method = args.trading
    config.disaggregation_method = args.disaggregation
    config.num_managers = args.managers
    
    # 解析用户数量
    try:
        config.users_per_manager = [int(n) for n in args.users.split(",")]
        if len(config.users_per_manager) < config.num_managers:
            # 如果用户数不足，则使用默认值补充
            default_users = [9] * (config.num_managers - len(config.users_per_manager))
            config.users_per_manager.extend(default_users)
    except:
        # 如果解析失败，则使用默认配置
        config.users_per_manager = [9] * config.num_managers
    
    # 输出配置信息
    print(f"运行ModelBased Pipeline:")
    print(f"- 聚合方法: {config.aggregation_method}")
    print(f"- 交易方法: {config.trading_method}")
    print(f"- 分解方法: {config.disaggregation_method}")
    print(f"- Manager数量: {config.num_managers}")
    print(f"- 用户分布: {config.users_per_manager} (总计 {sum(config.users_per_manager)} 个用户)")
    print(f"- 时间步数: {args.timesteps}")
    if args.seed is not None:
        print(f"- 随机种子: {args.seed}")
    
    # 运行pipeline，传递正确的参数
    results = run_pipeline(
        config_path=None, 
        num_timesteps=args.timesteps, 
        aggregation_method=config.aggregation_method,
        trading_method=config.trading_method, 
        disaggregation_method=config.disaggregation_method,
        save_results=True,
        seed=args.seed
    )
    
    # 输出总奖励
    total_reward = sum(results.get('total_rewards', []))
    print(f"\n运行完成!")
    print(f"总奖励: {total_reward:.4f}") 
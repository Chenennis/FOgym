from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import os
import sys
import random
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# 添加项目根目录到系统路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入fo_generate模块
from fo_generate.battery_model import BatteryModel, BatteryParameters, BatteryScheduleParams
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.uncertain_model import UncertainModel, UncertainParameters

# 导入标准FlexOffer结构
from fo_common.flexoffer import FlexOffer, FOSlice, FOFactory

# 导入新的聚合器
from .aggregator import FOAggregatorFactory, AggregatedFlexOffer, LongestProfileAggregator, DynamicProfileAggregator

# 创建日志记录器
logger = logging.getLogger(__name__)

@dataclass
class Device:
    """设备类"""
    device_id: str              # 设备ID
    device_type: str            # 设备类型: battery, heat_pump, uncertain
    params: Any                 # 设备参数
    model: Any = None           # 设备模型
    flex_offers: List[FlexOffer] = field(default_factory=list)  # 标准FlexOffer列表
    
    def __post_init__(self):
        # 根据设备类型创建相应的模型
        if self.model is None:
            if self.device_type == "battery":
                self.model = BatteryModel(self.params)
            elif self.device_type == "heat_pump":
                self.model = HeatPumpModel(self.params)
            elif self.device_type == "uncertain":
                self.model = UncertainModel(self.params)
    
    def clone(self):
        """创建设备的克隆"""
        return Device(
            device_id=self.device_id,
            device_type=self.device_type,
            params=self.params,
            model=None  # 让新设备自己创建模型
        )
    
    def get_parameters(self):
        """获取设备参数"""
        return self.params
    
    def set_allocation(self, allocation: float, step: int):
        """设置能源分配"""
        # 这个方法在实际应用中会处理能源分配，这里只是简单记录
        if not hasattr(self, 'allocations'):
            self.allocations = {}
        self.allocations[step] = allocation
    
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """生成标准FlexOffer"""
        if base_time is None:
            base_time = datetime.now()
        
        self.flex_offers = []
        
        # 为每小时生成一个FlexOffer
        for hour in range(time_horizon):
            # 根据设备类型生成不同的能量轮廓
            if self.device_type == "battery":
                # 电池：充放电轮廓
                energy_profile = self._generate_battery_profile()
            elif self.device_type == "heat_pump":
                # 热泵：供暖需求轮廓
                energy_profile = self._generate_heat_pump_profile(hour)
            elif self.device_type == "uncertain":
                # 不确定性设备：随机轮廓
                energy_profile = self._generate_uncertain_profile()
            else:
                # 默认轮廓
                energy_profile = [(1.0, 3.0)] * 30  # 30个2分钟时间片
            
            # 创建FlexOffer
            fo = FOFactory.create_hourly_fo(
                device_id=self.device_id,
                device_type=self.device_type,
                hour=hour,
                base_time=base_time,
                slices_per_hour=len(energy_profile),
                energy_profile=energy_profile
            )
            
            self.flex_offers.append(fo)
    
    def _generate_battery_profile(self) -> List[Tuple[float, float]]:
        """生成电池能量轮廓"""
        # 30个时间片，每片2分钟
        profile = []
        for i in range(30):
            # 模拟充放电模式：可放电（负值）和充电（正值）
            e_min = -2.0  # 可放电2kWh
            e_max = 1.5   # 可充电1.5kWh
            profile.append((e_min, e_max))
        return profile
    
    def _generate_heat_pump_profile(self, hour: int) -> List[Tuple[float, float]]:
        """生成热泵能量轮廓"""
        # 根据时间调整需求
        if 6 <= hour <= 22:  # 白天
            base_demand = 1.5
        else:  # 夜间
            base_demand = 0.8
        
        profile = []
        for i in range(30):
            # 热泵只消耗能量
            e_min = base_demand * 0.8
            e_max = base_demand * 1.2
            profile.append((e_min, e_max))
        return profile
    
    def _generate_uncertain_profile(self) -> List[Tuple[float, float]]:
        """生成不确定性设备能量轮廓"""
        profile = []
        for i in range(30):
            # 随机能量范围
            e_min = random.uniform(0.5, 1.5)
            e_max = e_min + random.uniform(0.5, 2.0)
            profile.append((e_min, e_max))
        return profile
    
    def get_flex_offers(self) -> List[FlexOffer]:
        """获取FlexOffer列表"""
        return self.flex_offers
    
    def visualize_flex_offers(self, save_path: Optional[str] = None):
        """可视化FlexOffer"""
        if not self.flex_offers:
            logger.warning(f"设备 {self.device_id} 没有FlexOffer可视化")
            return
        
        # 提取24小时的能量边界
        hours = []
        e_min_total = []
        e_max_total = []
        
        for fo in self.flex_offers:
            hours.append(fo.hour)
            e_min_total.append(fo.total_energy_min)
            e_max_total.append(fo.total_energy_max)
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.plot(hours, e_min_total, 'b-', label='最小总能量', marker='o')
        plt.plot(hours, e_max_total, 'r-', label='最大总能量', marker='s')
        plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
        plt.xlabel('小时')
        plt.ylabel('总能量 (kWh)')
        plt.title(f'{self.device_type} {self.device_id} 的24小时FlexOffer')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

@dataclass
class User:
    """用户类"""
    user_id: str                   # 用户ID
    user_type: str                 # 用户类型: prosumer, consumer, producer
    location: Tuple[float, float]  # 位置坐标
    devices: List[Device] = field(default_factory=list)  # 设备列表
    preferences: Dict[str, float] = field(default_factory=dict)  # 用户偏好
    
    def add_device(self, device: Device):
        """添加设备"""
        self.devices.append(device)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """生成所有设备的FlexOffer"""
        for device in self.devices:
            device.generate_flex_offers(time_horizon, base_time)
    
    def get_all_flex_offers(self) -> List[FlexOffer]:
        """获取所有设备的FlexOffer"""
        all_fos = []
        for device in self.devices:
            all_fos.extend(device.get_flex_offers())
        return all_fos
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """根据设备ID获取设备"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_allocation(self, step: int) -> Dict[str, float]:
        """获取某个时间步的能源分配"""
        allocations = {}
        for device in self.devices:
            if hasattr(device, 'allocations') and step in device.allocations:
                allocations[device.device_id] = device.allocations[step]
        return allocations

@dataclass
class Manager:
    """Manager类，管理多个用户和设备"""
    manager_id: str                      # 管理器ID
    location: Tuple[float, float]        # 位置坐标
    coverage_area: float                 # 覆盖范围（平方公里）
    users: List[User] = field(default_factory=list)  # 用户列表
    fo_aggregator: Optional[Any] = None              # FlexOffer聚合器
    aggregated_results: List[AggregatedFlexOffer] = field(default_factory=list)  # 聚合结果
    aggregation_method: str = "DP"       # 默认使用Dynamic Profile方法
    
    def __post_init__(self):
        # 初始化聚合器
        if self.fo_aggregator is None:
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
    
    def add_user(self, user: User):
        """添加用户"""
        self.users.append(user)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """生成所有用户的FlexOffer"""
        for user in self.users:
            user.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_flex_offers(self) -> List[AggregatedFlexOffer]:
        """聚合所有用户的FlexOffer"""
        # 收集所有FlexOffer
        all_fos = []
        for user in self.users:
            all_fos.extend(user.get_all_flex_offers())
        
        if not all_fos:
            logger.warning(f"Manager {self.manager_id} 没有FlexOffer需要聚合")
            return []
        
        # 检查聚合器是否存在
        if self.fo_aggregator is None:
            logger.error(f"Manager {self.manager_id} 聚合器未初始化")
            return []
        
        # 执行聚合
        self.aggregated_results = self.fo_aggregator.aggregate(all_fos)
        
        logger.info(f"Manager {self.manager_id} 聚合完成: "
                   f"输入{len(all_fos)}个FO, 输出{len(self.aggregated_results)}个AFO")
        
        return self.aggregated_results
    
    def set_aggregation_method(self, method: str):
        """设置聚合方法"""
        if method.upper() in ["LP", "DP"]:
            self.aggregation_method = method.upper()
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
            logger.info(f"Manager {self.manager_id} 聚合方法设置为: {self.aggregation_method}")
        else:
            logger.error(f"不支持的聚合方法: {method}")
    
    def get_aggregated_flex_offers(self) -> List[FlexOffer]:
        """获取聚合后的FlexOffer列表"""
        return [afo.aggregated_fo for afo in self.aggregated_results if afo.aggregated_fo]
    
    def visualize_aggregated_results(self, save_dir: Optional[str] = None):
        """可视化聚合结果"""
        if not self.aggregated_results:
            logger.warning(f"Manager {self.manager_id} 没有聚合结果可视化")
            return
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 为每个聚合结果创建图形
        for i, afo in enumerate(self.aggregated_results):
            plt.figure(figsize=(15, 10))
            
            # 获取聚合FlexOffer的24小时数据
            hours = []
            e_min_total = []
            e_max_total = []
            
            # 按小时分组聚合FlexOffer（如果有多个相同小时的FO）
            hourly_data = {}
            for slice in afo.aggregated_fo.slices:
                hour = slice.start_time.hour
                if hour not in hourly_data:
                    hourly_data[hour] = {'e_min': 0, 'e_max': 0}
                hourly_data[hour]['e_min'] += slice.energy_min
                hourly_data[hour]['e_max'] += slice.energy_max
            
            for hour in sorted(hourly_data.keys()):
                hours.append(hour)
                e_min_total.append(hourly_data[hour]['e_min'])
                e_max_total.append(hourly_data[hour]['e_max'])
            
            # 主图：能量轮廓
            plt.subplot(2, 2, 1)
            plt.plot(hours, e_min_total, 'b-', label='最小总能量', marker='o')
            plt.plot(hours, e_max_total, 'r-', label='最大总能量', marker='s')
            plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
            plt.xlabel('小时')
            plt.ylabel('总能量 (kWh)')
            plt.title(f'AFO {afo.afo_id} - 24小时能量轮廓')
            plt.grid(True)
            plt.legend()
            
            # 子图1：功率轮廓
            plt.subplot(2, 2, 2)
            p_min, p_max = afo.aggregated_fo.get_power_profile()
            slice_times = list(range(len(p_min)))
            plt.plot(slice_times, p_min, 'b-', label='最小功率', alpha=0.7)
            plt.plot(slice_times, p_max, 'r-', label='最大功率', alpha=0.7)
            plt.axhline(y=100, color='k', linestyle='--', label='目标功率阈值(100kW)')
            plt.xlabel('时间片')
            plt.ylabel('功率 (kW)')
            plt.title('功率轮廓')
            plt.grid(True)
            plt.legend()
            
            # 子图2：聚合统计
            plt.subplot(2, 2, 3)
            stats_data = [
                f"聚合方法: {afo.aggregation_method}",
                f"源FO数量: {len(afo.source_fo_ids)}",
                f"总能量范围: [{afo.total_energy_min:.1f}, {afo.total_energy_max:.1f}] kWh",
                f"功率RMSE: {afo.power_profile_rmse:.2f}",
                f"功率CV: {afo.power_profile_cv:.2f}",
                f"时间片数量: {afo.slice_count}"
            ]
            
            plt.text(0.1, 0.9, '\n'.join(stats_data), 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.axis('off')
            plt.title('聚合统计信息')
            
            # 子图3：源FO分布
            plt.subplot(2, 2, 4)
            # 简单的饼图显示设备类型分布
            device_types = [fo_id.split('_')[0] for fo_id in afo.source_fo_ids]
            type_counts = {}
            for dtype in device_types:
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            if type_counts:
                plt.pie(list(type_counts.values()), labels=list(type_counts.keys()), autopct='%1.1f%%')
                plt.title('源FlexOffer设备类型分布')
            
            plt.suptitle(f'Manager {self.manager_id} - 聚合结果 {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f'manager_{self.manager_id}_afo_{i+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"聚合结果图保存至: {save_path}")
            else:
                plt.show()
            plt.close()
    
    @classmethod
    def load_from_data(cls, manager_id: str, location: Tuple[float, float], coverage_area: float, 
                   num_users: int, data_dir: str = "../data", aggregation_method: str = "DP") -> 'Manager':
        """从数据文件加载Manager"""
        manager = cls(manager_id, location, coverage_area, aggregation_method=aggregation_method)
        
        # 加载用户和设备
        for i in range(num_users):
            user_id = f"user_{manager_id}_{i}"
            
            # 随机位置（在manager位置附近）
            x_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            y_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            user_location = (location[0] + x_offset, location[1] + y_offset)
            
            # 随机用户类型
            user_type = random.choice(["prosumer", "consumer", "producer"])
            
            # 创建用户
            user = User(user_id, user_type, user_location)
            
            # 随机设备数量（3-5个）
            num_devices = random.randint(3, 5)
            
            # 创建设备
            device_types = ["battery", "heat_pump", "uncertain"]
            for j in range(num_devices):
                device_type = random.choice(device_types)
                device_id = f"device_{user_id}_{j}"
                
                if device_type == "battery":
                    # 从CSV文件加载电池参数
                    try:
                        battery_model = BatteryModel.from_csv(
                            os.path.join(data_dir, "battery_base_parameters.csv"),
                            os.path.join(data_dir, "battery_dfo_input.csv"),
                            "BAT001"  # 随机选择一个电池ID，可以改进为从文件中随机选择
                        )
                        device = Device(device_id, device_type, battery_model.params, battery_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"加载电池设备 {device_id} 失败: {e}")
                        
                elif device_type == "heat_pump":
                    # 从CSV文件加载热泵参数
                    try:
                        heat_pump_model = HeatPumpModel.from_csv(
                            os.path.join(data_dir, "heat_pump_system.csv"),
                            "1-1-101-LR"  # 随机选择一个房间ID，可以改进为从文件中随机选择
                        )
                        device = Device(device_id, device_type, heat_pump_model.params, heat_pump_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"加载热泵设备 {device_id} 失败: {e}")
                        
                elif device_type == "uncertain":
                    # 从CSV文件加载不确定性参数
                    try:
                        uncertain_model = UncertainModel.from_csv(
                            os.path.join(data_dir, "uncertain_energy_data.csv"),
                            "光伏发电"  # 随机选择一个能源类型，可以改进为从文件中随机选择
                        )
                        device = Device(device_id, device_type, uncertain_model.params_list, uncertain_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"加载不确定性设备 {device_id} 失败: {e}")
            
            # 添加用户到管理器
            manager.add_user(user)
        
        return manager

@dataclass
class City:
    """城市类，管理多个Manager"""
    city_name: str                           # 城市名称
    width: float = 10.0                      # 城市宽度（公里）
    height: float = 10.0                     # 城市高度（公里）
    managers: List[Manager] = field(default_factory=list)  # Manager列表
    
    def add_manager(self, manager: Manager):
        """添加Manager"""
        self.managers.append(manager)
        
    def generate_managers(self, num_managers: int = 10, users_per_manager: int = 20, 
                        coverage_area: float = 2.0, data_dir: str = "../data", 
                        aggregation_method: str = "DP"):
        """生成指定数量的Manager"""
        for i in range(num_managers):
            # 随机位置
            location = (random.uniform(0, self.width), random.uniform(0, self.height))
            manager_id = f"manager_{i}"
            
            # 创建Manager
            manager = Manager.load_from_data(
                manager_id, location, coverage_area, users_per_manager, 
                data_dir, aggregation_method
            )
            self.add_manager(manager)
            
        logger.info(f"城市 {self.city_name} 生成了 {num_managers} 个Manager")
    
    def generate_all_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """生成所有Manager的FlexOffer"""
        for manager in self.managers:
            manager.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_all(self):
        """聚合所有Manager的FlexOffer"""
        for manager in self.managers:
            manager.aggregate_flex_offers()
    
    def visualize_city(self, save_dir: Optional[str] = None):
        """可视化整个城市的Manager分布和聚合结果"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 城市分布图
        plt.figure(figsize=(12, 8))
        
        for manager in self.managers:
            x, y = manager.location
            # 绘制Manager位置
            plt.scatter(x, y, s=100, c='red', marker='s', alpha=0.7)
            plt.text(x+0.1, y+0.1, manager.manager_id, fontsize=8)
            
            # 绘制覆盖范围
            circle = Circle((x, y), np.sqrt(manager.coverage_area/np.pi), 
                          fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # 绘制用户位置
            for user in manager.users:
                ux, uy = user.location
                plt.scatter(ux, uy, s=20, c='blue', alpha=0.6)
        
        plt.xlim(-1, self.width+1)
        plt.ylim(-1, self.height+1)
        plt.xlabel('距离 (km)')
        plt.ylabel('距离 (km)')
        plt.title(f'城市 {self.city_name} - Manager和用户分布')
        plt.grid(True, alpha=0.3)
        plt.legend(['Manager', '用户'], loc='upper right')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'city_{self.city_name}_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # 为每个Manager生成详细的聚合结果图
        for manager in self.managers:
            manager_save_dir = os.path.join(save_dir, manager.manager_id) if save_dir else None
            manager.visualize_aggregated_results(manager_save_dir) 
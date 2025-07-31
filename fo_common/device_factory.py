"""
FlexOffer统一设备工厂

本模块提供统一的设备创建和管理接口，
消除在多个文件中重复的设备创建逻辑。
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type, Union
from abc import ABC, abstractmethod
import logging

# 设备模型导入
from fo_generate.battery_model import BatteryModel, BatteryParameters
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior
from fo_generate.pv_model import PVModel, PVParameters
from fo_generate.dishwasher_model import DishwasherModel, DishwasherParameters, DishwasherUserBehavior

# MDP设备导入
from fo_generate.unified_mdp_env import (
    DeviceMDPInterface, DeviceType, 
    BatteryMDPDevice, HeatPumpMDPDevice, 
    EVMDPDevice, PVMDPDevice, DishwasherMDPDevice
)

logger = logging.getLogger(__name__)


class DeviceConfigTemplate:
    """设备配置模板"""
    
    @staticmethod
    def get_battery_defaults() -> Dict[str, Any]:
        """获取电池默认配置"""
        return {
            'capacity': 10.0,           # kWh
            'max_power': 5.0,           # kW
            'efficiency': 0.95,         # 效率
            'initial_state': 0.5,       # 初始SOC
            'param1': 0.1,              # soc_min
            'param2': 0.9,              # soc_max
            'can_interrupt': True,
            'priority': 3
        }
    
    @staticmethod
    def get_heat_pump_defaults() -> Dict[str, Any]:
        """获取热泵默认配置"""
        return {
            'max_power': 3.0,           # kW
            'efficiency': 3.5,          # COP
            'initial_state': 21.0,      # 初始温度
            'param1': 18.0,             # temp_min
            'param2': 26.0,             # temp_max
            'param3': 0.1,              # heat_loss_coef
            'can_interrupt': True,
            'priority': 4
        }
    
    @staticmethod
    def get_ev_defaults() -> Dict[str, Any]:
        """获取电动汽车默认配置"""
        return {
            'capacity': 60.0,           # kWh
            'max_power': 7.0,           # kW
            'efficiency': 0.9,          # 效率
            'initial_state': 0.3,       # 初始SOC
            'param1': 0.1,              # soc_min
            'param2': 0.95,             # soc_max
            'param3': 20.0,             # departure_hour
            'can_interrupt': True,
            'priority': 2
        }
    
    @staticmethod
    def get_pv_defaults() -> Dict[str, Any]:
        """获取光伏默认配置"""
        return {
            'max_power': 5.0,           # kW
            'efficiency': 0.18,         # 效率
            'param1': 35.0,             # tilt_angle
            'param2': 180.0,            # azimuth_angle
            'param3': 25.0,             # area
            'can_interrupt': False,
            'priority': 1
        }
    
    @staticmethod
    def get_dishwasher_defaults() -> Dict[str, Any]:
        """获取洗碗机默认配置"""
        return {
            'capacity': 3.0,            # 总能量需求 kWh
            'max_power': 2.0,           # kW
            'efficiency': 0.9,          # 效率
            'initial_state': 0.0,       # 初始状态：未部署
            'param1': 3.5,              # 运行时长 hours
            'param2': 0.5,              # 最小启动延迟 hours
            'param3': 6.0,              # 最大启动延迟 hours
            'can_interrupt': False,
            'priority': 3
        }


class DeviceFactory:
    """统一设备工厂"""
    
    @staticmethod
    def create_device_model(device_type: str, device_config: Dict[str, Any]) -> Any:
        """
        创建设备模型
        
        Args:
            device_type: 设备类型
            device_config: 设备配置
            
        Returns:
            设备模型实例
        """
        if device_type == DeviceType.BATTERY:
            return DeviceFactory._create_battery_model(device_config)
        elif device_type == DeviceType.HEAT_PUMP:
            return DeviceFactory._create_heat_pump_model(device_config)
        elif device_type == DeviceType.EV:
            return DeviceFactory._create_ev_model(device_config)
        elif device_type == DeviceType.PV:
            return DeviceFactory._create_pv_model(device_config)
        elif device_type == DeviceType.DISHWASHER:
            return DeviceFactory._create_dishwasher_model(device_config)
        else:
            raise ValueError(f"不支持的设备类型: {device_type}")
    
    @staticmethod
    def create_device_mdp(device_type: str, device_model: Any) -> DeviceMDPInterface:
        """
        创建设备MDP包装器
        
        Args:
            device_type: 设备类型
            device_model: 设备模型
            
        Returns:
            设备MDP接口实例
        """
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
            raise ValueError(f"不支持的设备类型: {device_type}")
    
    @staticmethod
    def create_complete_device(device_type: str, device_config: Dict[str, Any]) -> DeviceMDPInterface:
        """
        创建完整的设备（模型+MDP包装器）
        
        Args:
            device_type: 设备类型
            device_config: 设备配置
            
        Returns:
            设备MDP接口实例
        """
        # 填充默认配置
        config = DeviceFactory._fill_default_config(device_type, device_config)
        
        # 创建设备模型
        device_model = DeviceFactory.create_device_model(device_type, config)
        
        # 创建MDP包装器
        device_mdp = DeviceFactory.create_device_mdp(device_type, device_model)
        
        logger.info(f"创建设备成功: {device_type} - {config.get('device_id', 'unknown')}")
        return device_mdp
    
    @staticmethod
    def _fill_default_config(device_type: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """填充默认配置"""
        if device_type == DeviceType.BATTERY:
            defaults = DeviceConfigTemplate.get_battery_defaults()
        elif device_type == DeviceType.HEAT_PUMP:
            defaults = DeviceConfigTemplate.get_heat_pump_defaults()
        elif device_type == DeviceType.EV:
            defaults = DeviceConfigTemplate.get_ev_defaults()
        elif device_type == DeviceType.PV:
            defaults = DeviceConfigTemplate.get_pv_defaults()
        elif device_type == DeviceType.DISHWASHER:
            defaults = DeviceConfigTemplate.get_dishwasher_defaults()
        else:
            defaults = {}
        
        # 合并用户配置和默认配置
        config = defaults.copy()
        config.update(user_config)
        return config
    
    @staticmethod
    def _create_battery_model(config: Dict[str, Any]) -> BatteryModel:
        """创建电池模型"""
        params = BatteryParameters(
            battery_id=config.get('device_id', 'battery_default'),
            soc_min=config.get('param1', 0.1),
            soc_max=config.get('param2', 0.9),
            p_min=-config.get('max_power', 5.0),
            p_max=config.get('max_power', 5.0),
            efficiency=config.get('efficiency', 0.95),
            initial_soc=config.get('initial_state', 0.5),
            battery_type="lithium-ion",
            capacity_kwh=config.get('capacity', 10.0)
        )
        return BatteryModel(params)
    
    @staticmethod
    def _create_heat_pump_model(config: Dict[str, Any]) -> HeatPumpModel:
        """创建热泵模型"""
        params = HeatPumpParameters(
            room_id=config.get('device_id', 'room_default'),
            room_area=30.0,
            room_volume=75.0,
            temp_min=config.get('param1', 18.0),
            temp_max=config.get('param2', 26.0),
            initial_temp=config.get('initial_state', 21.0),
            cop=config.get('efficiency', 3.5),
            heat_loss_coef=config.get('param3', 0.1),
            primary_use_period="8:00-22:00",
            secondary_use_period="22:00-8:00",
            primary_target_temp=22.0,
            secondary_target_temp=19.0,
            max_power=config.get('max_power', 3.0)
        )
        return HeatPumpModel(params)
    
    @staticmethod
    def _create_ev_model(config: Dict[str, Any]) -> EVModel:
        """创建电动汽车模型"""
        params = EVParameters(
            ev_id=config.get('device_id', 'ev_default'),
            battery_capacity=config.get('capacity', 60.0),
            soc_min=config.get('param1', 0.1),
            soc_max=config.get('param2', 0.95),
            max_charging_power=config.get('max_power', 7.0),
            efficiency=config.get('efficiency', 0.9),
            initial_soc=config.get('initial_state', 0.3),
            fast_charge_capable=True
        )
        
        # 创建用户行为
        now = datetime.now()
        departure_hour = config.get('param3', 20.0)
        arrival_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
        departure_time = now.replace(hour=int(departure_hour), minute=0, second=0, microsecond=0)
        if departure_time <= arrival_time:
            departure_time += timedelta(days=1)
        
        behavior = EVUserBehavior(
            ev_id=config.get('device_id', 'ev_default'),
            connection_time=arrival_time,
            disconnection_time=departure_time,
            next_departure_time=departure_time,  # 使用departure_time作为下次出发时间
            target_soc=0.85,
            min_required_soc=0.6,
            fast_charge_preferred=False,
            location="home",
            priority=config.get('priority', 2)
        )
        
        return EVModel(params, behavior)
    
    @staticmethod
    def _create_pv_model(config: Dict[str, Any]) -> PVModel:
        """创建光伏模型"""
        params = PVParameters(
            pv_id=config.get('device_id', 'pv_default'),
            max_power=config.get('max_power', 5.0),
            efficiency=config.get('efficiency', 0.18),
            area=config.get('param3', 25.0),
            location="roof",
            tilt_angle=config.get('param1', 35.0),
            azimuth_angle=config.get('param2', 180.0),
            weather_dependent=True,
            forecast_accuracy=0.85
        )
        return PVModel(params)
    
    @staticmethod
    def _create_dishwasher_model(config: Dict[str, Any]) -> DishwasherModel:
        """创建洗碗机模型"""
        params = DishwasherParameters(
            dishwasher_id=config.get('device_id', 'dishwasher_default'),
            total_energy=config.get('capacity', 3.0),
            power_rating=config.get('max_power', 2.0),
            operation_hours=config.get('param1', 3.5),
            min_start_delay=config.get('param2', 0.5),
            max_start_delay=config.get('param3', 6.0),
            efficiency=config.get('efficiency', 0.9),
            can_interrupt=False
        )
        
        # 创建用户行为
        now = datetime.now()
        deployment_time = now + timedelta(hours=np.random.uniform(0, 2))
        
        behavior = DishwasherUserBehavior(
            dishwasher_id=config.get('device_id', 'dishwasher_default'),
            deployment_time=deployment_time,
            preferred_start_time=deployment_time + timedelta(hours=1),
            latest_completion_time=deployment_time + timedelta(hours=8),
            priority=config.get('priority', 3),
            user_tolerance=2.0
        )
        
        return DishwasherModel(params, behavior)
    
    @staticmethod
    def create_device_config_from_csv_row(device_type: str, csv_row: Dict[str, Any]) -> Dict[str, Any]:
        """从CSV行数据创建设备配置"""
        config = {
            'device_id': csv_row.get('device_id', f"{device_type}_default"),
            'device_type': device_type,
            'capacity': float(csv_row.get('capacity', 0)),
            'max_power': float(csv_row.get('max_power', 1)),
            'efficiency': float(csv_row.get('efficiency', 0.9)),
            'initial_state': float(csv_row.get('initial_state', 0.5)),
            'param1': float(csv_row.get('param1', 0)),
            'param2': float(csv_row.get('param2', 1)),
            'param3': float(csv_row.get('param3', 0)),
            'can_interrupt': bool(csv_row.get('can_interrupt', True)),
            'priority': int(csv_row.get('priority', 3))
        }
        return config
    
    @staticmethod
    def validate_device_config(device_type: str, config: Dict[str, Any]) -> bool:
        """验证设备配置"""
        required_fields = ['device_id', 'max_power', 'efficiency']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"设备配置缺少必要字段: {field}")
                return False
        
        # 类型特定验证
        if device_type == DeviceType.BATTERY and config.get('capacity', 0) <= 0:
            logger.error("电池容量必须大于0")
            return False
        
        if config.get('max_power', 0) <= 0:
            logger.error("最大功率必须大于0")
            return False
        
        if not 0 <= config.get('efficiency', 0) <= 1:
            logger.error("效率必须在0-1之间")
            return False
        
        return True
    
    @staticmethod
    def get_supported_device_types() -> list:
        """获取支持的设备类型列表"""
        return [
            DeviceType.BATTERY,
            DeviceType.HEAT_PUMP,
            DeviceType.EV,
            DeviceType.PV,
            DeviceType.DISHWASHER
        ]


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.devices = {}
        self.device_types = {}
    
    def add_device(self, device_id: str, device_type: str, device_config: Dict[str, Any]) -> bool:
        """添加设备"""
        try:
            if not DeviceFactory.validate_device_config(device_type, device_config):
                return False
            
            device_config['device_id'] = device_id
            device_mdp = DeviceFactory.create_complete_device(device_type, device_config)
            
            self.devices[device_id] = device_mdp
            self.device_types[device_id] = device_type
            
            logger.info(f"设备添加成功: {device_id} ({device_type})")
            return True
        except Exception as e:
            logger.error(f"设备添加失败: {device_id} - {str(e)}")
            return False
    
    def remove_device(self, device_id: str) -> bool:
        """移除设备"""
        if device_id in self.devices:
            del self.devices[device_id]
            del self.device_types[device_id]
            logger.info(f"设备移除成功: {device_id}")
            return True
        return False
    
    def get_device(self, device_id: str) -> Optional[DeviceMDPInterface]:
        """获取设备"""
        return self.devices.get(device_id)
    
    def get_device_type(self, device_id: str) -> Optional[str]:
        """获取设备类型"""
        return self.device_types.get(device_id)
    
    def list_devices(self) -> Dict[str, str]:
        """列出所有设备"""
        return self.device_types.copy()
    
    def get_devices_by_type(self, device_type: str) -> Dict[str, DeviceMDPInterface]:
        """按类型获取设备"""
        return {
            device_id: device 
            for device_id, device in self.devices.items()
            if self.device_types[device_id] == device_type
        }
    
    def get_device_count(self) -> int:
        """获取设备数量"""
        return len(self.devices)
    
    def clear_all_devices(self):
        """清空所有设备"""
        self.devices.clear()
        self.device_types.clear()
        logger.info("所有设备已清空") 
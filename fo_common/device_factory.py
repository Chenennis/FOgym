import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type, Union
from abc import ABC, abstractmethod
import logging

from fo_generate.battery_model import BatteryModel, BatteryParameters
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior
from fo_generate.pv_model import PVModel, PVParameters
from fo_generate.dishwasher_model import DishwasherModel, DishwasherParameters, DishwasherUserBehavior

from fo_generate.unified_mdp_env import (
    DeviceMDPInterface, DeviceType, 
    BatteryMDPDevice, HeatPumpMDPDevice, 
    EVMDPDevice, PVMDPDevice, DishwasherMDPDevice
)

logger = logging.getLogger(__name__)


class DeviceConfigTemplate:
    """device configuration template"""
    
    @staticmethod
    def get_battery_defaults() -> Dict[str, Any]:
        """get battery default configuration"""
        return {
            'capacity': 10.0,           # kWh
            'max_power': 5.0,           # kW
            'efficiency': 0.95,         # efficiency
            'initial_state': 0.5,       # initial SOC
            'param1': 0.1,              # soc_min
            'param2': 0.9,              # soc_max
            'can_interrupt': True,
            'priority': 3
        }
    
    @staticmethod
    def get_heat_pump_defaults() -> Dict[str, Any]:
        """get heat pump default configuration"""
        return {
            'max_power': 3.0,           # kW
            'efficiency': 3.5,          # COP
            'initial_state': 21.0,      # initial temperature
            'param1': 18.0,             # temp_min
            'param2': 26.0,             # temp_max
            'param3': 0.1,              # heat_loss_coef
            'can_interrupt': True,
            'priority': 4
        }
    
    @staticmethod
    def get_ev_defaults() -> Dict[str, Any]:
        """get electric vehicle default configuration"""
        return {
            'capacity': 60.0,           # kWh
            'max_power': 7.0,           # kW
            'efficiency': 0.9,          # efficiency
            'initial_state': 0.3,       # initial SOC
            'param1': 0.1,              # soc_min
            'param2': 0.95,             # soc_max
            'param3': 20.0,             # departure_hour
            'can_interrupt': True,
            'priority': 2
        }
    
    @staticmethod
    def get_pv_defaults() -> Dict[str, Any]:
        """get PV default configuration"""
        return {
            'max_power': 5.0,           # kW
            'efficiency': 0.18,         # efficiency
            'param1': 35.0,             # tilt_angle
            'param2': 180.0,            # azimuth_angle
            'param3': 25.0,             # area
            'can_interrupt': False,
            'priority': 1
        }
    
    @staticmethod
    def get_dishwasher_defaults() -> Dict[str, Any]:
        """get dishwasher default configuration"""
        return {
            'capacity': 3.0,            # total energy demand kWh
            'max_power': 2.0,           # kW
            'efficiency': 0.9,          # efficiency
            'initial_state': 0.0,       # initial state: not deployed
            'param1': 3.5,              # operation hours
            'param2': 0.5,              # minimum start delay hours
            'param3': 6.0,              # maximum start delay hours
            'can_interrupt': False,
            'priority': 3
        }


class DeviceFactory:
    """unified device factory"""
    
    @staticmethod
    def create_device_model(device_type: str, device_config: Dict[str, Any]) -> Any:
        """
        create device model
        
        Args:
            device_type: device type
            device_config: device configuration
            
        Returns:
            device model instance
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
            raise ValueError(f"unsupported device type: {device_type}")
    
    @staticmethod
    def create_device_mdp(device_type: str, device_model: Any) -> DeviceMDPInterface:
        """
        create device MDP wrapper
        
        Args:
            device_type: device type
            device_model: device model
            
        Returns:
            device MDP interface instance
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
            raise ValueError(f"unsupported device type: {device_type}")
    
    @staticmethod
    def create_complete_device(device_type: str, device_config: Dict[str, Any]) -> DeviceMDPInterface:
        """
        create complete device (model + MDP wrapper)
        
        Args:
            device_type: device type
            device_config: device configuration
            
        Returns:
            device MDP interface instance
        """
        # fill default configuration
        config = DeviceFactory._fill_default_config(device_type, device_config)
        
        # create device model
        device_model = DeviceFactory.create_device_model(device_type, config)
        
        # create MDP wrapper
        device_mdp = DeviceFactory.create_device_mdp(device_type, device_model)
        
        logger.info(f"create device successfully: {device_type} - {config.get('device_id', 'unknown')}")
        return device_mdp
    
    @staticmethod
    def _fill_default_config(device_type: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """fill default configuration"""
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
        
        # merge user configuration and default configuration
        config = defaults.copy()
        config.update(user_config)
        return config
    
    @staticmethod
    def _create_battery_model(config: Dict[str, Any]) -> BatteryModel:
        """create battery model"""
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
        """create heat pump model"""
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
        """create electric vehicle model"""
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
        
        # create user behavior
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
            next_departure_time=departure_time,  # use departure_time as next departure time
            target_soc=0.85,
            min_required_soc=0.6,
            fast_charge_preferred=False,
            location="home",
            priority=config.get('priority', 2)
        )
        
        return EVModel(params, behavior)
    
    @staticmethod
    def _create_pv_model(config: Dict[str, Any]) -> PVModel:
        """create PV model"""
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
        """create dishwasher model"""
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
        
        # create user behavior
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
        """create device configuration from CSV row"""
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
        """validate device configuration"""
        required_fields = ['device_id', 'max_power', 'efficiency']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"device configuration missing required field: {field}")
                return False
        
        # type specific validation
        if device_type == DeviceType.BATTERY and config.get('capacity', 0) <= 0:
            logger.error("battery capacity must be greater than 0")
            return False
        
        if config.get('max_power', 0) <= 0:
            logger.error("maximum power must be greater than 0")
            return False
        
        if not 0 <= config.get('efficiency', 0) <= 1:
            logger.error("efficiency must be between 0 and 1")
            return False
        
        return True
    
    @staticmethod
    def get_supported_device_types() -> list:
        """get supported device type list"""
        return [
            DeviceType.BATTERY,
            DeviceType.HEAT_PUMP,
            DeviceType.EV,
            DeviceType.PV,
            DeviceType.DISHWASHER
        ]


class DeviceManager:
    """device manager"""
    
    def __init__(self):
        self.devices = {}
        self.device_types = {}
    
    def add_device(self, device_id: str, device_type: str, device_config: Dict[str, Any]) -> bool:
        """add device"""
        try:
            if not DeviceFactory.validate_device_config(device_type, device_config):
                return False
            
            device_config['device_id'] = device_id
            device_mdp = DeviceFactory.create_complete_device(device_type, device_config)
            
            self.devices[device_id] = device_mdp
            self.device_types[device_id] = device_type
            
            logger.info(f"add device successfully: {device_id} ({device_type})")
            return True
        except Exception as e:
            logger.error(f"add device failed: {device_id} - {str(e)}")
            return False
    
    def remove_device(self, device_id: str) -> bool:
        """remove device"""
        if device_id in self.devices:
            del self.devices[device_id]
            del self.device_types[device_id]
            logger.info(f"remove device successfully: {device_id}")
            return True
        return False
    
    def get_device(self, device_id: str) -> Optional[DeviceMDPInterface]:
        """get device"""
        return self.devices.get(device_id)
    
    def get_device_type(self, device_id: str) -> Optional[str]:
        """get device type"""
        return self.device_types.get(device_id)
    
    def list_devices(self) -> Dict[str, str]:
        """list all devices"""
        return self.device_types.copy()
    
    def get_devices_by_type(self, device_type: str) -> Dict[str, DeviceMDPInterface]:
        """get devices by type"""
        return {
            device_id: device 
            for device_id, device in self.devices.items()
            if self.device_types[device_id] == device_type
        }
    
    def get_device_count(self) -> int:
        """get device count"""
        return len(self.devices)
    
    def clear_all_devices(self):
        """clear all devices"""
        self.devices.clear()
        self.device_types.clear()
        logger.info("all devices cleared") 
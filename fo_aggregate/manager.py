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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fo_generate.battery_model import BatteryModel, BatteryParameters, BatteryScheduleParams
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.uncertain_model import UncertainModel, UncertainParameters

from fo_common.flexoffer import FlexOffer, FOSlice, FOFactory

from .aggregator import FOAggregatorFactory, AggregatedFlexOffer, LongestProfileAggregator, DynamicProfileAggregator

logger = logging.getLogger(__name__)

@dataclass
class Device:
    """device class"""
    device_id: str              # device ID
    device_type: str            # device type: battery, heat_pump, uncertain
    params: Any                 # device parameters
    model: Any = None           # device model
    flex_offers: List[FlexOffer] = field(default_factory=list)  # standard FlexOffer list
    
    def __post_init__(self):
        # create corresponding model based on device type
        if self.model is None:
            if self.device_type == "battery":
                self.model = BatteryModel(self.params)
            elif self.device_type == "heat_pump":
                self.model = HeatPumpModel(self.params)
            elif self.device_type == "uncertain":
                self.model = UncertainModel(self.params)
    
    def clone(self):
        """create clone of device"""
        return Device(
            device_id=self.device_id,
            device_type=self.device_type,
            params=self.params,
            model=None  
        )
    
    def get_parameters(self):
        """get device parameters"""
        return self.params
    
    def set_allocation(self, allocation: float, step: int):
        """set energy allocation"""
        if not hasattr(self, 'allocations'):
            self.allocations = {}
        self.allocations[step] = allocation
    
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """generate standard FlexOffer"""
        if base_time is None:
            base_time = datetime.now()
        
        self.flex_offers = []
        
        # generate FlexOffer for each hour
        for hour in range(time_horizon):
            # generate different energy profiles based on device type
            if self.device_type == "battery":
                # battery: charge/discharge profile
                energy_profile = self._generate_battery_profile()
            elif self.device_type == "heat_pump":
                # heat pump: heating demand profile
                energy_profile = self._generate_heat_pump_profile(hour)
            elif self.device_type == "uncertain":
                # uncertain device: random profile
                energy_profile = self._generate_uncertain_profile()
            else:
                # default profile
                energy_profile = [(1.0, 3.0)] * 30  # 30 2-minute time slices
            
            # create FlexOffer
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
        """generate battery energy profile"""
        # 30 time slices, 2 minutes per slice
        profile = []
        for i in range(30):
            # simulate charge/discharge mode: discharge (negative) and charge (positive)
            e_min = -2.0  # discharge 2kWh
            e_max = 1.5   # charge 1.5kWh
            profile.append((e_min, e_max))
        return profile
    
    def _generate_heat_pump_profile(self, hour: int) -> List[Tuple[float, float]]:
        """generate heat pump energy profile"""
        # adjust demand based on time
        if 6 <= hour <= 22:  # day
            base_demand = 1.5
        else:  # night
            base_demand = 0.8
        
        profile = []
        for i in range(30):
            # heat pump only consumes energy
            e_min = base_demand * 0.8
            e_max = base_demand * 1.2
            profile.append((e_min, e_max))
        return profile
    
    def _generate_uncertain_profile(self) -> List[Tuple[float, float]]:
        """generate uncertain device energy profile"""
        profile = []
        for i in range(30):
            # random energy range
            e_min = random.uniform(0.5, 1.5)
            e_max = e_min + random.uniform(0.5, 2.0)
            profile.append((e_min, e_max))
        return profile
    
    def get_flex_offers(self) -> List[FlexOffer]:
        """get FlexOffer list"""
        return self.flex_offers
    
    def visualize_flex_offers(self, save_path: Optional[str] = None):
        """visualize FlexOffer"""
        if not self.flex_offers:
            logger.warning(f"device {self.device_id} has no FlexOffer visualization")
            return
        
        # extract energy boundaries for 24 hours
        hours = []
        e_min_total = []
        e_max_total = []
        
        for fo in self.flex_offers:
            hours.append(fo.hour)
            e_min_total.append(fo.total_energy_min)
            e_max_total.append(fo.total_energy_max)
        
        # create graph
        plt.figure(figsize=(12, 6))
        plt.plot(hours, e_min_total, 'b-', label='minimum total energy', marker='o')
        plt.plot(hours, e_max_total, 'r-', label='maximum total energy', marker='s')
        plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
        plt.xlabel('hour')
        plt.ylabel('total energy (kWh)')
        plt.title(f'{self.device_type} {self.device_id} 24-hour FlexOffer')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

@dataclass
class User:
    """user class"""
    user_id: str                   # user ID
    user_type: str                 # user type: prosumer, consumer, producer
    location: Tuple[float, float]  # location coordinates
    devices: List[Device] = field(default_factory=list)  # device list
    preferences: Dict[str, float] = field(default_factory=dict)  # user preferences
    
    def add_device(self, device: Device):
        """add device"""
        self.devices.append(device)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """generate all devices' FlexOffer"""
        for device in self.devices:
            device.generate_flex_offers(time_horizon, base_time)
    
    def get_all_flex_offers(self) -> List[FlexOffer]:
        """get all devices' FlexOffer"""
        all_fos = []
        for device in self.devices:
            all_fos.extend(device.get_flex_offers())
        return all_fos
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """get device by device ID"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_allocation(self, step: int) -> Dict[str, float]:
        """get energy allocation at a specific time step"""
        allocations = {}
        for device in self.devices:
            if hasattr(device, 'allocations') and step in device.allocations:
                allocations[device.device_id] = device.allocations[step]
        return allocations

@dataclass
class Manager:
    """Manager class, managing multiple users and devices"""
    manager_id: str                      # manager ID
    location: Tuple[float, float]        # location coordinates
    coverage_area: float                 # coverage area (square kilometers)
    users: List[User] = field(default_factory=list)  # user list
    fo_aggregator: Optional[Any] = None              # FlexOffer aggregator
    aggregated_results: List[AggregatedFlexOffer] = field(default_factory=list)  # aggregated results
    aggregation_method: str = "DP"       # default using Dynamic Profile method
    
    def __post_init__(self):
        # initialize aggregator
        if self.fo_aggregator is None:
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
    
    def add_user(self, user: User):
        """add user"""
        self.users.append(user)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """generate all users' FlexOffer"""
        for user in self.users:
            user.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_flex_offers(self) -> List[AggregatedFlexOffer]:
        """aggregate all users' FlexOffer"""
        # collect all FlexOffer
        all_fos = []
        for user in self.users:
            all_fos.extend(user.get_all_flex_offers())
        
        if not all_fos:
            logger.warning(f"Manager {self.manager_id} has no FlexOffer to aggregate")
            return []
        
        # check if aggregator exists
        if self.fo_aggregator is None:
            logger.error(f"Manager {self.manager_id} aggregator not initialized")
            return []
        
        # execute aggregation
        self.aggregated_results = self.fo_aggregator.aggregate(all_fos)
        
        logger.info(f"Manager {self.manager_id} aggregation completed: "
                   f"input {len(all_fos)} FO, output {len(self.aggregated_results)} AFO")
        
        return self.aggregated_results
    
    def set_aggregation_method(self, method: str):
        """set aggregation method"""
        if method.upper() in ["LP", "DP"]:
            self.aggregation_method = method.upper()
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
            logger.info(f"Manager {self.manager_id} aggregation method set to: {self.aggregation_method}")
        else:
            logger.error(f"unsupported aggregation method: {method}")
    
    def get_aggregated_flex_offers(self) -> List[FlexOffer]:
        """get aggregated FlexOffer list"""
        return [afo.aggregated_fo for afo in self.aggregated_results if afo.aggregated_fo]
    
    def visualize_aggregated_results(self, save_dir: Optional[str] = None):
        """visualize aggregated results"""
        if not self.aggregated_results:
            logger.warning(f"Manager {self.manager_id} has no aggregated results visualization")
            return
        
        # create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # create graph for each aggregated result
        for i, afo in enumerate(self.aggregated_results):
            plt.figure(figsize=(15, 10))
            
            # get 24-hour data of aggregated FlexOffer
            hours = []
            e_min_total = []
            e_max_total = []
            
            # group aggregated FlexOffer by hour (if there are multiple FOs with the same hour)
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
            
            # main graph: energy profile
            plt.subplot(2, 2, 1)
            plt.plot(hours, e_min_total, 'b-', label='minimum total energy', marker='o')
            plt.plot(hours, e_max_total, 'r-', label='maximum total energy', marker='s')
            plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
            plt.xlabel('hour')
            plt.ylabel('total energy (kWh)')
            plt.title(f'AFO {afo.afo_id} - 24-hour energy profile')
            plt.grid(True)
            plt.legend()
            
            # subplot 1: power profile
            plt.subplot(2, 2, 2)
            p_min, p_max = afo.aggregated_fo.get_power_profile()
            slice_times = list(range(len(p_min)))
            plt.plot(slice_times, p_min, 'b-', label='minimum power', alpha=0.7)
            plt.plot(slice_times, p_max, 'r-', label='maximum power', alpha=0.7)
            plt.axhline(y=100, color='k', linestyle='--', label='target power threshold (100kW)')
            plt.xlabel('time slice')
            plt.ylabel('power (kW)')
            plt.title('power profile')
            plt.grid(True)
            plt.legend()
            
            # subplot 2: aggregated statistics
            plt.subplot(2, 2, 3)
            stats_data = [
                f"aggregation method: {afo.aggregation_method}",
                f"source FO count: {len(afo.source_fo_ids)}",
                f"total energy range: [{afo.total_energy_min:.1f}, {afo.total_energy_max:.1f}] kWh",
                f"power RMSE: {afo.power_profile_rmse:.2f}",
                f"power CV: {afo.power_profile_cv:.2f}",
                f"time slice count: {afo.slice_count}"
            ]
            
            plt.text(0.1, 0.9, '\n'.join(stats_data), 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.axis('off')
            plt.title('aggregated statistics')
            
            # subplot 3: source FO distribution
            plt.subplot(2, 2, 4)
            # pie chart to show device type distribution
            device_types = [fo_id.split('_')[0] for fo_id in afo.source_fo_ids]
            type_counts = {}
            for dtype in device_types:
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            if type_counts:
                plt.pie(list(type_counts.values()), labels=list(type_counts.keys()), autopct='%1.1f%%')
                plt.title('source FlexOffer device type distribution')
            
            plt.suptitle(f'Manager {self.manager_id} - aggregated result {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f'manager_{self.manager_id}_afo_{i+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"aggregated result graph saved to: {save_path}")
            else:
                plt.show()
            plt.close()
    
    @classmethod
    def load_from_data(cls, manager_id: str, location: Tuple[float, float], coverage_area: float, 
                   num_users: int, data_dir: str = "../data", aggregation_method: str = "DP") -> 'Manager':
        """load Manager from data file"""
        manager = cls(manager_id, location, coverage_area, aggregation_method=aggregation_method)
        
        # load users and devices
        for i in range(num_users):
            user_id = f"user_{manager_id}_{i}"
            
            # random location (near manager location)
            x_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            y_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            user_location = (location[0] + x_offset, location[1] + y_offset)
            
            # random user type
            user_type = random.choice(["prosumer", "consumer", "producer"])
            
            # create user
            user = User(user_id, user_type, user_location)
            
            # random device count (3-5)
            num_devices = random.randint(3, 5)
            
            # create devices
            device_types = ["battery", "heat_pump", "uncertain"]
            for j in range(num_devices):
                device_type = random.choice(device_types)
                device_id = f"device_{user_id}_{j}"
                
                if device_type == "battery":
                    # load battery parameters from CSV file
                    try:
                        battery_model = BatteryModel.from_csv(
                            os.path.join(data_dir, "battery_base_parameters.csv"),
                            os.path.join(data_dir, "battery_dfo_input.csv"),
                            "BAT001"  # randomly select a battery ID, can be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, battery_model.params, battery_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"failed to load battery device {device_id}: {e}")
                        
                elif device_type == "heat_pump":
                    # load heat pump parameters from CSV file
                    try:
                        heat_pump_model = HeatPumpModel.from_csv(
                            os.path.join(data_dir, "heat_pump_system.csv"),
                            "1-1-101-LR"  # randomly select a room ID, can be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, heat_pump_model.params, heat_pump_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"failed to load heat pump device {device_id}: {e}")
                        
                elif device_type == "uncertain":
                    # load uncertain parameters from CSV file
                    try:
                        uncertain_model = UncertainModel.from_csv(
                            os.path.join(data_dir, "uncertain_energy_data.csv"),
                            "photovoltaic_generation"  # randomly select an energy type, can be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, uncertain_model.params_list, uncertain_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"failed to load uncertain device {device_id}: {e}")
            
            # add user to manager
            manager.add_user(user)
        
        return manager

@dataclass
class City:
    """city class, managing multiple Managers"""
    city_name: str                           # city name
    width: float = 10.0                      # city width (kilometers)
    height: float = 10.0                     # city height (kilometers)
    managers: List[Manager] = field(default_factory=list)  # Manager list
    
    def add_manager(self, manager: Manager):
        """add Manager"""
        self.managers.append(manager)
        
    def generate_managers(self, num_managers: int = 10, users_per_manager: int = 20, 
                        coverage_area: float = 2.0, data_dir: str = "../data", 
                        aggregation_method: str = "DP"):
        """generate specified number of Managers"""
        for i in range(num_managers):
            # random location
            location = (random.uniform(0, self.width), random.uniform(0, self.height))
            manager_id = f"manager_{i}"
            
            # create Manager
            manager = Manager.load_from_data(
                manager_id, location, coverage_area, users_per_manager, 
                data_dir, aggregation_method
            )
            self.add_manager(manager)
            
        logger.info(f"city {self.city_name} generated {num_managers} Managers")
    
    def generate_all_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """generate all Managers' FlexOffer"""
        for manager in self.managers:
            manager.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_all(self):
        """aggregate all Managers' FlexOffer"""
        for manager in self.managers:
            manager.aggregate_flex_offers()
    
    def visualize_city(self, save_dir: Optional[str] = None):
        """visualize the distribution of Managers and aggregated results in the city"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # city distribution graph
        plt.figure(figsize=(12, 8))
        
        for manager in self.managers:
            x, y = manager.location
            # draw Manager location
            plt.scatter(x, y, s=100, c='red', marker='s', alpha=0.7)
            plt.text(x+0.1, y+0.1, manager.manager_id, fontsize=8)
            
            # draw coverage area
            circle = Circle((x, y), np.sqrt(manager.coverage_area/np.pi), 
                          fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # draw user location
            for user in manager.users:
                ux, uy = user.location
                plt.scatter(ux, uy, s=20, c='blue', alpha=0.6)
        
        plt.xlim(-1, self.width+1)
        plt.ylim(-1, self.height+1)
        plt.xlabel('distance (km)')
        plt.ylabel('distance (km)')
        plt.title(f'city {self.city_name} - Manager and user distribution')
        plt.grid(True, alpha=0.3)
        plt.legend(['Manager', 'user'], loc='upper right')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'city_{self.city_name}_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # generate detailed aggregated result graph for each Manager
        for manager in self.managers:
            manager_save_dir = os.path.join(save_dir, manager.manager_id) if save_dir else None
            manager.visualize_aggregated_results(manager_save_dir) 
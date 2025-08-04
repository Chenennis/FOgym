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

# Add project root directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fo_generate modules
from fo_generate.battery_model import BatteryModel, BatteryParameters, BatteryScheduleParams
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.uncertain_model import UncertainModel, UncertainParameters

# Import standard FlexOffer structure
from fo_common.flexoffer import FlexOffer, FOSlice, FOFactory

# Import new aggregator
from .aggregator import FOAggregatorFactory, AggregatedFlexOffer, LongestProfileAggregator, DynamicProfileAggregator

# Create logger
logger = logging.getLogger(__name__)

@dataclass
class Device:
    """Device class"""
    device_id: str              # Device ID
    device_type: str            # Device type: battery, heat_pump, uncertain
    params: Any                 # Device parameters
    model: Any = None           # Device model
    flex_offers: List[FlexOffer] = field(default_factory=list)  # Standard FlexOffer list
    
    def __post_init__(self):
        # Create appropriate model based on device type
        if self.model is None:
            if self.device_type == "battery":
                self.model = BatteryModel(self.params)
            elif self.device_type == "heat_pump":
                self.model = HeatPumpModel(self.params)
            elif self.device_type == "uncertain":
                self.model = UncertainModel(self.params)
    
    def clone(self):
        """Create a clone of the device"""
        return Device(
            device_id=self.device_id,
            device_type=self.device_type,
            params=self.params,
            model=None  # Let the new device create its own model
        )
    
    def get_parameters(self):
        """Get device parameters"""
        return self.params
    
    def set_allocation(self, allocation: float, step: int):
        """Set energy allocation"""
        # This method would handle energy allocation in a real application, here it just records it
        if not hasattr(self, 'allocations'):
            self.allocations = {}
        self.allocations[step] = allocation
    
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """Generate standard FlexOffers"""
        if base_time is None:
            base_time = datetime.now()
        
        self.flex_offers = []
        
        # Generate a FlexOffer for each hour
        for hour in range(time_horizon):
            # Generate different energy profiles based on device type
            if self.device_type == "battery":
                # Battery: charge/discharge profile
                energy_profile = self._generate_battery_profile()
            elif self.device_type == "heat_pump":
                # Heat pump: heating demand profile
                energy_profile = self._generate_heat_pump_profile(hour)
            elif self.device_type == "uncertain":
                # Uncertain device: random profile
                energy_profile = self._generate_uncertain_profile()
            else:
                # Default profile
                energy_profile = [(1.0, 3.0)] * 30  # 30 time slices of 2 minutes each
            
            # Create FlexOffer
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
        """Generate battery energy profile"""
        # 30 time slices, 2 minutes each
        profile = []
        for i in range(30):
            # Simulate charge/discharge pattern: discharge (negative) and charge (positive)
            e_min = -2.0  # Can discharge 2kWh
            e_max = 1.5   # Can charge 1.5kWh
            profile.append((e_min, e_max))
        return profile
    
    def _generate_heat_pump_profile(self, hour: int) -> List[Tuple[float, float]]:
        """Generate heat pump energy profile"""
        # Adjust demand based on time
        if 6 <= hour <= 22:  # Daytime
            base_demand = 1.5
        else:  # Nighttime
            base_demand = 0.8
        
        profile = []
        for i in range(30):
            # Heat pump only consumes energy
            e_min = base_demand * 0.8
            e_max = base_demand * 1.2
            profile.append((e_min, e_max))
        return profile
    
    def _generate_uncertain_profile(self) -> List[Tuple[float, float]]:
        """Generate uncertain device energy profile"""
        profile = []
        for i in range(30):
            # Random energy range
            e_min = random.uniform(0.5, 1.5)
            e_max = e_min + random.uniform(0.5, 2.0)
            profile.append((e_min, e_max))
        return profile
    
    def get_flex_offers(self) -> List[FlexOffer]:
        """Get FlexOffer list"""
        return self.flex_offers
    
    def visualize_flex_offers(self, save_path: Optional[str] = None):
        """Visualize FlexOffers"""
        if not self.flex_offers:
            logger.warning(f"Device {self.device_id} has no FlexOffers to visualize")
            return
        
        # Extract 24-hour energy bounds
        hours = []
        e_min_total = []
        e_max_total = []
        
        for fo in self.flex_offers:
            hours.append(fo.hour)
            e_min_total.append(fo.total_energy_min)
            e_max_total.append(fo.total_energy_max)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(hours, e_min_total, 'b-', label='Minimum Total Energy', marker='o')
        plt.plot(hours, e_max_total, 'r-', label='Maximum Total Energy', marker='s')
        plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
        plt.xlabel('Hour')
        plt.ylabel('Total Energy (kWh)')
        plt.title(f'24-hour FlexOffer for {self.device_type} {self.device_id}')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

@dataclass
class User:
    """User class"""
    user_id: str                   # User ID
    user_type: str                 # User type: prosumer, consumer, producer
    location: Tuple[float, float]  # Location coordinates
    devices: List[Device] = field(default_factory=list)  # Device list
    preferences: Dict[str, float] = field(default_factory=dict)  # User preferences
    
    def add_device(self, device: Device):
        """Add device"""
        self.devices.append(device)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """Generate FlexOffers for all devices"""
        for device in self.devices:
            device.generate_flex_offers(time_horizon, base_time)
    
    def get_all_flex_offers(self) -> List[FlexOffer]:
        """Get FlexOffers from all devices"""
        all_fos = []
        for device in self.devices:
            all_fos.extend(device.get_flex_offers())
        return all_fos
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by device ID"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_allocation(self, step: int) -> Dict[str, float]:
        """Get energy allocation for a specific time step"""
        allocations = {}
        for device in self.devices:
            if hasattr(device, 'allocations') and step in device.allocations:
                allocations[device.device_id] = device.allocations[step]
        return allocations

@dataclass
class Manager:
    """Manager class, manages multiple users and devices"""
    manager_id: str                      # Manager ID
    location: Tuple[float, float]        # Location coordinates
    coverage_area: float                 # Coverage area (square kilometers)
    users: List[User] = field(default_factory=list)  # User list
    fo_aggregator: Optional[Any] = None              # FlexOffer aggregator
    aggregated_results: List[AggregatedFlexOffer] = field(default_factory=list)  # Aggregation results
    aggregation_method: str = "DP"       # Default to Dynamic Profile method
    
    def __post_init__(self):
        # Initialize aggregator
        if self.fo_aggregator is None:
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
    
    def add_user(self, user: User):
        """Add user"""
        self.users.append(user)
        
    def generate_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """Generate FlexOffers for all users"""
        for user in self.users:
            user.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_flex_offers(self) -> List[AggregatedFlexOffer]:
        """Aggregate FlexOffers from all users"""
        # Collect all FlexOffers
        all_fos = []
        for user in self.users:
            all_fos.extend(user.get_all_flex_offers())
        
        if not all_fos:
            logger.warning(f"Manager {self.manager_id} has no FlexOffers to aggregate")
            return []
        
        # Check if aggregator exists
        if self.fo_aggregator is None:
            logger.error(f"Manager {self.manager_id} aggregator not initialized")
            return []
        
        # Perform aggregation
        self.aggregated_results = self.fo_aggregator.aggregate(all_fos)
        
        logger.info(f"Manager {self.manager_id} aggregation complete: "
                   f"input {len(all_fos)} FOs, output {len(self.aggregated_results)} AFOs")
        
        return self.aggregated_results
    
    def set_aggregation_method(self, method: str):
        """Set aggregation method"""
        if method.upper() in ["LP", "DP"]:
            self.aggregation_method = method.upper()
            self.fo_aggregator = FOAggregatorFactory.create_aggregator(self.aggregation_method)
            logger.info(f"Manager {self.manager_id} aggregation method set to: {self.aggregation_method}")
        else:
            logger.error(f"Unsupported aggregation method: {method}")
    
    def get_aggregated_flex_offers(self) -> List[FlexOffer]:
        """Get list of aggregated FlexOffers"""
        return [afo.aggregated_fo for afo in self.aggregated_results if afo.aggregated_fo]
    
    def visualize_aggregated_results(self, save_dir: Optional[str] = None):
        """Visualize aggregation results"""
        if not self.aggregated_results:
            logger.warning(f"Manager {self.manager_id} has no aggregation results to visualize")
            return
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create figure for each aggregation result
        for i, afo in enumerate(self.aggregated_results):
            plt.figure(figsize=(15, 10))
            
            # Get 24-hour data from aggregated FlexOffer
            hours = []
            e_min_total = []
            e_max_total = []
            
            # Group aggregated FlexOffer by hour (if multiple FOs for the same hour)
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
            
            # Main plot: Energy profile
            plt.subplot(2, 2, 1)
            plt.plot(hours, e_min_total, 'b-', label='Minimum Total Energy', marker='o')
            plt.plot(hours, e_max_total, 'r-', label='Maximum Total Energy', marker='s')
            plt.fill_between(hours, e_min_total, e_max_total, alpha=0.2)
            plt.xlabel('Hour')
            plt.ylabel('Total Energy (kWh)')
            plt.title(f'AFO {afo.afo_id} - 24-hour Energy Profile')
            plt.grid(True)
            plt.legend()
            
            # Subplot 1: Power profile
            plt.subplot(2, 2, 2)
            p_min, p_max = afo.aggregated_fo.get_power_profile()
            slice_times = list(range(len(p_min)))
            plt.plot(slice_times, p_min, 'b-', label='Minimum Power', alpha=0.7)
            plt.plot(slice_times, p_max, 'r-', label='Maximum Power', alpha=0.7)
            plt.axhline(y=100, color='k', linestyle='--', label='Target Power Threshold (100kW)')
            plt.xlabel('Time Slice')
            plt.ylabel('Power (kW)')
            plt.title('Power Profile')
            plt.grid(True)
            plt.legend()
            
            # Subplot 2: Aggregation statistics
            plt.subplot(2, 2, 3)
            stats_data = [
                f"Aggregation Method: {afo.aggregation_method}",
                f"Source FO Count: {len(afo.source_fo_ids)}",
                f"Total Energy Range: [{afo.total_energy_min:.1f}, {afo.total_energy_max:.1f}] kWh",
                f"Power RMSE: {afo.power_profile_rmse:.2f}",
                f"Power CV: {afo.power_profile_cv:.2f}",
                f"Time Slice Count: {afo.slice_count}"
            ]
            
            plt.text(0.1, 0.9, '\n'.join(stats_data), 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.axis('off')
            plt.title('Aggregation Statistics')
            
            # Subplot 3: Source FO distribution
            plt.subplot(2, 2, 4)
            # Simple pie chart showing device type distribution
            device_types = [fo_id.split('_')[0] for fo_id in afo.source_fo_ids]
            type_counts = {}
            for dtype in device_types:
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            if type_counts:
                plt.pie(list(type_counts.values()), labels=list(type_counts.keys()), autopct='%1.1f%%')
                plt.title('Source FlexOffer Device Type Distribution')
            
            plt.suptitle(f'Manager {self.manager_id} - Aggregation Result {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f'manager_{self.manager_id}_afo_{i+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Aggregation result figure saved to: {save_path}")
            else:
                plt.show()
            plt.close()
    
    @classmethod
    def load_from_data(cls, manager_id: str, location: Tuple[float, float], coverage_area: float, 
                   num_users: int, data_dir: str = "../data", aggregation_method: str = "DP") -> 'Manager':
        """Load Manager from data files"""
        manager = cls(manager_id, location, coverage_area, aggregation_method=aggregation_method)
        
        # Load users and devices
        for i in range(num_users):
            user_id = f"user_{manager_id}_{i}"
            
            # Random location (near manager location)
            x_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            y_offset = random.uniform(-1, 1) * np.sqrt(coverage_area) / 2
            user_location = (location[0] + x_offset, location[1] + y_offset)
            
            # Random user type
            user_type = random.choice(["prosumer", "consumer", "producer"])
            
            # Create user
            user = User(user_id, user_type, user_location)
            
            # Random number of devices (3-5)
            num_devices = random.randint(3, 5)
            
            # Create devices
            device_types = ["battery", "heat_pump", "uncertain"]
            for j in range(num_devices):
                device_type = random.choice(device_types)
                device_id = f"device_{user_id}_{j}"
                
                if device_type == "battery":
                    # Load battery parameters from CSV file
                    try:
                        battery_model = BatteryModel.from_csv(
                            os.path.join(data_dir, "battery_base_parameters.csv"),
                            os.path.join(data_dir, "battery_dfo_input.csv"),
                            "BAT001"  # Randomly select a battery ID, could be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, battery_model.params, battery_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"Failed to load battery device {device_id}: {e}")
                        
                elif device_type == "heat_pump":
                    # Load heat pump parameters from CSV file
                    try:
                        heat_pump_model = HeatPumpModel.from_csv(
                            os.path.join(data_dir, "heat_pump_system.csv"),
                            "1-1-101-LR"  # Randomly select a room ID, could be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, heat_pump_model.params, heat_pump_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"Failed to load heat pump device {device_id}: {e}")
                        
                elif device_type == "uncertain":
                    # Load uncertain parameters from CSV file
                    try:
                        uncertain_model = UncertainModel.from_csv(
                            os.path.join(data_dir, "uncertain_energy_data.csv"),
                            "PV Generation"  # Randomly select an energy type, could be improved to randomly select from file
                        )
                        device = Device(device_id, device_type, uncertain_model.params_list, uncertain_model)
                        user.add_device(device)
                    except Exception as e:
                        logger.error(f"Failed to load uncertain device {device_id}: {e}")
            
            # Add user to manager
            manager.add_user(user)
        
        return manager

@dataclass
class City:
    """City class, manages multiple Managers"""
    city_name: str                           # City name
    width: float = 10.0                      # City width (kilometers)
    height: float = 10.0                     # City height (kilometers)
    managers: List[Manager] = field(default_factory=list)  # Manager list
    
    def add_manager(self, manager: Manager):
        """Add Manager"""
        self.managers.append(manager)
        
    def generate_managers(self, num_managers: int = 10, users_per_manager: int = 20, 
                        coverage_area: float = 2.0, data_dir: str = "../data", 
                        aggregation_method: str = "DP"):
        """Generate specified number of Managers"""
        for i in range(num_managers):
            # Random location
            location = (random.uniform(0, self.width), random.uniform(0, self.height))
            manager_id = f"manager_{i}"
            
            # Create Manager
            manager = Manager.load_from_data(
                manager_id, location, coverage_area, users_per_manager, 
                data_dir, aggregation_method
            )
            self.add_manager(manager)
            
        logger.info(f"City {self.city_name} generated {num_managers} Managers")
    
    def generate_all_flex_offers(self, time_horizon: int = 24, base_time: Optional[datetime] = None):
        """Generate FlexOffers for all Managers"""
        for manager in self.managers:
            manager.generate_flex_offers(time_horizon, base_time)
    
    def aggregate_all(self):
        """Aggregate FlexOffers for all Managers"""
        for manager in self.managers:
            manager.aggregate_flex_offers()
    
    def visualize_city(self, save_dir: Optional[str] = None):
        """Visualize Manager distribution and aggregation results for the entire city"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # City distribution figure
        plt.figure(figsize=(12, 8))
        
        for manager in self.managers:
            x, y = manager.location
            # Draw Manager location
            plt.scatter(x, y, s=100, c='red', marker='s', alpha=0.7)
            plt.text(x+0.1, y+0.1, manager.manager_id, fontsize=8)
            
            # Draw coverage area
            circle = Circle((x, y), np.sqrt(manager.coverage_area/np.pi), 
                          fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # Draw user locations
            for user in manager.users:
                ux, uy = user.location
                plt.scatter(ux, uy, s=20, c='blue', alpha=0.6)
        
        plt.xlim(-1, self.width+1)
        plt.ylim(-1, self.height+1)
        plt.xlabel('Distance (km)')
        plt.ylabel('Distance (km)')
        plt.title(f'City {self.city_name} - Manager and User Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend(['Manager', 'User'], loc='upper right')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'city_{self.city_name}_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Generate detailed aggregation result figures for each Manager
        for manager in self.managers:
            manager_save_dir = os.path.join(save_dir, manager.manager_id) if save_dir else None
            manager.visualize_aggregated_results(manager_save_dir) 
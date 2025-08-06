import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import sys
import time

# handle import method
try:
    # try to import as part of the package
    from .config import PipelineConfig, ModelBasedConfig
    from .model_based_controller import ModelBasedController, DeviceModel, BatteryModel, HeatPumpModel
except (ImportError, SystemError):
    # import method when running script directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import PipelineConfig, ModelBasedConfig
    from model_based_controller import ModelBasedController, DeviceModel, BatteryModel, HeatPumpModel

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelBasedPipeline')


class FlexOffer:
    """FlexOffer class, represents a flexible offer"""
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
        """create FlexOffer from dictionary"""
        return cls(
            device_id=fo_dict.get('device_id'),
            device_type=fo_dict.get('device_type'),
            energy_profile=fo_dict.get('energy_profile', [0.0]),
            time_flexibility=fo_dict.get('time_flexibility', 0),
            manager_id=fo_dict.get('manager_id')
        )
    
    def to_dict(self) -> Dict:
        """convert to dictionary"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'energy_profile': self.energy_profile,
            'time_flexibility': self.time_flexibility,
            'manager_id': self.manager_id,
            'time_horizon': self.time_horizon
        }


class Manager:
    """Manager class, manages multiple devices"""
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.devices = {}  # device_id -> device_config
    
    def add_device(self, device_id: str, device_config: Dict):
        """add device"""
        self.devices[device_id] = device_config
    
    def get_device_ids(self) -> List[str]:
        """get all device IDs"""
        return list(self.devices.keys()) 


class ModelBasedPipeline:
    """model-based FlexOffer Pipeline"""
    
    def __init__(self, config: PipelineConfig):
        """initialize Pipeline"""
        self.config = config
        self.time_horizon = config.time_horizon
        self.time_step = config.time_step
        self.aggregation_method = config.aggregation_method
        self.trading_method = config.trading_method
        self.disaggregation_method = config.disaggregation_method
        
        if config.seed is not None:
            np.random.seed(config.seed)
            import random
            random.seed(config.seed)
            logger.info(f"set random seed: {config.seed}")
        
        # create results directory
        os.makedirs(config.results_dir, exist_ok=True)
        
        # generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # load device config and price data
        self._load_device_config()
        self._load_price_data()
        
        # initialize Manager and model controllers
        self._setup_managers()
        self._setup_model_controllers()
        
        # initialize result storage
        self.results = {
            'manager_rewards': {},
            'timestep_details': [],
            'total_rewards': [],
            'aggregated_fo_count': [],  # aggregated FO count per time step
            'traded_fo_count': [],      # traded FO count per time step
            'traded_fo_value': [],      # traded FO total value per time step
            'disaggregate_count': []    # disaggregated FO count per time step
        }
        
        logger.info(f"ModelBasedPipeline initialized, experiment ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """generate experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        managers_count = self.config.num_managers
        users_count = sum(self.config.users_per_manager)
        
        seed_str = f"_seed{self.config.seed}" if self.config.seed is not None else ""
        
        return f"MODELBASED_m{managers_count}_u{users_count}{seed_str}_{timestamp}" 

    def _load_device_config(self):
        """load device config"""
        try:
            self.device_config = pd.read_csv(self.config.device_config_file)
            logger.info(f"loaded {len(self.device_config)} device configs")
        except Exception as e:
            logger.warning(f"failed to load device configs: {e}, using default device configs")
            # create default device configs
            self.device_config = self._create_default_device_config()
    
    def _create_default_device_config(self) -> pd.DataFrame:
        """create default device configs"""
        device_data = []
        
        # total users
        total_users = sum(self.config.users_per_manager)
        
        # create devices for each user
        for user_idx in range(total_users):
            # battery device
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
            
            # heat pump device
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
            
            # electric vehicle device
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
        """load price data"""
        try:
            self.price_data = pd.read_csv(self.config.price_data_file)
            logger.info(f"loaded {len(self.price_data)} price data")
        except Exception as e:
            logger.warning(f"failed to load price data: {e}, using default price data")
            self.price_data = self._create_default_price_data()
    
    def _create_default_price_data(self) -> pd.DataFrame:
        """create default price data"""
        hours = list(range(24))
        
        prices = []
        for hour in hours:
            if 0 <= hour < 6:  
                prices.append(0.05)  
            elif 6 <= hour < 9:  
                prices.append(0.15)  
            elif 9 <= hour < 17:  
                prices.append(0.12)  
            elif 17 <= hour < 21:  
                prices.append(0.18)  
            else:  
                prices.append(0.08)  
        
        return pd.DataFrame({
            'hour': hours,
            'price': prices
        }) 

    def _setup_managers(self):
        """setup Manager"""
        self.managers = {}
        
        # create Manager and assign devices
        for manager_idx in range(self.config.num_managers):
            manager_id = f"manager_{manager_idx+1}"
            self.managers[manager_id] = Manager(manager_id)
            
            manager_filter = f"user_manager_{manager_idx+1}"
            manager_devices = self.device_config[self.device_config['user_id'].str.startswith(manager_filter)]
            
            # record users under this manager
            unique_users = manager_devices['user_id'].unique()
            
            # add each device to Manager
            for _, device_config in manager_devices.iterrows():
                device_id = device_config['device_id']
                self.managers[manager_id].add_device(device_id, device_config.to_dict())
            
            logger.info(f"created Manager {manager_id}, managing {len(self.managers[manager_id].devices)} devices, {len(unique_users)} users")
        
        logger.info(f"created {len(self.managers)} Managers")
    
    def _setup_model_controllers(self):
        """setup model controllers"""
        self.model_controllers = {}
        
        # create a controller for each Manager
        for manager_id, manager in self.managers.items():
            # create controller
            controller = ModelBasedController(
                manager_id=manager_id,
                time_horizon=self.time_horizon,
                time_step=self.time_step,
                config=self.config.model_config
            )
            
            # add device models to controller
            for device_id, device_config in manager.devices.items():
                device_type = str(device_config['device_type'])
                # convert device config to device model parameters
                device_params = self._convert_device_config(device_config)
                # add to controller
                controller.add_device_model(device_id, device_type, device_params)
            
            self.model_controllers[manager_id] = controller
            logger.info(f"created model controller for Manager {manager_id}, configured {len(manager.devices)} devices")
    
    def _convert_device_config(self, device_config: Dict) -> Dict:
        """convert device config to device model parameters"""
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
                'outdoor_temp': 5.0,  
                'thermal_mass': 5000.0,  
                'heat_transfer_coeff': 100.0  
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
            # for other device types, keep original config
            device_params = device_config.copy()
        
        return device_params 

    def generate_flexoffers(self, timestep: int) -> Dict[str, Dict[str, Any]]:
        fo_systems = {}  # manager_id -> {device_id: fo_object}
        
        prices = self._get_prices_for_horizon(timestep)
        
        for manager_id, manager in self.managers.items():
            fo_systems[manager_id] = {}
            
            if manager_id in self.model_controllers:
                controller = self.model_controllers[manager_id]
                
                fo_dict = controller.generate_flex_offers(prices)
                
                for device_id, fo_data in fo_dict.items():
                    if device_id in manager.devices:
                        device_config = manager.devices[device_id]
                        device_type = str(device_config['device_type'])
                        
                        energy_profile = fo_data.get('energy_profile', [0.0] * self.time_horizon)
                        time_flexibility = fo_data.get('time_flexibility', 1)
                        
                        fo = FlexOffer(
                            device_id=device_id,
                            device_type=device_type,
                            energy_profile=energy_profile,
                            time_flexibility=time_flexibility,
                            manager_id=manager_id
                        )
                        
                        fo_systems[manager_id][device_id] = fo
        
        total_fo_count = sum(len(devices) for devices in fo_systems.values())
        logger.info(f"generated {total_fo_count} FlexOffers for time step {timestep}")
        
        return fo_systems
    
    def _get_prices_for_horizon(self, start_timestep):
        """get prices for time horizon starting from start_timestep"""
        prices = []
        
        for t in range(self.time_horizon):
            hour = (start_timestep + t) % 24
            hour_price = self.price_data[self.price_data['hour'] == hour]['price_usd_kwh'].values
            
            if len(hour_price) > 0:
                prices.append(hour_price[0])
            else:
                if 0 <= hour < 6:
                    prices.append(0.05)  
                elif 17 <= hour < 21:
                    prices.append(0.18)  
                else:
                    prices.append(0.12)  
        
        return prices
    
    def aggregate_flexoffers(self, fo_systems, timestep):
        """aggregate FlexOffers"""
        aggregated_results = {}
        
        # aggregate FlexOffers for each Manager
        for manager_id, devices in fo_systems.items():
            if not devices:
                continue
                
            # collect all FOs for this Manager
            flexoffers = list(devices.values())
            
            # aggregate FlexOffers based on aggregation method
            if self.aggregation_method == "LP":
                aggregated_fos = self._aggregate_lp(flexoffers, manager_id)
            else:
                aggregated_fos = self._aggregate_dp(flexoffers, manager_id)
            
            # save aggregated results
            aggregated_results[manager_id] = {
                'aggregated_fos': aggregated_fos,  
                'original_fos': flexoffers,
                'timestep': timestep
            }
        
        # calculate total aggregated FO count
        total_aggregated_fos = sum(len(result.get('aggregated_fos', [])) for result in aggregated_results.values())
        logger.info(f"aggregated {total_aggregated_fos} FlexOffers for time step {timestep}")
        return aggregated_results 

    def _aggregate_lp(self, flexoffers, manager_id):
        # if no FO, return empty list
        if not flexoffers:
            return []
        
        # energy limit (KWh)
        ENERGY_LIMIT = 100.0
        
        # calculate profile size (non-zero energy time steps) for each FlexOffer
        profile_sizes = {}
        for i, fo in enumerate(flexoffers):
            # calculate non-zero energy time steps
            non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
            profile_sizes[i] = non_zero_count
        
        remaining_fos = list(flexoffers)
        
        # store final aggregated results
        aggregated_fos = []
        
        # process remaining FlexOffers until done
        while remaining_fos:
            # find FlexOffers with maximum profile size
            profile_sizes = {}
            for i, fo in enumerate(remaining_fos):
                non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
                profile_sizes[i] = non_zero_count
                
            max_size = max(profile_sizes.values(), default=0)
            if max_size == 0:
                break  # no valid FOs
                
            max_size_fos = [i for i, size in profile_sizes.items() if size == max_size]
            
            initial_fo_idx = max(max_size_fos, key=lambda i: remaining_fos[i].time_flexibility)
            initial_fo = remaining_fos[initial_fo_idx]
            
            # calculate aggregated energy profile
            time_horizon = initial_fo.time_horizon
            agg_profile = np.array(initial_fo.energy_profile)
            
            # calculate initial RMSE and CV
            rmse = 0.0  # initial RMSE is 0
            cv = np.std(agg_profile) / (np.mean(abs(agg_profile)) + 1e-10)  # coefficient of variation
            
            # remove selected FO from remaining_fos
            selected_fos = [initial_fo]
            remaining_fos.pop(initial_fo_idx)
            
            # calculate current energy total
            current_energy = sum(abs(e) for e in agg_profile)
            
            # iterate to add other FlexOffers, until energy limit is reached or all FOs are processed
            i = 0
            while i < len(remaining_fos):
                fo = remaining_fos[i]
                
                # calculate current FO energy total
                fo_energy = sum(abs(e) for e in fo.energy_profile)
                
                # check if energy limit is exceeded
                if current_energy + fo_energy > ENERGY_LIMIT:
                    i += 1  # try next FO
                    continue
                
                # calculate binary aggregation
                temp_profile = agg_profile + np.array(fo.energy_profile)
                
                # calculate new RMSE and CV
                mean_profile = np.mean(abs(temp_profile))
                new_cv = np.std(temp_profile) / (mean_profile + 1e-10)
                
                # calculate RMSE between original FOs and new aggregated FO
                new_rmse = 0.0
                for orig_fo in selected_fos + [fo]:
                    # calculate target profile based on energy ratio
                    orig_weight = np.sum(abs(np.array(orig_fo.energy_profile))) / (np.sum(abs(temp_profile)) + 1e-10)
                    target_profile = temp_profile * orig_weight
                    
                    # calculate RMSE between target profile and original profile
                    error = np.mean((target_profile - np.array(orig_fo.energy_profile)) ** 2)
                    new_rmse += error
                
                # if new aggregation improves RMSE and CV, accept it
                if new_rmse <= rmse or new_cv < cv:
                    agg_profile = temp_profile
                    rmse = new_rmse
                    cv = new_cv
                    selected_fos.append(fo)
                    current_energy += fo_energy
                    remaining_fos.pop(i)  # remove added FO
                else:
                    i += 1  # try next FO
            
            # calculate time flexibility - use weighted average of selected FOs
            total_energy = sum(abs(np.sum(fo.energy_profile)) for fo in selected_fos)
            if total_energy > 0:
                time_flexibility = sum(fo.time_flexibility * abs(np.sum(fo.energy_profile)) 
                                    for fo in selected_fos) / total_energy
            else:
                time_flexibility = 0
            
            # create aggregated FlexOffer
            aggregated_fo = FlexOffer(
                device_id=f"aggregated_{manager_id}_{len(aggregated_fos)}",
                device_type="AGGREGATED",
                energy_profile=agg_profile.tolist(),
                time_flexibility=int(time_flexibility),
                manager_id=manager_id
            )
            
            # add to aggregated results list
            aggregated_fos.append(aggregated_fo)
            
            logger.info(f"created aggregated FO: {aggregated_fo.device_id}, containing {len(selected_fos)} FOs, total energy: {current_energy:.2f}kWh")
        
        logger.info(f"Manager {manager_id}: created {len(aggregated_fos)} aggregated FOs")
        
        # if no aggregated results, return empty list
        if not aggregated_fos:
            return []
            
        return aggregated_fos
    
    def _aggregate_dp(self, flexoffers, manager_id):
        # if no FO, return empty list
        if not flexoffers:
            return []
        
        # energy limit (KWh)
        ENERGY_LIMIT = 100.0
            
        # calculate profile size for each FlexOffer
        profile_sizes = {}
        for i, fo in enumerate(flexoffers):
            non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
            profile_sizes[i] = non_zero_count
        
        # calculate quartiles
        sizes = list(profile_sizes.values())
        q1 = np.percentile(sizes, 25) if sizes else 0
        q3 = np.percentile(sizes, 75) if sizes else 0
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        # filter out outliers
        filtered_fos = [flexoffers[i] for i, size in profile_sizes.items() if size <= upper_fence]
        
        # if no FO after filtering, use original list
        if not filtered_fos:
            filtered_fos = list(flexoffers)
            
        remaining_fos = list(filtered_fos)
        
        # store final aggregated results
        aggregated_fos = []
        
        # process remaining FlexOffers until done
        while remaining_fos:
            # re-calculate profile sizes
            profile_sizes = {}
            for i, fo in enumerate(remaining_fos):
                non_zero_count = sum(1 for e in fo.energy_profile if abs(e) > 1e-6)
                profile_sizes[i] = non_zero_count
                
            if not profile_sizes:
                break  # no valid FOs
                
            # select FO with maximum profile size
            max_size = max(profile_sizes.values(), default=0)
            if max_size == 0:
                break
                
            max_size_fos = [i for i, size in profile_sizes.items() if size == max_size]
            
            # select FO with highest time flexibility from FOs with maximum profile size
            initial_fo_idx = max(max_size_fos, key=lambda i: remaining_fos[i].time_flexibility)
            initial_fo = remaining_fos[initial_fo_idx]
            
            # initialize aggregated profile
            agg_profile = np.array(initial_fo.energy_profile)
            
            # calculate initial RMSE and CV
            rmse = 0.0
            cv = np.std(agg_profile) / (np.mean(abs(agg_profile)) + 1e-10)
            
            # remove selected FO from remaining_fos
            selected_fos = [initial_fo]
            remaining_fos.pop(initial_fo_idx)
            
            # calculate current energy total
            current_energy = sum(abs(e) for e in agg_profile)
            
            # iterate to add other FlexOffers, until energy limit is reached or all FOs are processed
            i = 0
            while i < len(remaining_fos):
                fo = remaining_fos[i]
                
                # calculate current FO energy total
                fo_energy = sum(abs(e) for e in fo.energy_profile)
                
                # check if energy limit is exceeded
                if current_energy + fo_energy > ENERGY_LIMIT:
                    i += 1  # try next FO
                    continue
                
                # calculate binary aggregation
                temp_profile = agg_profile + np.array(fo.energy_profile)
                
                # calculate new CV and RMSE
                mean_profile = np.mean(abs(temp_profile))
                new_cv = np.std(temp_profile) / (mean_profile + 1e-10)
                
                # calculate RMSE between original FOs and new aggregated FO
                new_rmse = 0.0
                for orig_fo in selected_fos + [fo]:
                    orig_weight = np.sum(abs(np.array(orig_fo.energy_profile))) / (np.sum(abs(temp_profile)) + 1e-10)
                    target_profile = temp_profile * orig_weight
                    
                    error = np.mean((target_profile - np.array(orig_fo.energy_profile)) ** 2)
                    new_rmse += error
                
                # if new aggregation improves metrics, accept it
                if new_rmse <= rmse or new_cv < cv:
                    agg_profile = temp_profile
                    rmse = new_rmse
                    cv = new_cv
                    selected_fos.append(fo)
                    current_energy += fo_energy
                    remaining_fos.pop(i)  
                else:
                    i += 1 
            
            # calculate time flexibility
            total_energy = sum(abs(np.sum(fo.energy_profile)) for fo in selected_fos)
            if total_energy > 0:
                time_flexibility = sum(fo.time_flexibility * abs(np.sum(fo.energy_profile))
                                    for fo in selected_fos) / total_energy
            else:
                time_flexibility = 0
            
            # create aggregated FlexOffer
            aggregated_fo = FlexOffer(
                device_id=f"aggregated_{manager_id}_{len(aggregated_fos)}",
                device_type="AGGREGATED",
                energy_profile=agg_profile.tolist(),
                time_flexibility=int(time_flexibility),
                manager_id=manager_id
            )
            
            # add to aggregated results list
            aggregated_fos.append(aggregated_fo)
            
            logger.info(f"created aggregated FO: {aggregated_fo.device_id}, containing {len(selected_fos)} FOs, total energy: {current_energy:.2f}kWh")
        
        logger.info(f"Manager {manager_id}: created {len(aggregated_fos)} aggregated FOs")
        
        # if no aggregated results, return empty list
        if not aggregated_fos:
            return []
            
        return aggregated_fos
    
    def trade_flexoffers(self, aggregated_results, timestep):
        """trade FlexOffers"""
        trading_results = {}
        prices = self._get_prices_for_horizon(timestep)
        
        # track traded FO IDs
        traded_fo_ids = []
        
        # trade aggregated FOs for each Manager
        for manager_id, agg_data in aggregated_results.items():
            aggregated_fos = agg_data.get('aggregated_fos', [])
            
            if not aggregated_fos:
                continue
            
            # trade each aggregated FO
            for aggregated_fo in aggregated_fos:
                # trade aggregated FO based on trading method
                if self.trading_method == "bidding":
                    # bidding method
                    schedule, revenue = self._trade_bidding(aggregated_fo, prices)
                else:
                    # market clearing method
                    schedule, revenue = self._trade_market_clearing(aggregated_fo, prices)
                
                fo_id = aggregated_fo.device_id
                trading_results[fo_id] = {
                    'schedule': schedule,
                    'revenue': revenue,
                    'original_fo': aggregated_fo,
                    'manager_id': manager_id,
                    'timestep': timestep
                }
                
                # record traded FO ID
                traded_fo_ids.append(fo_id)
        
        logger.info(f"time step {timestep} trading completed: {len(traded_fo_ids)} traded FOs")
        return trading_results
    
    def _trade_bidding(self, flexoffer, prices):
        energy_profile = np.array(flexoffer.energy_profile)
        time_flexibility = flexoffer.time_flexibility
        time_horizon = len(energy_profile)
        
        energy_consumption = np.where(energy_profile > 0, energy_profile, 0)
        energy_production = np.where(energy_profile < 0, -energy_profile, 0)

        market_adj = 0.05  # market adjustment factor
        random_factor = 0.015  # random factor range
        bias = 0.02  # preference factor
        
        bid_volumes = energy_consumption  
        ask_volumes = energy_production   
        
        # calculate bid prices
        random_values = np.random.uniform(-random_factor, random_factor, time_horizon)
        
        bid_prices = np.array(prices) * (1 + market_adj + random_values + bias)
        ask_prices = np.array(prices) * (1 - market_adj + random_values - bias)
        
        max_revenue = float('-inf')
        best_schedule = energy_profile.copy()
        
        for shift in range(time_flexibility + 1):
            shifted_consumption = np.roll(energy_consumption, shift)
            shifted_production = np.roll(energy_production, shift)
            
            sell_income = sum(shifted_production * prices)
            buy_cost = sum(shifted_consumption * prices)
            trade_factor = 4.0
            revenue = trade_factor * (buy_cost-sell_income)
            
            if revenue > max_revenue:
                max_revenue = revenue
                best_schedule = np.roll(energy_profile, shift)
        
        return best_schedule.tolist(), max_revenue
    
    def _trade_market_clearing(self, flexoffer, prices):
        energy_profile = np.array(flexoffer.energy_profile)
        time_flexibility = flexoffer.time_flexibility
        time_horizon = len(energy_profile)
        
        consumer_surplus_weight = 0.5  
        producer_surplus_weight = 0.5  
        
        sorted_price_indices = np.argsort(prices)  
        
        low_price_indices = sorted_price_indices[:time_horizon//3]  
        high_price_indices = sorted_price_indices[-time_horizon//3:]  
        
        schedule = energy_profile.copy()
        
        consumer_surplus = 0
        producer_surplus = 0
        
        consumption_volume = sum(energy_consumption for energy_consumption in energy_profile if energy_consumption > 0)
        production_volume = sum(-energy_production for energy_production in energy_profile if energy_production < 0)
        
        optimized_schedule = np.zeros(time_horizon)
        
        remaining_production = production_volume
        for idx in reversed(high_price_indices):  
            if remaining_production <= 0:
                break
            
            max_production_at_step = min(remaining_production, 10.0)  
            optimized_schedule[idx] = -max_production_at_step  
            remaining_production -= max_production_at_step
        
        remaining_consumption = consumption_volume
        for idx in low_price_indices:  
            if remaining_consumption <= 0:
                break
            
            max_consumption_at_step = min(remaining_consumption, 10.0)  
            optimized_schedule[idx] = max_consumption_at_step  
            remaining_consumption -= max_consumption_at_step
        
        new_consumer_surplus = sum(max(0, 0.2 - prices[i]) * optimized_schedule[i] 
                              for i in range(time_horizon) if optimized_schedule[i] > 0)
        new_producer_surplus = sum(max(0, prices[i] - 0.05) * (-optimized_schedule[i]) 
                              for i in range(time_horizon) if optimized_schedule[i] < 0)
        new_social_welfare = consumer_surplus_weight * new_consumer_surplus + producer_surplus_weight * new_producer_surplus
        
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
        """disaggregate schedules"""
        disaggregated_results = {}
        
        # organize trading results by manager_id
        trading_by_manager = {}
        for fo_id, trade_data in trading_results.items():
            manager_id = trade_data.get('manager_id')
            if manager_id not in trading_by_manager:
                trading_by_manager[manager_id] = []
            trading_by_manager[manager_id].append(trade_data)
        
        # disaggregate schedules for each Manager
        for manager_id, trade_data_list in trading_by_manager.items():
            # get original FOs for this Manager
            if manager_id not in aggregated_results:
                continue
                
            agg_data = aggregated_results[manager_id]
            original_fos = agg_data.get('original_fos', [])
            
            if not original_fos:
                continue
            
            # store all device schedules for this Manager
            all_device_schedules = {}
            total_revenue = 0.0
            
            # disaggregate each trade result
            for trade_data in trade_data_list:
                schedule = trade_data.get('schedule', [])
                revenue = trade_data.get('revenue', 0.0)
                aggregated_fo = trade_data.get('original_fo')
                
                if not aggregated_fo or not schedule:
                    continue
                
                # disaggregate based on method
                if self.disaggregation_method == "proportional":
                    # proportional disaggregation
                    device_schedules = self._disaggregate_proportional(schedule, aggregated_fo, original_fos)
                else:
                    # average disaggregation
                    device_schedules = self._disaggregate_average(schedule, aggregated_fo, original_fos)
                
                for device_id, device_data in device_schedules.items():
                    if device_id in all_device_schedules:
                        existing_data = all_device_schedules[device_id]
                        existing_schedule = existing_data.get('schedule', [])
                        
                        min_len = min(len(existing_schedule), len(device_data.get('schedule', [])))
                        
                        if min_len > 0:
                            merged_schedule = []
                            for i in range(min_len):
                                merged_schedule.append((existing_schedule[i] + device_data.get('schedule', [])[i]) / 2)
                            
                            # update schedule
                            existing_data['schedule'] = merged_schedule
                    else:
                        # if new device, add directly
                        all_device_schedules[device_id] = device_data
                
                # accumulate revenue
                total_revenue += revenue
            
            # save disaggregated results for this Manager
            disaggregated_results[manager_id] = {
                'device_schedules': all_device_schedules,
                'revenue': total_revenue
            }
        
        return disaggregated_results
    
    def _disaggregate_proportional(self, schedule, aggregated_fo, original_fos):
        device_schedules = {}
        
        if not original_fos:
            return device_schedules
            
        time_horizon = len(schedule)
        schedule_array = np.array(schedule)
        
        total_energy_needs = {}  
        for t in range(time_horizon):
            consumption = sum(max(0, fo.energy_profile[t]) for fo in original_fos)
            production = sum(abs(min(0, fo.energy_profile[t])) for fo in original_fos)
            total_energy_needs[t] = {'consumption': consumption, 'production': production}
        
        for fo in original_fos:
            device_id = fo.device_id
            orig_profile = np.array(fo.energy_profile)
            
            device_schedule = np.zeros(time_horizon)
            for t in range(time_horizon):
                if schedule_array[t] > 0:
                    if total_energy_needs[t]['consumption'] > 0:
                        weight = max(0, orig_profile[t]) / total_energy_needs[t]['consumption']
                        device_schedule[t] = schedule_array[t] * weight
                elif schedule_array[t] < 0:
                    if total_energy_needs[t]['production'] > 0:
                        weight = abs(min(0, orig_profile[t])) / total_energy_needs[t]['production']
                        device_schedule[t] = schedule_array[t] * weight
            
            device_schedules[device_id] = {
                'schedule': device_schedule.tolist(),
                'original_fo': fo.to_dict()
            }
        
        return device_schedules
    
    def _disaggregate_average(self, schedule, aggregated_fo, original_fos):
        device_schedules = {}
        
        if not original_fos:
            return device_schedules
            
        time_horizon = len(schedule)
        schedule_array = np.array(schedule)
        
        device_types = {}
        for fo in original_fos:
            device_type = fo.device_type
            if device_type not in device_types:
                device_types[device_type] = []
            device_types[device_type].append(fo)
        
        for device_type, fos in device_types.items():
            num_devices = len(fos)
            for t in range(time_horizon):
                if schedule_array[t] > 0:  
                    type_consumption = sum(max(0, fo.energy_profile[t]) for fo in fos)
                    if type_consumption > 0:
                        type_ratio = type_consumption / sum(max(0, fo.energy_profile[t]) for fo in original_fos)
                        type_energy = schedule_array[t] * type_ratio
                        device_energy = type_energy / num_devices
                    else:
                        device_energy = 0
                elif schedule_array[t] < 0:  
                    type_production = sum(abs(min(0, fo.energy_profile[t])) for fo in fos)
                    if type_production > 0:
                        type_ratio = type_production / sum(abs(min(0, fo.energy_profile[t])) for fo in original_fos)
                        type_energy = schedule_array[t] * type_ratio
                        device_energy = type_energy / num_devices
                    else:
                        device_energy = 0
                else:
                    device_energy = 0
                
                # update schedule for each device
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
        """calculate rewards"""
        rewards = {}
        
        for manager_id, disagg_data in disaggregated_results.items():
            device_schedules = disagg_data.get('device_schedules', {})
            revenue = disagg_data.get('revenue', 0.0)
            
            if not device_schedules:
                rewards[manager_id] = 0.0
                continue
            
            # get original energy profiles
            original_profiles = {}
            for device_id, device_data in device_schedules.items():
                original_fo = device_data.get('original_fo', {})
                original_profiles[device_id] = original_fo.get('energy_profile', [0.0])
            
            # get schedules
            schedules = {}
            for device_id, device_data in device_schedules.items():
                schedules[device_id] = device_data.get('schedule', [0.0])
            
            # calculate rewards using model controllers
            if manager_id in self.model_controllers:
                controller = self.model_controllers[manager_id]
                reward = controller.calculate_reward(schedules, revenue, original_profiles)
                rewards[manager_id] = reward
            else:
                # if no corresponding controller, use default reward calculation
                reward = self._calculate_default_reward(schedules, revenue, original_profiles)
                rewards[manager_id] = reward
        
        return rewards
    
    def _calculate_default_reward(self, schedules, revenue, original_profiles):
        satisfaction = 0.0
        profile_count = 0
        
        for device_id, schedule in schedules.items():
            if device_id in original_profiles:
                original = original_profiles[device_id]
                min_len = min(len(schedule), len(original))
                
                if min_len > 0:
                    schedule_np = np.array(schedule[:min_len])
                    original_np = np.array(original[:min_len])
                    
                    # avoid division by zero
                    total_energy = np.sum(np.abs(original_np))
                    if total_energy > 0:
                        error = np.sum(np.abs(schedule_np - original_np)) / total_energy
                        similarity = max(0, 1 - error)  # convert to similarity
                    else:
                        similarity = 1.0  # if original energy is 0, consider it as fully satisfied
                    
                    satisfaction += similarity
                    profile_count += 1
        
        # calculate average satisfaction
        avg_satisfaction = satisfaction / max(1, profile_count)
        
        # normalize revenue (assume maximum possible revenue is device count * 10)
        max_possible_revenue = len(schedules) * 10
        normalized_revenue = min(1.0, revenue / max(0.1, max_possible_revenue))
        
        satisfaction_weight = 0.7
        revenue_weight = 0.3
        
        reward = (satisfaction_weight * avg_satisfaction + revenue_weight * normalized_revenue) * 36.0
        
        return reward 

    def run(self, num_timesteps=1):
        """run pipeline"""
        logger.info(f"running ModelBased Pipeline, experiment ID: {self.experiment_id}, total time steps: {num_timesteps}")
        
        # initialize results
        total_rewards = []
        
        for timestep in range(num_timesteps):
            logger.info(f"==== time step {timestep} ====")
            
            # step 1: generate FlexOffers
            logger.info(f"step 1: generate FlexOffers...")
            fo_systems = self.generate_flexoffers(timestep)
            
            # step 2: aggregate FlexOffers
            logger.info(f"step 2: aggregate FlexOffers...")
            aggregated_results = self.aggregate_flexoffers(fo_systems, timestep)
            
            # collect aggregated FO count
            aggregated_fo_count = sum(len(result.get('aggregated_fos', [])) for result in aggregated_results.values())
            self.results['aggregated_fo_count'].append(aggregated_fo_count)
            logger.info(f"aggregated FO count: {aggregated_fo_count} (energy limit 100KWh)")
            
            # step 3: trade FlexOffers
            logger.info(f"step 3: trade FlexOffers...")
            trading_results = self.trade_flexoffers(aggregated_results, timestep)
            
            # collect traded FO count and total value
            traded_fo_count = len(trading_results)
            traded_fo_value = sum(result.get('revenue', 0.0) for result in trading_results.values())
            self.results['traded_fo_count'].append(traded_fo_count)
            self.results['traded_fo_value'].append(traded_fo_value)
            logger.info(f"traded FO count: {traded_fo_count}, traded FO total value: {traded_fo_value:.4f}")
            
            # step 4: disaggregate schedules
            logger.info(f"step 4: disaggregate schedules...")
            disaggregated_results = self.disaggregate_schedules(trading_results, aggregated_results)
            
            # collect disaggregate FO count - count all devices, not managers
            disaggregate_count = sum(len(result.get('device_schedules', {})) 
                                    for result in disaggregated_results.values())
            self.results['disaggregate_count'].append(disaggregate_count)
            logger.info(f"disaggregate FO count: {disaggregate_count} (disaggregated to each device)")
            
            # step 5: calculate rewards
            logger.info(f"step 5: calculate rewards...")
            rewards = self.calculate_rewards(disaggregated_results)
            
            # summarize time step results
            timestep_reward = sum(rewards.values())
            total_rewards.append(timestep_reward)
            
            # save time step results
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
            
            for manager_id, reward in rewards.items():
                if manager_id not in self.results['manager_rewards']:
                    self.results['manager_rewards'][manager_id] = []
                self.results['manager_rewards'][manager_id].append(reward)
            
            logger.info(f"time step {timestep} completed, total reward: {timestep_reward:.4f}")
        
        self.results['total_rewards'] = total_rewards
        
        final_reward = sum(total_rewards)
        logger.info(f"Pipeline completed, total reward: {final_reward:.4f}")
        
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """save results"""
        # create result directory
        result_dir = os.path.join(self.config.results_dir, self.experiment_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # save time step details
        timestep_df = pd.DataFrame(self.results['timestep_details'])
        timestep_file = os.path.join(result_dir, "timestep_details.csv")
        timestep_df.to_csv(timestep_file, index=False)
        logger.info(f"time step details saved to: {timestep_file}")
        
        # save manager rewards
        manager_rewards = {}
        for manager_id, rewards in self.results['manager_rewards'].items():
            manager_rewards[f"manager_{manager_id}_rewards"] = rewards
        
        manager_df = pd.DataFrame(manager_rewards)
        manager_file = os.path.join(result_dir, "manager_rewards.csv")
        manager_df.to_csv(manager_file, index=False)
        logger.info(f"manager rewards saved to: {manager_file}")
        
        # save total reward and trading metrics
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
        logger.info(f"metrics data saved to: {metrics_file}")
        
        # save config
        config_file = os.path.join(result_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"config saved to: {config_file}")
        
        # save statistics
        stats = self._get_statistics()
        stats_file = os.path.join(result_dir, "statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"statistics saved to: {stats_file}")
        
        return result_dir
    
    def _get_statistics(self):
        """get statistics"""
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
            # add new metrics
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
        
        # add device type statistics
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
    # load config
    try:
        from .config import load_config
    except (ImportError, SystemError):
        from config import load_config
        
    config = load_config(config_path)
    
    # apply command line arguments to override config
    if aggregation_method:
        config.aggregation_method = aggregation_method
    if trading_method:
        config.trading_method = trading_method
    if disaggregation_method:
        config.disaggregation_method = disaggregation_method
    
    if seed is not None:
        config.seed = seed
        print(f"using random seed: {seed}")
    
    # print selected algorithms
    print(f"using algorithm combination: {config.aggregation_method} + {config.trading_method} + {config.disaggregation_method}")
    
    # create and run pipeline
    pipeline = ModelBasedPipeline(config)
    results = pipeline.run(num_timesteps)
    
    return results


if __name__ == "__main__":
    import argparse
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description="run ModelBased FlexOffer Pipeline")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--timesteps", type=int, default=24, help="time steps")
    parser.add_argument("--aggregation", type=str, default="LP", choices=["LP", "DP"], help="aggregation method")
    parser.add_argument("--trading", type=str, default="bidding", choices=["bidding", "market-clearing"], help="trading method")
    parser.add_argument("--disaggregation", type=str, default="proportional", choices=["proportional", "average"], help="disaggregation method")
    parser.add_argument("--managers", type=int, default=4, help="number of managers")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="number of users per manager, comma separated")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducibility")
    
    args = parser.parse_args()
    
    # import dependencies
    try:
        from .config import load_config, PipelineConfig
    except (ImportError, SystemError):
        from config import load_config, PipelineConfig
    
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()
    
    # apply command line arguments
    config.aggregation_method = args.aggregation
    config.trading_method = args.trading
    config.disaggregation_method = args.disaggregation
    config.num_managers = args.managers
    
    # parse number of users
    try:
        config.users_per_manager = [int(n) for n in args.users.split(",")]
        if len(config.users_per_manager) < config.num_managers:
            default_users = [9] * (config.num_managers - len(config.users_per_manager))
            config.users_per_manager.extend(default_users)
    except:
        config.users_per_manager = [9] * config.num_managers
    
    # print config information
    print(f"running ModelBased Pipeline:")
    print(f"- aggregation method: {config.aggregation_method}")
    print(f"- trading method: {config.trading_method}")
    print(f"- disaggregation method: {config.disaggregation_method}")
    print(f"- number of managers: {config.num_managers}")
    print(f"- user distribution: {config.users_per_manager} (total {sum(config.users_per_manager)} users)")
    print(f"- time steps: {args.timesteps}")
    if args.seed is not None:
        print(f"- random seed: {args.seed}")
    
    results = run_pipeline(
        config_path=None, 
        num_timesteps=args.timesteps, 
        aggregation_method=config.aggregation_method,
        trading_method=config.trading_method, 
        disaggregation_method=config.disaggregation_method,
        save_results=True,
        seed=args.seed
    )
    
    # print total reward
    total_reward = sum(results.get('total_rewards', []))
    print(f"\ncompleted!")
    print(f"total reward: {total_reward:.4f}") 
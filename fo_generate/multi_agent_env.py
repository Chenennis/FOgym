import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import logging
import math
from abc import ABC, abstractmethod

from fo_generate.unified_mdp_env import (
    DeviceMDPInterface, BatteryMDPDevice, HeatPumpMDPDevice, 
    EVMDPDevice, PVMDPDevice, DishwasherMDPDevice, DeviceType, EnvironmentDynamics
)
from fo_generate.data_loader import DataLoader
from fo_generate.battery_model import BatteryModel, BatteryParameters
from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters
from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior
from fo_generate.pv_model import PVModel, PVParameters
from fo_generate.dishwasher_model import DishwasherModel, DishwasherParameters, DishwasherUserBehavior
from fo_generate.dfo import DFOSystem, DFOSlice
from fo_common.dec_pomdp_config import DecPOMDPConfig, DecPOMDPObservationSpace
from fo_common.dynamic_observation_quality import DynamicObservationQuality

logger = logging.getLogger(__name__)

class ManagerAgent:
    """Manager class, manage a group of users and devices"""
    
    def __init__(self, manager_id: str, manager_config: Dict, users: List[Dict], devices: List[Dict]):
        self.manager_id = manager_id
        self.config = manager_config
        self.users = users
        self.devices = devices
        
        # location and coverage information
        self.location = (manager_config['location_x'], manager_config['location_y'])
        self.coverage_area = manager_config['coverage_area']
        self.district_type = manager_config['district_type']
        
        # device MDP objects
        self.device_mdps: Dict[str, DeviceMDPInterface] = {}
        self.device_types: Dict[str, str] = {}
        self.controllable_devices: List[str] = []
        
        # aggregated user preferences
        self.aggregated_preferences = self._aggregate_user_preferences()
        
        # initialize devices
        self._initialize_devices()
        
        # Markov history
        self.markov_history = {
            'prev_actions': np.zeros(len(self.controllable_devices)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0,
            'user_satisfaction': 0.0
        }
    
    def _aggregate_user_preferences(self) -> Dict[str, float]:
        """aggregate user preferences"""
        if not self.users:
            return {'economic': 0.33, 'comfort': 0.33, 'environmental': 0.34}
        
        total_economic = sum(user.get('economic_pref', 0.33) for user in self.users)
        total_comfort = sum(user.get('comfort_pref', 0.33) for user in self.users)
        total_environmental = sum(user.get('environmental_pref', 0.34) for user in self.users)
        
        total = total_economic + total_comfort + total_environmental
        
        return {
            'economic': total_economic / total,
            'comfort': total_comfort / total,
            'environmental': total_environmental / total
        }
    
    def _initialize_devices(self):
        """initialize device MDP objects"""
        for device in self.devices:
            device_id = device['device_id']
            device_type = device['device_type']
            
            # create device model
            device_model = self._create_device_model(device_type, device)
            
            # create device MDP
            device_mdp = self._create_device_mdp(device_type, device_model)
            
            self.device_mdps[device_id] = device_mdp
            self.device_types[device_id] = device_type
            
            # record controllable devices
            if device_type not in [DeviceType.PV]:  # PV is not controllable
                self.controllable_devices.append(device_id)
        
        logger.info(f"Manager {self.manager_id}: initialize {len(self.device_mdps)} devices, "
                   f"among which {len(self.controllable_devices)} are controllable")
    
    def _create_device_model(self, device_type: str, device_config: Dict):
        """create device model"""
        if device_type == DeviceType.BATTERY:
            params = BatteryParameters(
                battery_id=device_config['device_id'],
                soc_min=device_config.get('param1', 0.1),
                soc_max=device_config.get('param2', 0.9),
                p_min=-device_config['max_power'],
                p_max=device_config['max_power'],
                efficiency=device_config['efficiency'],
                initial_soc=device_config['initial_state'],
                battery_type="lithium-ion",
                capacity_kwh=device_config['capacity']
            )
            return BatteryModel(params)
            
        elif device_type == DeviceType.HEAT_PUMP:
            params = HeatPumpParameters(
                room_id=device_config['device_id'],
                room_area=30.0,
                room_volume=75.0,
                temp_min=device_config.get('param1', 18.0),
                temp_max=device_config.get('param2', 26.0),
                initial_temp=device_config['initial_state'],
                cop=device_config['efficiency'],
                heat_loss_coef=device_config.get('param3', 0.1),
                primary_use_period="8:00-22:00",
                secondary_use_period="22:00-8:00",
                primary_target_temp=22.0,
                secondary_target_temp=19.0,
                max_power=device_config['max_power']
            )
            return HeatPumpModel(params)
            
        elif device_type == DeviceType.EV:
            params = EVParameters(
                ev_id=device_config['device_id'],
                battery_capacity=device_config['capacity'],
                soc_min=device_config.get('param1', 0.1),
                soc_max=device_config.get('param2', 0.95),
                max_charging_power=device_config['max_power'],
                efficiency=device_config['efficiency'],
                initial_soc=device_config['initial_state'],
                fast_charge_capable=True
            )
            
            # create user behavior
            now = datetime.now()
            connection_time = datetime(now.year, now.month, now.day, 18, 0)
            disconnection_time = datetime(now.year, now.month, now.day + 1, 7, 30)
            next_departure_time = datetime(now.year, now.month, now.day + 1, 8, 0)
            
            behavior = EVUserBehavior(
                ev_id=device_config['device_id'],
                connection_time=connection_time,
                disconnection_time=disconnection_time,
                next_departure_time=next_departure_time,
                target_soc=0.85,
                min_required_soc=0.6,
                fast_charge_preferred=False,
                location="home",
                priority=3,
                charge_flexibility=0.8
            )
            
            model = EVModel(params)
            model.user_behavior = behavior
            return model
            
        elif device_type == DeviceType.PV:
            params = PVParameters(
                pv_id=device_config['device_id'],
                max_power=device_config['max_power'],
                efficiency=device_config['efficiency'],
                area=device_config.get('param3', 25.0),
                location="roof",
                tilt_angle=device_config.get('param1', 30.0),
                azimuth_angle=device_config.get('param2', 180.0),
                weather_dependent=True,
                forecast_accuracy=0.8
            )
            return PVModel(params)
            
        elif device_type == DeviceType.DISHWASHER:
            params = DishwasherParameters(
                dishwasher_id=device_config['device_id'],
                total_energy=device_config.get('capacity', 3.0),  # total energy demand
                power_rating=device_config['max_power'],
                operation_hours=device_config.get('param1', 3.5),  # operation hours
                min_start_delay=device_config.get('param2', 0.5),  # minimum start delay
                max_start_delay=device_config.get('param3', 6.0),  # maximum start delay
                efficiency=device_config['efficiency'],
                can_interrupt=False  # dishwasher cannot be interrupted
            )
            
            # create user behavior
            now = datetime.now()
            deployment_time = now + timedelta(hours=np.random.uniform(0, 2))  # random deployment time
            
            behavior = DishwasherUserBehavior(
                dishwasher_id=device_config['device_id'],
                deployment_time=deployment_time,
                preferred_start_time=deployment_time + timedelta(hours=1),
                latest_completion_time=deployment_time + timedelta(hours=8),
                priority=3,
                user_tolerance=2.0
            )
            
            model = DishwasherModel(params, behavior)
            return model
        
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
    
    def _create_device_mdp(self, device_type: str, device_model) -> DeviceMDPInterface:
        """create device MDP"""
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
            raise ValueError(f"Unsupported device type: {device_type}")
    
    def get_state_features(self, standardized: bool = True) -> np.ndarray:
        if standardized:
            return self._get_standardized_state_features()
        else:
            return self._get_legacy_state_features()
    
    def _get_standardized_state_features(self) -> np.ndarray:
        """get standardized state features"""
        
        # 1. device features module (organized and standardized by type)
        device_features = self._get_standardized_device_features()
        
        # 2. manager management features module (standardized)
        management_features = self._get_standardized_management_features()
        
        # 3. user preference features module (standardized)
        preference_features = self._get_standardized_preference_features()
        
        # 4. system state features module (standardized)
        system_features = self._get_standardized_system_features()
        
        # merge all feature modules
        all_features = np.concatenate([
            device_features,      # device state (the most important features)
            management_features,  # management features
            preference_features,  # user preferences
            system_features      # system state
        ])
        
        return all_features.astype(np.float32)
    
    def _get_standardized_device_features(self) -> np.ndarray:
        """get standardized device features - efficient version"""
        # new design: aggregate features instead of fixed dimension filling
        device_aggregation = {
            DeviceType.BATTERY: [],
            DeviceType.HEAT_PUMP: [],
            DeviceType.EV: [],
            DeviceType.PV: [],
            DeviceType.DISHWASHER: []
        }
        
        # collect enhanced features of each type of device
        for device_id, mdp in self.device_mdps.items():
            device_type = self.device_types[device_id]
            enhanced_features = self._get_enhanced_device_features(device_id, device_type, mdp)
            if device_type in device_aggregation:
                device_aggregation[device_type].append(enhanced_features)
        
        # generate aggregated features instead of fixed dimension filling
        aggregated_features = []
        
        for device_type in sorted(device_aggregation.keys()):
            features_list = device_aggregation[device_type]
            if features_list:
                # aggregate features of the same type of device: [number, average, max, min, std]
                stacked = np.stack(features_list)
                aggregated = np.array([
                    float(len(features_list)),   # number of devices
                    float(np.mean(stacked)),     # average
                    float(np.max(stacked)),      # max
                    float(np.min(stacked)),      # min
                    float(np.std(stacked))       # std
                ])
            else:
                # no devices of this type
                aggregated = np.zeros(5)
            
            aggregated_features.append(aggregated)
        
        return np.concatenate(aggregated_features)
    
    def _get_enhanced_device_features(self, device_id: str, device_type: str, mdp) -> np.ndarray:
        """get enhanced device features - efficient version"""
        base_features = mdp.get_state_features()
        
        # fast standardized processing, reduce complex calculation
        if device_type == DeviceType.BATTERY:
            # battery features: [SOC, power status, health]
            soc = base_features[0] if len(base_features) > 0 else 0.5
            power_status = 1.0 if len(base_features) > 3 and abs(base_features[3]) > 0.1 else 0.0
            health = base_features[1] if len(base_features) > 1 else 1.0
            return np.array([soc, power_status, health])
            
        elif device_type == DeviceType.HEAT_PUMP:
            # heat pump features: [temperature standardized, running status]
            temp = base_features[0] if len(base_features) > 0 else 20.0
            temp_normalized = np.clip((temp - 15.0) / 15.0, 0.0, 1.0)
            running = 1.0 if len(base_features) > 1 and abs(base_features[1] - temp) > 1.0 else 0.0
            return np.array([temp_normalized, running])
            
        elif device_type == DeviceType.EV:
            # EV features: [SOC, connection status]
            soc = base_features[0] if len(base_features) > 0 else 0.5
            connected = base_features[1] if len(base_features) > 1 else 1.0
            return np.array([soc, connected])
            
        elif device_type == DeviceType.PV:
            # PV features: [generation potential] (based on simple time-based model)
            hour = 12  
            generation_potential = 0.8 if 8 <= hour <= 16 else 0.2
            return np.array([generation_potential])
            
        elif device_type == DeviceType.DISHWASHER:
            # dishwasher features: [running status, progress]
            running = 1.0 if len(base_features) > 1 and base_features[1] > 0 else 0.0
            progress = base_features[3] if len(base_features) > 3 else 0.0
            return np.array([running, progress])
        
        else:
            # unknown device type, return the first two features or fill with zeros
            if len(base_features) >= 2:
                return base_features[:2]
            elif len(base_features) == 1:
                return np.array([base_features[0], 0.0])
            else:
                return np.array([0.0, 0.0])
    
    def _get_standardized_management_features(self) -> np.ndarray:
        """get standardized management features"""
        # basic management statistics
        total_users = len(self.users)
        total_devices = len(self.device_mdps)
        controllable_devices = len(self.controllable_devices)
        
        # standardized management features (relative to system size)
        user_density = min(1.0, total_users / 15.0)  # assume maximum 15 users
        device_density = min(1.0, total_devices / 35.0)  # assume maximum 35 devices
        control_ratio = controllable_devices / max(1, total_devices)
        
        # coverage area standardized
        area_normalized = min(1.0, self.coverage_area / 1000.0)  # assume maximum 1000 square meters
        
        # district type encoded (one-hot encoding)
        is_residential = 1.0 if self.district_type == 'residential' else 0.0
        is_commercial = 1.0 if self.district_type == 'commercial' else 0.0
        is_mixed = 1.0 if self.district_type == 'mixed' else 0.0
        
        return np.array([
            user_density, device_density, control_ratio, area_normalized,
            is_residential, is_commercial, is_mixed
        ])
    
    def _get_standardized_preference_features(self) -> np.ndarray:
        """get standardized preference features"""
        return np.array([
            self.aggregated_preferences['economic'],
            self.aggregated_preferences['comfort'],
            self.aggregated_preferences['environmental']
        ])
    
    def _get_standardized_system_features(self) -> np.ndarray:
        """get standardized system features"""
        # system load features
        total_capacity = sum(
            1.0 for device_type in self.device_types.values()
            if device_type in [DeviceType.BATTERY, DeviceType.EV]
        )
        total_generation = sum(
            1.0 for device_type in self.device_types.values()
            if device_type == DeviceType.PV
        )
        total_load = sum(
            1.0 for device_type in self.device_types.values()
            if device_type in [DeviceType.HEAT_PUMP, DeviceType.DISHWASHER]
        )
        
        # standardized system features
        capacity_ratio = total_capacity / max(1, len(self.device_mdps))
        generation_ratio = total_generation / max(1, len(self.device_mdps))
        load_ratio = total_load / max(1, len(self.device_mdps))
        
        # system balance index
        balance_index = min(1.0, (total_capacity + total_generation) / max(1, total_load))
        
        return np.array([capacity_ratio, generation_ratio, load_ratio, balance_index])
    
    def _get_legacy_state_features(self) -> np.ndarray:
        """get legacy state features (keep backward compatibility)"""
        # device state features
        device_features = []
        for device_id in sorted(self.device_mdps.keys()):
            device_state = self.device_mdps[device_id].get_state_features()
            device_features.append(device_state)
        
        if device_features:
            device_features = np.concatenate(device_features)
        else:
            device_features = np.array([])
        
        # user preference features
        preference_features = np.array([
            self.aggregated_preferences['economic'],
            self.aggregated_preferences['comfort'],
            self.aggregated_preferences['environmental']
        ])
        
        # manager features
        manager_features = np.array([
            len(self.users),  # number of users
            len(self.device_mdps),  # number of devices
            len(self.controllable_devices),  # number of controllable devices
            self.coverage_area,  # coverage area
            1.0 if self.district_type == 'residential' else 0.0,
            1.0 if self.district_type == 'commercial' else 0.0,
            1.0 if self.district_type == 'mixed' else 0.0
        ])
        
        # merge all features
        if len(device_features) > 0:
            return np.concatenate([device_features, preference_features, manager_features])
        else:
            return np.concatenate([preference_features, manager_features])
    
    def get_action_space_size(self) -> int:
        """
        get action space size - modified to FlexOffer parameter generation
        
        each controllable device action contains:
        - time window flexibility (2D): [start_flex, end_flex] 
        - energy range adjustment (2D): [energy_min_factor, energy_max_factor]
        - priority weight (1D): [priority_weight]
        
        total 5D action for each device
        """
        # each controllable device needs 5 FlexOffer parameters
        fo_params_per_device = 5
        total_action_dim = len(self.controllable_devices) * fo_params_per_device
        
        # ensure at least basic action dimension
        return max(total_action_dim, 10)
    
    def reset(self):
        """reset manager state"""
        # reset all devices
        for device_mdp in self.device_mdps.values():
            device_mdp.reset_state()
        
        # reset Markov history
        self.markov_history = {
            'prev_actions': np.zeros(len(self.controllable_devices)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0,
            'user_satisfaction': 0.0
        }
    
    def step(self, actions: np.ndarray, env_state: Dict) -> Tuple[float, Dict]:
        """
        execute one step action
        
        new process:
        1. map actions to FlexOffer parameters
        2. generate device-level FlexOffer
        3. execute Pipeline process: aggregate → trade → disaggregate → schedule
        4. calculate Pipeline execution reward
        """
        # Step 1: map actions to FlexOffer parameters
        fo_params = self._map_actions_to_fo_params(actions)
        
        # Step 2: generate device-level FlexOffer
        device_flexoffers = self._generate_device_flexoffers(fo_params, env_state)
        
        # Step 3: execute complete Pipeline process
        pipeline_results = self._execute_full_pipeline(device_flexoffers, env_state)
        
        # Step 4: calculate Pipeline reward
        pipeline_reward, reward_info = self._calculate_pipeline_reward(
            pipeline_results, env_state
        )
        
        # update history
        self.markov_history['prev_actions'] = actions.copy()
        self.markov_history['prev_reward'] = pipeline_reward
        self.markov_history['cumulative_cost'] += reward_info.get('total_cost', 0.0)
        self.markov_history['cumulative_energy'] += reward_info.get('total_energy', 0.0)
        self.markov_history['user_satisfaction'] = reward_info.get('user_satisfaction', 0.0)
        
        info = {
            'pipeline_results': pipeline_results,
            'reward_components': reward_info,
            'fo_params': fo_params,
            'device_flexoffers': device_flexoffers
        }
        
        return pipeline_reward, info
    
    def _map_actions_to_fo_params(self, actions: np.ndarray) -> Dict[str, Dict]:
        """map agent actions to FlexOffer parameters"""
        fo_params = {}
        fo_params_per_device = 5
        
        for i, device_id in enumerate(self.controllable_devices):
            start_idx = i * fo_params_per_device
            
            if start_idx + fo_params_per_device <= len(actions):
                device_actions = actions[start_idx:start_idx + fo_params_per_device]
                
                # map actions to FlexOffer parameters
                fo_params[device_id] = {
                    'start_flex': np.clip(device_actions[0], -1.0, 1.0),  # start time flexibility
                    'end_flex': np.clip(device_actions[1], -1.0, 1.0),    # end time flexibility
                    'energy_min_factor': np.clip(device_actions[2], 0.1, 1.0),  # minimum energy factor
                    'energy_max_factor': np.clip(device_actions[3], 1.0, 2.0),  # maximum energy factor
                    'priority_weight': np.clip(device_actions[4], 0.1, 2.0)     # priority weight
                }
            else:
                # default parameters
                fo_params[device_id] = {
                    'start_flex': 0.0,
                    'end_flex': 0.0,
                    'energy_min_factor': 1.0,
                    'energy_max_factor': 1.0,
                    'priority_weight': 1.0
                }
        
        return fo_params
    
    def _generate_device_flexoffers(self, fo_params: Dict, env_state: Dict) -> Dict:
        """generate device-level FlexOffer based on FlexOffer parameters"""
        device_flexoffers = {}
        
        from fo_generate.dfo import DFOSystem, DFOSlice
        from datetime import datetime, timedelta
        
        for device_id, params in fo_params.items():
            if device_id in self.device_mdps:
                device_mdp = self.device_mdps[device_id]
                
                # get device action bounds (no need for current state)
                p_min, p_max = device_mdp.get_action_bounds()
                
                # adjust FlexOffer based on action parameters
                base_energy_min = p_min * 1.0  # 1 hour baseline
                base_energy_max = p_max * 1.0
                
                # apply energy factor adjustment
                energy_min = base_energy_min * params['energy_min_factor']
                energy_max = base_energy_max * params['energy_max_factor']
                
                # create time window (based on flexibility parameters)
                current_time = datetime.now()
                start_offset = int(params['start_flex'] * 2)  # -2 to +2 hours
                end_offset = 1 + int(params['end_flex'] * 2)   # 1 to 3 hours
                
                start_time = current_time + timedelta(hours=start_offset)
                end_time = current_time + timedelta(hours=end_offset)
                
                # create DFO system
                dfo_system = DFOSystem(
                    time_horizon=max(1, end_offset - start_offset),
                    device_id=device_id,
                    device_type=getattr(device_mdp, 'device_type', 'unknown')
                )
                
                # add time slice 
                dfo_slice = DFOSlice(
                    time_step=0,
                    energy_min=energy_min,
                    energy_max=energy_max,
                    constraints=[],
                    power_min=p_min,
                    power_max=p_max,
                    start_time=start_time,
                    end_time=end_time,
                    flexibility_factor=params['priority_weight'],
                    device_type=getattr(device_mdp, 'device_type', 'unknown'),
                    device_id=device_id
                )
                dfo_system.add_slice(dfo_slice)
                
                device_flexoffers[device_id] = dfo_system
        
        return device_flexoffers
    
    def _execute_full_pipeline(self, device_flexoffers: Dict, env_state: Dict) -> Dict:
        """execute complete Pipeline process - integrate real modules"""
        pipeline_results = {
            'flexoffers': device_flexoffers,
            'aggregated': [],
            'trades': [],
            'disaggregated': [],
            'scheduled': {},
            'stats': {}
        }
        
        try:
            if not device_flexoffers:
                pipeline_results['stats'] = {
                    'num_flexoffers': 0, 'num_trades': 0, 
                    'avg_satisfaction': 0.0, 'total_cost': 0.0
                }
                return pipeline_results
            
            # Step 1: aggregate FlexOffer (using fo_aggregate module)
            aggregated_results = self._aggregate_flexoffers(device_flexoffers, env_state)
            pipeline_results['aggregated'] = aggregated_results
            
            # Step 2: trade FlexOffer (using fo_trading module)
            trade_results = self._trade_flexoffers(aggregated_results, env_state)
            pipeline_results['trades'] = trade_results
            
            # Step 3: disaggregate FlexOffer (using fo_schedule module)
            disaggregated_results = self._disaggregate_flexoffers(
                trade_results, device_flexoffers, env_state
            )
            pipeline_results['disaggregated'] = disaggregated_results
            
            # Step 4: schedule execution (using fo_schedule module)
            scheduled_results = self._schedule_flexoffers(disaggregated_results, env_state)
            pipeline_results['scheduled'] = scheduled_results
            
            # calculate statistics
            pipeline_results['stats'] = self._calculate_pipeline_stats(
                pipeline_results, env_state
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # return failed results
            pipeline_results['stats'] = {
                'num_flexoffers': 0,
                'num_trades': 0,
                'avg_satisfaction': 0.0,
                'total_cost': 1000.0  # high cost as penalty
            }
        
        return pipeline_results
    
    def _aggregate_flexoffers(self, device_flexoffers: Dict, env_state: Dict) -> List:
        """aggregate FlexOffer - call fo_aggregate module"""
        try:
            from fo_aggregate.aggregator import FOAggregatorFactory
            from fo_common.flexoffer import FlexOffer, FOSlice
            
            # convert DFO to FlexOffer format
            flex_offers = []
            for device_id, dfo_system in device_flexoffers.items():
                for i, slice in enumerate(dfo_system.slices):
                    # create FOSlice
                    start_time = slice.start_time or datetime.now()
                    end_time = slice.end_time or (datetime.now() + timedelta(hours=1))
                    duration_minutes = (end_time - start_time).total_seconds() / 60.0
                    
                    fo_slice = FOSlice(
                        slice_id=i,
                        start_time=start_time,
                        end_time=end_time,
                        energy_min=slice.energy_min,
                        energy_max=slice.energy_max,
                        duration_minutes=duration_minutes,
                        device_type=slice.device_type,
                        device_id=device_id
                    )
                    
                    # create FlexOffer
                    flex_offer = FlexOffer(
                        fo_id=f"fo_{device_id}_{slice.time_step}",
                        hour=slice.time_step,
                        start_time=slice.start_time or datetime.now(),
                        end_time=slice.end_time or (datetime.now() + timedelta(hours=1)),
                        device_id=device_id,
                        device_type=slice.device_type,
                        slices=[fo_slice]
                    )
                    flex_offers.append(flex_offer)
            
            # select aggregation algorithm (configurable)
            aggregation_method = getattr(self, 'aggregation_method', 'LP')  # default LP
            
            # add detailed logging for aggregation method selection
            logger.info(f"using aggregation method: {aggregation_method}")
            logger.info(f"number of device FlexOffer: {len(flex_offers)}")
            
            # ensure aggregation method is uppercase to match FOAggregatorFactory logic
            aggregation_method = aggregation_method.upper()
            
            # create aggregator
            aggregator = FOAggregatorFactory.create_aggregator(aggregation_method)
            logger.info(f"created aggregator type: {aggregator.__class__.__name__}")
            
            # execute aggregation
            if flex_offers:
                aggregated_result = aggregator.aggregate(flex_offers)
                
                # add aggregated result logging
                if aggregated_result:
                    logger.info(f"aggregation successful: got {len(aggregated_result)} aggregated FlexOffer")
                    for i, afo in enumerate(aggregated_result):
                        logger.info(f"aggregated result #{i+1}: method={afo.aggregation_method}, "
                                   f"source FO number={len(afo.source_fo_ids)}, "
                                   f"total energy range=[{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}]")
                else:
                    logger.warning("aggregation process completed, but no aggregated result")
                
                return aggregated_result
            else:
                logger.warning("no FlexOffer to aggregate")
                return []
                
        except Exception as e:
            logger.error(f"FlexOffer aggregation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _trade_flexoffers(self, aggregated_results: List, env_state: Dict) -> List:
        """trade FlexOffer - call real algorithm from fo_trading module"""
        try:
            if not aggregated_results:
                return []
            
            # select trading algorithm (from environment state or configuration)
            trading_method = env_state.get('trading_algorithm', getattr(self, 'trading_method', 'market_clearing'))
            
            if trading_method == 'market_clearing':
                return self._trade_with_market_clearing(aggregated_results, env_state)
            else:
                return self._trade_with_bidding(aggregated_results, env_state)
            
        except Exception as e:
            logger.error(f"FlexOffer trading failed: {e}")
            return []
    
    def _trade_with_bidding(self, aggregated_results: List, env_state: Dict) -> List:
        """trade FlexOffer with Bidding algorithm - generate trade when both prices match"""
        try:
            from fo_trading.pool import BiddingAlgorithm, Bid
            
            bidding_algo = BiddingAlgorithm()
            bids = []
            
            # create buy and sell bids for each aggregated FO
            for i, aggregated_fo in enumerate(aggregated_results):
                total_energy = 0.0
                if hasattr(aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.total_energy_max
                elif hasattr(aggregated_fo, 'aggregated_fo') and hasattr(aggregated_fo.aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.aggregated_fo.total_energy_max
                else:
                    total_energy = getattr(aggregated_fo, 'total_energy', 10.0)
                
                # ensure total energy is at least 1.0, avoid zero energy trading
                total_energy = max(1.0, total_energy)
                
                logger.info(f"aggregated FO {i} total energy: {total_energy:.2f} kWh")
                
                base_price = env_state.get('price', 0.15)
                
                sell_price = base_price * (0.85 + 0.15 * np.random.random())  
                sell_bid = Bid(
                    bid_id=f"sell_{self.manager_id}_{i}",
                    participant_id=self.manager_id,
                    price=sell_price,
                    quantity=total_energy,
                    time_step=0,
                    side="sell"
                )
                
                buy_price = base_price * (1.05 + 0.25 * np.random.random())  
                buy_quantity = total_energy * (0.8 + 0.4 * np.random.random())  
                buy_bid = Bid(
                    bid_id=f"buy_market_{i}",
                    participant_id=f"buyer_{i}",
                    price=buy_price,
                    quantity=buy_quantity,
                    time_step=0,
                    side="buy"
                )
                
                bids.extend([sell_bid, buy_bid])
                bidding_algo.submit_bid(sell_bid)
                bidding_algo.submit_bid(buy_bid)
            
            # Bidding algorithm: check if buy and sell prices match, generate trade if they do
            trades = []
            processed_bids = bidding_algo.get_bids_by_timestep(0)
            
            # separate buy and sell bids
            buy_bids = [bid for bid in processed_bids if bid.side == "buy"]
            sell_bids = [bid for bid in processed_bids if bid.side == "sell"]
            
            # sort by price
            buy_bids.sort(key=lambda x: x.price, reverse=True)  # buy bids from high to low
            sell_bids.sort(key=lambda x: x.price)  # sell bids from low to high
            
            # match buy and sell bids: buy bid >= sell bid
            for sell_bid in sell_bids:
                if sell_bid.participant_id != self.manager_id:
                    continue  # only process sell bids for current Manager
                    
                for buy_bid in buy_bids:
                    if buy_bid.price >= sell_bid.price:  # price match
                        # calculate trade quantity 
                        trade_quantity = min(sell_bid.quantity, buy_bid.quantity)
                        
                        # calculate trade price 
                        trade_price = (buy_bid.price + sell_bid.price) / 2
                        
                        # create trade record
                        trade = {
                            'trade_id': f"trade_{sell_bid.bid_id}_{buy_bid.bid_id}",
                            'buyer_id': buy_bid.participant_id,
                            'seller_id': sell_bid.participant_id,
                            'aggregated_fo': aggregated_results[0] if aggregated_results else None,
                            'trade_price': trade_price,
                            'trade_volume': trade_quantity,
                            'success': True,
                            'algorithm': 'bidding',
                            'buy_bid_price': buy_bid.price,
                            'sell_bid_price': sell_bid.price
                        }
                        trades.append(trade)
                        
                        # update bid quantity
                        sell_bid.quantity -= trade_quantity
                        buy_bid.quantity -= trade_quantity
                        
                        logger.info(f"success trade(Bidding): buyer {buy_bid.participant_id}({buy_bid.price:.4f}) "
                                  f"vs seller {sell_bid.participant_id}({sell_bid.price:.4f}), "
                                  f"trade price {trade_price:.4f}, quantity {trade_quantity:.2f}")
                        
                        # if buy bid quantity is satisfied, jump to next buy bid
                        if buy_bid.quantity <= 0:
                            break
                    
                    # if sell bid quantity is used up, jump to next sell bid
                    if sell_bid.quantity <= 0:
                        break
            
            logger.info(f"Bidding algorithm completed: processed {len(processed_bids)} bids, generated {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Bidding algorithm trading failed: {e}")
            return []
    
    def _trade_with_market_clearing(self, aggregated_results: List, env_state: Dict) -> List:
        """trade FlexOffer with Market Clearing algorithm"""
        try:
            from fo_trading.pool import MarketClearingAlgorithm, Bid
            
            # create market clearing algorithm instance
            clearing_algo = MarketClearingAlgorithm(clearing_method="uniform_price")
            bids = []
            
            # create buy and sell bids for each aggregated FO
            for i, aggregated_fo in enumerate(aggregated_results):
                # fix: correct energy of aggregated FlexOffer
                total_energy = 0.0
                if hasattr(aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.total_energy_max
                elif hasattr(aggregated_fo, 'aggregated_fo') and hasattr(aggregated_fo.aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.aggregated_fo.total_energy_max
                else:
                    total_energy = getattr(aggregated_fo, 'total_energy', 10.0)
                
                # ensure total energy is at least 1.0, avoid zero energy trading
                total_energy = max(1.0, total_energy)
                
                logger.info(f"aggregated FO {i} total energy: {total_energy:.2f} kWh")
                
                base_price = env_state.get('price', 0.15)
                
                # create sell bid (current Manager) - same price range as bidding algorithm
                sell_bid = Bid(
                    bid_id=f"sell_{self.manager_id}_{i}",
                    participant_id=self.manager_id,
                    price=base_price * (0.85 + 0.15 * np.random.random()),  
                    quantity=total_energy,
                    time_step=0,
                    side="sell"
                )
                
                for j in range(2):  
                    buy_bid = Bid(
                        bid_id=f"buy_market_{i}_{j}",
                        participant_id=f"buyer_{i}_{j}",
                        price=base_price * (1.05 + 0.25 * np.random.random()),  
                        quantity=total_energy * (0.4 + 0.4 * np.random.random()),  
                        time_step=0,
                        side="buy"
                    )
                    bids.append(buy_bid)
                
                bids.append(sell_bid)
            
            # execute market clearing
            clearing_results = clearing_algo.process_bids(bids)
            
            # generate trades
            generated_trades = clearing_algo.generate_trades(clearing_results, bids)
            
            # convert to unified format
            trades = []
            for trade in generated_trades:
                if trade.seller_id == self.manager_id:  # only process trades for current Manager
                    trades.append({
                        'trade_id': trade.trade_id,
                        'buyer_id': trade.buyer_id,
                        'seller_id': trade.seller_id,
                        'aggregated_fo': aggregated_results[0] if aggregated_results else None,
                        'trade_price': trade.price,
                        'trade_volume': trade.quantity,
                        'success': trade.status == "completed",
                        'algorithm': 'market_clearing',
                        'clearing_result_id': trade.clearing_result_id
                    })
            
            logger.info(f"Market Clearing algorithm completed: {len(clearing_results)} clearing results, {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Market Clearing algorithm trading failed: {e}")
            return []
    
    def _disaggregate_flexoffers(self, trade_results: List, original_flexoffers: Dict, env_state: Dict) -> List:
        """disaggregate FlexOffer - call fo_schedule module"""
        try:
            if not trade_results:
                return []
            
            from fo_schedule.scheduler import AggregatedResultDisaggregator
            
            # select disaggregation method (configurable)
            disaggregation_method = getattr(self, 'disaggregation_method', 'proportional')
            disaggregator = AggregatedResultDisaggregator(
                time_horizon=24, 
                default_algorithm=disaggregation_method
            )
            
            disaggregated_results = []
            
            for trade in trade_results:
                if trade.get('success', False):
                    # prepare disaggregation data
                    original_data = []
                    for device_id, dfo_system in original_flexoffers.items():
                        for slice in dfo_system.slices:
                            original_data.append({
                                'device_id': device_id,
                                'energy_min': slice.energy_min,
                                'energy_max': slice.energy_max,
                                'weight': slice.flexibility_factor,
                                'energy': slice.energy_max  # add energy field for disaggregation algorithm
                            })
                    
                    if original_data:
                        # execute disaggregation
                        total_energy = trade.get('trade_volume', 0.0)
                        
                        # fix: check if total energy is 0 or negative
                        if total_energy <= 0:
                            logger.info(f"trade volume is 0 or negative ({total_energy}), allocate zero energy to all devices")
                            for data in original_data:
                                disaggregated_results.append({
                                    'device_id': data['device_id'],
                                    'allocated_energy': 0.0,
                                    'method': 'zero_energy',
                                    'allocation_ratio': 0.0
                                })
                            continue
                        
                        try:
                            disaggregated = disaggregator.disaggregate(
                                aggregated_result=trade.get('aggregated_fo'),
                                original_data=original_data, 
                                time_step=0
                            )
                            disaggregated_results.extend(disaggregated)
                        except Exception as e:
                            logger.warning(f"disaggregation failed, use average allocation: {e}")
                            # fallback to average allocation
                            avg_energy = total_energy / len(original_data)
                            for data in original_data:
                                disaggregated_results.append({
                                    'device_id': data['device_id'],
                                    'allocated_energy': avg_energy,
                                    'method': 'average_fallback'
                                })
            
            return disaggregated_results
            
        except Exception as e:
            logger.error(f"FlexOffer disaggregation failed: {e}")
            return []
    
    def _schedule_flexoffers(self, disaggregated_results: List, env_state: Dict) -> Dict:
        """schedule FlexOffer - execute final device control"""
        try:
            scheduled_results = {}
            total_satisfaction = 0.0
            device_count = 0
            
            for result in disaggregated_results:
                device_id = result.get('device_id')
                allocated_energy = result.get('allocated_energy', 0.0)
                
                if device_id and device_id in self.device_mdps:
                    device_mdp = self.device_mdps[device_id]
                    
                    # convert allocated energy to power control signal
                    power_signal = allocated_energy / 1.0  # 1 hour
                    
                    # limit within device power range
                    p_min, p_max = device_mdp.get_action_bounds()
                    power_signal = np.clip(power_signal, p_min, p_max)
                    
                    # execute device state transition
                    next_state = device_mdp.transition_state(power_signal, env_state)
                    
                    # calculate device satisfaction
                    device_satisfaction = self._calculate_device_satisfaction(
                        device_mdp, power_signal, next_state, env_state
                    )
                    
                    scheduled_results[device_id] = {
                        'power_signal': power_signal,
                        'allocated_energy': allocated_energy,
                        'device_state': next_state,
                        'satisfaction': device_satisfaction
                    }
                    
                    total_satisfaction += device_satisfaction
                    device_count += 1
            
            # calculate average satisfaction
            avg_satisfaction = total_satisfaction / max(device_count, 1)
            scheduled_results['_summary'] = {
                'avg_satisfaction': avg_satisfaction,
                'total_devices': device_count
            }
            
            return scheduled_results
            
        except Exception as e:
            logger.error(f"FlexOffer scheduling failed: {e}")
            return {'_summary': {'avg_satisfaction': 0.0, 'total_devices': 0}}
    
    def _calculate_device_satisfaction(self, device_mdp, power_signal: float, 
                                     device_state: Dict, env_state: Dict) -> float:
        """calculate device satisfaction (based on actual device state)"""
        try:
            # use device reward function to evaluate satisfaction
            device_reward, reward_components = device_mdp.calculate_reward(
                power_signal, device_state, env_state
            )
            
            # convert reward to satisfaction (0-1 range)
            # assume positive reward corresponds to high satisfaction, negative reward corresponds to low satisfaction
            if device_reward > 0:
                satisfaction = min(device_reward / 10.0, 1.0)  # normalize to 0-1
            else:
                satisfaction = max(0.5 + device_reward / 20.0, 0.0)  # negative reward reduces satisfaction
            
            return np.clip(satisfaction, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"device satisfaction calculation failed: {e}")
            return 0.5  
    
    def _calculate_pipeline_stats(self, pipeline_results: Dict, env_state: Dict) -> Dict:
        """calculate Pipeline statistics"""
        stats = {}
        
        # basic statistics
        stats['num_flexoffers'] = len(pipeline_results.get('flexoffers', {}))
        stats['num_aggregated'] = len(pipeline_results.get('aggregated', []))
        stats['num_trades'] = len(pipeline_results.get('trades', []))
        stats['num_disaggregated'] = len(pipeline_results.get('disaggregated', []))
        
        # satisfaction statistics
        scheduled = pipeline_results.get('scheduled', {})
        summary = scheduled.get('_summary', {})
        stats['avg_satisfaction'] = summary.get('avg_satisfaction', 0.0)
        
        # cost statistics
        trades = pipeline_results.get('trades', [])
        total_cost = 0.0
        total_revenue = 0.0
        
        for trade in trades:
            if trade.get('success', False):
                volume = trade.get('trade_volume', 0.0)
                price = trade.get('trade_price', env_state.get('price', 0.15))
                
                if trade.get('seller_id') == self.manager_id:
                    total_revenue += volume * price
                else:
                    total_cost += volume * price
        
        stats['total_cost'] = total_cost
        stats['total_revenue'] = total_revenue
        stats['net_benefit'] = total_revenue - total_cost
        
        return stats
    
    def _calculate_pipeline_reward(self, pipeline_results: Dict, env_state: Dict) -> Tuple[float, Dict]:
        """calculate reward based on Pipeline execution results"""
        stats = pipeline_results.get('stats', {})
        
        total_cost = stats.get('total_cost', 0.0)
        total_revenue = stats.get('total_revenue', 0.0)
        net_benefit = total_revenue - total_cost
        
        if net_benefit > 20:
            economic_reward = 70.0 + min(net_benefit - 20, 30.0)  # 70-100
        elif net_benefit > 5:
            economic_reward = 40.0 + (net_benefit - 5) * 2.0  # 40-70
        elif net_benefit > 0:
            economic_reward = 20.0 + net_benefit * 4.0  # 20-40
        elif net_benefit > -5:
            economic_reward = 10.0 + (net_benefit + 5) * 2.0  # 0-20
        else:
            economic_reward = max(0.0, 10.0 + net_benefit)  # minimum 0, avoid large negative values
        
        satisfaction_base = stats.get('avg_satisfaction', 0.5)
        
        if satisfaction_base > 0.8:
            satisfaction_reward = 60.0 + (satisfaction_base - 0.8) * 100  # 60-80
        elif satisfaction_base > 0.6:
            satisfaction_reward = 40.0 + (satisfaction_base - 0.6) * 100  # 40-60
        elif satisfaction_base > 0.4:
            satisfaction_reward = 20.0 + (satisfaction_base - 0.4) * 100  # 20-40
        elif satisfaction_base > 0.2:
            satisfaction_reward = 10.0 + (satisfaction_base - 0.2) * 50  # 10-20
        else:
            satisfaction_reward = max(10.0, satisfaction_base * 50)  # minimum 10
        
        # 3. coordination reward
        num_trades = stats.get('num_trades', 0)
        num_flexoffers = stats.get('num_flexoffers', 1)
        trade_success_rate = num_trades / max(num_flexoffers, 1)
        
        if trade_success_rate > 0.7:
            coordination_reward = 40.0 + (trade_success_rate - 0.7) * 66.7  # 40-60
        elif trade_success_rate > 0.4:
            coordination_reward = 20.0 + (trade_success_rate - 0.4) * 66.7  # 20-40
        elif trade_success_rate > 0.1:
            coordination_reward = 5.0 + (trade_success_rate - 0.1) * 50  # 5-20
        else:
            coordination_reward = max(5.0, trade_success_rate * 50)  # minimum 5
        
        # 4. strategy consistency reward
        strategy_consistency_reward = 15.0  # default base
        if hasattr(self, 'markov_history') and 'prev_actions' in self.markov_history:
            prev_actions = self.markov_history['prev_actions']
            if prev_actions is not None and len(prev_actions) > 0:
                action_mean = np.mean(prev_actions)
                action_std = np.std(prev_actions)
                
                # 奖励合理的策略
                if 0.2 <= action_std <= 0.6 and 0.3 <= action_mean <= 0.7:
                    strategy_consistency_reward = 30.0  # excellent strategy
                elif 0.1 <= action_std <= 0.8 and 0.2 <= action_mean <= 0.8:
                    strategy_consistency_reward = 20.0  # good strategy
                elif action_std < 0.05:  # too conservative
                    strategy_consistency_reward = 5.0
                elif action_std > 0.9:  # too random
                    strategy_consistency_reward = 8.0
                else:
                    strategy_consistency_reward = 15.0  # average strategy
        
        # target range: 30-270
        total_reward = (
            0.4 * economic_reward +           # 40% weight: 0-40
            0.3 * satisfaction_reward +       # 30% weight: 3-24
            0.2 * coordination_reward +       # 20% weight: 1-12
            0.1 * strategy_consistency_reward # 10% weight: 0-3
        )  # total range: 4-79, plus base reward
        
        # add base reward, ensure total reward is positive
        base_reward = 50.0  # base reward 50
        total_reward += base_reward  # final range: 54-129
        
        # add small adjustment based on learning progress (instead of large reward)
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
        else:
            self.episode_count = 1
            
        # learning progress reward: small adjustment instead of large change
        learning_progress_reward = 0.0
        if self.episode_count <= 20:
            # early stage: encourage exploration, small random adjustment
            learning_progress_reward = np.random.uniform(-5, 15)  # -5到+15分
        elif self.episode_count <= 50:
            # mid stage: reward stable strategy
            if economic_reward > 60 and satisfaction_reward > 50:
                learning_progress_reward = 10.0
            else:
                learning_progress_reward = 5.0
        else:
            # late stage: reward excellent strategy
            if total_reward > 100:
                learning_progress_reward = 15.0
            else:
                learning_progress_reward = 8.0
                
        total_reward += learning_progress_reward
        
        # ensure reward is within reasonable range
        total_reward = max(30.0, min(200.0, total_reward))  # limit within 30-200
        
        reward_info = {
            'economic': economic_reward,
            'satisfaction': satisfaction_reward,
            'coordination': coordination_reward,
            'strategy_consistency': strategy_consistency_reward,
            'learning_progress': learning_progress_reward,
            'base_reward': base_reward,
            'total_cost': total_cost,
            'total_revenue': stats.get('total_revenue', 0.0),
            'net_benefit': net_benefit,
            'user_satisfaction': satisfaction_base,
            'trade_success_rate': trade_success_rate,
            'episode_count': getattr(self, 'episode_count', 1)
        }
        
        return total_reward, reward_info
    
    def _apply_user_preferences(self, base_reward: float, reward_components: Dict) -> float:
        """apply user preference weights"""
        # extract different types of rewards
        economic_reward = 0.0
        comfort_reward = 0.0
        environmental_reward = 0.0
        
        for device_rewards in reward_components.values():
            economic_reward += device_rewards.get('economic', 0.0)
            comfort_reward += device_rewards.get('comfort', 0.0)
            environmental_reward += device_rewards.get('efficiency', 0.0)  
        
        # apply user preference weights
        weighted_reward = (
            self.aggregated_preferences['economic'] * economic_reward +
            self.aggregated_preferences['comfort'] * comfort_reward +
            self.aggregated_preferences['environmental'] * environmental_reward
        )
        
        return weighted_reward
    
    def _calculate_user_satisfaction(self, reward_components: Dict) -> float:
        """calculate user satisfaction"""
        # calculate satisfaction based on comfort reward
        comfort_rewards = []
        for device_rewards in reward_components.values():
            if 'comfort' in device_rewards:
                comfort_rewards.append(device_rewards['comfort'])
        
        if comfort_rewards:
            return float(np.mean(comfort_rewards))
        else:
            return 0.5  # default satisfaction
    
    def generate_dfo(self, time_horizon: int) -> Dict[str, DFOSystem]:
        """generate DFO system"""
        dfo_systems = {}
        
        for device_id in self.controllable_devices:
            device_mdp = self.device_mdps[device_id]
            dfo = DFOSystem(time_horizon)
            
            for t in range(time_horizon):
                # get action bounds
                p_min, p_max = device_mdp.get_action_bounds()
                
                # create time slice
                dfo_slice = DFOSlice(
                    time_step=t,
                    energy_min=p_min,
                    energy_max=p_max,
                    constraints=[]
                )
                
                dfo.add_slice(dfo_slice)
            
            dfo_systems[device_id] = dfo
        
        return dfo_systems

    def get_observation(self):
        """get current observation"""
        device_states = []
        user_states = []
        
        # collect device states
        if isinstance(self.devices, dict):
            for device in self.devices.values():
                if hasattr(device, 'env'):
                    state = device.env.get_state()
                    device_states.extend(state)
        else:
            # self.devices is a list
            for device in self.devices:
                if hasattr(device, 'env'):
                    state = device['env'].get_state()  # type: ignore
                    device_states.extend(state)
        
        # collect user states  
        for user in self.users:
            # process user device count (if user has device attribute, use it, otherwise default to 0)
            user_device_count = 0
            if 'devices' in user:
                user_device_count = len(user['devices'])
            elif 'device_count' in user:
                user_device_count = user['device_count']
            
            # process user preferences (if user has preference attribute, use it, otherwise use default value)
            preferences = getattr(user, 'preferences', {})
            
            user_state = [
                user_device_count,  # device count
                preferences.get('economic', 0.25),  # economic preference
                preferences.get('comfort', 0.25),   # comfort preference
                preferences.get('self_sufficient', 0.25),  # self-sufficient preference
                preferences.get('environmental', 0.25)     # environmental preference
            ]
            user_states.extend(user_state)
        
        # combine observations
        observation = device_states + user_states
        
        # ensure observation dimension consistency (here no constraint, because observation dimension is dynamic)
        return np.array(observation, dtype=np.float32)

class MultiAgentFlexOfferEnv(gym.Env):
    """multi-agent FlexOffer environment"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 time_horizon: int = 24,
                 time_step: float = 1.0,
                 start_time: Optional[datetime] = None,
                 dec_pomdp_config: Optional[DecPOMDPConfig] = None,
                 aggregation_method: str = "LP",
                 trading_method: str = "bidding", 
                 disaggregation_method: str = "proportional"):
        """
        initialize multi-agent FlexOffer environment
        
        Args:
            data_dir: data directory
            time_horizon: time horizon
            time_step: time step
            start_time: start time
            dec_pomdp_config: Dec-POMDP configuration
            aggregation_method: aggregation algorithm ("LP", "DP")
            trading_method: trading algorithm ("bidding", "market_clearing")
            disaggregation_method: disaggregation algorithm ("average", "proportional")
        """
        
        self.data_dir = data_dir
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time or datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # algorithm configuration
        self.aggregation_method = aggregation_method
        self.trading_method = trading_method
        self.disaggregation_method = disaggregation_method
        
        # validate algorithm choices
        self._validate_algorithm_choices()
        
        # Dec-POMDP configuration
        self.dec_pomdp_config = dec_pomdp_config or DecPOMDPConfig()
        self.dec_pomdp_obs_space = DecPOMDPObservationSpace(self.dec_pomdp_config)
        
        # observation history (for information delay)
        self.observation_history: Dict[str, List[np.ndarray]] = {}
        
        # dynamic observation quality manager (if enable observation noise, enable dynamic quality)
        if self.dec_pomdp_config.enable_observation_noise:
            self.dynamic_quality_manager = DynamicObservationQuality()
        else:
            self.dynamic_quality_manager = None
        
        # data loader
        self.data_loader = DataLoader(data_dir)
        
        # load configuration data
        self._load_configuration_data()
        
        # environment dynamics
        self.env_dynamics = EnvironmentDynamics(
            price_data=self.price_data,
            weather_data=self.weather_data
        )
        
        # create Manager agents
        self._create_manager_agents()
        
        # set algorithm configuration for each Manager
        self._configure_manager_algorithms()
        
        # set observation and action spaces
        self._setup_spaces()
        
        # time state
        self.current_time = self.start_time
        self.current_step = 0
        
        logger.info(f"multi-agent environment initialized: {len(self.manager_agents)} Managers")
        logger.info(f"algorithm configuration - aggregation: {aggregation_method}, trading: {trading_method}, disaggregation: {disaggregation_method}")
        logger.info(f"Dec-POMDP mode: {self.dec_pomdp_config.enable_observation_noise}, "
                   f"dynamic quality management: {self.dynamic_quality_manager is not None}")
    
    def _validate_algorithm_choices(self):
        """validate algorithm choices"""
        # validate aggregation algorithm
        valid_aggregation = ["LP", "DP"]
        if self.aggregation_method not in valid_aggregation:
            logger.warning(f"invalid aggregation algorithm '{self.aggregation_method}', using default 'LP'")
            self.aggregation_method = "LP"
        
        # validate trading algorithm
        valid_trading = ["bidding", "market_clearing"]
        if self.trading_method not in valid_trading:
            logger.warning(f"invalid trading algorithm '{self.trading_method}', using default 'bidding'")
            self.trading_method = "bidding"
        
        # validate disaggregation algorithm
        valid_disaggregation = ["average", "proportional"]
        if self.disaggregation_method not in valid_disaggregation:
            logger.warning(f"invalid disaggregation algorithm '{self.disaggregation_method}', using default 'proportional'")
            self.disaggregation_method = "proportional"
        
        logger.info(f"algorithm validation completed - aggregation: {self.aggregation_method}, "
                   f"trading: {self.trading_method}, disaggregation: {self.disaggregation_method}")
    
    def _configure_manager_algorithms(self):
        """configure algorithm selection for each Manager"""
        for manager_id, manager in self.manager_agents.items():
            # pass algorithm configuration to Manager (use setattr to dynamically set attributes)
            setattr(manager, 'aggregation_method', self.aggregation_method)
            setattr(manager, 'trading_method', self.trading_method)
            setattr(manager, 'disaggregation_method', self.disaggregation_method)
            
            logger.debug(f"Manager {manager_id} algorithm configuration completed")
    
    def set_algorithms(self, aggregation: Optional[str] = None, trading: Optional[str] = None, disaggregation: Optional[str] = None):
        """dynamically set algorithm selection"""
        if aggregation:
            self.aggregation_method = aggregation
        if trading:
            self.trading_method = trading
        if disaggregation:
            self.disaggregation_method = disaggregation
        
        # re-validate and configure
        self._validate_algorithm_choices()
        self._configure_manager_algorithms()
        
        logger.info(f"algorithm configuration updated - aggregation: {self.aggregation_method}, "
                   f"trading: {self.trading_method}, disaggregation: {self.disaggregation_method}")
    
    def get_algorithm_config(self) -> Dict[str, str]:
        """get current algorithm configuration"""
        return {
            'aggregation': self.aggregation_method,
            'trading': self.trading_method,
            'disaggregation': self.disaggregation_method
        }
    
    def _load_configuration_data(self):
        """load configuration data"""
        # load external data
        self.weather_data = self.data_loader.load_weather_data(
            start_time=self.start_time, hours=self.time_horizon * 2
        )
        self.price_data = self.data_loader.load_price_data(
            start_time=self.start_time, hours=self.time_horizon * 2
        )
        self.pv_forecast_data = self.data_loader.load_pv_forecast_data(
            start_time=self.start_time, hours=self.time_horizon * 2
        )
        self.calendar_data = self.data_loader.load_calendar_data()
        
        # load Manager and user configuration - use actual file names
        self.manager_config_df = self.data_loader.load_manager_config("manager_config_36users.csv")
        self.user_config_df = self.data_loader.load_user_config("user_config_36users.csv")
        self.device_config_df = self.data_loader.load_device_config("device_config_36users.csv")
        
        logger.info("configuration data loaded")
    
    def _create_manager_agents(self):
        """create Manager agents"""
        self.manager_agents: Dict[str, ManagerAgent] = {}
        
        for _, manager_row in self.manager_config_df.iterrows():
            manager_id = manager_row['manager_id']
            
            # get users of this Manager
            manager_users = self.user_config_df[
                self.user_config_df['manager_id'] == manager_id
            ].to_dict('records')  # type: ignore
            
            # get devices of this Manager's users - handle user ID format mismatch
            user_ids = [user['user_id'] for user in manager_users]
            
            # generate possible user ID formats for matching
            # format 1: user_01, user_02, ... (user configuration file format)
            # format 2: user_manager_1_1, user_manager_1_2, ... (device configuration file format)
            extended_user_ids = set(user_ids)  # original user IDs
            
            # generate possible device configuration formats for each user ID
            for user_id in user_ids:
                # extract manager and user number from user_01
                if user_id.startswith('user_'):
                    try:
                        user_num_str = user_id.split('_')[1]  # get "01", "02" etc.
                        user_num = int(user_num_str)  # convert to number
                        
                        # generate corresponding device configuration user ID format based on manager_id
                        manager_num = str(manager_id).split('_')[1]  # get "1" from manager_1
                        
                        # calculate local user number within Manager
                        # Manager 1: user_01-06 -> user_manager_1_1-6
                        # Manager 2: user_07-16 -> user_manager_2_1-10  
                        # Manager 3: user_17-24 -> user_manager_3_1-8
                        # Manager 4: user_25-36 -> user_manager_4_1-12
                        
                        if manager_id == "manager_1" and 1 <= user_num <= 6:
                            local_user_num = user_num
                        elif manager_id == "manager_2" and 7 <= user_num <= 16:
                            local_user_num = user_num - 6
                        elif manager_id == "manager_3" and 17 <= user_num <= 24:
                            local_user_num = user_num - 16
                        elif manager_id == "manager_4" and 25 <= user_num <= 36:
                            local_user_num = user_num - 24
                        else:
                            continue  # user does not belong to current Manager
                        
                        # generate user ID in device configuration format
                        device_user_id = f"user_manager_{manager_num}_{local_user_num}"
                        extended_user_ids.add(device_user_id)
                        
                    except (ValueError, IndexError):
                        continue  # cannot parse user ID format, skip
            
            # use extended user ID list to match devices
            manager_devices = self.device_config_df[
                self.device_config_df['user_id'].isin(list(extended_user_ids))
            ].to_dict('records')  # type: ignore
            
            logger.debug(f"{manager_id}: original user IDs {user_ids}, extended user IDs {list(extended_user_ids)}, matched devices {len(manager_devices)}")
            
            # create Manager agent
            manager_agent = ManagerAgent(
                manager_id=str(manager_id),
                manager_config=manager_row.to_dict(),
                users=manager_users,
                devices=manager_devices
            )
            
            self.manager_agents[str(manager_id)] = manager_agent
        
        self.manager_ids = list(self.manager_agents.keys())
        logger.info(f"created {len(self.manager_agents)} Manager agents")
    
    def _setup_spaces(self):
        """set observation and action spaces - update to FlexOffer parameter generation mode"""
        # 计算观测空间维度
        obs_dims = {}
        action_dims = {}
        
        for manager_id, manager in self.manager_agents.items():
            # get state feature dimension of single Manager
            state_features = manager.get_state_features()
            
            # environment feature dimension (time 4 + price 5 + weather 4 = 13)
            env_dim = 13
            
            # other Manager information dimension (each Manager enhanced information dimension)
            # base information 5D + enhanced information 9D = 14D for each Manager
            enhanced_manager_info_dim = 14
            other_managers_dim = (len(self.manager_agents) - 1) * enhanced_manager_info_dim
            
            # market state feature dimension (16D)
            market_state_dim = 16
            
            # total observation dimension
            total_obs_dim = len(state_features) + env_dim + other_managers_dim + market_state_dim
            
            obs_dims[manager_id] = total_obs_dim
            
            # new action space: FlexOffer parameter generation
            # each controllable device needs 5 FlexOffer parameters: [start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight]
            action_dims[manager_id] = manager.get_action_space_size()
        
        # create observation and action spaces
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for manager_id in self.manager_ids:
            self.observation_spaces[manager_id] = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_dims[manager_id],), 
                dtype=np.float32
            )
            
            # new action space definition: FlexOffer parameter range
            manager = self.manager_agents[manager_id]
            num_controllable_devices = len(manager.controllable_devices)
            fo_params_per_device = 5
            total_action_dim = num_controllable_devices * fo_params_per_device
            
            if total_action_dim > 0:
                # reasonable range of FlexOffer parameters
                action_low = []
                action_high = []
                
                for _ in range(num_controllable_devices):
                    action_low.extend([-1.0, -1.0, 0.1, 1.0, 0.1])   # [start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight]
                    action_high.extend([1.0, 1.0, 1.0, 2.0, 2.0])    # corresponding upper limits
                
                self.action_spaces[manager_id] = spaces.Box(
                    low=np.array(action_low, dtype=np.float32),
                    high=np.array(action_high, dtype=np.float32),
                    dtype=np.float32
                )
                
                logger.info(f"Manager {manager_id} action space: {total_action_dim}D "
                          f"({num_controllable_devices} devices × {fo_params_per_device} parameters/device)")
            else:
                # if no controllable devices, create a virtual action space
                self.action_spaces[manager_id] = spaces.Box(
                    low=-1.0, high=1.0, shape=(5,), dtype=np.float32
                )
                logger.warning(f"Manager {manager_id} has no controllable devices, using virtual action space")
        
        logger.info("observation and action spaces set - FlexOffer parameter generation mode")
    
    def reset(self, seed=None, options=None):
        """reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_time = self.start_time
        self.current_step = 0
        
        # reset public information cache
        self._cached_public_features = None
        self._cache_time_step = -1
        
        # reset environment state cache
        self._cached_env_state = None
        self._env_state_cache_time = -1
        
        # reset environment dynamics
        self.env_dynamics.price_history = []
        self.env_dynamics.weather_history = []
        
        # reset all Managers
        for manager in self.manager_agents.values():
            manager.reset()
        
        # get initial observations
        observations = self._get_observations()
        infos = {manager_id: {'time': self.current_time, 'step': self.current_step} 
                for manager_id in self.manager_ids}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]):
        """execute one step"""
        # get current environment state
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # add algorithm configuration to environment state
        env_state['trading_algorithm'] = self.trading_method
        
        # execute actions of all Managers
        rewards = {}
        infos = {}
        
        for manager_id, action in actions.items():
            if manager_id in self.manager_agents:
                manager = self.manager_agents[manager_id]
                reward, info = manager.step(action, env_state)
                rewards[manager_id] = reward
                infos[manager_id] = info
        
        # update time
        self.current_time += timedelta(hours=self.time_step)
        self.current_step += 1
        
        # check termination conditions
        done = self.current_step >= self.time_horizon
        dones = {manager_id: done for manager_id in self.manager_ids}
        dones['__all__'] = done
        
        # get next observations
        next_observations = self._get_observations()
        
        # add environment information
        for manager_id in self.manager_ids:
            infos[manager_id].update({
                'time': self.current_time,
                'step': self.current_step,
                'env_state': env_state
            })
        
        return next_observations, rewards, dones, False, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:

        observations = {}
        
        # ensure all Managers get the same public environment features (no noise) at the same time step
        if not hasattr(self, '_cached_public_features') or self._cache_time_step != self.current_step:
            self._cached_public_features = self._get_dec_pomdp_public_features()
            self._cache_time_step = self.current_step
        
        public_features = self._cached_public_features
        
        # get simplified Manager-to-Manager collaboration information (limited and noisy)
        limited_collaboration_info = self._get_limited_collaboration_info()
        
        # update dynamic observation quality (if enabled)
        if self.dynamic_quality_manager:
            self.dynamic_quality_manager.step()
        
        for manager_id, manager in self.manager_agents.items():
            # 1. private information layer: Manager's complete state (no noise)
            private_features = manager.get_state_features()
            
            # 2. public information layer: environment state (no noise, all Managers visible)
            # public_features already obtained above
            
            # 3. limited others information layer: extremely simplified collaboration information (configurable noise and quality degradation)
            limited_others_features = self.dec_pomdp_obs_space.compute_limited_other_manager_info(
                limited_collaboration_info, manager_id
            )
            
            # apply dynamic observation quality degradation (if enabled)
            if self.dynamic_quality_manager:
                # calculate current observation quality
                other_manager_ids = [mid for mid in self.manager_ids if mid != manager_id]
                quality_metrics = self.dynamic_quality_manager.calculate_observation_quality(
                    manager_id, other_manager_ids
                )
                
                # apply quality degradation to others information
                if len(limited_others_features) > 0:
                    limited_others_features = self.dynamic_quality_manager.apply_quality_degradation(
                        limited_others_features, quality_metrics
                    )
                
                # update quality history
                self.dynamic_quality_manager.update_quality_history(manager_id, quality_metrics)
            
            # Dec-POMDP observation layer combination (ensure public information consistency)
            # private information: Manager's own state (can be slightly processed)
            processed_private_features = private_features  # temporarily not processed, maintain completeness
            
            # others information: apply information transmission mechanism (may have noise and delay)
            if len(limited_others_features) > 0:
                # apply information transmission mechanism only to others information
                processed_others_features = self._apply_enhanced_information_mechanisms(
                    limited_others_features, manager_id
                )
                
                # combine three layers of observations: private + public (no noise) + processed others
                arrays_to_concat = []
                if processed_private_features is not None:
                    arrays_to_concat.append(processed_private_features)
                if public_features is not None:
                    arrays_to_concat.append(public_features)
                if processed_others_features is not None:
                    arrays_to_concat.append(processed_others_features)
                
                dec_pomdp_observation = np.concatenate(arrays_to_concat) if arrays_to_concat else np.array([])
            else:
                # if other Manager information is disabled, only include private and public information
                arrays_to_concat = []
                if processed_private_features is not None:
                    arrays_to_concat.append(processed_private_features)
                if public_features is not None:
                    arrays_to_concat.append(public_features)
                
                dec_pomdp_observation = np.concatenate(arrays_to_concat) if arrays_to_concat else np.array([])
            
            # ensure observation dimension is 73
            current_dim = len(dec_pomdp_observation)
            if current_dim < 73:
                # if dimension is less than 73, pad to 73
                padding = np.zeros(73 - current_dim)
                dec_pomdp_observation = np.concatenate([dec_pomdp_observation, padding])
            elif current_dim > 73:
                # if dimension is greater than 73, truncate to 73
                dec_pomdp_observation = dec_pomdp_observation[:73]
            
            # update observation history
            if manager_id not in self.observation_history:
                self.observation_history[manager_id] = []
            self.observation_history[manager_id].append(dec_pomdp_observation.copy())
            
            # limit history length
            max_history_len = max(10, self.dec_pomdp_config.max_delay_steps + 5)
            if len(self.observation_history[manager_id]) > max_history_len:
                self.observation_history[manager_id] = self.observation_history[manager_id][-max_history_len:]
            
            observations[manager_id] = dec_pomdp_observation.astype(np.float32)
        
        return observations
    
    def _get_dec_pomdp_public_features(self) -> np.ndarray:

        # 1. time information layer (6D) - standardized time representation
        time_layer = self._get_standardized_time_features()
        
        # 2. market information layer (7D) - complete market state
        market_layer = self._get_standardized_market_features()
        
        # 3. environment information layer (5D) - standardized environment state
        environment_layer = self._get_standardized_environment_features()
        
        # combine all public information layers
        public_features = np.concatenate([time_layer, market_layer, environment_layer])
        
        # validate public information layer completeness
        if self.dec_pomdp_config.enable_observation_noise:  # only validate in debug mode
            self._validate_public_information_layer(public_features, time_layer, market_layer, environment_layer)
        
        return public_features
    
    def _get_standardized_time_features(self) -> np.ndarray:

        hour = self.current_time.hour
        day_of_year = self.current_time.timetuple().tm_yday
        
        # periodic hour encoding
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # weekday/weekend encoding
        is_weekday = 1.0 if self.current_time.weekday() < 5 else 0.0
        
        # seasonal encoding (based on day of year)
        season_progress = math.sin(2 * math.pi * day_of_year / 365)
        
        # task time progress (standardized to [0,1])
        # ensure time progress does not exceed 1.0, current_step should be in [0, time_horizon-1] range
        time_progress = min(1.0, self.current_step / max(1, self.time_horizon))
        
        # remaining time urgency
        time_urgency = min(1.0, (self.time_horizon - self.current_step) / max(1, self.time_horizon * 0.1))
        
        return np.array([hour_sin, hour_cos, is_weekday, season_progress, time_progress, time_urgency])
    
    def _get_standardized_market_features(self) -> np.ndarray:
 
        # ensure using cached environment state, avoid inconsistency caused by randomness
        if not hasattr(self, '_cached_env_state') or self._env_state_cache_time != self.current_step:
            self._cached_env_state = self.env_dynamics.get_current_state(self.current_time)
            self._env_state_cache_time = self.current_step
        
        env_state = self._cached_env_state
        
        # standardized current price (relative to base price)
        base_price = 0.12  # base price $/kWh
        if env_state is None:
            env_state = {'price': base_price, 'price_trend': 0.0, 'future_prices': [base_price] * 3}
        
        normalized_price = env_state['price'] / base_price  # relative price
        
        # price trend strength (standardized)
        price_trend_strength = np.tanh(env_state['price_trend'])  # use tanh to limit to [-1,1]
        
        # future price prediction (standardized)
        future_prices = env_state.get('future_prices', [env_state['price']] * 3)
        future_price_1 = future_prices[0] / base_price if len(future_prices) > 0 else normalized_price
        future_price_2 = future_prices[1] / base_price if len(future_prices) > 1 else normalized_price
        future_price_3 = future_prices[2] / base_price if len(future_prices) > 2 else normalized_price
        
        # market phase clearly identified
        hour = self.current_time.hour

        is_peak_period = 1.0 if (7 <= hour <= 9) or (18 <= hour <= 21) else 0.0

        is_valley_period = 1.0 if (hour >= 23) or (hour <= 6) else 0.0
        
        return np.array([
            normalized_price, price_trend_strength, 
            future_price_1, future_price_2, future_price_3,
            is_peak_period, is_valley_period
        ])
    
    def _get_standardized_environment_features(self) -> np.ndarray:

        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # standardized temperature (relative to comfort temperature 20°C)
        comfort_temp = 20.0
        normalized_temperature = (env_state['temperature'] - comfort_temp) / 15.0  # assume ±15°C is reasonable range
        
        # standardized solar irradiance (relative to standard test condition 1000 W/m²)
        standard_irradiance = 1000.0
        normalized_irradiance = env_state['solar_irradiance'] / standard_irradiance
        
        # environment trend standardized
        temp_trend = np.tanh(env_state['weather_trend']['temperature_trend'])  # limit to [-1,1]
        irradiance_trend = np.tanh(env_state['weather_trend']['irradiance_trend'])  # limit to [-1,1]
        
        # daylight quality indicator (considering irradiance and time)
        hour = self.current_time.hour
        daylight_quality = float(normalized_irradiance * max(0, math.sin(math.pi * (hour - 6) / 12))) if 6 <= hour <= 18 else 0.0
        
        return np.array([
            float(normalized_temperature), float(normalized_irradiance), 
            float(temp_trend), float(irradiance_trend), float(daylight_quality)
        ])
    
    def _validate_public_information_layer(self, public_features: np.ndarray, 
                                         time_layer: np.ndarray, 
                                         market_layer: np.ndarray, 
                                         environment_layer: np.ndarray):

        # dimension validation
        expected_dims = {
            'time_layer': 6,
            'market_layer': 7,
            'environment_layer': 5,
            'total': 18
        }
        
        assert len(time_layer) == expected_dims['time_layer'], f"time layer dimension error: {len(time_layer)} != {expected_dims['time_layer']}"
        assert len(market_layer) == expected_dims['market_layer'], f"market layer dimension error: {len(market_layer)} != {expected_dims['market_layer']}"
        assert len(environment_layer) == expected_dims['environment_layer'], f"environment layer dimension error: {len(environment_layer)} != {expected_dims['environment_layer']}"
        assert len(public_features) == expected_dims['total'], f"public information total dimension error: {len(public_features)} != {expected_dims['total']}"
        
        # value range validation
        # time layer validation: sin/cos values should be in [-1,1], others should be in [0,1]
        assert -1.1 <= time_layer[0] <= 1.1, f"hour sin encoding out of range: {time_layer[0]}"
        assert -1.1 <= time_layer[1] <= 1.1, f"hour cos encoding out of range: {time_layer[1]}"
        assert 0 <= time_layer[4] <= 1.1, f"time progress out of range: {time_layer[4]}"
        
        # market layer validation: price should be positive, trend should be in [-1,1]
        assert market_layer[0] > 0, f"standardized price should be positive: {market_layer[0]}"
        assert -1.1 <= market_layer[1] <= 1.1, f"price trend out of range: {market_layer[1]}"
        
        # environment layer validation: irradiance should be non-negative, trend should be in [-1,1]
        assert environment_layer[1] >= 0, f"standardized irradiance should be non-negative: {environment_layer[1]}"
        assert -1.1 <= environment_layer[2] <= 1.1, f"temperature trend out of range: {environment_layer[2]}"
        
        # temporal consistency validation
        hour = self.current_time.hour
        expected_hour_sin = math.sin(2 * math.pi * hour / 24)
        assert abs(time_layer[0] - expected_hour_sin) < 0.01, f"hour encoding inconsistent: {time_layer[0]} vs {expected_hour_sin}"
    
    def _get_limited_collaboration_info(self) -> Dict[str, List[float]]:

        manager_info = {}
        
        # calculate system-level aggregation statistics
        system_stats = self._calculate_system_aggregation_stats()
        
        # calculate dynamic aggregation weights
        aggregation_weights = self._calculate_dynamic_aggregation_weights()
        
        for manager_id, manager in self.manager_agents.items():
            # 1. basic aggregation metrics (scale-related)
            scale_metrics = self._aggregate_scale_metrics(manager, system_stats)
            
            # 2. performance aggregation metrics (efficiency-related)
            performance_metrics = self._aggregate_performance_metrics(manager, system_stats, aggregation_weights)
            
            # 3. collaboration aggregation metrics (system-related)
            collaboration_metrics = self._aggregate_collaboration_metrics(manager, system_stats)
            
            # 4. temporal aggregation metrics (trend-related)
            temporal_metrics = self._aggregate_temporal_metrics(manager, manager_id)
            
            # 5. adaptive aggregation strategy
            adaptive_metrics = self._apply_adaptive_aggregation_strategy(
                manager, manager_id, system_stats
            )
            
            # combine all aggregation metrics
            aggregated_info = (
                scale_metrics + 
                performance_metrics + 
                collaboration_metrics + 
                temporal_metrics + 
                adaptive_metrics
            )
            
            manager_info[manager_id] = aggregated_info
        
        return manager_info
    
    def _calculate_system_aggregation_stats(self) -> Dict[str, float]:
        """calculate system-level aggregation statistics"""
        all_users = [len(m.users) for m in self.manager_agents.values()]
        all_devices = [len(m.device_mdps) for m in self.manager_agents.values()]
        all_energies = [m.markov_history['cumulative_energy'] for m in self.manager_agents.values()]
        all_costs = [m.markov_history['cumulative_cost'] for m in self.manager_agents.values()]
        all_satisfactions = [m.markov_history['user_satisfaction'] for m in self.manager_agents.values()]
        
        return {
            'total_users': int(sum(all_users)),
            'total_devices': int(sum(all_devices)),
            'total_energy': float(sum(all_energies)),
            'total_cost': float(sum(all_costs)),
            'avg_satisfaction': float(np.mean(all_satisfactions)) if all_satisfactions else 0.0,
            'energy_std': float(np.std(all_energies)) if len(all_energies) > 1 else 0.0,
            'satisfaction_std': float(np.std(all_satisfactions)) if len(all_satisfactions) > 1 else 0.0,
            'system_balance': 1.0 - (float(np.std(all_energies)) / max(float(np.mean(all_energies)), 1.0)) if all_energies else 1.0,
        }
    
    def _calculate_dynamic_aggregation_weights(self) -> Dict[str, float]:
        """calculate dynamic aggregation weights"""
        # based on current time step and system state
        time_progress = self.current_step / max(1, self.time_horizon)
        
        # early stage more focused on scale and settings, later stage more focused on performance
        weights = {
            'scale_weight': max(0.3, 1.0 - time_progress),  # early stage weight high
            'performance_weight': min(1.0, 0.5 + time_progress),  # later stage weight high
            'collaboration_weight': 0.6 + 0.2 * math.sin(time_progress * math.pi),  # mid-stage weight high
            'temporal_weight': min(1.0, time_progress * 2),  # increase with time
        }
        
        return weights
    
    def _aggregate_scale_metrics(self, manager: 'ManagerAgent', system_stats: Dict[str, float]) -> List[float]:
        """aggregate scale-related metrics"""
        # relative scale (standardized to [0,1])
        user_ratio = len(manager.users) / max(1, system_stats['total_users'])
        device_ratio = len(manager.device_mdps) / max(1, system_stats['total_devices'])
        
        # relative capacity (estimated based on device number)
        capacity_indicator = min(1.0, len(manager.device_mdps) / 30.0)  # assume 30 as high capacity threshold
        
        return [user_ratio, device_ratio, capacity_indicator]
    
    def _aggregate_performance_metrics(self, manager: 'ManagerAgent', 
                                     system_stats: Dict[str, float],
                                     weights: Dict[str, float]) -> List[float]:
        """aggregate performance-related metrics"""
        # energy efficiency metric (energy consumption relative to scale)
        manager_energy = manager.markov_history['cumulative_energy']
        manager_users = len(manager.users)
        
        if manager_users > 0 and system_stats['total_energy'] > 0:
            energy_efficiency = 1.0 - (manager_energy / manager_users) / (system_stats['total_energy'] / system_stats['total_users'])
            energy_efficiency = np.clip(energy_efficiency, -1.0, 1.0)
        else:
            energy_efficiency = 0.0
        
        # relative satisfaction level
        satisfaction_level = manager.markov_history['user_satisfaction'] - system_stats['avg_satisfaction']
        satisfaction_level = np.clip(satisfaction_level, -1.0, 1.0)
        
        # comprehensive performance metric
        performance_score = (energy_efficiency * 0.6 + satisfaction_level * 0.4) * weights['performance_weight']
        
        return [energy_efficiency, satisfaction_level, performance_score]
    
    def _aggregate_collaboration_metrics(self, manager: 'ManagerAgent', 
                                       system_stats: Dict[str, float]) -> List[float]:
        """aggregate collaboration-related metrics"""
        # system contribution (based on energy consumption ratio)
        if system_stats['total_energy'] > 0:
            contribution_ratio = manager.markov_history['cumulative_energy'] / system_stats['total_energy']
        else:
            contribution_ratio = 0.0
        
        # system balance impact (the impact of this Manager on system balance)
        balance_impact = 1.0 - abs(contribution_ratio - (1.0 / len(self.manager_agents)))
        
        # collaboration activity level (based on whether there is actual activity)
        activity_level = 1.0 if manager.markov_history['cumulative_energy'] > 0 else 0.0
        
        return [contribution_ratio, balance_impact, activity_level]
    
    def _aggregate_temporal_metrics(self, manager: 'ManagerAgent', manager_id: str) -> List[float]:
        """aggregate temporal-related metrics"""
        # get history data (using user satisfaction history in markov_history)
        # ManagerAgent class uses markov_history to store history information, not performance_history
        history = [manager.markov_history['user_satisfaction']]
        
        # trend indicator
        if len(history) >= 2:
            recent_trend = history[-1] - history[-2]
            trend_indicator = np.clip(recent_trend, -1.0, 1.0)
        else:
            trend_indicator = 0.0
        
        # stability indicator
        if len(history) >= 3:
            stability = 1.0 - np.std(history[-3:]) / max(np.mean(history[-3:]), 0.1)
            stability = np.clip(stability, 0.0, 1.0)
        else:
            stability = 1.0
        
        # improvement potential (based on historical best performance)
        if history:
            current_performance = history[-1]
            best_performance = max(history)
            improvement_potential = max(0.0, best_performance - current_performance)
        else:
            improvement_potential = 0.5  # neutral value
        
        return [float(trend_indicator), float(stability), float(improvement_potential)]
    
    def _apply_adaptive_aggregation_strategy(self, manager: 'ManagerAgent', 
                                           manager_id: str,
                                           system_stats: Dict[str, float]) -> List[float]:
        """apply adaptive aggregation strategy"""
        # adjust aggregation strategy based on system state
        
        # 1. system pressure indicator
        if system_stats['total_energy'] > 0:
            system_load = system_stats['total_energy'] / (system_stats['total_users'] * 24)  # average energy consumption per user per hour
            pressure_indicator = min(1.0, system_load / 5.0)  # assume 5kWh/user/hour as high pressure
        else:
            pressure_indicator = 0.0
        
        # 2. coordination need indicator
        coordination_need = 1.0 - system_stats['system_balance']  # system imbalance, higher coordination need
        
        # 3. adaptive response capability
        manager_flexibility = len(manager.controllable_devices) / max(1, len(manager.device_mdps))
        
        # 4. relative importance (based on scale and performance)
        scale_importance = len(manager.users) / max(1, system_stats['total_users'])
        performance_importance = abs(manager.markov_history['user_satisfaction'] - system_stats['avg_satisfaction'])
        relative_importance = (scale_importance * 0.6 + performance_importance * 0.4)
        
        return [pressure_indicator, coordination_need, manager_flexibility, relative_importance]
    
    def _apply_enhanced_information_mechanisms(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        processed_observation = observation.copy()
        
        # 1. apply multi-level delay mechanism
        processed_observation = self._apply_multi_level_delay(processed_observation, manager_id)
        
        # 2. apply intelligent information loss mechanism
        processed_observation = self._apply_intelligent_information_loss(processed_observation, manager_id)
        
        # 3. apply network interruption simulation
        processed_observation = self._apply_network_interruption_simulation(processed_observation, manager_id)
        
        # 4. apply transmission quality degradation
        processed_observation = self._apply_transmission_quality_degradation(processed_observation, manager_id)
        
        # 5. apply information retransmission and recovery mechanism
        processed_observation = self._apply_information_recovery_mechanism(processed_observation, manager_id)
        
        return processed_observation
    
    def _apply_multi_level_delay(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        if not self.dec_pomdp_config.enable_info_delay:
            return observation
        
        delayed_observation = observation.copy()
        
        # 1. fixed delay (basic configuration)
        if manager_id in self.observation_history and len(self.observation_history[manager_id]) > 0:
            fixed_delay_steps = self.dec_pomdp_config.max_delay_steps
            if len(self.observation_history[manager_id]) >= fixed_delay_steps:
                delayed_observation = self.observation_history[manager_id][-fixed_delay_steps].copy()
        
        # 2. random delay (simulate network jitter)
        if np.random.random() < 0.3:  # 30% probability of random delay
            random_delay = np.random.randint(1, min(3, len(self.observation_history[manager_id])) + 1)
            if manager_id in self.observation_history and len(self.observation_history[manager_id]) >= random_delay:
                delayed_observation = self.observation_history[manager_id][-random_delay].copy()
        
        # 3. network delay (based on dynamic quality manager)
        if self.dynamic_quality_manager:
            network_conditions = getattr(self.dynamic_quality_manager, 'network_history', [])
            if network_conditions:
                from fo_common.dynamic_observation_quality import NetworkCondition
                current_condition = network_conditions[-1] if network_conditions else NetworkCondition.GOOD
                
                # network condition worse, delay larger
                network_delay_prob = {
                    NetworkCondition.EXCELLENT: 0.05,
                    NetworkCondition.GOOD: 0.1,
                    NetworkCondition.FAIR: 0.25,
                    NetworkCondition.POOR: 0.5,
                    NetworkCondition.CRITICAL: 0.8
                }.get(current_condition, 0.1)
                
                if np.random.random() < network_delay_prob:
                    network_delay_steps = {
                        NetworkCondition.EXCELLENT: 1,
                        NetworkCondition.GOOD: 1,
                        NetworkCondition.FAIR: 2,
                        NetworkCondition.POOR: 3,
                        NetworkCondition.CRITICAL: 4
                    }.get(current_condition, 1)
                    
                    if (manager_id in self.observation_history and 
                        len(self.observation_history[manager_id]) >= network_delay_steps):
                        delayed_observation = self.observation_history[manager_id][-network_delay_steps].copy()
        
        # 4. load delay (based on system load)
        system_load = len(self.manager_agents) * self.current_step / max(1, self.time_horizon)
        if system_load > 0.7:  # high load, increase delay
            load_delay_prob = (system_load - 0.7) * 0.5
            if np.random.random() < load_delay_prob:
                if manager_id in self.observation_history and len(self.observation_history[manager_id]) >= 2:
                    delayed_observation = self.observation_history[manager_id][-2].copy()
        
        return delayed_observation
    
    def _apply_intelligent_information_loss(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        if not self.dec_pomdp_config.enable_info_missing:
            return observation
        
        lost_observation = observation.copy()
        missing_prob = self.dec_pomdp_config.missing_probability
        
        # 1. choosen disable
        # assume the first 1/3 of the observation vector is the most important private information, should not be lost
        protected_length = len(observation) // 3
        vulnerable_start = protected_length
        
        # 2. time-based missing
        time_factor = math.sin(self.current_step * 0.1) * 0.1 + 1.0  # periodic change
        adjusted_missing_prob = missing_prob * time_factor
        
        # 3. component missing
        for i in range(len(lost_observation)):
            if i >= vulnerable_start:  # protect private information
                # the probability of losing other information is higher
                component_missing_prob = adjusted_missing_prob * 1.5
                
                # 4. cumulative missing (the farther from the current time, the more likely to lose information)
                distance_factor = 1.0 + (i - vulnerable_start) * 0.1
                final_missing_prob = min(0.8, component_missing_prob * distance_factor)
                
                if np.random.random() < final_missing_prob:
                    lost_observation[i] = 0.0  # information loss
        
        # 5. batch missing (simulate connection interruption)
        if np.random.random() < 0.05:  # 5% probability of batch missing
            batch_start = max(vulnerable_start, np.random.randint(0, len(lost_observation) - 5))
            batch_length = min(5, len(lost_observation) - batch_start)
            lost_observation[batch_start:batch_start + batch_length] = 0.0
        
        return lost_observation
    
    def _apply_network_interruption_simulation(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        interrupted_observation = observation.copy()
        
        # initialize Manager network state (if not exist)
        if not hasattr(self, 'manager_network_states'):
            self.manager_network_states = {mid: 'connected' for mid in self.manager_ids}
        
        # 1. instantaneous interruption (short-term complete interruption)
        interruption_prob = 0.02  # 2% probability of instantaneous interruption
        if np.random.random() < interruption_prob:
            self.manager_network_states[manager_id] = 'interrupted'
            # during instantaneous interruption, only keep private information
            private_length = len(interrupted_observation) // 3
            interrupted_observation[private_length:] = 0.0
        
        # 2. intermittent interruption (based on sine wave pattern)
        intermittent_factor = math.sin(self.current_step * 0.3) + 1.0
        if intermittent_factor < 0.5 and np.random.random() < 0.1:
            self.manager_network_states[manager_id] = 'intermittent'
            # during intermittent interruption, randomly lose 50% of other information
            private_length = len(interrupted_observation) // 3
            for i in range(private_length, len(interrupted_observation)):
                if np.random.random() < 0.5:
                    interrupted_observation[i] = 0.0
        
        # 3. partition interruption (simulate Manager connection problem) - commented, no noise environment test
        # if hasattr(self, 'network_partition_active'):
        #     if self.network_partition_active and manager_id in getattr(self, 'partitioned_managers', []):
        #         # 分区中的Manager无法获得其他Manager信息
        #         private_length = len(interrupted_observation) // 3
        #         interrupted_observation[private_length:] = interrupted_observation[private_length:] * 0.1
        
        # 4. degradation interruption (connection quality seriously degraded)
        if self.dynamic_quality_manager:
            network_history = getattr(self.dynamic_quality_manager, 'network_history', [])
            if network_history:
                from fo_common.dynamic_observation_quality import NetworkCondition
                current_condition = network_history[-1]
                if current_condition == NetworkCondition.CRITICAL:
                    # when severely degraded, significantly reduce the quality of other information
                    private_length = len(interrupted_observation) // 3
                    degradation_factor = 0.2
                    interrupted_observation[private_length:] = interrupted_observation[private_length:] * degradation_factor
        
        # network state recovery mechanism
        recovery_prob = 0.3  # 30% probability of recovery connection
        if (self.manager_network_states[manager_id] != 'connected' and 
            np.random.random() < recovery_prob):
            self.manager_network_states[manager_id] = 'connected'
        
        return interrupted_observation
    
    def _apply_transmission_quality_degradation(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        degraded_observation = observation.copy()
        
        # 1. SNR degradation (transmission noise) - commented, no noise environment test
        # if hasattr(self.dec_pomdp_config, 'enable_transmission_noise'):
        #     enable_noise = self.dec_pomdp_config.enable_transmission_noise
        # else:
        #     enable_noise = True  # default enabled
        enable_noise = False  # no noise environment test
        
        if enable_noise:
            # adjust noise level based on network conditions
            base_noise_level = 0.01
            if self.dynamic_quality_manager:
                network_history = getattr(self.dynamic_quality_manager, 'network_history', [])
                if network_history:
                    from fo_common.dynamic_observation_quality import NetworkCondition
                    current_condition = network_history[-1]
                    noise_multiplier = {
                        NetworkCondition.EXCELLENT: 0.5,
                        NetworkCondition.GOOD: 1.0,
                        NetworkCondition.FAIR: 2.0,
                        NetworkCondition.POOR: 4.0,
                        NetworkCondition.CRITICAL: 8.0
                    }.get(current_condition, 1.0)
                    
                    noise_level = base_noise_level * noise_multiplier
                    transmission_noise = np.random.normal(0, noise_level, degraded_observation.shape)
                    degraded_observation += transmission_noise
        
        # 2. quantization degradation (numerical precision下降)
        if np.random.random() < 0.1:  # 10% probability of quantization degradation
            quantization_levels = 256  # 8-bit quantization
            degraded_observation = np.round(degraded_observation * quantization_levels) / quantization_levels
        
        # 3. compression degradation (simulate data compression loss)
        if np.random.random() < 0.05:  # 5% probability of compression degradation
            compression_factor = 0.95
            degraded_observation = degraded_observation * compression_factor
        
        # 4. attenuation degradation (signal attenuation related to distance and time)
        distance_factor = 1.0  # assume all Managers are at the same distance
        time_factor = 1.0 - (self.current_step / self.time_horizon) * 0.05  # time attenuation
        attenuation_factor = distance_factor * time_factor
        
        if attenuation_factor < 1.0:
            degraded_observation = degraded_observation * attenuation_factor
        
        return degraded_observation
    
    def _apply_information_recovery_mechanism(self, observation: np.ndarray, manager_id: str) -> np.ndarray:

        recovered_observation = observation.copy()
        
        # initialize recovery cache
        if not hasattr(self, 'recovery_cache'):
            self.recovery_cache = {}
        if manager_id not in self.recovery_cache:
            self.recovery_cache[manager_id] = []
        
        # 1. cache recovery (use the most recent valid value)
        valid_indices = np.where(np.abs(recovered_observation) > 1e-8)[0]  # non-zero value is considered valid
        invalid_indices = np.where(np.abs(recovered_observation) <= 1e-8)[0]  # zero value is considered lost
        
        if len(self.recovery_cache[manager_id]) > 0 and len(invalid_indices) > 0:
            last_valid_observation = self.recovery_cache[manager_id][-1]
            
            for idx in invalid_indices:
                if idx < len(last_valid_observation):
                    # 2. estimated recovery (based on historical trend)
                    if len(self.recovery_cache[manager_id]) >= 2:
                        recent_values = [cache[idx] for cache in self.recovery_cache[manager_id][-2:] 
                                       if idx < len(cache)]
                        if len(recent_values) >= 2:
                            trend = recent_values[-1] - recent_values[-2]
                            estimated_value = recent_values[-1] + trend * 0.5  # conservative estimate
                            recovered_observation[idx] = estimated_value
                        else:
                            recovered_observation[idx] = last_valid_observation[idx]
                    else:
                        recovered_observation[idx] = last_valid_observation[idx]
                else:
                    # 3. default value recovery
                    recovered_observation[idx] = 0.0
        
        # 4. interpolation recovery (interpolate for continuous loss)
        for i in range(1, len(recovered_observation) - 1):
            if (abs(recovered_observation[i]) <= 1e-8 and 
                abs(recovered_observation[i-1]) > 1e-8 and 
                abs(recovered_observation[i+1]) > 1e-8):
                # linear interpolation
                recovered_observation[i] = (recovered_observation[i-1] + recovered_observation[i+1]) / 2.0
        
        # update recovery cache
        self.recovery_cache[manager_id].append(recovered_observation.copy())
        
        # limit cache size
        if len(self.recovery_cache[manager_id]) > 10:
            self.recovery_cache[manager_id] = self.recovery_cache[manager_id][-10:]
        
        return recovered_observation
    
    def _get_environment_features(self) -> np.ndarray:

        # time features
        hour = self.current_time.hour
        time_features = np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if self.current_time.weekday() < 5 else 0.0,
            self.current_step / self.time_horizon
        ])
        
        # environment state
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # price features
        price_features = np.array([
            env_state['price'],
            env_state['price_trend'],
            env_state.get('future_prices', [0, 0, 0])[0] if env_state.get('future_prices') else 0,
            env_state.get('future_prices', [0, 0, 0])[1] if env_state.get('future_prices') else 0,
            env_state.get('future_prices', [0, 0, 0])[2] if env_state.get('future_prices') else 0
        ])
        
        # weather features
        weather_features = np.array([
            env_state['temperature'],
            env_state['solar_irradiance'],
            env_state['weather_trend']['temperature_trend'],
            env_state['weather_trend']['irradiance_trend']
        ])
        
        return np.concatenate([time_features, price_features, weather_features])
    
    def _get_all_manager_aggregated_info(self) -> Dict[str, List[float]]:

        # if other Manager information is explicitly disabled, return empty information
        if not self.dec_pomdp_config.enable_other_manager_info:
            return {manager_id: [] for manager_id in self.manager_agents.keys()}
        
        # determine information sharing level based on configuration - commented, no noise environment test
        # if hasattr(self.dec_pomdp_config, 'information_sharing_level'):
        #     sharing_level = self.dec_pomdp_config.information_sharing_level
        # else:
        #     sharing_level = 'limited'  # default is limited mode
        sharing_level = 'full'  # no noise environment test, use full information sharing
        
        if sharing_level == 'none':
            # no information sharing mode: completely isolated POMDP
            return {manager_id: [] for manager_id in self.manager_agents.keys()}
        
        elif sharing_level == 'minimal':
            # minimal information sharing: only existence indicator
            minimal_info = {}
            for manager_id in self.manager_agents.keys():
                minimal_info[manager_id] = [1.0]  # only represent Manager existence
            return minimal_info
        
        elif sharing_level == 'limited':
            # limited information sharing: call new limited collaboration information method
            return self._get_limited_collaboration_info()
        
        elif sharing_level == 'legacy':
            # traditional mode: keep original detailed information (only for backward compatibility and debugging)
            return self._get_legacy_detailed_manager_info()
        
        else:
            # default use limited mode
            return self._get_limited_collaboration_info()
    
    def _get_legacy_detailed_manager_info(self) -> Dict[str, List[float]]:

        manager_info = {}
        
        # calculate total system statistics
        total_users = sum(len(m.users) for m in self.manager_agents.values())
        total_devices = sum(len(m.device_mdps) for m in self.manager_agents.values())
        total_cost = sum(m.markov_history['cumulative_cost'] for m in self.manager_agents.values())
        total_energy = sum(m.markov_history['cumulative_energy'] for m in self.manager_agents.values())
        avg_satisfaction = np.mean([m.markov_history['user_satisfaction'] for m in self.manager_agents.values()])
        
        for manager_id, manager in self.manager_agents.items():
            # traditional detailed information (deprecated, but keep compatibility)
            legacy_info = [
                len(manager.users),  # absolute number of users
                len(manager.device_mdps),  # absolute number of devices
                manager.markov_history['cumulative_cost'],  # exact cost
                manager.markov_history['cumulative_energy'],  # exact energy
                manager.markov_history['user_satisfaction'],  # exact satisfaction
                
                # relative indicators (slightly limited)
                len(manager.users) / max(1, total_users),
                len(manager.device_mdps) / max(1, total_devices),
                manager.markov_history['cumulative_cost'] / max(1, total_cost) if total_cost > 0 else 0,
                manager.markov_history['cumulative_energy'] / max(1, total_energy) if total_energy > 0 else 0,
                manager.markov_history['user_satisfaction'] - avg_satisfaction,
            ]
            
            manager_info[manager_id] = legacy_info
        
        return manager_info
    
    def _get_market_state_features(self) -> np.ndarray:

        # overall market state
        total_devices = sum(len(m.device_mdps) for m in self.manager_agents.values())
        total_controllable = sum(len(m.controllable_devices) for m in self.manager_agents.values())
        total_users = sum(len(m.users) for m in self.manager_agents.values())
        
        # calculate overall system capability indicators
        avg_devices_per_user = total_devices / max(1, total_users)
        controllability_ratio = total_controllable / max(1, total_devices)
        
        # calculate competition intensity between Managers
        manager_count = len(self.manager_agents)
        user_distribution_variance = 0.0
        if manager_count > 1:
            user_counts = [len(m.users) for m in self.manager_agents.values()]
            user_distribution_variance = np.var(user_counts) / max(1, np.mean(user_counts))
        
        # calculate satisfaction distribution
        satisfactions = [m.markov_history['user_satisfaction'] for m in self.manager_agents.values()]
        satisfaction_mean = np.mean(satisfactions)
        satisfaction_std = np.std(satisfactions)
        
        # calculate energy distribution
        energies = [m.markov_history['cumulative_energy'] for m in self.manager_agents.values()]
        energy_balance = np.std(energies) / max(1, np.mean(energies)) if np.mean(energies) > 0 else 0
        
        # time-related market state
        time_progress = self.current_step / max(1, self.time_horizon)
        is_peak_hour = 1.0 if 7 <= self.current_time.hour <= 9 or 18 <= self.current_time.hour <= 21 else 0.0
        is_off_peak = 1.0 if 23 <= self.current_time.hour or self.current_time.hour <= 6 else 0.0
        
        # historical trend features (simplified version based on current available data)
        recent_activity = min(1.0, self.current_step / 5.0)  # activity indicator
        
        market_features = np.array([
            # system size features
            total_users,
            total_devices,
            total_controllable,
            avg_devices_per_user,
            controllability_ratio,
            
            # competition and distribution features
            manager_count,
            user_distribution_variance,
            energy_balance,
            satisfaction_mean,
            satisfaction_std,
            
            # time and activity features
            time_progress,
            is_peak_hour,
            is_off_peak,
            recent_activity,
            
            # system state indicators
            1.0 if satisfaction_mean > 0.5 else 0.0,  # whether the system satisfaction is good
            1.0 if energy_balance < 0.5 else 0.0,     # whether the energy consumption is balanced
        ])
        
        return market_features.astype(np.float32)
    
    def generate_all_dfos(self) -> Dict[str, Dict[str, DFOSystem]]:
        """generate all Manager's DFO systems"""
        all_dfos = {}
        
        for manager_id, manager in self.manager_agents.items():
            manager_dfos = manager.generate_dfo(self.time_horizon)
            all_dfos[manager_id] = manager_dfos
        
        return all_dfos
    
    def get_manager_count(self) -> int:
        """get Manager count"""
        return len(self.manager_agents)
    
    def get_total_user_count(self) -> int:
        """get total user count"""
        return sum(len(manager.users) for manager in self.manager_agents.values())
    
    def get_total_device_count(self) -> int:
        """get total device count"""
        return sum(len(manager.device_mdps) for manager in self.manager_agents.values())
    
    def get_current_observations(self):
        """get current time step observations"""
        obs = {}
        for manager_id, agent in self.manager_agents.items():
            obs[manager_id] = agent.get_observation()
        return obs
    
    def generate_current_dfos(self, timestep):
        """generate current time step DFO systems"""
        dfo_systems = {}
        for manager_id, agent in self.manager_agents.items():
            agent_dfos = {}
            
            # process device list
            if isinstance(agent.devices, dict):
                devices_list = list(agent.devices.values())
            else:
                devices_list = agent.devices
            
            # generate DFO for each device
            for device in devices_list:
                device_id = getattr(device, 'device_id', f"{manager_id}_device_{len(agent_dfos)}")
                
                # generate FlexOffer based on device type - include core features
                from fo_generate.dfo import DFOSystem, DFOSlice
                from datetime import datetime, timedelta
                
                # generate basic FlexOffer parameters
                energy_min = np.random.uniform(5, 20)     # minimum energy demand
                energy_max = np.random.uniform(20, 50)    # maximum energy demand
                power_min = np.random.uniform(-10, 0)     # minimum power (negative value means discharge)
                power_max = np.random.uniform(5, 15)      # maximum power (positive value means charge)
                flexibility = np.random.uniform(0.2, 0.8) # flexibility factor
                
                # create time window
                current_time = datetime.now() + timedelta(hours=timestep)
                start_time = current_time  # time window start
                end_time = current_time + timedelta(hours=1)  # time window end
                
                # create DFO system
                dfo_system = DFOSystem(
                    time_horizon=1,
                    device_id=device_id,
                    device_type=getattr(device, 'device_type', 'unknown')
                )
                
                # create DFO slice
                dfo_slice = DFOSlice(
                    time_step=timestep,
                    energy_min=energy_min,
                    energy_max=energy_max,
                    constraints=[],  # basic constraints, can be extended later
                    power_min=power_min,
                    power_max=power_max,
                    start_time=start_time,
                    end_time=end_time,
                    flexibility_factor=flexibility,
                    device_type=getattr(device, 'device_type', 'unknown'),
                    device_id=device_id
                )
                
                # add slice to DFO system
                dfo_system.add_slice(dfo_slice)
                
                agent_dfos[device_id] = dfo_system
            
            if agent_dfos:
                dfo_systems[manager_id] = agent_dfos
        
        # merge DFO generation information into one line
        total_dfos = sum(len(dfos) for dfos in dfo_systems.values())
        manager_dfo_counts = [f"{manager_id}:{len(dfos)}" for manager_id, dfos in dfo_systems.items()]
        logger.info(f"时间步 {timestep} DFO生成: {', '.join(manager_dfo_counts)}, 总计 {total_dfos} 个")
        return dfo_systems
    
    def update_user_states(self, user_satisfied_energy, timestep):
        """update user states based on allocated energy"""
        try:
            # update device states based on user satisfied energy
            for manager_id, agent in self.manager_agents.items():
                for user in agent.users:
                    # process user object: users may be a list of dictionaries
                    if isinstance(user, dict):
                        user_id = user.get('user_id', '')
                        user_devices = user.get('devices', [])
                    else:
                        # if it is an object, directly access the attributes
                        user_id = getattr(user, 'user_id', '')
                        user_devices = getattr(user, 'devices', [])
                    
                    if user_id:
                        try:
                            # parse user ID format: support user_X and user_manager_X_Y format
                            if 'manager_' in user_id:
                                # format: user_manager_X_Y, need to calculate global user index
                                parts = user_id.split('_')
                                if len(parts) >= 4:
                                    manager_num = int(parts[2])  # manager number (1, 2, 3, 4)
                                    user_local_num = int(parts[3])  # user number in manager (1, 2, ...)
                                    
                                    # calculate global user index based on Manager distribution
                                    # Manager 1: 6 users (index 0-5), Manager 2: 10 users (index 6-15), 
                                    # Manager 3: 8 users (index 16-23), Manager 4: 12 users (index 24-35)
                                    user_distributions = [6, 10, 8, 12]
                                    base_index = sum(user_distributions[:manager_num-1])
                                    user_idx = base_index + (user_local_num - 1)
                                else:
                                    logger.warning(f"无法解析user_manager格式的ID: {user_id}")
                                    continue
                            else:
                                # format: user_X
                                user_idx = int(user_id.split('_')[1])
                            
                            if user_idx < user_satisfied_energy.shape[0]:
                                satisfied_energy = user_satisfied_energy[user_idx, timestep]
                                # allocate satisfied energy to user's devices
                                if user_devices and satisfied_energy > 0:
                                    energy_per_device = satisfied_energy / len(user_devices)
                                    for device in user_devices:
                                        # process device object: devices may be a list of dictionaries
                                        if isinstance(device, dict):
                                            device_id = device.get('device_id', '')
                                        else:
                                            device_id = getattr(device, 'device_id', '')
                                        
                                        if device_id:
                                            device_key = f"{user_id}_{device_id}"
                                            if device_key in agent.device_mdps:
                                                mdp_device = agent.device_mdps[device_key]
                                                if hasattr(mdp_device, 'env'):
                                                    # update device state to reflect the obtained energy
                                                    device_env = getattr(mdp_device, 'env', None)
                                                    if device_env is not None:
                                                        self._update_device_state_with_energy(device_env, energy_per_device)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"error parsing user ID: {user_id}: {e}")
        except Exception as e:
            logger.error(f"error updating user states: {e}")
    
    def _update_device_state_with_energy(self, device_env, allocated_energy):
        """update device state with allocated energy"""
        try:
            # update device state based on device type
            if hasattr(device_env, 'device_type'):
                if device_env.device_type == DeviceType.BATTERY:
                    # battery: update SOC state
                    if hasattr(device_env, 'battery_device') and hasattr(device_env.battery_device, 'charge'):
                        # use allocated energy for charging
                        device_env.battery_device.charge(allocated_energy, device_env.time_step)
                elif device_env.device_type == DeviceType.EV:
                    # electric vehicle: update charging state
                    if hasattr(device_env, 'ev_device') and hasattr(device_env.ev_device, 'charge'):
                        device_env.ev_device.charge(allocated_energy, device_env.time_step)
                # other device types can be updated here
        except Exception as e:
            logger.warning(f"warning updating device state: {e}") 
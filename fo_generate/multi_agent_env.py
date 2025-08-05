"""
多智能体FlexOffer环境 - Manager级别的多智能体系统

每个Manager作为一个独立的agent：
- 观测：自己管理的所有用户设备状态 + 其他Manager的聚合信息
- 动作：控制管理范围内所有可控设备
- 奖励：基于经济性、环保性和用户满意度的多目标优化
- 协作：通过观测其他Manager信息实现协作优化
"""

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
    """Manager代理类，管理一组用户和设备"""
    
    def __init__(self, manager_id: str, manager_config: Dict, users: List[Dict], devices: List[Dict]):
        self.manager_id = manager_id
        self.config = manager_config
        self.users = users
        self.devices = devices
        
        # 位置和覆盖信息
        self.location = (manager_config['location_x'], manager_config['location_y'])
        self.coverage_area = manager_config['coverage_area']
        self.district_type = manager_config['district_type']
        
        # 设备MDP对象
        self.device_mdps: Dict[str, DeviceMDPInterface] = {}
        self.device_types: Dict[str, str] = {}
        self.controllable_devices: List[str] = []
        
        # 用户偏好聚合
        self.aggregated_preferences = self._aggregate_user_preferences()
        
        # 初始化设备
        self._initialize_devices()
        
        # 马尔可夫历史
        self.markov_history = {
            'prev_actions': np.zeros(len(self.controllable_devices)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0,
            'user_satisfaction': 0.0
        }
    
    def _aggregate_user_preferences(self) -> Dict[str, float]:
        """聚合用户偏好"""
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
        """初始化设备MDP对象"""
        for device in self.devices:
            device_id = device['device_id']
            device_type = device['device_type']
            
            # 创建设备模型
            device_model = self._create_device_model(device_type, device)
            
            # 创建设备MDP
            device_mdp = self._create_device_mdp(device_type, device_model)
            
            self.device_mdps[device_id] = device_mdp
            self.device_types[device_id] = device_type
            
            # 记录可控设备
            if device_type not in [DeviceType.PV]:  # PV是不可控的
                self.controllable_devices.append(device_id)
        
        logger.info(f"Manager {self.manager_id}: 初始化 {len(self.device_mdps)} 个设备，"
                   f"其中 {len(self.controllable_devices)} 个可控")
    
    def _create_device_model(self, device_type: str, device_config: Dict):
        """创建设备模型"""
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
            
            # 创建用户行为
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
                total_energy=device_config.get('capacity', 3.0),  # 总能量需求
                power_rating=device_config['max_power'],
                operation_hours=device_config.get('param1', 3.5),  # 运行时长
                min_start_delay=device_config.get('param2', 0.5),  # 最小启动延迟
                max_start_delay=device_config.get('param3', 6.0),  # 最大启动延迟
                efficiency=device_config['efficiency'],
                can_interrupt=False  # 洗碗机不可中断
            )
            
            # 创建用户行为
            now = datetime.now()
            deployment_time = now + timedelta(hours=np.random.uniform(0, 2))  # 随机部署时间
            
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
            raise ValueError(f"不支持的设备类型: {device_type}")
    
    def _create_device_mdp(self, device_type: str, device_model) -> DeviceMDPInterface:
        """创建设备MDP"""
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
    
    def get_state_features(self, standardized: bool = True) -> np.ndarray:
        """
        获取Manager的状态特征 - 优化版本
        
        优化内容：
        1. 标准化的维度管理：确保所有Manager具有一致的状态向量维度
        2. 增强的特征工程：改进设备状态特征的表示方式
        3. 统一的特征缩放：保证数值范围的一致性
        4. 语义明确的特征编码：减少零值特征，提高信息密度
        5. 分层特征组织：按功能模块组织特征，便于理解和调试
        
        Args:
            standardized: 是否应用标准化处理
            
        Returns:
            np.ndarray: 标准化的Manager状态特征向量
        """
        if standardized:
            return self._get_standardized_state_features()
        else:
            return self._get_legacy_state_features()
    
    def _get_standardized_state_features(self) -> np.ndarray:
        """获取标准化的状态特征"""
        
        # 1. 设备特征模块（按类型组织和标准化）
        device_features = self._get_standardized_device_features()
        
        # 2. Manager管理特征模块（标准化）
        management_features = self._get_standardized_management_features()
        
        # 3. 用户偏好特征模块（已标准化）
        preference_features = self._get_standardized_preference_features()
        
        # 4. 系统状态特征模块（标准化）
        system_features = self._get_standardized_system_features()
        
        # 合并所有特征模块
        all_features = np.concatenate([
            device_features,      # 设备状态（最主要的特征）
            management_features,  # 管理特征
            preference_features,  # 用户偏好
            system_features      # 系统状态
        ])
        
        return all_features.astype(np.float32)
    
    def _get_standardized_device_features(self) -> np.ndarray:
        """获取标准化的设备特征 - 高效版本"""
        # 新设计：聚合特征而不是固定维度填充
        device_aggregation = {
            DeviceType.BATTERY: [],
            DeviceType.HEAT_PUMP: [],
            DeviceType.EV: [],
            DeviceType.PV: [],
            DeviceType.DISHWASHER: []
        }
        
        # 收集各类型设备的增强特征
        for device_id, mdp in self.device_mdps.items():
            device_type = self.device_types[device_id]
            enhanced_features = self._get_enhanced_device_features(device_id, device_type, mdp)
            if device_type in device_aggregation:
                device_aggregation[device_type].append(enhanced_features)
        
        # 生成聚合特征而非固定维度
        aggregated_features = []
        
        for device_type in sorted(device_aggregation.keys()):
            features_list = device_aggregation[device_type]
            if features_list:
                # 聚合同类设备特征：[数量, 平均值, 最大值, 最小值, 标准差]
                stacked = np.stack(features_list)
                aggregated = np.array([
                    float(len(features_list)),   # 设备数量
                    float(np.mean(stacked)),     # 平均值
                    float(np.max(stacked)),      # 最大值 
                    float(np.min(stacked)),      # 最小值
                    float(np.std(stacked))       # 标准差
                ])
            else:
                # 无该类型设备
                aggregated = np.zeros(5)
            
            aggregated_features.append(aggregated)
        
        # 固定25维设备聚合特征 (5类型 × 5维/类型)
        return np.concatenate(aggregated_features)
    
    def _get_enhanced_device_features(self, device_id: str, device_type: str, mdp) -> np.ndarray:
        """获取增强的设备特征 - 高效版本"""
        base_features = mdp.get_state_features()
        
        # 快速标准化处理，减少复杂计算
        if device_type == DeviceType.BATTERY:
            # 简化电池特征：[SOC, 功率状态, 健康度]
            soc = base_features[0] if len(base_features) > 0 else 0.5
            power_status = 1.0 if len(base_features) > 3 and abs(base_features[3]) > 0.1 else 0.0
            health = base_features[1] if len(base_features) > 1 else 1.0
            return np.array([soc, power_status, health])
            
        elif device_type == DeviceType.HEAT_PUMP:
            # 简化热泵特征：[温度标准化, 运行状态]
            temp = base_features[0] if len(base_features) > 0 else 20.0
            temp_normalized = np.clip((temp - 15.0) / 15.0, 0.0, 1.0)
            running = 1.0 if len(base_features) > 1 and abs(base_features[1] - temp) > 1.0 else 0.0
            return np.array([temp_normalized, running])
            
        elif device_type == DeviceType.EV:
            # 简化EV特征：[SOC, 连接状态]
            soc = base_features[0] if len(base_features) > 0 else 0.5
            connected = base_features[1] if len(base_features) > 1 else 1.0
            return np.array([soc, connected])
            
        elif device_type == DeviceType.PV:
            # 简化PV特征：[发电能力]（基于时间的简单模型）
            hour = 12  # 简化为固定时间
            generation_potential = 0.8 if 8 <= hour <= 16 else 0.2
            return np.array([generation_potential])
            
        elif device_type == DeviceType.DISHWASHER:
            # 简化洗碗机特征：[运行状态, 进度]
            running = 1.0 if len(base_features) > 1 and base_features[1] > 0 else 0.0
            progress = base_features[3] if len(base_features) > 3 else 0.0
            return np.array([running, progress])
        
        else:
            # 未知设备类型，返回前两个特征或填充
            if len(base_features) >= 2:
                return base_features[:2]
            elif len(base_features) == 1:
                return np.array([base_features[0], 0.0])
            else:
                return np.array([0.0, 0.0])
    
    def _get_standardized_management_features(self) -> np.ndarray:
        """获取标准化的管理特征"""
        # 基础管理统计
        total_users = len(self.users)
        total_devices = len(self.device_mdps)
        controllable_devices = len(self.controllable_devices)
        
        # 标准化管理特征（相对于系统规模）
        user_density = min(1.0, total_users / 15.0)  # 假设最大15个用户
        device_density = min(1.0, total_devices / 35.0)  # 假设最大35个设备
        control_ratio = controllable_devices / max(1, total_devices)
        
        # 覆盖面积标准化
        area_normalized = min(1.0, self.coverage_area / 1000.0)  # 假设最大1000平方米
        
        # 区域类型编码（独热编码）
        is_residential = 1.0 if self.district_type == 'residential' else 0.0
        is_commercial = 1.0 if self.district_type == 'commercial' else 0.0
        is_mixed = 1.0 if self.district_type == 'mixed' else 0.0
        
        return np.array([
            user_density, device_density, control_ratio, area_normalized,
            is_residential, is_commercial, is_mixed
        ])
    
    def _get_standardized_preference_features(self) -> np.ndarray:
        """获取标准化的偏好特征"""
        return np.array([
            self.aggregated_preferences['economic'],
            self.aggregated_preferences['comfort'],
            self.aggregated_preferences['environmental']
        ])
    
    def _get_standardized_system_features(self) -> np.ndarray:
        """获取标准化的系统特征"""
        # 系统负载特征
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
        
        # 标准化系统特征
        capacity_ratio = total_capacity / max(1, len(self.device_mdps))
        generation_ratio = total_generation / max(1, len(self.device_mdps))
        load_ratio = total_load / max(1, len(self.device_mdps))
        
        # 系统平衡指标
        balance_index = min(1.0, (total_capacity + total_generation) / max(1, total_load))
        
        return np.array([capacity_ratio, generation_ratio, load_ratio, balance_index])
    
    def _get_legacy_state_features(self) -> np.ndarray:
        """获取传统的状态特征（保持向后兼容）"""
        # 设备状态特征
        device_features = []
        for device_id in sorted(self.device_mdps.keys()):
            device_state = self.device_mdps[device_id].get_state_features()
            device_features.append(device_state)
        
        if device_features:
            device_features = np.concatenate(device_features)
        else:
            device_features = np.array([])
        
        # 用户偏好特征
        preference_features = np.array([
            self.aggregated_preferences['economic'],
            self.aggregated_preferences['comfort'],
            self.aggregated_preferences['environmental']
        ])
        
        # Manager特征
        manager_features = np.array([
            len(self.users),  # 用户数量
            len(self.device_mdps),  # 设备数量
            len(self.controllable_devices),  # 可控设备数量
            self.coverage_area,  # 覆盖面积
            1.0 if self.district_type == 'residential' else 0.0,
            1.0 if self.district_type == 'commercial' else 0.0,
            1.0 if self.district_type == 'mixed' else 0.0
        ])
        
        # 合并所有特征
        if len(device_features) > 0:
            return np.concatenate([device_features, preference_features, manager_features])
        else:
            return np.concatenate([preference_features, manager_features])
    
    def get_action_space_size(self) -> int:
        """
        获取动作空间大小 - 修改为FlexOffer参数生成
        
        每个可控设备的动作包含：
        - 时间窗口灵活性 (2维): [start_flex, end_flex] 
        - 能量范围调整 (2维): [energy_min_factor, energy_max_factor]
        - 优先级权重 (1维): [priority_weight]
        
        总共每个设备5维动作
        """
        # 每个可控设备需要5维FlexOffer参数
        fo_params_per_device = 5
        total_action_dim = len(self.controllable_devices) * fo_params_per_device
        
        # 确保至少有基本的动作维度
        return max(total_action_dim, 10)
    
    def reset(self):
        """重置Manager状态"""
        # 重置所有设备
        for device_mdp in self.device_mdps.values():
            device_mdp.reset_state()
        
        # 重置马尔可夫历史
        self.markov_history = {
            'prev_actions': np.zeros(len(self.controllable_devices)),
            'prev_reward': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_energy': 0.0,
            'user_satisfaction': 0.0
        }
    
    def step(self, actions: np.ndarray, env_state: Dict) -> Tuple[float, Dict]:
        """
        执行一步动作 - 修改为完整Pipeline流程
        
        新的流程：
        1. 将动作映射为FlexOffer参数
        2. 生成设备级FlexOffer
        3. 执行Pipeline流程：aggregate → trade → disaggregate → schedule
        4. 计算Pipeline执行奖励
        """
        # Step 1: 将动作映射为FlexOffer参数
        fo_params = self._map_actions_to_fo_params(actions)
        
        # Step 2: 生成设备级FlexOffer
        device_flexoffers = self._generate_device_flexoffers(fo_params, env_state)
        
        # Step 3: 执行完整Pipeline流程
        pipeline_results = self._execute_full_pipeline(device_flexoffers, env_state)
        
        # Step 4: 计算Pipeline奖励
        pipeline_reward, reward_info = self._calculate_pipeline_reward(
            pipeline_results, env_state
        )
        
        # 更新马尔可夫历史
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
        """将Agent动作映射为FlexOffer参数"""
        fo_params = {}
        fo_params_per_device = 5
        
        for i, device_id in enumerate(self.controllable_devices):
            start_idx = i * fo_params_per_device
            
            if start_idx + fo_params_per_device <= len(actions):
                device_actions = actions[start_idx:start_idx + fo_params_per_device]
                
                # 将动作映射为FlexOffer参数
                fo_params[device_id] = {
                    'start_flex': np.clip(device_actions[0], -1.0, 1.0),  # 开始时间灵活性
                    'end_flex': np.clip(device_actions[1], -1.0, 1.0),    # 结束时间灵活性
                    'energy_min_factor': np.clip(device_actions[2], 0.1, 1.0),  # 最小能量因子
                    'energy_max_factor': np.clip(device_actions[3], 1.0, 2.0),  # 最大能量因子
                    'priority_weight': np.clip(device_actions[4], 0.1, 2.0)     # 优先级权重
                }
            else:
                # 默认参数
                fo_params[device_id] = {
                    'start_flex': 0.0,
                    'end_flex': 0.0,
                    'energy_min_factor': 1.0,
                    'energy_max_factor': 1.0,
                    'priority_weight': 1.0
                }
        
        return fo_params
    
    def _generate_device_flexoffers(self, fo_params: Dict, env_state: Dict) -> Dict:
        """基于FlexOffer参数生成设备级FlexOffer"""
        device_flexoffers = {}
        
        from fo_generate.dfo import DFOSystem, DFOSlice
        from datetime import datetime, timedelta
        
        for device_id, params in fo_params.items():
            if device_id in self.device_mdps:
                device_mdp = self.device_mdps[device_id]
                
                # 获取设备动作边界（不需要当前状态）
                p_min, p_max = device_mdp.get_action_bounds()
                
                # 基于动作参数调整FlexOffer
                base_energy_min = p_min * 1.0  # 1小时基准
                base_energy_max = p_max * 1.0
                
                # 应用能量因子调整
                energy_min = base_energy_min * params['energy_min_factor']
                energy_max = base_energy_max * params['energy_max_factor']
                
                # 创建时间窗口（基于灵活性参数）
                current_time = datetime.now()
                start_offset = int(params['start_flex'] * 2)  # -2到+2小时
                end_offset = 1 + int(params['end_flex'] * 2)   # 1到3小时
                
                start_time = current_time + timedelta(hours=start_offset)
                end_time = current_time + timedelta(hours=end_offset)
                
                # 创建DFO系统
                dfo_system = DFOSystem(
                    time_horizon=max(1, end_offset - start_offset),
                    device_id=device_id,
                    device_type=getattr(device_mdp, 'device_type', 'unknown')
                )
                
                # 添加时间片（使用正确的DFOSlice构造函数参数）
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
        """执行完整的Pipeline流程 - 集成真实模块"""
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
            
            # Step 1: 聚合FlexOffer (使用fo_aggregate模块)
            aggregated_results = self._aggregate_flexoffers(device_flexoffers, env_state)
            pipeline_results['aggregated'] = aggregated_results
            
            # Step 2: 交易FlexOffer (使用fo_trading模块)
            trade_results = self._trade_flexoffers(aggregated_results, env_state)
            pipeline_results['trades'] = trade_results
            
            # Step 3: 分解FlexOffer (使用fo_schedule模块)
            disaggregated_results = self._disaggregate_flexoffers(
                trade_results, device_flexoffers, env_state
            )
            pipeline_results['disaggregated'] = disaggregated_results
            
            # Step 4: 调度执行 (使用fo_schedule模块)
            scheduled_results = self._schedule_flexoffers(disaggregated_results, env_state)
            pipeline_results['scheduled'] = scheduled_results
            
            # 计算统计信息
            pipeline_results['stats'] = self._calculate_pipeline_stats(
                pipeline_results, env_state
            )
            
        except Exception as e:
            logger.error(f"Pipeline执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回失败的结果
            pipeline_results['stats'] = {
                'num_flexoffers': 0,
                'num_trades': 0,
                'avg_satisfaction': 0.0,
                'total_cost': 1000.0  # 高成本作为惩罚
            }
        
        return pipeline_results
    
    def _aggregate_flexoffers(self, device_flexoffers: Dict, env_state: Dict) -> List:
        """聚合FlexOffer - 调用fo_aggregate模块"""
        try:
            from fo_aggregate.aggregator import FOAggregatorFactory
            from fo_common.flexoffer import FlexOffer, FOSlice
            
            # 转换DFO为FlexOffer格式
            flex_offers = []
            for device_id, dfo_system in device_flexoffers.items():
                for i, slice in enumerate(dfo_system.slices):
                    # 创建FOSlice
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
                    
                    # 创建FlexOffer
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
            
            # 选择聚合算法（可配置）
            # 修复：确保正确获取聚合方法
            aggregation_method = getattr(self, 'aggregation_method', 'LP')  # 默认LP
            
            # 添加详细日志记录聚合方法的选择
            logger.info(f"使用聚合方法: {aggregation_method}")
            logger.info(f"设备FlexOffer数量: {len(flex_offers)}")
            
            # 确保聚合方法是大写的，以匹配FOAggregatorFactory中的逻辑
            aggregation_method = aggregation_method.upper()
            
            # 创建聚合器
            aggregator = FOAggregatorFactory.create_aggregator(aggregation_method)
            logger.info(f"创建的聚合器类型: {aggregator.__class__.__name__}")
            
            # 执行聚合
            if flex_offers:
                aggregated_result = aggregator.aggregate(flex_offers)
                
                # 添加聚合结果日志
                if aggregated_result:
                    logger.info(f"聚合成功: 得到{len(aggregated_result)}个聚合FlexOffer")
                    for i, afo in enumerate(aggregated_result):
                        logger.info(f"聚合结果 #{i+1}: 方法={afo.aggregation_method}, "
                                   f"源FO数量={len(afo.source_fo_ids)}, "
                                   f"总能量范围=[{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}]")
                else:
                    logger.warning("聚合过程完成，但没有得到聚合结果")
                
                return aggregated_result
            else:
                logger.warning("没有FlexOffer可聚合")
                return []
                
        except Exception as e:
            logger.error(f"FlexOffer聚合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _trade_flexoffers(self, aggregated_results: List, env_state: Dict) -> List:
        """交易FlexOffer - 调用fo_trading模块的真实算法"""
        try:
            if not aggregated_results:
                return []
            
            # 选择交易算法（从环境状态或配置中获取）
            trading_method = env_state.get('trading_algorithm', getattr(self, 'trading_method', 'market_clearing'))
            
            if trading_method == 'market_clearing':
                return self._trade_with_market_clearing(aggregated_results, env_state)
            else:
                return self._trade_with_bidding(aggregated_results, env_state)
            
        except Exception as e:
            logger.error(f"FlexOffer交易失败: {e}")
            return []
    
    def _trade_with_bidding(self, aggregated_results: List, env_state: Dict) -> List:
        """使用Bidding算法交易 - 双方价格匹配时产生交易"""
        try:
            from fo_trading.pool import BiddingAlgorithm, Bid
            
            bidding_algo = BiddingAlgorithm()
            bids = []
            
            # 为每个聚合FO创建买卖双方报价
            for i, aggregated_fo in enumerate(aggregated_results):
                # 修复：正确获取聚合FlexOffer的能量
                total_energy = 0.0
                if hasattr(aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.total_energy_max
                elif hasattr(aggregated_fo, 'aggregated_fo') and hasattr(aggregated_fo.aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.aggregated_fo.total_energy_max
                else:
                    total_energy = getattr(aggregated_fo, 'total_energy', 10.0)
                
                # 确保总能量至少为1.0，避免零能量交易
                total_energy = max(1.0, total_energy)
                
                logger.info(f"聚合FO {i} 总能量: {total_energy:.2f} kWh")
                
                base_price = env_state.get('price', 0.15)
                
                # 创建卖方报价（当前Manager）- 降低卖方价格增加交易概率
                sell_price = base_price * (0.85 + 0.15 * np.random.random())  # 0.85-1.0倍基准价，更低的卖价
                sell_bid = Bid(
                    bid_id=f"sell_{self.manager_id}_{i}",
                    participant_id=self.manager_id,
                    price=sell_price,
                    quantity=total_energy,
                    time_step=0,
                    side="sell"
                )
                
                # 创建模拟买方报价 - 提高买方价格增加交易概率
                buy_price = base_price * (1.05 + 0.25 * np.random.random())  # 1.05-1.3倍基准价，更高的买价
                buy_quantity = total_energy * (0.8 + 0.4 * np.random.random())  # 0.8-1.2倍能量需求
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
            
            # Bidding算法：检查买卖双方价格是否匹配，如果匹配则产生交易
            trades = []
            processed_bids = bidding_algo.get_bids_by_timestep(0)
            
            # 分离买方和卖方报价
            buy_bids = [bid for bid in processed_bids if bid.side == "buy"]
            sell_bids = [bid for bid in processed_bids if bid.side == "sell"]
            
            # 按价格排序
            buy_bids.sort(key=lambda x: x.price, reverse=True)  # 买方出价从高到低
            sell_bids.sort(key=lambda x: x.price)  # 卖方出价从低到高
            
            # 匹配买卖双方：买方出价 >= 卖方出价时成交
            for sell_bid in sell_bids:
                if sell_bid.participant_id != self.manager_id:
                    continue  # 只处理当前Manager的卖方报价
                    
                for buy_bid in buy_bids:
                    if buy_bid.price >= sell_bid.price:  # 价格匹配
                        # 计算交易数量（取两者最小值）
                        trade_quantity = min(sell_bid.quantity, buy_bid.quantity)
                        
                        # 计算交易价格（买卖价格的中间值）
                        trade_price = (buy_bid.price + sell_bid.price) / 2
                        
                        # 创建交易记录
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
                        
                        # 更新报价数量（简化处理，实际应该更复杂）
                        sell_bid.quantity -= trade_quantity
                        buy_bid.quantity -= trade_quantity
                        
                        logger.info(f"交易成功(Bidding): 买方{buy_bid.participant_id}({buy_bid.price:.4f}) "
                                  f"vs 卖方{sell_bid.participant_id}({sell_bid.price:.4f}), "
                                  f"成交价{trade_price:.4f}, 数量{trade_quantity:.2f}")
                        
                        # 如果买方需求已满足，跳到下一个买方
                        if buy_bid.quantity <= 0:
                            break
                    
                    # 如果卖方供给已用完，跳到下一个卖方
                    if sell_bid.quantity <= 0:
                        break
            
            logger.info(f"交易算法(Bidding)完成: 处理{len(processed_bids)}个报价, 产生{len(trades)}笔交易")
            return trades
            
        except Exception as e:
            logger.error(f"Bidding算法交易失败: {e}")
            return []
    
    def _trade_with_market_clearing(self, aggregated_results: List, env_state: Dict) -> List:
        """使用Market Clearing算法交易"""
        try:
            from fo_trading.pool import MarketClearingAlgorithm, Bid
            
            # 创建市场出清算法实例
            clearing_algo = MarketClearingAlgorithm(clearing_method="uniform_price")
            bids = []
            
            # 为每个聚合FO创建买卖双方报价
            for i, aggregated_fo in enumerate(aggregated_results):
                # 修复：正确获取聚合FlexOffer的能量
                total_energy = 0.0
                if hasattr(aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.total_energy_max
                elif hasattr(aggregated_fo, 'aggregated_fo') and hasattr(aggregated_fo.aggregated_fo, 'total_energy_max'):
                    total_energy = aggregated_fo.aggregated_fo.total_energy_max
                else:
                    total_energy = getattr(aggregated_fo, 'total_energy', 10.0)
                
                # 确保总能量至少为1.0，避免零能量交易
                total_energy = max(1.0, total_energy)
                
                logger.info(f"聚合FO {i} 总能量: {total_energy:.2f} kWh")
                
                base_price = env_state.get('price', 0.15)
                
                # 创建卖方报价（当前Manager）- 与bidding算法保持一致的价格范围
                sell_bid = Bid(
                    bid_id=f"sell_{self.manager_id}_{i}",
                    participant_id=self.manager_id,
                    price=base_price * (0.85 + 0.15 * np.random.random()),  # 0.85-1.0倍基准价
                    quantity=total_energy,
                    time_step=0,
                    side="sell"
                )
                
                # 创建多个模拟买方报价（增加成交概率）- 扩大价格范围
                for j in range(2):  # 每个FO创建2个买方
                    buy_bid = Bid(
                        bid_id=f"buy_market_{i}_{j}",
                        participant_id=f"buyer_{i}_{j}",
                        price=base_price * (1.05 + 0.25 * np.random.random()),  # 1.05-1.3倍基准价
                        quantity=total_energy * (0.4 + 0.4 * np.random.random()),  # 0.4-0.8倍需求量
                        time_step=0,
                        side="buy"
                    )
                    bids.append(buy_bid)
                
                bids.append(sell_bid)
            
            # 执行市场出清
            clearing_results = clearing_algo.process_bids(bids)
            
            # 生成交易
            generated_trades = clearing_algo.generate_trades(clearing_results, bids)
            
            # 转换为统一格式
            trades = []
            for trade in generated_trades:
                if trade.seller_id == self.manager_id:  # 只关心当前Manager的交易
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
            
            logger.info(f"交易算法(Market Clearing)完成: {len(clearing_results)}个出清结果, {len(trades)}笔交易")
            return trades
            
        except Exception as e:
            logger.error(f"Market Clearing算法交易失败: {e}")
            return []
    
    def _disaggregate_flexoffers(self, trade_results: List, original_flexoffers: Dict, env_state: Dict) -> List:
        """分解FlexOffer - 调用fo_schedule模块"""
        try:
            if not trade_results:
                return []
            
            from fo_schedule.scheduler import AggregatedResultDisaggregator
            
            # 选择分解方法（可配置）
            disaggregation_method = getattr(self, 'disaggregation_method', 'proportional')
            disaggregator = AggregatedResultDisaggregator(
                time_horizon=24, 
                default_algorithm=disaggregation_method
            )
            
            disaggregated_results = []
            
            for trade in trade_results:
                if trade.get('success', False):
                    # 准备分解数据
                    original_data = []
                    for device_id, dfo_system in original_flexoffers.items():
                        for slice in dfo_system.slices:
                            original_data.append({
                                'device_id': device_id,
                                'energy_min': slice.energy_min,
                                'energy_max': slice.energy_max,
                                'weight': slice.flexibility_factor,
                                'energy': slice.energy_max  # 添加energy字段供分解算法使用
                            })
                    
                    if original_data:
                        # 执行分解
                        total_energy = trade.get('trade_volume', 0.0)
                        
                        # 修复：检查总能量是否为0或负值
                        if total_energy <= 0:
                            logger.info(f"交易量为0或负值({total_energy})，分配零能量给所有设备")
                            # 直接分配零能量给所有设备，避免分解算法报错
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
                            logger.warning(f"分解失败，使用平均分配: {e}")
                            # 回退到平均分配
                            avg_energy = total_energy / len(original_data)
                            for data in original_data:
                                disaggregated_results.append({
                                    'device_id': data['device_id'],
                                    'allocated_energy': avg_energy,
                                    'method': 'average_fallback'
                                })
            
            return disaggregated_results
            
        except Exception as e:
            logger.error(f"FlexOffer分解失败: {e}")
            return []
    
    def _schedule_flexoffers(self, disaggregated_results: List, env_state: Dict) -> Dict:
        """调度FlexOffer - 执行最终的设备控制"""
        try:
            scheduled_results = {}
            total_satisfaction = 0.0
            device_count = 0
            
            for result in disaggregated_results:
                device_id = result.get('device_id')
                allocated_energy = result.get('allocated_energy', 0.0)
                
                if device_id and device_id in self.device_mdps:
                    device_mdp = self.device_mdps[device_id]
                    
                    # 将分配的能量转换为功率控制信号
                    # 简化：假设1小时内均匀分配
                    power_signal = allocated_energy / 1.0  # 1小时
                    
                    # 限制在设备功率范围内
                    p_min, p_max = device_mdp.get_action_bounds()
                    power_signal = np.clip(power_signal, p_min, p_max)
                    
                    # 执行设备状态转移
                    next_state = device_mdp.transition_state(power_signal, env_state)
                    
                    # 计算设备满意度（简化版）
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
            
            # 计算平均满意度
            avg_satisfaction = total_satisfaction / max(device_count, 1)
            scheduled_results['_summary'] = {
                'avg_satisfaction': avg_satisfaction,
                'total_devices': device_count
            }
            
            return scheduled_results
            
        except Exception as e:
            logger.error(f"FlexOffer调度失败: {e}")
            return {'_summary': {'avg_satisfaction': 0.0, 'total_devices': 0}}
    
    def _calculate_device_satisfaction(self, device_mdp, power_signal: float, 
                                     device_state: Dict, env_state: Dict) -> float:
        """计算设备满意度（基于实际设备状态）"""
        try:
            # 使用设备的奖励函数来评估满意度
            device_reward, reward_components = device_mdp.calculate_reward(
                power_signal, device_state, env_state
            )
            
            # 将奖励转换为满意度 (0-1范围)
            # 假设正奖励对应高满意度，负奖励对应低满意度
            if device_reward > 0:
                satisfaction = min(device_reward / 10.0, 1.0)  # 归一化到0-1
            else:
                satisfaction = max(0.5 + device_reward / 20.0, 0.0)  # 负奖励降低满意度
            
            return np.clip(satisfaction, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"设备满意度计算失败: {e}")
            return 0.5  # 默认中等满意度
    
    def _calculate_pipeline_stats(self, pipeline_results: Dict, env_state: Dict) -> Dict:
        """计算Pipeline统计信息"""
        stats = {}
        
        # 基本统计
        stats['num_flexoffers'] = len(pipeline_results.get('flexoffers', {}))
        stats['num_aggregated'] = len(pipeline_results.get('aggregated', []))
        stats['num_trades'] = len(pipeline_results.get('trades', []))
        stats['num_disaggregated'] = len(pipeline_results.get('disaggregated', []))
        
        # 满意度统计
        scheduled = pipeline_results.get('scheduled', {})
        summary = scheduled.get('_summary', {})
        stats['avg_satisfaction'] = summary.get('avg_satisfaction', 0.0)
        
        # 成本统计
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
        """计算基于Pipeline执行结果的奖励 - 🔧 紧急修复：确保奖励合理"""
        stats = pipeline_results.get('stats', {})
        
        # 🔧 紧急修复1：重新设计奖励基准，确保主要为正值
        
        # 1. 经济性奖励：重新设计为正值导向 (0 到 100)
        total_cost = stats.get('total_cost', 0.0)
        total_revenue = stats.get('total_revenue', 0.0)
        net_benefit = total_revenue - total_cost
        
        # 🔧 修复：以正奖励为主，避免大幅负值
        if net_benefit > 20:
            economic_reward = 70.0 + min(net_benefit - 20, 30.0)  # 70-100分
        elif net_benefit > 5:
            economic_reward = 40.0 + (net_benefit - 5) * 2.0  # 40-70分
        elif net_benefit > 0:
            economic_reward = 20.0 + net_benefit * 4.0  # 20-40分
        elif net_benefit > -5:
            economic_reward = 10.0 + (net_benefit + 5) * 2.0  # 0-20分
        else:
            economic_reward = max(0.0, 10.0 + net_benefit)  # 最低0分，避免大幅负值
        
        # 2. 用户满意度奖励：重新设计为正值导向 (10 到 80)
        satisfaction_base = stats.get('avg_satisfaction', 0.5)
        
        # 🔧 修复：确保满意度奖励主要为正
        if satisfaction_base > 0.8:
            satisfaction_reward = 60.0 + (satisfaction_base - 0.8) * 100  # 60-80分
        elif satisfaction_base > 0.6:
            satisfaction_reward = 40.0 + (satisfaction_base - 0.6) * 100  # 40-60分
        elif satisfaction_base > 0.4:
            satisfaction_reward = 20.0 + (satisfaction_base - 0.4) * 100  # 20-40分
        elif satisfaction_base > 0.2:
            satisfaction_reward = 10.0 + (satisfaction_base - 0.2) * 50  # 10-20分
        else:
            satisfaction_reward = max(10.0, satisfaction_base * 50)  # 最低10分
        
        # 3. 协调奖励：重新设计为正值导向 (5 到 60)
        num_trades = stats.get('num_trades', 0)
        num_flexoffers = stats.get('num_flexoffers', 1)
        trade_success_rate = num_trades / max(num_flexoffers, 1)
        
        # 🔧 修复：协调奖励以正值为主
        if trade_success_rate > 0.7:
            coordination_reward = 40.0 + (trade_success_rate - 0.7) * 66.7  # 40-60分
        elif trade_success_rate > 0.4:
            coordination_reward = 20.0 + (trade_success_rate - 0.4) * 66.7  # 20-40分
        elif trade_success_rate > 0.1:
            coordination_reward = 5.0 + (trade_success_rate - 0.1) * 50  # 5-20分
        else:
            coordination_reward = max(5.0, trade_success_rate * 50)  # 最低5分
        
        # 4. 策略一致性奖励：重新设计为正值导向 (0 到 30)
        strategy_consistency_reward = 15.0  # 默认基础分
        if hasattr(self, 'markov_history') and 'prev_actions' in self.markov_history:
            prev_actions = self.markov_history['prev_actions']
            if prev_actions is not None and len(prev_actions) > 0:
                action_mean = np.mean(prev_actions)
                action_std = np.std(prev_actions)
                
                # 奖励合理的策略
                if 0.2 <= action_std <= 0.6 and 0.3 <= action_mean <= 0.7:
                    strategy_consistency_reward = 30.0  # 优秀策略
                elif 0.1 <= action_std <= 0.8 and 0.2 <= action_mean <= 0.8:
                    strategy_consistency_reward = 20.0  # 良好策略
                elif action_std < 0.05:  # 策略过于保守
                    strategy_consistency_reward = 5.0
                elif action_std > 0.9:  # 策略过于随机
                    strategy_consistency_reward = 8.0
                else:
                    strategy_consistency_reward = 15.0  # 平均策略
        
        # 🔧 关键修复2：使用加权平均而不是直接相加，控制总奖励范围
        # 目标范围：30-270分
        total_reward = (
            0.4 * economic_reward +           # 40%权重：0-40分
            0.3 * satisfaction_reward +       # 30%权重：3-24分  
            0.2 * coordination_reward +       # 20%权重：1-12分
            0.1 * strategy_consistency_reward # 10%权重：0-3分
        )  # 总范围：4-79分，再加上基础分
        
        # 🔧 修复3：添加基础奖励，确保总奖励为正
        base_reward = 50.0  # 基础奖励50分
        total_reward += base_reward  # 最终范围：54-129分
        
        # 🔧 修复4：添加基于学习进度的小幅调整（而不是大幅奖励）
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
        else:
            self.episode_count = 1
            
        # 学习进度奖励：小幅调整而不是大幅变化
        learning_progress_reward = 0.0
        if self.episode_count <= 20:
            # 早期阶段：鼓励探索，小幅随机调整
            learning_progress_reward = np.random.uniform(-5, 15)  # -5到+15分
        elif self.episode_count <= 50:
            # 中期阶段：奖励稳定策略
            if economic_reward > 60 and satisfaction_reward > 50:
                learning_progress_reward = 10.0
            else:
                learning_progress_reward = 5.0
        else:
            # 后期阶段：奖励优秀策略
            if total_reward > 100:
                learning_progress_reward = 15.0
            else:
                learning_progress_reward = 8.0
                
        total_reward += learning_progress_reward
        
        # 🔧 修复5：确保奖励在合理范围内
        total_reward = max(30.0, min(200.0, total_reward))  # 限制在30-200分范围内
        
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
        """应用用户偏好权重"""
        # 提取不同类型的奖励
        economic_reward = 0.0
        comfort_reward = 0.0
        environmental_reward = 0.0
        
        for device_rewards in reward_components.values():
            economic_reward += device_rewards.get('economic', 0.0)
            comfort_reward += device_rewards.get('comfort', 0.0)
            environmental_reward += device_rewards.get('efficiency', 0.0)  # 效率作为环保指标
        
        # 应用用户偏好权重
        weighted_reward = (
            self.aggregated_preferences['economic'] * economic_reward +
            self.aggregated_preferences['comfort'] * comfort_reward +
            self.aggregated_preferences['environmental'] * environmental_reward
        )
        
        return weighted_reward
    
    def _calculate_user_satisfaction(self, reward_components: Dict) -> float:
        """计算用户满意度"""
        # 基于舒适度奖励计算满意度
        comfort_rewards = []
        for device_rewards in reward_components.values():
            if 'comfort' in device_rewards:
                comfort_rewards.append(device_rewards['comfort'])
        
        if comfort_rewards:
            return float(np.mean(comfort_rewards))
        else:
            return 0.5  # 默认满意度
    
    def generate_dfo(self, time_horizon: int) -> Dict[str, DFOSystem]:
        """生成DFO系统"""
        dfo_systems = {}
        
        for device_id in self.controllable_devices:
            device_mdp = self.device_mdps[device_id]
            dfo = DFOSystem(time_horizon)
            
            for t in range(time_horizon):
                # 获取动作边界
                p_min, p_max = device_mdp.get_action_bounds()
                
                # 创建时间片
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
        """获取当前观测"""
        device_states = []
        user_states = []
        
        # 收集设备状态
        if isinstance(self.devices, dict):
            for device in self.devices.values():
                if hasattr(device, 'env'):
                    state = device.env.get_state()
                    device_states.extend(state)
        else:
            # self.devices 是列表
            for device in self.devices:
                if hasattr(device, 'env'):
                    state = device['env'].get_state()  # type: ignore
                    device_states.extend(state)
        
        # 收集用户状态  
        for user in self.users:
            # 处理用户设备数量（如果用户有设备属性则使用，否则默认为0）
            user_device_count = 0
            if 'devices' in user:
                user_device_count = len(user['devices'])
            elif 'device_count' in user:
                user_device_count = user['device_count']
            
            # 处理用户偏好（如果有偏好属性则使用，否则使用默认值）
            preferences = getattr(user, 'preferences', {})
            
            user_state = [
                user_device_count,  # 设备数量
                preferences.get('economic', 0.25),  # 经济偏好
                preferences.get('comfort', 0.25),   # 舒适偏好
                preferences.get('self_sufficient', 0.25),  # 自给自足偏好
                preferences.get('environmental', 0.25)     # 环保偏好
            ]
            user_states.extend(user_state)
        
        # 组合观测
        observation = device_states + user_states
        
        # 确保观测维度一致性（这里不做强制约束，因为观测维度是动态的）
        return np.array(observation, dtype=np.float32)

class MultiAgentFlexOfferEnv(gym.Env):
    """多智能体FlexOffer环境"""
    
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
        初始化多智能体FlexOffer环境
        
        Args:
            data_dir: 数据目录
            time_horizon: 时间范围
            time_step: 时间步长
            start_time: 开始时间
            dec_pomdp_config: Dec-POMDP配置
            aggregation_method: 聚合算法 ("LP", "DP")
            trading_method: 交易算法 ("bidding", "market_clearing")
            disaggregation_method: 分解算法 ("average", "proportional")
        """
        
        self.data_dir = data_dir
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time or datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # 算法配置
        self.aggregation_method = aggregation_method
        self.trading_method = trading_method
        self.disaggregation_method = disaggregation_method
        
        # 验证算法选择
        self._validate_algorithm_choices()
        
        # Dec-POMDP配置
        self.dec_pomdp_config = dec_pomdp_config or DecPOMDPConfig()
        self.dec_pomdp_obs_space = DecPOMDPObservationSpace(self.dec_pomdp_config)
        
        # 观测历史（用于信息延迟）
        self.observation_history: Dict[str, List[np.ndarray]] = {}
        
        # 动态观测质量管理器（如果启用观测噪声，则启用动态质量）
        if self.dec_pomdp_config.enable_observation_noise:
            self.dynamic_quality_manager = DynamicObservationQuality()
        else:
            self.dynamic_quality_manager = None
        
        # 数据加载器
        self.data_loader = DataLoader(data_dir)
        
        # 加载配置数据
        self._load_configuration_data()
        
        # 环境动态
        self.env_dynamics = EnvironmentDynamics(
            price_data=self.price_data,
            weather_data=self.weather_data
        )
        
        # 创建Manager代理
        self._create_manager_agents()
        
        # 为每个Manager设置算法配置
        self._configure_manager_algorithms()
        
        # 设置观测和动作空间
        self._setup_spaces()
        
        # 时间状态
        self.current_time = self.start_time
        self.current_step = 0
        
        logger.info(f"多智能体环境初始化完成: {len(self.manager_agents)} 个Manager")
        logger.info(f"算法配置 - 聚合: {aggregation_method}, 交易: {trading_method}, 分解: {disaggregation_method}")
        logger.info(f"Dec-POMDP模式: {self.dec_pomdp_config.enable_observation_noise}, "
                   f"动态质量管理: {self.dynamic_quality_manager is not None}")
    
    def _validate_algorithm_choices(self):
        """验证算法选择的有效性"""
        # 验证聚合算法
        valid_aggregation = ["LP", "DP"]
        if self.aggregation_method not in valid_aggregation:
            logger.warning(f"无效的聚合算法 '{self.aggregation_method}'，使用默认 'LP'")
            self.aggregation_method = "LP"
        
        # 验证交易算法
        valid_trading = ["bidding", "market_clearing"]
        if self.trading_method not in valid_trading:
            logger.warning(f"无效的交易算法 '{self.trading_method}'，使用默认 'bidding'")
            self.trading_method = "bidding"
        
        # 验证分解算法
        valid_disaggregation = ["average", "proportional"]
        if self.disaggregation_method not in valid_disaggregation:
            logger.warning(f"无效的分解算法 '{self.disaggregation_method}'，使用默认 'proportional'")
            self.disaggregation_method = "proportional"
        
        logger.info(f"算法验证完成 - 聚合: {self.aggregation_method}, "
                   f"交易: {self.trading_method}, 分解: {self.disaggregation_method}")
    
    def _configure_manager_algorithms(self):
        """为每个Manager配置算法选择"""
        for manager_id, manager in self.manager_agents.items():
            # 将算法配置传递给Manager（使用setattr动态设置属性）
            setattr(manager, 'aggregation_method', self.aggregation_method)
            setattr(manager, 'trading_method', self.trading_method)
            setattr(manager, 'disaggregation_method', self.disaggregation_method)
            
            logger.debug(f"Manager {manager_id} 算法配置完成")
    
    def set_algorithms(self, aggregation: Optional[str] = None, trading: Optional[str] = None, disaggregation: Optional[str] = None):
        """动态设置算法选择"""
        if aggregation:
            self.aggregation_method = aggregation
        if trading:
            self.trading_method = trading
        if disaggregation:
            self.disaggregation_method = disaggregation
        
        # 重新验证和配置
        self._validate_algorithm_choices()
        self._configure_manager_algorithms()
        
        logger.info(f"算法配置已更新 - 聚合: {self.aggregation_method}, "
                   f"交易: {self.trading_method}, 分解: {self.disaggregation_method}")
    
    def get_algorithm_config(self) -> Dict[str, str]:
        """获取当前算法配置"""
        return {
            'aggregation': self.aggregation_method,
            'trading': self.trading_method,
            'disaggregation': self.disaggregation_method
        }
    
    def _load_configuration_data(self):
        """加载配置数据"""
        # 加载外部数据
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
        
        # 加载Manager和用户配置 - 使用实际存在的文件名
        self.manager_config_df = self.data_loader.load_manager_config("manager_config_36users.csv")
        self.user_config_df = self.data_loader.load_user_config("user_config_36users.csv")
        self.device_config_df = self.data_loader.load_device_config("device_config_36users.csv")
        
        logger.info("配置数据加载完成")
    
    def _create_manager_agents(self):
        """创建Manager代理"""
        self.manager_agents: Dict[str, ManagerAgent] = {}
        
        for _, manager_row in self.manager_config_df.iterrows():
            manager_id = manager_row['manager_id']
            
            # 获取该Manager的用户
            manager_users = self.user_config_df[
                self.user_config_df['manager_id'] == manager_id
            ].to_dict('records')  # type: ignore
            
            # 获取该Manager用户的设备 - 处理用户ID格式不匹配问题
            user_ids = [user['user_id'] for user in manager_users]
            
            # 生成可能的用户ID格式进行匹配
            # 格式1: user_01, user_02, ... (用户配置文件格式)
            # 格式2: user_manager_1_1, user_manager_1_2, ... (设备配置文件格式)
            extended_user_ids = set(user_ids)  # 原始用户ID
            
            # 为每个用户ID生成可能的设备配置格式
            for user_id in user_ids:
                # 从user_01提取manager和用户编号
                if user_id.startswith('user_'):
                    try:
                        user_num_str = user_id.split('_')[1]  # 获取"01", "02"等
                        user_num = int(user_num_str)  # 转换为数字
                        
                        # 根据manager_id生成对应的设备配置用户ID格式
                        manager_num = str(manager_id).split('_')[1]  # 从manager_1获取"1"
                        
                        # 计算该用户在Manager内的局部编号
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
                            continue  # 用户不属于当前Manager
                        
                        # 生成设备配置格式的用户ID
                        device_user_id = f"user_manager_{manager_num}_{local_user_num}"
                        extended_user_ids.add(device_user_id)
                        
                    except (ValueError, IndexError):
                        continue  # 无法解析的用户ID格式，跳过
            
            # 使用扩展的用户ID列表匹配设备
            manager_devices = self.device_config_df[
                self.device_config_df['user_id'].isin(list(extended_user_ids))
            ].to_dict('records')  # type: ignore
            
            logger.debug(f"{manager_id}: 原始用户IDs {user_ids}, 扩展用户IDs {list(extended_user_ids)}, 匹配设备 {len(manager_devices)} 个")
            
            # 创建Manager代理
            manager_agent = ManagerAgent(
                manager_id=str(manager_id),
                manager_config=manager_row.to_dict(),
                users=manager_users,
                devices=manager_devices
            )
            
            self.manager_agents[str(manager_id)] = manager_agent
        
        self.manager_ids = list(self.manager_agents.keys())
        logger.info(f"创建 {len(self.manager_agents)} 个Manager代理")
    
    def _setup_spaces(self):
        """设置观测和动作空间 - 更新为FlexOffer参数生成模式"""
        # 计算观测空间维度
        obs_dims = {}
        action_dims = {}
        
        for manager_id, manager in self.manager_agents.items():
            # 获取单个Manager的状态特征维度
            state_features = manager.get_state_features()
            
            # 环境特征维度 (时间4 + 价格5 + 天气4 = 13)
            env_dim = 13
            
            # 其他Manager信息维度 (每个Manager增强后的信息维度)
            # 基础信息5维 + 增强信息9维 = 14维每个Manager
            enhanced_manager_info_dim = 14
            other_managers_dim = (len(self.manager_agents) - 1) * enhanced_manager_info_dim
            
            # 市场状态特征维度 (16维)
            market_state_dim = 16
            
            # 总观测维度
            total_obs_dim = len(state_features) + env_dim + other_managers_dim + market_state_dim
            
            obs_dims[manager_id] = total_obs_dim
            
            # 新的动作空间：FlexOffer参数生成
            # 每个可控设备需要5维FlexOffer参数：[start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight]
            action_dims[manager_id] = manager.get_action_space_size()
        
        # 创建观测和动作空间
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for manager_id in self.manager_ids:
            self.observation_spaces[manager_id] = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_dims[manager_id],), 
                dtype=np.float32
            )
            
            # 新的动作空间定义：FlexOffer参数范围
            manager = self.manager_agents[manager_id]
            num_controllable_devices = len(manager.controllable_devices)
            fo_params_per_device = 5
            total_action_dim = num_controllable_devices * fo_params_per_device
            
            if total_action_dim > 0:
                # FlexOffer参数的合理范围
                action_low = []
                action_high = []
                
                for _ in range(num_controllable_devices):
                    action_low.extend([-1.0, -1.0, 0.1, 1.0, 0.1])   # [start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight]
                    action_high.extend([1.0, 1.0, 1.0, 2.0, 2.0])    # 对应的上限
                
                self.action_spaces[manager_id] = spaces.Box(
                    low=np.array(action_low, dtype=np.float32),
                    high=np.array(action_high, dtype=np.float32),
                    dtype=np.float32
                )
                
                logger.info(f"Manager {manager_id} 动作空间: {total_action_dim}维 "
                          f"({num_controllable_devices} 设备 × {fo_params_per_device} 参数/设备)")
            else:
                # 如果没有可控设备，创建一个虚拟动作空间
                self.action_spaces[manager_id] = spaces.Box(
                    low=-1.0, high=1.0, shape=(5,), dtype=np.float32
                )
                logger.warning(f"Manager {manager_id} 没有可控设备，使用虚拟动作空间")
        
        logger.info("观测和动作空间设置完成 - FlexOffer参数生成模式")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_time = self.start_time
        self.current_step = 0
        
        # 重置公共信息缓存
        self._cached_public_features = None
        self._cache_time_step = -1
        
        # 重置环境状态缓存
        self._cached_env_state = None
        self._env_state_cache_time = -1
        
        # 重置环境动态
        self.env_dynamics.price_history = []
        self.env_dynamics.weather_history = []
        
        # 重置所有Manager
        for manager in self.manager_agents.values():
            manager.reset()
        
        # 获取初始观测
        observations = self._get_observations()
        infos = {manager_id: {'time': self.current_time, 'step': self.current_step} 
                for manager_id in self.manager_ids}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]):
        """执行一步"""
        # 获取当前环境状态
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # 🔧 添加交易算法配置到环境状态
        env_state['trading_algorithm'] = self.trading_method
        
        # 执行所有Manager的动作
        rewards = {}
        infos = {}
        
        for manager_id, action in actions.items():
            if manager_id in self.manager_agents:
                manager = self.manager_agents[manager_id]
                reward, info = manager.step(action, env_state)
                rewards[manager_id] = reward
                infos[manager_id] = info
        
        # 更新时间
        self.current_time += timedelta(hours=self.time_step)
        self.current_step += 1
        
        # 检查终止条件
        done = self.current_step >= self.time_horizon
        dones = {manager_id: done for manager_id in self.manager_ids}
        dones['__all__'] = done
        
        # 获取下一个观测
        next_observations = self._get_observations()
        
        # 添加环境信息
        for manager_id in self.manager_ids:
            infos[manager_id].update({
                'time': self.current_time,
                'step': self.current_step,
                'env_state': env_state
            })
        
        return next_observations, rewards, dones, False, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        获取所有Manager的观测 - 增强Dec-POMDP版本
        实现严格的分层观测空间：O_i = [O_private_i, O_public, O_limited_others_i]
        
        核心改进：
        1. 严格限制其他Manager信息共享
        2. 集成动态观测质量管理器
        3. 增强信息传递延迟和丢失机制
        4. 添加网络状况对观测质量的影响
        """
        observations = {}
        
        # 确保同一时间步内所有Manager获得相同的公共环境特征（无噪声）
        if not hasattr(self, '_cached_public_features') or self._cache_time_step != self.current_step:
            self._cached_public_features = self._get_dec_pomdp_public_features()
            self._cache_time_step = self.current_step
        
        public_features = self._cached_public_features
        
        # 获取简化的Manager间协作信息（有限且带噪声）
        limited_collaboration_info = self._get_limited_collaboration_info()
        
        # 更新动态观测质量（如果启用）
        if self.dynamic_quality_manager:
            self.dynamic_quality_manager.step()
        
        for manager_id, manager in self.manager_agents.items():
            # 1. 私有信息层：Manager自身的完整状态（无噪声）
            private_features = manager.get_state_features()
            
            # 2. 公共信息层：环境状态（无噪声，所有Manager可见）
            # public_features已经在上面获取
            
            # 3. 有限他者信息层：极度简化的协作信息（可配置噪声和质量降级）
            limited_others_features = self.dec_pomdp_obs_space.compute_limited_other_manager_info(
                limited_collaboration_info, manager_id
            )
            
            # 应用动态观测质量降级（如果启用）
            if self.dynamic_quality_manager:
                # 计算当前观测质量
                other_manager_ids = [mid for mid in self.manager_ids if mid != manager_id]
                quality_metrics = self.dynamic_quality_manager.calculate_observation_quality(
                    manager_id, other_manager_ids
                )
                
                # 应用质量降级到他者信息
                if len(limited_others_features) > 0:
                    limited_others_features = self.dynamic_quality_manager.apply_quality_degradation(
                        limited_others_features, quality_metrics
                    )
                
                # 更新质量历史
                self.dynamic_quality_manager.update_quality_history(manager_id, quality_metrics)
            
            # Dec-POMDP观测层组合（保证公共信息一致性）
            # 私有信息：Manager自身状态（可以轻微处理）
            processed_private_features = private_features  # 暂时不处理，保持完整性
            
            # 他者信息：应用信息传递机制（可能有噪声和延迟）
            if len(limited_others_features) > 0:
                # 只对他者信息应用信息传递机制
                processed_others_features = self._apply_enhanced_information_mechanisms(
                    limited_others_features, manager_id
                )
                
                # 合并三层观测信息：私有 + 公共（无噪声） + 处理后的他者
                arrays_to_concat = []
                if processed_private_features is not None:
                    arrays_to_concat.append(processed_private_features)
                if public_features is not None:
                    arrays_to_concat.append(public_features)
                if processed_others_features is not None:
                    arrays_to_concat.append(processed_others_features)
                
                dec_pomdp_observation = np.concatenate(arrays_to_concat) if arrays_to_concat else np.array([])
            else:
                # 如果禁用了其他Manager信息，只包含私有和公共信息
                arrays_to_concat = []
                if processed_private_features is not None:
                    arrays_to_concat.append(processed_private_features)
                if public_features is not None:
                    arrays_to_concat.append(public_features)
                
                dec_pomdp_observation = np.concatenate(arrays_to_concat) if arrays_to_concat else np.array([])
            
            # 确保观测维度为73
            current_dim = len(dec_pomdp_observation)
            if current_dim < 73:
                # 如果维度小于73，填充到73
                padding = np.zeros(73 - current_dim)
                dec_pomdp_observation = np.concatenate([dec_pomdp_observation, padding])
            elif current_dim > 73:
                # 如果维度大于73，截断到73
                dec_pomdp_observation = dec_pomdp_observation[:73]
            
            # 更新观测历史
            if manager_id not in self.observation_history:
                self.observation_history[manager_id] = []
            self.observation_history[manager_id].append(dec_pomdp_observation.copy())
            
            # 限制历史长度
            max_history_len = max(10, self.dec_pomdp_config.max_delay_steps + 5)
            if len(self.observation_history[manager_id]) > max_history_len:
                self.observation_history[manager_id] = self.observation_history[manager_id][-max_history_len:]
            
            observations[manager_id] = dec_pomdp_observation.astype(np.float32)
        
        return observations
    
    def _get_dec_pomdp_public_features(self) -> np.ndarray:
        """
        获取Dec-POMDP的公共信息特征 - 优化版本
        
        公共信息层设计原则：
        1. 所有Manager都能无噪声观测
        2. 环境状态的完整表示
        3. 标准化和归一化的信息编码
        4. 语义明确的特征组织
        5. 可配置的信息粒度
        
        公共信息层结构：
        - 时间信息层（6维）：周期性时间编码 + 进度指标
        - 市场信息层（7维）：价格状态 + 趋势 + 预测 + 市场阶段
        - 环境信息层（5维）：天气状态 + 趋势 + 季节性
        
        Returns:
            np.ndarray: 18维标准化公共信息向量
        """
        # 1. 时间信息层（6维）- 标准化时间表示
        time_layer = self._get_standardized_time_features()
        
        # 2. 市场信息层（7维）- 完整市场状态
        market_layer = self._get_standardized_market_features()
        
        # 3. 环境信息层（5维）- 标准化环境状态
        environment_layer = self._get_standardized_environment_features()
        
        # 组合所有公共信息层
        public_features = np.concatenate([time_layer, market_layer, environment_layer])
        
        # 验证公共信息层完整性
        if self.dec_pomdp_config.enable_observation_noise:  # 仅在调试模式下验证
            self._validate_public_information_layer(public_features, time_layer, market_layer, environment_layer)
        
        return public_features
    
    def _get_standardized_time_features(self) -> np.ndarray:
        """
        获取标准化的时间特征层
        
        时间信息编码原则：
        1. 周期性时间编码：sin/cos避免边界问题
        2. 多尺度时间表示：小时、日、周、季节
        3. 进度指标：任务完成度和剩余时间
        
        Returns:
            np.ndarray: 6维时间特征向量
        """
        hour = self.current_time.hour
        day_of_year = self.current_time.timetuple().tm_yday
        
        # 周期性小时编码
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # 工作日/周末编码
        is_weekday = 1.0 if self.current_time.weekday() < 5 else 0.0
        
        # 季节性编码（基于年内天数）
        season_progress = math.sin(2 * math.pi * day_of_year / 365)
        
        # 任务时间进度（标准化到[0,1]）
        # 确保时间进度不超过1.0，current_step应该在[0, time_horizon-1]范围内
        time_progress = min(1.0, self.current_step / max(1, self.time_horizon))
        
        # 剩余时间紧急度
        time_urgency = min(1.0, (self.time_horizon - self.current_step) / max(1, self.time_horizon * 0.1))
        
        return np.array([hour_sin, hour_cos, is_weekday, season_progress, time_progress, time_urgency])
    
    def _get_standardized_market_features(self) -> np.ndarray:
        """
        获取标准化的市场特征层
        
        市场信息编码原则：
        1. 价格标准化：相对于历史均值的标准化
        2. 趋势量化：价格变化率和方向
        3. 预测信息：未来价格趋势预期
        4. 市场阶段：峰谷时段的明确标识
        
        Returns:
            np.ndarray: 7维市场特征向量
        """
        # 确保使用缓存的环境状态，避免随机性导致的不一致
        if not hasattr(self, '_cached_env_state') or self._env_state_cache_time != self.current_step:
            self._cached_env_state = self.env_dynamics.get_current_state(self.current_time)
            self._env_state_cache_time = self.current_step
        
        env_state = self._cached_env_state
        
        # 标准化当前价格（相对于基准价格）
        base_price = 0.12  # 基准电价 $/kWh
        if env_state is None:
            env_state = {'price': base_price, 'price_trend': 0.0, 'future_prices': [base_price] * 3}
        
        normalized_price = env_state['price'] / base_price  # 相对价格
        
        # 价格趋势强度（标准化）
        price_trend_strength = np.tanh(env_state['price_trend'])  # 使用tanh限制到[-1,1]
        
        # 未来价格预测（标准化）
        future_prices = env_state.get('future_prices', [env_state['price']] * 3)
        future_price_1 = future_prices[0] / base_price if len(future_prices) > 0 else normalized_price
        future_price_2 = future_prices[1] / base_price if len(future_prices) > 1 else normalized_price
        future_price_3 = future_prices[2] / base_price if len(future_prices) > 2 else normalized_price
        
        # 市场阶段明确标识
        hour = self.current_time.hour
        # 峰时：7-9时、18-21时
        is_peak_period = 1.0 if (7 <= hour <= 9) or (18 <= hour <= 21) else 0.0
        # 谷时：23-6时
        is_valley_period = 1.0 if (hour >= 23) or (hour <= 6) else 0.0
        
        return np.array([
            normalized_price, price_trend_strength, 
            future_price_1, future_price_2, future_price_3,
            is_peak_period, is_valley_period
        ])
    
    def _get_standardized_environment_features(self) -> np.ndarray:
        """
        获取标准化的环境特征层
        
        环境信息编码原则：
        1. 温度标准化：相对于舒适温度的偏差
        2. 辐照标准化：相对于标准测试条件
        3. 趋势量化：环境变化的方向和强度
        4. 季节性考虑：环境参数的季节调整
        
        Returns:
            np.ndarray: 5维环境特征向量
        """
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # 标准化温度（相对于舒适温度20°C）
        comfort_temp = 20.0
        normalized_temperature = (env_state['temperature'] - comfort_temp) / 15.0  # 假设±15°C为合理范围
        
        # 标准化太阳辐照度（相对于标准测试条件1000 W/m²）
        standard_irradiance = 1000.0
        normalized_irradiance = env_state['solar_irradiance'] / standard_irradiance
        
        # 环境趋势标准化
        temp_trend = np.tanh(env_state['weather_trend']['temperature_trend'])  # 限制到[-1,1]
        irradiance_trend = np.tanh(env_state['weather_trend']['irradiance_trend'])  # 限制到[-1,1]
        
        # 日照质量指标（综合考虑辐照度和时间）
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
        """
        验证公共信息层的完整性和一致性
        
        验证原则：
        1. 维度一致性：确保各层维度符合预期
        2. 数值范围：确保标准化值在合理范围内
        3. 语义完整：确保关键信息没有缺失
        4. 时序一致：确保时间相关特征的一致性
        """
        # 维度验证
        expected_dims = {
            'time_layer': 6,
            'market_layer': 7,
            'environment_layer': 5,
            'total': 18
        }
        
        assert len(time_layer) == expected_dims['time_layer'], f"时间层维度错误: {len(time_layer)} != {expected_dims['time_layer']}"
        assert len(market_layer) == expected_dims['market_layer'], f"市场层维度错误: {len(market_layer)} != {expected_dims['market_layer']}"
        assert len(environment_layer) == expected_dims['environment_layer'], f"环境层维度错误: {len(environment_layer)} != {expected_dims['environment_layer']}"
        assert len(public_features) == expected_dims['total'], f"公共信息总维度错误: {len(public_features)} != {expected_dims['total']}"
        
        # 数值范围验证
        # 时间层验证：sin/cos值应在[-1,1]，其他应在[0,1]
        assert -1.1 <= time_layer[0] <= 1.1, f"小时sin编码超范围: {time_layer[0]}"
        assert -1.1 <= time_layer[1] <= 1.1, f"小时cos编码超范围: {time_layer[1]}"
        assert 0 <= time_layer[4] <= 1.1, f"时间进度超范围: {time_layer[4]}"
        
        # 市场层验证：价格应为正值，趋势应在[-1,1]
        assert market_layer[0] > 0, f"标准化价格应为正值: {market_layer[0]}"
        assert -1.1 <= market_layer[1] <= 1.1, f"价格趋势超范围: {market_layer[1]}"
        
        # 环境层验证：辐照度应非负，趋势应在[-1,1]
        assert environment_layer[1] >= 0, f"标准化辐照度应非负: {environment_layer[1]}"
        assert -1.1 <= environment_layer[2] <= 1.1, f"温度趋势超范围: {environment_layer[2]}"
        
        # 时序一致性验证
        hour = self.current_time.hour
        expected_hour_sin = math.sin(2 * math.pi * hour / 24)
        assert abs(time_layer[0] - expected_hour_sin) < 0.01, f"小时编码不一致: {time_layer[0]} vs {expected_hour_sin}"
    
    def _get_limited_collaboration_info(self) -> Dict[str, List[float]]:
        """
        获取智能聚合的Manager间协作信息 - 重设计版本
        
        新的聚合信息设计原则：
        1. 自适应聚合：根据系统状态调整信息粒度
        2. 层次化聚合：提供不同抽象层次的信息
        3. 动态权重：基于相关性和重要性调整信息权重
        4. 时序聚合：考虑历史趋势而非仅当前状态
        
        Returns:
            Dict[str, List[float]]: 每个Manager的智能聚合信息
        """
        manager_info = {}
        
        # 计算系统级聚合统计
        system_stats = self._calculate_system_aggregation_stats()
        
        # 计算动态聚合权重
        aggregation_weights = self._calculate_dynamic_aggregation_weights()
        
        for manager_id, manager in self.manager_agents.items():
            # 1. 基础聚合指标（规模相关）
            scale_metrics = self._aggregate_scale_metrics(manager, system_stats)
            
            # 2. 性能聚合指标（效率相关）
            performance_metrics = self._aggregate_performance_metrics(manager, system_stats, aggregation_weights)
            
            # 3. 协作聚合指标（系统相关）
            collaboration_metrics = self._aggregate_collaboration_metrics(manager, system_stats)
            
            # 4. 时序聚合指标（趋势相关）
            temporal_metrics = self._aggregate_temporal_metrics(manager, manager_id)
            
            # 5. 自适应聚合策略
            adaptive_metrics = self._apply_adaptive_aggregation_strategy(
                manager, manager_id, system_stats
            )
            
            # 组合所有聚合指标
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
        """计算系统级聚合统计信息"""
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
        """计算动态聚合权重"""
        # 基于当前时间步和系统状态计算权重
        time_progress = self.current_step / max(1, self.time_horizon)
        
        # 早期阶段更关注规模和设置，后期更关注性能
        weights = {
            'scale_weight': max(0.3, 1.0 - time_progress),  # 早期权重高
            'performance_weight': min(1.0, 0.5 + time_progress),  # 后期权重高
            'collaboration_weight': 0.6 + 0.2 * math.sin(time_progress * math.pi),  # 中期权重高
            'temporal_weight': min(1.0, time_progress * 2),  # 随时间增加
        }
        
        return weights
    
    def _aggregate_scale_metrics(self, manager: 'ManagerAgent', system_stats: Dict[str, float]) -> List[float]:
        """聚合规模相关指标"""
        # 相对规模（标准化到[0,1]）
        user_ratio = len(manager.users) / max(1, system_stats['total_users'])
        device_ratio = len(manager.device_mdps) / max(1, system_stats['total_devices'])
        
        # 相对容量（基于设备数量估算）
        capacity_indicator = min(1.0, len(manager.device_mdps) / 30.0)  # 假设30为高容量阈值
        
        return [user_ratio, device_ratio, capacity_indicator]
    
    def _aggregate_performance_metrics(self, manager: 'ManagerAgent', 
                                     system_stats: Dict[str, float],
                                     weights: Dict[str, float]) -> List[float]:
        """聚合性能相关指标"""
        # 能效指标（能耗相对于规模）
        manager_energy = manager.markov_history['cumulative_energy']
        manager_users = len(manager.users)
        
        if manager_users > 0 and system_stats['total_energy'] > 0:
            energy_efficiency = 1.0 - (manager_energy / manager_users) / (system_stats['total_energy'] / system_stats['total_users'])
            energy_efficiency = np.clip(energy_efficiency, -1.0, 1.0)
        else:
            energy_efficiency = 0.0
        
        # 满意度相对水平
        satisfaction_level = manager.markov_history['user_satisfaction'] - system_stats['avg_satisfaction']
        satisfaction_level = np.clip(satisfaction_level, -1.0, 1.0)
        
        # 综合性能指标
        performance_score = (energy_efficiency * 0.6 + satisfaction_level * 0.4) * weights['performance_weight']
        
        return [energy_efficiency, satisfaction_level, performance_score]
    
    def _aggregate_collaboration_metrics(self, manager: 'ManagerAgent', 
                                       system_stats: Dict[str, float]) -> List[float]:
        """聚合协作相关指标"""
        # 系统贡献度（基于能耗占比）
        if system_stats['total_energy'] > 0:
            contribution_ratio = manager.markov_history['cumulative_energy'] / system_stats['total_energy']
        else:
            contribution_ratio = 0.0
        
        # 系统平衡度影响（该Manager对系统平衡的影响）
        balance_impact = 1.0 - abs(contribution_ratio - (1.0 / len(self.manager_agents)))
        
        # 协作活跃度（基于是否有实际活动）
        activity_level = 1.0 if manager.markov_history['cumulative_energy'] > 0 else 0.0
        
        return [contribution_ratio, balance_impact, activity_level]
    
    def _aggregate_temporal_metrics(self, manager: 'ManagerAgent', manager_id: str) -> List[float]:
        """聚合时序相关指标"""
        # 获取历史数据（使用markov_history中的满意度历史）
        # ManagerAgent类使用markov_history存储历史信息，而不是performance_history
        history = [manager.markov_history['user_satisfaction']]
        
        # 趋势指标
        if len(history) >= 2:
            recent_trend = history[-1] - history[-2]
            trend_indicator = np.clip(recent_trend, -1.0, 1.0)
        else:
            trend_indicator = 0.0
        
        # 稳定性指标
        if len(history) >= 3:
            stability = 1.0 - np.std(history[-3:]) / max(np.mean(history[-3:]), 0.1)
            stability = np.clip(stability, 0.0, 1.0)
        else:
            stability = 1.0
        
        # 改进潜力（基于历史最佳表现）
        if history:
            current_performance = history[-1]
            best_performance = max(history)
            improvement_potential = max(0.0, best_performance - current_performance)
        else:
            improvement_potential = 0.5  # 中性值
        
        return [float(trend_indicator), float(stability), float(improvement_potential)]
    
    def _apply_adaptive_aggregation_strategy(self, manager: 'ManagerAgent', 
                                           manager_id: str,
                                           system_stats: Dict[str, float]) -> List[float]:
        """应用自适应聚合策略"""
        # 基于系统状态调整聚合策略
        
        # 1. 系统压力指标
        if system_stats['total_energy'] > 0:
            system_load = system_stats['total_energy'] / (system_stats['total_users'] * 24)  # 平均每用户每小时能耗
            pressure_indicator = min(1.0, system_load / 5.0)  # 假设5kWh/用户/小时为高压力
        else:
            pressure_indicator = 0.0
        
        # 2. 协调需求指标
        coordination_need = 1.0 - system_stats['system_balance']  # 系统越不平衡，协调需求越高
        
        # 3. 自适应响应能力
        manager_flexibility = len(manager.controllable_devices) / max(1, len(manager.device_mdps))
        
        # 4. 相对重要性（基于规模和性能）
        scale_importance = len(manager.users) / max(1, system_stats['total_users'])
        performance_importance = abs(manager.markov_history['user_satisfaction'] - system_stats['avg_satisfaction'])
        relative_importance = (scale_importance * 0.6 + performance_importance * 0.4)
        
        return [pressure_indicator, coordination_need, manager_flexibility, relative_importance]
    
    def _apply_enhanced_information_mechanisms(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用增强的信息传递机制 - 完整Dec-POMDP版本
        
        集成全面的Dec-POMDP特性：
        1. 多层次信息延迟机制（固定延迟、随机延迟、网络延迟）
        2. 智能信息缺失机制（选择性丢失、时序丢失、重要性丢失）
        3. 网络中断模拟（间歇性连接、分区容忍）
        4. 传递质量降级（噪声、干扰、衰减）
        5. 信息重传和恢复机制
        
        Args:
            observation: 原始观测向量
            manager_id: Manager标识
            
        Returns:
            np.ndarray: 经过完整信息传递处理的观测向量
        """
        processed_observation = observation.copy()
        
        # 1. 应用多层次延迟机制
        processed_observation = self._apply_multi_level_delay(processed_observation, manager_id)
        
        # 2. 应用智能信息缺失机制
        processed_observation = self._apply_intelligent_information_loss(processed_observation, manager_id)
        
        # 3. 应用网络中断模拟
        processed_observation = self._apply_network_interruption_simulation(processed_observation, manager_id)
        
        # 4. 应用传递质量降级
        processed_observation = self._apply_transmission_quality_degradation(processed_observation, manager_id)
        
        # 5. 应用信息重传和恢复机制
        processed_observation = self._apply_information_recovery_mechanism(processed_observation, manager_id)
        
        return processed_observation
    
    def _apply_multi_level_delay(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用多层次延迟机制
        
        延迟类型：
        1. 固定延迟：基于配置的确定性延迟
        2. 随机延迟：基于概率分布的随机延迟
        3. 网络延迟：基于网络状况的动态延迟
        4. 负载延迟：基于系统负载的自适应延迟
        """
        if not self.dec_pomdp_config.enable_info_delay:
            return observation
        
        delayed_observation = observation.copy()
        
        # 1. 固定延迟（基础配置）
        if manager_id in self.observation_history and len(self.observation_history[manager_id]) > 0:
            fixed_delay_steps = self.dec_pomdp_config.max_delay_steps
            if len(self.observation_history[manager_id]) >= fixed_delay_steps:
                delayed_observation = self.observation_history[manager_id][-fixed_delay_steps].copy()
        
        # 2. 随机延迟（模拟网络抖动）
        if np.random.random() < 0.3:  # 30%概率发生随机延迟
            random_delay = np.random.randint(1, min(3, len(self.observation_history[manager_id])) + 1)
            if manager_id in self.observation_history and len(self.observation_history[manager_id]) >= random_delay:
                delayed_observation = self.observation_history[manager_id][-random_delay].copy()
        
        # 3. 网络延迟（基于动态质量管理器）
        if self.dynamic_quality_manager:
            network_conditions = getattr(self.dynamic_quality_manager, 'network_history', [])
            if network_conditions:
                from fo_common.dynamic_observation_quality import NetworkCondition
                current_condition = network_conditions[-1] if network_conditions else NetworkCondition.GOOD
                
                # 网络状况越差，延迟越大
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
        
        # 4. 负载延迟（基于系统负载）
        system_load = len(self.manager_agents) * self.current_step / max(1, self.time_horizon)
        if system_load > 0.7:  # 高负载时增加延迟
            load_delay_prob = (system_load - 0.7) * 0.5
            if np.random.random() < load_delay_prob:
                if manager_id in self.observation_history and len(self.observation_history[manager_id]) >= 2:
                    delayed_observation = self.observation_history[manager_id][-2].copy()
        
        return delayed_observation
    
    def _apply_intelligent_information_loss(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用智能信息缺失机制
        
        缺失类型：
        1. 选择性丢失：重要信息优先保留
        2. 时序丢失：基于时间模式的丢失
        3. 分量丢失：按照观测分量类型的丢失
        4. 累积丢失：随时间累积的信息损失
        """
        if not self.dec_pomdp_config.enable_info_missing:
            return observation
        
        lost_observation = observation.copy()
        missing_prob = self.dec_pomdp_config.missing_probability
        
        # 1. 选择性丢失（重要信息保护）
        # 假设观测向量的前1/3是最重要的私有信息，不应丢失
        protected_length = len(observation) // 3
        vulnerable_start = protected_length
        
        # 2. 时序丢失（基于时间模式）
        time_factor = math.sin(self.current_step * 0.1) * 0.1 + 1.0  # 周期性变化
        adjusted_missing_prob = missing_prob * time_factor
        
        # 3. 分量丢失（他者信息更容易丢失）
        for i in range(len(lost_observation)):
            if i >= vulnerable_start:  # 保护私有信息
                # 他者信息的丢失概率更高
                component_missing_prob = adjusted_missing_prob * 1.5
                
                # 4. 累积丢失（距离当前时间越远的信息越容易丢失）
                distance_factor = 1.0 + (i - vulnerable_start) * 0.1
                final_missing_prob = min(0.8, component_missing_prob * distance_factor)
                
                if np.random.random() < final_missing_prob:
                    lost_observation[i] = 0.0  # 信息丢失
        
        # 5. 批量丢失（模拟连接中断）
        if np.random.random() < 0.05:  # 5%概率发生批量丢失
            batch_start = max(vulnerable_start, np.random.randint(0, len(lost_observation) - 5))
            batch_length = min(5, len(lost_observation) - batch_start)
            lost_observation[batch_start:batch_start + batch_length] = 0.0
        
        return lost_observation
    
    def _apply_network_interruption_simulation(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用网络中断模拟
        
        中断类型：
        1. 瞬时中断：短时间完全断开
        2. 间歇性中断：周期性连接问题
        3. 分区中断：部分Manager间连接中断
        4. 降级中断：连接质量严重下降
        """
        interrupted_observation = observation.copy()
        
        # 初始化Manager网络状态（如果不存在）
        if not hasattr(self, 'manager_network_states'):
            self.manager_network_states = {mid: 'connected' for mid in self.manager_ids}
        
        # 1. 瞬时中断（短时间完全断开）
        interruption_prob = 0.02  # 2%概率发生瞬时中断
        if np.random.random() < interruption_prob:
            self.manager_network_states[manager_id] = 'interrupted'
            # 瞬时中断期间，只保留私有信息
            private_length = len(interrupted_observation) // 3
            interrupted_observation[private_length:] = 0.0
        
        # 2. 间歇性中断（基于正弦波模式）
        intermittent_factor = math.sin(self.current_step * 0.3) + 1.0
        if intermittent_factor < 0.5 and np.random.random() < 0.1:
            self.manager_network_states[manager_id] = 'intermittent'
            # 间歇性中断期间，随机丢失50%的他者信息
            private_length = len(interrupted_observation) // 3
            for i in range(private_length, len(interrupted_observation)):
                if np.random.random() < 0.5:
                    interrupted_observation[i] = 0.0
        
        # 3. 分区中断（模拟Manager间连接问题）- 已注释，无噪声环境测试
        # if hasattr(self, 'network_partition_active'):
        #     if self.network_partition_active and manager_id in getattr(self, 'partitioned_managers', []):
        #         # 分区中的Manager无法获得其他Manager信息
        #         private_length = len(interrupted_observation) // 3
        #         interrupted_observation[private_length:] = interrupted_observation[private_length:] * 0.1
        
        # 4. 降级中断（连接质量严重下降）
        if self.dynamic_quality_manager:
            network_history = getattr(self.dynamic_quality_manager, 'network_history', [])
            if network_history:
                from fo_common.dynamic_observation_quality import NetworkCondition
                current_condition = network_history[-1]
                if current_condition == NetworkCondition.CRITICAL:
                    # 严重降级时，大幅减少他者信息质量
                    private_length = len(interrupted_observation) // 3
                    degradation_factor = 0.2
                    interrupted_observation[private_length:] = interrupted_observation[private_length:] * degradation_factor
        
        # 网络状态恢复机制
        recovery_prob = 0.3  # 30%概率恢复连接
        if (self.manager_network_states[manager_id] != 'connected' and 
            np.random.random() < recovery_prob):
            self.manager_network_states[manager_id] = 'connected'
        
        return interrupted_observation
    
    def _apply_transmission_quality_degradation(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用传递质量降级
        
        降级类型：
        1. 信噪比降级：添加传输噪声
        2. 量化降级：减少数值精度
        3. 压缩降级：信息压缩损失
        4. 衰减降级：信号衰减
        """
        degraded_observation = observation.copy()
        
        # 1. 信噪比降级（传输噪声）- 已注释，无噪声环境测试
        # if hasattr(self.dec_pomdp_config, 'enable_transmission_noise'):
        #     enable_noise = self.dec_pomdp_config.enable_transmission_noise
        # else:
        #     enable_noise = True  # 默认启用
        enable_noise = False  # 无噪声环境测试
        
        if enable_noise:
            # 基于网络状况调整噪声水平
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
        
        # 2. 量化降级（数值精度下降）
        if np.random.random() < 0.1:  # 10%概率发生量化降级
            quantization_levels = 256  # 8-bit量化
            degraded_observation = np.round(degraded_observation * quantization_levels) / quantization_levels
        
        # 3. 压缩降级（模拟数据压缩损失）
        if np.random.random() < 0.05:  # 5%概率发生压缩降级
            compression_factor = 0.95
            degraded_observation = degraded_observation * compression_factor
        
        # 4. 衰减降级（距离和时间相关的信号衰减）
        distance_factor = 1.0  # 假设所有Manager距离相等
        time_factor = 1.0 - (self.current_step / self.time_horizon) * 0.05  # 时间衰减
        attenuation_factor = distance_factor * time_factor
        
        if attenuation_factor < 1.0:
            degraded_observation = degraded_observation * attenuation_factor
        
        return degraded_observation
    
    def _apply_information_recovery_mechanism(self, observation: np.ndarray, manager_id: str) -> np.ndarray:
        """
        应用信息重传和恢复机制
        
        恢复策略：
        1. 缓存恢复：使用历史缓存填补丢失信息
        2. 估计恢复：基于历史趋势估计丢失值
        3. 插值恢复：使用邻近值进行插值
        4. 默认值恢复：使用安全默认值
        """
        recovered_observation = observation.copy()
        
        # 初始化恢复缓存
        if not hasattr(self, 'recovery_cache'):
            self.recovery_cache = {}
        if manager_id not in self.recovery_cache:
            self.recovery_cache[manager_id] = []
        
        # 1. 缓存恢复（使用最近的有效值）
        valid_indices = np.where(np.abs(recovered_observation) > 1e-8)[0]  # 非零值认为有效
        invalid_indices = np.where(np.abs(recovered_observation) <= 1e-8)[0]  # 零值认为丢失
        
        if len(self.recovery_cache[manager_id]) > 0 and len(invalid_indices) > 0:
            last_valid_observation = self.recovery_cache[manager_id][-1]
            
            for idx in invalid_indices:
                if idx < len(last_valid_observation):
                    # 2. 估计恢复（基于历史趋势）
                    if len(self.recovery_cache[manager_id]) >= 2:
                        recent_values = [cache[idx] for cache in self.recovery_cache[manager_id][-2:] 
                                       if idx < len(cache)]
                        if len(recent_values) >= 2:
                            trend = recent_values[-1] - recent_values[-2]
                            estimated_value = recent_values[-1] + trend * 0.5  # 保守估计
                            recovered_observation[idx] = estimated_value
                        else:
                            recovered_observation[idx] = last_valid_observation[idx]
                    else:
                        recovered_observation[idx] = last_valid_observation[idx]
                else:
                    # 3. 默认值恢复
                    recovered_observation[idx] = 0.0
        
        # 4. 插值恢复（对连续丢失进行插值）
        for i in range(1, len(recovered_observation) - 1):
            if (abs(recovered_observation[i]) <= 1e-8 and 
                abs(recovered_observation[i-1]) > 1e-8 and 
                abs(recovered_observation[i+1]) > 1e-8):
                # 线性插值
                recovered_observation[i] = (recovered_observation[i-1] + recovered_observation[i+1]) / 2.0
        
        # 更新恢复缓存
        self.recovery_cache[manager_id].append(recovered_observation.copy())
        
        # 限制缓存大小
        if len(self.recovery_cache[manager_id]) > 10:
            self.recovery_cache[manager_id] = self.recovery_cache[manager_id][-10:]
        
        return recovered_observation
    
    def _get_environment_features(self) -> np.ndarray:
        """获取环境特征"""
        # 时间特征
        hour = self.current_time.hour
        time_features = np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if self.current_time.weekday() < 5 else 0.0,
            self.current_step / self.time_horizon
        ])
        
        # 环境状态
        env_state = self.env_dynamics.get_current_state(self.current_time)
        
        # 价格特征
        price_features = np.array([
            env_state['price'],
            env_state['price_trend'],
            env_state.get('future_prices', [0, 0, 0])[0] if env_state.get('future_prices') else 0,
            env_state.get('future_prices', [0, 0, 0])[1] if env_state.get('future_prices') else 0,
            env_state.get('future_prices', [0, 0, 0])[2] if env_state.get('future_prices') else 0
        ])
        
        # 天气特征
        weather_features = np.array([
            env_state['temperature'],
            env_state['solar_irradiance'],
            env_state['weather_trend']['temperature_trend'],
            env_state['weather_trend']['irradiance_trend']
        ])
        
        return np.concatenate([time_features, price_features, weather_features])
    
    def _get_all_manager_aggregated_info(self) -> Dict[str, List[float]]:
        """
        获取Manager聚合信息 - Dec-POMDP受限版本
        
        这是对原有方法的Dec-POMDP重构，现在：
        1. 默认返回受限的协作信息
        2. 不再提供详细的竞争力指标和地理信息
        3. 遵循Dec-POMDP的信息共享限制原则
        
        注意：为了保持向后兼容性，这个方法保留了原有的接口，
        但内部实现已经修改为调用受限信息方法
        """
        # 如果明确禁用了其他Manager信息，返回空信息
        if not self.dec_pomdp_config.enable_other_manager_info:
            return {manager_id: [] for manager_id in self.manager_agents.keys()}
        
        # 根据配置决定信息共享级别 - 已注释，无噪声环境测试
        # if hasattr(self.dec_pomdp_config, 'information_sharing_level'):
        #     sharing_level = self.dec_pomdp_config.information_sharing_level
        # else:
        #     sharing_level = 'limited'  # 默认为受限模式
        sharing_level = 'full'  # 无噪声环境测试，使用完整信息共享
        
        if sharing_level == 'none':
            # 无信息共享模式：完全隔离的POMDP
            return {manager_id: [] for manager_id in self.manager_agents.keys()}
        
        elif sharing_level == 'minimal':
            # 最小信息共享：只有存在性指示
            minimal_info = {}
            for manager_id in self.manager_agents.keys():
                minimal_info[manager_id] = [1.0]  # 仅表示Manager存在
            return minimal_info
        
        elif sharing_level == 'limited':
            # 受限信息共享：调用新的受限协作信息方法
            return self._get_limited_collaboration_info()
        
        elif sharing_level == 'legacy':
            # 传统模式：保留原有详细信息（仅用于向后兼容和调试）
            return self._get_legacy_detailed_manager_info()
        
        else:
            # 默认使用受限模式
            return self._get_limited_collaboration_info()
    
    def _get_legacy_detailed_manager_info(self) -> Dict[str, List[float]]:
        """
        获取传统的详细Manager信息 - 仅用于向后兼容
        
        警告：此方法提供详细信息，违背Dec-POMDP原则，
        仅应在调试或特殊兼容性需求时使用
        """
        manager_info = {}
        
        # 计算全系统统计信息
        total_users = sum(len(m.users) for m in self.manager_agents.values())
        total_devices = sum(len(m.device_mdps) for m in self.manager_agents.values())
        total_cost = sum(m.markov_history['cumulative_cost'] for m in self.manager_agents.values())
        total_energy = sum(m.markov_history['cumulative_energy'] for m in self.manager_agents.values())
        avg_satisfaction = np.mean([m.markov_history['user_satisfaction'] for m in self.manager_agents.values()])
        
        for manager_id, manager in self.manager_agents.items():
            # 传统的详细信息（已弃用，但保留兼容性）
            legacy_info = [
                len(manager.users),  # 绝对用户数
                len(manager.device_mdps),  # 绝对设备数
                manager.markov_history['cumulative_cost'],  # 精确成本
                manager.markov_history['cumulative_energy'],  # 精确能耗
                manager.markov_history['user_satisfaction'],  # 精确满意度
                
                # 相对指标（略微受限）
                len(manager.users) / max(1, total_users),
                len(manager.device_mdps) / max(1, total_devices),
                manager.markov_history['cumulative_cost'] / max(1, total_cost) if total_cost > 0 else 0,
                manager.markov_history['cumulative_energy'] / max(1, total_energy) if total_energy > 0 else 0,
                manager.markov_history['user_satisfaction'] - avg_satisfaction,
            ]
            
            manager_info[manager_id] = legacy_info
        
        return manager_info
    
    def _get_market_state_features(self) -> np.ndarray:
        """获取全局市场状态特征"""
        # 系统总体供需状态
        total_devices = sum(len(m.device_mdps) for m in self.manager_agents.values())
        total_controllable = sum(len(m.controllable_devices) for m in self.manager_agents.values())
        total_users = sum(len(m.users) for m in self.manager_agents.values())
        
        # 计算系统总体能力指标
        avg_devices_per_user = total_devices / max(1, total_users)
        controllability_ratio = total_controllable / max(1, total_devices)
        
        # 计算Manager间的竞争强度
        manager_count = len(self.manager_agents)
        user_distribution_variance = 0.0
        if manager_count > 1:
            user_counts = [len(m.users) for m in self.manager_agents.values()]
            user_distribution_variance = np.var(user_counts) / max(1, np.mean(user_counts))
        
        # 计算满意度分布情况
        satisfactions = [m.markov_history['user_satisfaction'] for m in self.manager_agents.values()]
        satisfaction_mean = np.mean(satisfactions)
        satisfaction_std = np.std(satisfactions)
        
        # 计算能耗分布情况
        energies = [m.markov_history['cumulative_energy'] for m in self.manager_agents.values()]
        energy_balance = np.std(energies) / max(1, np.mean(energies)) if np.mean(energies) > 0 else 0
        
        # 时间相关的市场状态
        time_progress = self.current_step / max(1, self.time_horizon)
        is_peak_hour = 1.0 if 7 <= self.current_time.hour <= 9 or 18 <= self.current_time.hour <= 21 else 0.0
        is_off_peak = 1.0 if 23 <= self.current_time.hour or self.current_time.hour <= 6 else 0.0
        
        # 历史趋势特征（基于当前可用数据的简化版）
        recent_activity = min(1.0, self.current_step / 5.0)  # 活跃度指标
        
        market_features = np.array([
            # 系统规模特征
            total_users,
            total_devices,
            total_controllable,
            avg_devices_per_user,
            controllability_ratio,
            
            # 竞争和分布特征
            manager_count,
            user_distribution_variance,
            energy_balance,
            satisfaction_mean,
            satisfaction_std,
            
            # 时间和活跃度特征
            time_progress,
            is_peak_hour,
            is_off_peak,
            recent_activity,
            
            # 系统状态指标
            1.0 if satisfaction_mean > 0.5 else 0.0,  # 系统满意度是否良好
            1.0 if energy_balance < 0.5 else 0.0,     # 能耗是否均衡
        ])
        
        return market_features.astype(np.float32)
    
    def generate_all_dfos(self) -> Dict[str, Dict[str, DFOSystem]]:
        """生成所有Manager的DFO系统"""
        all_dfos = {}
        
        for manager_id, manager in self.manager_agents.items():
            manager_dfos = manager.generate_dfo(self.time_horizon)
            all_dfos[manager_id] = manager_dfos
        
        return all_dfos
    
    def get_manager_count(self) -> int:
        """获取Manager数量"""
        return len(self.manager_agents)
    
    def get_total_user_count(self) -> int:
        """获取总用户数量"""
        return sum(len(manager.users) for manager in self.manager_agents.values())
    
    def get_total_device_count(self) -> int:
        """获取总设备数量"""
        return sum(len(manager.device_mdps) for manager in self.manager_agents.values())
    
    def get_current_observations(self):
        """获取当前时间步的观测"""
        obs = {}
        for manager_id, agent in self.manager_agents.items():
            obs[manager_id] = agent.get_observation()
        return obs
    
    def generate_current_dfos(self, timestep):
        """生成当前时间步的DFO系统"""
        dfo_systems = {}
        for manager_id, agent in self.manager_agents.items():
            agent_dfos = {}
            
            # 处理设备列表
            if isinstance(agent.devices, dict):
                devices_list = list(agent.devices.values())
            else:
                devices_list = agent.devices
            
            # 为每个设备生成DFO
            for device in devices_list:
                device_id = getattr(device, 'device_id', f"{manager_id}_device_{len(agent_dfos)}")
                
                # 根据设备类型生成FlexOffer - 包含核心特征
                from fo_generate.dfo import DFOSystem, DFOSlice
                from datetime import datetime, timedelta
                
                # 生成基本的FlexOffer参数
                energy_min = np.random.uniform(5, 20)     # 最小能量需求
                energy_max = np.random.uniform(20, 50)    # 最大能量需求
                power_min = np.random.uniform(-10, 0)     # 最小功率（负值表示放电）
                power_max = np.random.uniform(5, 15)      # 最大功率（正值表示充电）
                flexibility = np.random.uniform(0.2, 0.8) # 灵活性因子
                
                # 创建时间窗口
                current_time = datetime.now() + timedelta(hours=timestep)
                start_time = current_time  # 时间窗口开始
                end_time = current_time + timedelta(hours=1)  # 时间窗口结束
                
                # 创建DFO系统
                dfo_system = DFOSystem(
                    time_horizon=1,
                    device_id=device_id,
                    device_type=getattr(device, 'device_type', 'unknown')
                )
                
                # 创建DFO片段
                dfo_slice = DFOSlice(
                    time_step=timestep,
                    energy_min=energy_min,
                    energy_max=energy_max,
                    constraints=[],  # 基本约束，可以后续扩展
                    power_min=power_min,
                    power_max=power_max,
                    start_time=start_time,
                    end_time=end_time,
                    flexibility_factor=flexibility,
                    device_type=getattr(device, 'device_type', 'unknown'),
                    device_id=device_id
                )
                
                # 添加片段到DFO系统
                dfo_system.add_slice(dfo_slice)
                
                agent_dfos[device_id] = dfo_system
            
            if agent_dfos:
                dfo_systems[manager_id] = agent_dfos
        
        # 合并DFO生成信息到一行
        total_dfos = sum(len(dfos) for dfos in dfo_systems.values())
        manager_dfo_counts = [f"{manager_id}:{len(dfos)}" for manager_id, dfos in dfo_systems.items()]
        logger.info(f"时间步 {timestep} DFO生成: {', '.join(manager_dfo_counts)}, 总计 {total_dfos} 个")
        return dfo_systems
    
    def update_user_states(self, user_satisfied_energy, timestep):
        """更新用户状态基于已分配的能源"""
        try:
            # 根据用户满足的能源更新设备状态
            for manager_id, agent in self.manager_agents.items():
                for user in agent.users:
                    # 处理用户对象：users可能是字典列表
                    if isinstance(user, dict):
                        user_id = user.get('user_id', '')
                        user_devices = user.get('devices', [])
                    else:
                        # 如果是对象，则直接访问属性
                        user_id = getattr(user, 'user_id', '')
                        user_devices = getattr(user, 'devices', [])
                    
                    if user_id:
                        try:
                            # 解析用户ID格式：支持user_X和user_manager_X_Y格式
                            if 'manager_' in user_id:
                                # 格式：user_manager_X_Y，需要计算全局用户索引
                                parts = user_id.split('_')
                                if len(parts) >= 4:
                                    manager_num = int(parts[2])  # manager编号 (1, 2, 3, 4)
                                    user_local_num = int(parts[3])  # manager内用户编号 (1, 2, ...)
                                    
                                    # 根据Manager分布计算全局用户索引
                                    # Manager 1: 6用户 (索引0-5), Manager 2: 10用户 (索引6-15), 
                                    # Manager 3: 8用户 (索引16-23), Manager 4: 12用户 (索引24-35)
                                    user_distributions = [6, 10, 8, 12]
                                    base_index = sum(user_distributions[:manager_num-1])
                                    user_idx = base_index + (user_local_num - 1)
                                else:
                                    logger.warning(f"无法解析user_manager格式的ID: {user_id}")
                                    continue
                            else:
                                # 格式：user_X
                                user_idx = int(user_id.split('_')[1])
                            
                            if user_idx < user_satisfied_energy.shape[0]:
                                satisfied_energy = user_satisfied_energy[user_idx, timestep]
                                # 将满足的能源分配给用户的设备
                                if user_devices and satisfied_energy > 0:
                                    energy_per_device = satisfied_energy / len(user_devices)
                                    for device in user_devices:
                                        # 处理设备对象：devices可能是字典
                                        if isinstance(device, dict):
                                            device_id = device.get('device_id', '')
                                        else:
                                            device_id = getattr(device, 'device_id', '')
                                        
                                        if device_id:
                                            device_key = f"{user_id}_{device_id}"
                                            if device_key in agent.device_mdps:
                                                mdp_device = agent.device_mdps[device_key]
                                                if hasattr(mdp_device, 'env'):
                                                    # 更新设备状态以反映获得的能源
                                                    device_env = getattr(mdp_device, 'env', None)
                                                    if device_env is not None:
                                                        self._update_device_state_with_energy(device_env, energy_per_device)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析用户ID时出错 {user_id}: {e}")
        except Exception as e:
            logger.error(f"更新用户状态时出错: {e}")
    
    def _update_device_state_with_energy(self, device_env, allocated_energy):
        """使用分配的能源更新设备状态"""
        try:
            # 根据设备类型更新状态
            if hasattr(device_env, 'device_type'):
                if device_env.device_type == DeviceType.BATTERY:
                    # 电池：更新SOC状态
                    if hasattr(device_env, 'battery_device') and hasattr(device_env.battery_device, 'charge'):
                        # 将分配的能源用于充电
                        device_env.battery_device.charge(allocated_energy, device_env.time_step)
                elif device_env.device_type == DeviceType.EV:
                    # 电动车：更新充电状态
                    if hasattr(device_env, 'ev_device') and hasattr(device_env.ev_device, 'charge'):
                        device_env.ev_device.charge(allocated_energy, device_env.time_step)
                # 其他设备类型的状态更新可以在这里添加
        except Exception as e:
            logger.warning(f"更新设备状态时出现警告: {e}") 
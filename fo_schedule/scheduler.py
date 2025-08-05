import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import sys
import random
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlexScheduler")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fo_aggregate.manager import Manager, User, Device
from fo_aggregate.aggregator import AggregatedFlexOffer
from fo_trading.pool import TradingPool, WeatherModel, DemandModel, Trade



@dataclass
class DisaggregationRequest:
    """FO分解请求数据结构"""
    aggregated_result: Any  
    original_data: List[Dict]  
    total_energy: float  
    time_step: int  
    metadata: Dict[str, Any]  
    
    def __post_init__(self):
        """validate input data"""
        # check original data list
        if self.original_data is None:
            logger.warning("original data list is None, initialize as empty list")
            self.original_data = []
        
        # check total energy
        if self.total_energy is None:
            logger.warning("total energy is None, set to 0")
            self.total_energy = 0.0
        elif self.total_energy < 0:
            logger.warning(f"total energy is negative({self.total_energy}), set to 0")
            self.total_energy = 0.0
        
        # check time step
        if self.time_step is None:
            logger.warning("time step is None, set to 0")
            self.time_step = 0
        elif self.time_step < 0:
            logger.warning(f"time step is negative({self.time_step}), set to 0")
            self.time_step = 0
        
        # ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class DisaggregationResult:
    """FO disaggregation result data structure"""
    disaggregated_data: List[Dict]  
    algorithm_used: str  
    allocation_ratios: List[float]  
    total_allocated_energy: float  
    metadata: Dict[str, Any]  
    
    def __post_init__(self):
        """validate result data"""
        if len(self.disaggregated_data) != len(self.allocation_ratios):
            raise ValueError("disaggregated data and allocation ratios do not match")
        if self.total_allocated_energy < 0:
            raise ValueError("total allocated energy cannot be negative")


class DisaggregationAlgorithm(ABC):
    """FO disaggregation algorithm abstract base class"""
    
    def __init__(self, algorithm_name: str):

        self.algorithm_name = algorithm_name
        self.total_requests = 0
        self.total_energy_processed = 0.0
        self.performance_metrics = {}
    
    @abstractmethod
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:

        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """get algorithm information"""
        return {
            "name": self.algorithm_name,
            "total_requests": self.total_requests,
            "total_energy_processed": self.total_energy_processed,
            "performance_metrics": self.performance_metrics
        }
    
    def _validate_request(self, request: DisaggregationRequest) -> bool:
        """validate disaggregation request"""
        if not isinstance(request, DisaggregationRequest):
            raise ValueError("invalid disaggregation request type")
        return True
    
    def _update_metrics(self, request: DisaggregationRequest, result: DisaggregationResult):
        """update performance metrics"""
        self.total_requests += 1
        self.total_energy_processed += request.total_energy
        
        # calculate allocation efficiency
        efficiency = result.total_allocated_energy / request.total_energy if request.total_energy > 0 else 0
        if 'allocation_efficiency' not in self.performance_metrics:
            self.performance_metrics['allocation_efficiency'] = []
        self.performance_metrics['allocation_efficiency'].append(efficiency)


class AverageDisaggregationAlgorithm(DisaggregationAlgorithm):
    """average disaggregation algorithm: E_i = E/N"""
    
    def __init__(self):
        super().__init__("average")
        logger.info("initialize average disaggregation algorithm")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:

        self._validate_request(request)
        
        logger.info(f"start average disaggregation, original data number: {len(request.original_data)}, total energy: {request.total_energy:.2f}")
        
        # calculate average allocated energy: E_i = E/N
        num_participants = len(request.original_data)
        
        # check participant number, avoid division by zero
        if num_participants == 0:
            logger.error("cannot execute average disaggregation: no participants")
            # return empty result
            return DisaggregationResult(
                disaggregated_data=[],
                algorithm_used=self.algorithm_name,
                allocation_ratios=[],
                total_allocated_energy=0.0,
                metadata={
                    'error': 'no_participants',
                    'time_step': request.time_step
                }
            )
            
        average_energy = request.total_energy / num_participants
        
        # create disaggregation result
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # copy original data
            new_item = copy.deepcopy(item)
            
            # allocate average energy
            new_item['allocated_energy'] = average_energy
            new_item['allocation_method'] = 'average'
            # avoid division by zero
            original_energy = item.get('energy', 0)
            new_item['allocation_ratio'] = average_energy / original_energy if original_energy > 0 else 1.0
            
            disaggregated_data.append(new_item)
            # avoid division by zero
            allocation_ratio = average_energy / request.total_energy if request.total_energy > 0 else 1.0 / num_participants
            allocation_ratios.append(allocation_ratio)
        
        # create result object
        result = DisaggregationResult(
            disaggregated_data=disaggregated_data,
            algorithm_used=self.algorithm_name,
            allocation_ratios=allocation_ratios,
            total_allocated_energy=request.total_energy,
            metadata={
                'average_energy_per_participant': average_energy,
                'num_participants': num_participants,
                'time_step': request.time_step
            }
        )
        
        # update performance metrics
        self._update_metrics(request, result)
        
        logger.info(f"average disaggregation completed, each participant allocated: {average_energy:.2f} kWh")
        return result


class ProportionalDisaggregationAlgorithm(DisaggregationAlgorithm):
    """proportional disaggregation algorithm: E_i = (w_i/W) * E"""
    
    def __init__(self, weight_key: str = 'energy'):
        super().__init__("proportional")
        self.weight_key = weight_key
        logger.info(f"initialize proportional disaggregation algorithm, weight key: {weight_key}")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:

        self._validate_request(request)
        
        logger.info(f"start proportional disaggregation, original data number: {len(request.original_data)}, total energy: {request.total_energy:.2f}")
        
        # calculate total weight: W = Σw_i
        total_weight = sum(item.get(self.weight_key, 1.0) for item in request.original_data)
        
        if total_weight <= 0:
            logger.warning("total weight is 0, fallback to average allocation")
            # if total weight is 0, fallback to average allocation
            average_algo = AverageDisaggregationAlgorithm()
            return average_algo.disaggregate(request)
        
        # create disaggregation result
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # copy original data
            new_item = copy.deepcopy(item)
            
            # calculate weight ratio: w_i/W
            weight = item.get(self.weight_key, 1.0)
            weight_ratio = weight / total_weight
            
            # allocate energy proportionally: E_i = (w_i/W) * E
            allocated_energy = weight_ratio * request.total_energy
            
            new_item['allocated_energy'] = allocated_energy
            new_item['allocation_method'] = 'proportional'
            new_item['weight_used'] = weight
            new_item['weight_ratio'] = weight_ratio
            new_item['allocation_ratio'] = allocated_energy / item.get('energy', 1.0) if item.get('energy', 0) > 0 else 1.0
            
            disaggregated_data.append(new_item)
            allocation_ratios.append(weight_ratio)
        
        # create result object
        result = DisaggregationResult(
            disaggregated_data=disaggregated_data,
            algorithm_used=self.algorithm_name,
            allocation_ratios=allocation_ratios,
            total_allocated_energy=request.total_energy,
            metadata={
                'total_weight': total_weight,
                'weight_key_used': self.weight_key,
                'time_step': request.time_step
            }
        )
        
        # update performance metrics
        self._update_metrics(request, result)
        
        logger.info(f"proportional disaggregation completed, total weight: {total_weight:.2f}")
        return result


class DisaggregationAlgorithmFactory:
    """FO disaggregation algorithm factory"""
    
    _algorithms = {}
    _initialized = False
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type, **kwargs):

        cls._algorithms[name] = {
            'class': algorithm_class,
            'kwargs': kwargs
        }
        logger.info(f"FO disaggregation algorithm {name} registered")
    
    @classmethod
    def create_algorithm(cls, name: str, **override_kwargs) -> DisaggregationAlgorithm:
    
        if name not in cls._algorithms:
            raise ValueError(f"unknown disaggregation algorithm: {name}")
        
        algo_info = cls._algorithms[name]
        algo_class = algo_info['class']
        
        # merge parameters
        kwargs = algo_info['kwargs'].copy()
        kwargs.update(override_kwargs)
        
        return algo_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """get available algorithms list"""
        return list(cls._algorithms.keys())
    
    @classmethod
    def initialize_default_algorithms(cls):
        """initialize default algorithms"""
        if cls._initialized:
            return
        
        # register average disaggregation algorithm
        cls.register_algorithm("average", AverageDisaggregationAlgorithm)
        
        # register proportional disaggregation algorithm
        cls.register_algorithm("proportional", ProportionalDisaggregationAlgorithm, weight_key='energy')
        cls.register_algorithm("equal_proportion", ProportionalDisaggregationAlgorithm, weight_key='energy')
        
        # register equal disaggregation algorithm for compatibility
        cls.register_algorithm("equal", AverageDisaggregationAlgorithm)
        cls.register_algorithm("priority", ProportionalDisaggregationAlgorithm, weight_key='priority')
        
        cls._initialized = True
        logger.info("default FO disaggregation algorithms initialized")

# initialize default algorithms
DisaggregationAlgorithmFactory.initialize_default_algorithms()

class FlexOffer:
    def __init__(self, resource_id=None, resource_type=None, location=None, 
                 time_horizon=24, time_interval=1, quantity=0, price=0, 
                 time_window=None, device_type=None, constraints=None):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.location = location
        self.time_horizon = time_horizon
        self.time_interval = time_interval
        self.quantity = quantity
        self.price = price
        self.time_window = time_window or (0, 24)
        self.device_type = device_type
        self.constraints = constraints or {}
        self.power_profile = np.zeros((time_horizon, 2))
        self.baseline_profile = np.zeros(time_horizon)
        self.reliability = 1.0
    
    def set_power_profile(self, profile):
        self.power_profile = profile
    
    def set_baseline_profile(self, profile):
        self.baseline_profile = profile
    
    def set_reliability(self, reliability):
        self.reliability = reliability

class FlexOfferManager:
    def __init__(self, manager_id=None, location=None):
        self.manager_id = manager_id
        self.location = location
        self.offers = []
    
    def add_offer(self, offer):
        self.offers.append(offer)

class FlexOfferDisaggregator:
    """aggregated FlexOffer disaggregator, disaggregate aggregated FlexOffer back to original FlexOffer"""
    
    def __init__(self, time_horizon: int = 24):
        """
        initialize disaggregator
        
        Args:
            time_horizon: time horizon
        """
        self.time_horizon = time_horizon
    
    def disaggregate(self, 
                     aggregated_offer: FlexOffer, 
                     original_offers: List[FlexOffer]) -> List[FlexOffer]:

        if not original_offers:
            raise ValueError("No original offers to disaggregate to")
        
        # create disaggregated FlexOffer list
        disaggregated_offers = []
        
        # total power and baseline
        total_min_power = np.zeros(self.time_horizon)
        total_max_power = np.zeros(self.time_horizon)
        total_baseline = np.zeros(self.time_horizon)
        
        for fo in original_offers:
            total_min_power += fo.power_profile[:, 0]
            total_max_power += fo.power_profile[:, 1]
            total_baseline += fo.baseline_profile
        
        # calculate aggregated FlexOffer's actual power
        aggregated_power = aggregated_offer.baseline_profile
        
        # allocate power
        for fo in original_offers:
            # create new FlexOffer
            new_fo = FlexOffer(
                resource_id=fo.resource_id,
                resource_type=fo.resource_type,
                location=fo.location,
                time_horizon=fo.time_horizon,
                time_interval=fo.time_interval
            )
            
            # copy power range
            new_fo.set_power_profile(fo.power_profile.copy())
            
            # allocate power proportionally to original contribution
            baseline_ratio = np.zeros(self.time_horizon)
            for t in range(self.time_horizon):
                if total_baseline[t] > 0:
                    baseline_ratio[t] = fo.baseline_profile[t] / total_baseline[t]
                else:
                    baseline_ratio[t] = 1.0 / len(original_offers)
            
            # calculate new baseline
            new_baseline = aggregated_power * baseline_ratio
            new_fo.set_baseline_profile(new_baseline)
            
            # set reliability
            new_fo.set_reliability(fo.reliability)
            
            disaggregated_offers.append(new_fo)
        
        return disaggregated_offers


class UserScheduler:
    """user scheduler, allocate energy to users according to user demands"""
    
    def __init__(self, 
                 num_users: int = 20,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1):
        """
        initialize scheduler
        
        Args:
            num_users: number of users
            time_horizon: time horizon (hours)
            time_steps_per_hour: number of time steps per hour
        """
        self.num_users = num_users
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        
        # user demands
        self.user_demands = np.zeros((num_users, self.total_steps))
        
        # user allocations
        self.user_allocations = np.zeros((num_users, self.total_steps))
        
        # user energy sources
        self.user_sources = {}
        
        # user configurations (can specify user priority, preferences, etc.)
        self.user_configs = [
            {'id': i, 'priority': random.uniform(0, 1), 'preferences': {}} 
            for i in range(num_users)
        ]
        
        logger.info(f"initialize user scheduler, number of users: {num_users}, time horizon: {time_horizon} hours")
    
    def set_user_demands(self, demands: np.ndarray):

        assert demands.shape == (self.num_users, self.total_steps), f"demand dimension mismatch: {demands.shape} vs {(self.num_users, self.total_steps)}"
        self.user_demands = demands
    
    def get_user_demand(self, user_id: int, step: int) -> float:

        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(f"user ID {user_id} out of range [0, {self.num_users-1}]")
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"time step {step} out of range [0, {self.total_steps-1}]")
        
        return self.user_demands[user_id, step]
    
    def schedule(self, 
                energy_resources: List[Dict], 
                step: int,
                method: str = 'priority') -> Dict[int, List[Dict]]:

        if step < 0 or step >= self.total_steps:
            raise ValueError(f"time step {step} out of range [0, {self.total_steps-1}]")
        
        # get current time step's user demands
        current_demands = self.user_demands[:, step].copy()
        
        # sort users according to the selected method
        if method == 'priority':
            # sort by priority (users with higher priority first)
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: self.user_configs[i]['priority'],
                reverse=True
            )
        elif method == 'fairness':
            # sort by historical satisfaction rate (users with lower satisfaction rate first)
            satisfaction_rates = []
            for user_id in range(self.num_users):
                total_demand = np.sum(self.user_demands[user_id, :step+1])
                total_allocation = np.sum(self.user_allocations[user_id, :step+1])
                
                if total_demand > 0:
                    rate = total_allocation / total_demand
                else:
                    rate = 1.0
                
                satisfaction_rates.append(rate)
            
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: satisfaction_rates[i]
            )
        elif method == 'cost':
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: current_demands[i],
                reverse=True
            )
        else:
            # default sort by user ID
            sorted_user_indices = list(range(self.num_users))
        
        # filter users with demand
        sorted_user_indices = [i for i in sorted_user_indices if current_demands[i] > 0]
        
        # count total available energy
        total_available_energy = sum(item.get('allocated_energy', 0) for item in energy_resources)
        
        # count total current demand
        total_demand = sum(current_demands)
        
        logger.info(f"time step {step}: number of users={len(sorted_user_indices)}, "
                   f"total demand={total_demand:.2f} kWh, total available energy={total_available_energy:.2f} kWh")
        
        # allocation result
        allocations = {user_id: [] for user_id in range(self.num_users)}
        
        # allocate energy resources
        for user_id in sorted_user_indices:
            user_demand = current_demands[user_id]
            
            # if user has no demand, skip
            if user_demand <= 0:
                continue
            
            # allocate resources to user
            remaining_demand = user_demand
            
            for resource in energy_resources:
                # check if resource has available energy
                available_energy = resource.get('allocated_energy', 0)
                if available_energy <= 0:
                    continue
                
                # allocation amount = min(user remaining demand, available energy)
                allocation_amount = min(remaining_demand, available_energy)
                
                if allocation_amount > 0:
                    # update available energy of resource
                    resource['allocated_energy'] -= allocation_amount
                    
                    # update user remaining demand
                    remaining_demand -= allocation_amount
                    
                    # record allocation result
                    allocation = {
                        'resource_id': resource.get('resource_id', ''),
                        'energy_type': resource.get('energy_type', ''),
                        'amount': allocation_amount,
                        'price': resource.get('price', 0.0)
                    }
                    
                    allocations[user_id].append(allocation)
                    
                    # update user allocation record
                    self.user_allocations[user_id, step] += allocation_amount
                
                # if user demand is satisfied, end loop
                if remaining_demand <= 0:
                    break
        
        # update user energy source record
        self.user_sources[step] = allocations
        
        # calculate energy resource utilization rate
        total_allocated = sum(self.user_allocations[:, step])
        utilization_rate = total_allocated / total_available_energy if total_available_energy > 0 else 0.0
        
        logger.info(f"time step {step}: allocation completed, total allocated={total_allocated:.2f} kWh, "
                   f"utilization rate={utilization_rate*100:.2f}%")
        
        return allocations
    
    def get_user_satisfaction(self, step: int) -> np.ndarray:
        """
        get user satisfaction (demand satisfaction rate)
        
        Args:
            step: time step
            
        Returns:
            np.ndarray: user satisfaction, range [0,1]
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"time step {step} out of range [0, {self.total_steps-1}]")
        
        # calculate user satisfaction
        satisfaction = np.zeros(self.num_users)
        for user_id in range(self.num_users):
            user_demand = self.user_demands[user_id, step]
            if user_demand > 0:
                satisfaction[user_id] = min(1.0, self.user_allocations[user_id, step] / user_demand)
            else:
                satisfaction[user_id] = 1.0  # no demand, default satisfied
        
        return satisfaction
    
    def get_overall_satisfaction(self) -> float:
        """
        get overall satisfaction (total demand satisfaction rate)
        
        Returns:
            float: overall satisfaction, range [0,1]
        """
        total_demand = np.sum(self.user_demands)
        total_allocation = np.sum(self.user_allocations)
        
        if total_demand > 0:
            return float(min(1.0, total_allocation / total_demand))
        else:
            return 1.0
    
    def visualize_allocation(self, step: Optional[int] = None, save_path: Optional[str] = None):

        plt.figure(figsize=(12, 6))
        
        if step is not None:
            if step < 0 or step >= self.total_steps:
                raise ValueError(f"time step {step} out of range [0, {self.total_steps-1}]")
            
            # show allocation of specific time step
            demands = self.user_demands[:, step]
            allocations = self.user_allocations[:, step]
            
            # calculate satisfaction
            satisfaction = self.get_user_satisfaction(step)
            
            # set x-axis and chart
            x = np.arange(self.num_users)
            width = 0.4
            
            # plot demand and allocation
            plt.bar(x - width/2, demands, width, label='demand')
            plt.bar(x + width/2, allocations, width, label='allocation')
            
            # plot satisfaction line
            plt.plot(x, satisfaction, 'r-', label='satisfaction')
            
            plt.xlabel('user ID')
            plt.ylabel('energy (kWh)')
            plt.title(f'user energy allocation at time step {step}')
            plt.xticks(x)
            plt.legend()
            
        else:
            # show total allocation of all time steps
            total_demands = np.sum(self.user_demands, axis=1)
            total_allocations = np.sum(self.user_allocations, axis=1)
            
            # calculate overall satisfaction
            satisfaction = []
            for user_id in range(self.num_users):
                if total_demands[user_id] > 0:
                    satisfaction.append(min(1.0, total_allocations[user_id] / total_demands[user_id]))
                else:
                    satisfaction.append(1.0)
            
            # set x-axis and chart
            x = np.arange(self.num_users)
            width = 0.4
            
            # plot total demand and allocation
            plt.bar(x - width/2, total_demands, width, label='total demand')
            plt.bar(x + width/2, total_allocations, width, label='total allocation')
            
            # plot satisfaction line
            plt.plot(x, satisfaction, 'r-', label='overall satisfaction')
            
            plt.xlabel('user ID')
            plt.ylabel('energy (kWh)')
            plt.title('total user energy allocation')
            plt.xticks(x)
            plt.legend()
        
        # save or show chart
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def update_cumulative_demands(self, cumulative_demands, timestep):
        """update cumulative demands"""
        try:
            if cumulative_demands.shape[0] == self.num_users:
                # update to current time step's cumulative demands
                if hasattr(self, 'cumulative_user_demands'):
                    # update existing cumulative demands
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                else:
                    # initialize cumulative demands
                    self.cumulative_user_demands = np.zeros((self.num_users, self.total_steps))
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                
                logger.debug(f"UserScheduler cumulative demands updated to time step {timestep}")  # change to DEBUG level to avoid repetition
            else:
                logger.warning(f"cumulative demands data dimension mismatch: expected {self.num_users} users, actual {cumulative_demands.shape[0]} users")
        except Exception as e:
            logger.error(f"error updating cumulative demands: {e}")


class ScheduleManager:
    """schedule manager, coordinate energy resource disaggregation and user scheduling"""
    
    def __init__(self, 
                 managers: List[Manager],
                 trading_pool: TradingPool,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1,
                 disaggregation_algorithm: str = 'proportional'):
        """
        initialize schedule manager
        
        Args:
            managers: Manager list
            trading_pool: trading pool
            time_horizon: time horizon (hours)
            time_steps_per_hour: number of time steps per hour
            disaggregation_algorithm: disaggregation algorithm, optional 'average', 'proportional', 'equal_proportion'
        """
        self.managers = managers
        self.trading_pool = trading_pool
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        self.disaggregation_algorithm = disaggregation_algorithm
        
        # create disaggregator (using new algorithm architecture)
        self.disaggregator = AggregatedResultDisaggregator(
            time_horizon=time_horizon,
            default_algorithm=disaggregation_algorithm
        )
        
        # create user scheduler (one per manager, based on actual user number)
        self.user_schedulers = {}
        for manager in managers:
            actual_users = len(manager.users)  # use actual user number
            scheduler = UserScheduler(
                num_users=actual_users,
                time_horizon=time_horizon,
                time_steps_per_hour=time_steps_per_hour
            )
            self.user_schedulers[manager.manager_id] = scheduler
            logger.info(f"create user scheduler for manager {manager.manager_id}, number of users: {actual_users}")
        
        # user demands data
        self.user_demands = None
        
        # satisfaction history
        self.satisfaction_history = []
        
        # trade history cache
        self.processed_trades = set()
        
        logger.info(f"initialize schedule manager, number of managers: {len(managers)}, time horizon: {time_horizon} hours, disaggregation algorithm: {disaggregation_algorithm}")
        logger.info(f"available disaggregation algorithms: {self.disaggregator.get_available_algorithms()}")
    
    def set_disaggregation_algorithm(self, algorithm_name: str):
        """
        set disaggregation algorithm
        
        Args:
            algorithm_name: algorithm name
        """
        self.disaggregation_algorithm = algorithm_name
        self.disaggregator.set_default_algorithm(algorithm_name)
        logger.info(f"switched disaggregation algorithm to: {algorithm_name}")
    
    def get_disaggregation_performance(self) -> Dict[str, Any]:
        """get disaggregation algorithm performance statistics"""
        return self.disaggregator.get_performance_summary()
    
    def set_user_demands(self, demands: np.ndarray):
        """
        set user demands
        
        Args:
            demands: user demands, dimension: [actual_total_users, total_steps]
        """
        # calculate actual total number of users
        actual_total_users = sum(len(manager.users) for manager in self.managers)
        expected_shape = (actual_total_users, self.total_steps)
        
        logger.info(f"expected demand dimension: {expected_shape}, actual input dimension: {demands.shape}")
        
        if demands.shape != expected_shape:
            logger.warning(f"demand dimension mismatch: {demands.shape} vs {expected_shape}, will try to adjust")
            
            # if user number mismatch, adjust or truncate/fill
            if demands.shape[0] > actual_total_users:
                demands = demands[:actual_total_users, :]
                logger.info(f"truncate demand data to first {actual_total_users} users")
            elif demands.shape[0] < actual_total_users:
                padding = np.zeros((actual_total_users - demands.shape[0], self.total_steps))
                demands = np.vstack([demands, padding])
                logger.info(f"fill zero demand for {actual_total_users - demands.shape[0]} missing users")
                
            # if time step number mismatch, adjust
            if demands.shape[1] > self.total_steps:
                demands = demands[:, :self.total_steps]
            elif demands.shape[1] < self.total_steps:
                padding = np.zeros((demands.shape[0], self.total_steps - demands.shape[1]))
                demands = np.hstack([demands, padding])
        
        self.user_demands = demands
        
        # update user demands for each scheduler based on actual user distribution
        current_user_index = 0
        for i, manager in enumerate(self.managers):
            manager_users = len(manager.users)
            
            if current_user_index < demands.shape[0]:
                end_user_index = min(current_user_index + manager_users, demands.shape[0])
                actual_assigned_users = end_user_index - current_user_index
                
                # get user demands for this manager
                manager_demands = demands[current_user_index:end_user_index]
                
                # if demand data is insufficient, fill with zero
                if manager_demands.shape[0] < manager_users:
                    padding_users = manager_users - manager_demands.shape[0]
                    padding = np.zeros((padding_users, self.total_steps))
                    manager_demands = np.vstack([manager_demands, padding])
                    logger.info(f"Manager {manager.manager_id}: actual {manager_users} users, allocated {actual_assigned_users} user demands, filled {padding_users} zero demands")
                
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    scheduler.set_user_demands(manager_demands)
                    avg_demand = np.mean(manager_demands)
                    total_demand = np.sum(manager_demands)
                    logger.info(f"set {manager_users} user demands for Manager {manager.manager_id} (user index {current_user_index}-{end_user_index-1}), average demand: {avg_demand:.2f} kWh, total demand: {total_demand:.2f} kWh")
                
                current_user_index = end_user_index
            else:
                # if no more demand data, set zero demand for remaining managers
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    zero_demands = np.zeros((manager_users, self.total_steps))
                    scheduler.set_user_demands(zero_demands)
                    logger.warning(f"set zero demand for {manager_users} users of Manager {manager.manager_id} (data insufficient)")
        
        logger.info(f"set user demands completed, total demand: {np.sum(demands):.2f} kWh")
    
    def process_trades(self, step: int) -> Dict:

        # validate step range
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"time step {step} out of range [0, {self.total_steps-1}]")
        
        # get current weather data
        current_weather = self.trading_pool.weather_model.get_current_weather()
        
        # get current trade history
        trade_history = self.trading_pool.trade_history
        
        # group by buyer
        trades_by_buyer = {}
        
        # get new trades
        new_trades = []
        for trade in trade_history:
            # skip processed trades
            if trade.trade_id in self.processed_trades:
                continue
            
            # only process completed trades
            if trade.status != "completed":
                continue
            
            # add to new trades list
            new_trades.append(trade)
            self.processed_trades.add(trade.trade_id)
            
            # group by buyer ID
            buyer_id = trade.buyer_id
            if buyer_id not in trades_by_buyer:
                trades_by_buyer[buyer_id] = []
            
            trades_by_buyer[buyer_id].append(trade)
        
        logger.info(f"time step {step}: processed {len(new_trades)} new trades")
        
        # process trades for each buyer
        all_disaggregated = {}
        
        for buyer_id, trades in trades_by_buyer.items():
            # find buyer's corresponding manager
            buyer_manager = None
            for manager in self.managers:
                if manager.manager_id == buyer_id:
                    buyer_manager = manager
                    break
            
            if not buyer_manager:
                logger.warning(f"buyer manager {buyer_id} not found, skip trade processing")
                continue
            
            # process each trade
            buyer_resources = []
            
            for trade in trades:
                # get trade resources
                energy_type = trade.energy_type
                quantity = trade.quantity
                price = trade.price
                
                # create resource object
                resource = {
                    'resource_id': trade.trade_id,
                    'energy_type': energy_type,
                    'allocated_energy': quantity,
                    'price': price,
                    'trade_time': trade.trade_time,
                    'seller_id': trade.seller_id
                }
                
                buyer_resources.append(resource)
            
            # store buyer resources
            all_disaggregated[buyer_id] = buyer_resources
        
        # schedule users for each buyer
        allocations = {}
        
        for buyer_id, resources in all_disaggregated.items():
            scheduler = self.user_schedulers.get(buyer_id)
            if scheduler and resources:
                # schedule users
                allocations[buyer_id] = scheduler.schedule(
                    energy_resources=resources,
                    step=step,
                    method='priority'  # can choose different scheduling methods
                )
        
        # calculate satisfaction
        satisfaction = {}
        overall_satisfaction = 0.0
        
        for buyer_id, scheduler in self.user_schedulers.items():
            user_satisfaction = scheduler.get_user_satisfaction(step)
            satisfaction[buyer_id] = user_satisfaction
            overall_satisfaction += np.mean(user_satisfaction)
        
        if self.user_schedulers:
            overall_satisfaction /= len(self.user_schedulers)
        
        self.satisfaction_history.append(overall_satisfaction)
        
        # return result
        return {
            'disaggregated_resources': all_disaggregated,
            'allocations': allocations,
            'satisfaction': satisfaction,
            'overall_satisfaction': overall_satisfaction
        }
    
    def get_satisfaction_history(self) -> List[float]:
        """
        get satisfaction history
        
        Returns:
            List[float]: satisfaction history
        """
        return self.satisfaction_history
    
    def get_overall_satisfaction(self) -> float:
        """
        get overall satisfaction
        
        Returns:
            float: overall satisfaction
        """
        all_satisfaction = 0.0
        for scheduler in self.user_schedulers.values():
            all_satisfaction += scheduler.get_overall_satisfaction()
        
        if self.user_schedulers:
            return all_satisfaction / len(self.user_schedulers)
        else:
            return 0.0
    
    def visualize_satisfaction(self, save_path: Optional[str] = None):
        """
        visualize satisfaction history
        
        Args:
            save_path: save path, if None, show chart
        """
        if not self.satisfaction_history:
            logger.warning("no satisfaction history data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.satisfaction_history, 'b-')
        plt.xlabel('time step')
        plt.ylabel('overall satisfaction')
        plt.title('user satisfaction history')
        plt.grid(True)
        
        # add overall average satisfaction line
        avg_satisfaction = float(np.mean(self.satisfaction_history))
        plt.axhline(y=avg_satisfaction, color='r', linestyle='--', 
                   label=f'average satisfaction: {avg_satisfaction:.2f}')
        
        plt.legend()
        
        # save or show chart
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_report(self, output_directory: Optional[str] = None):
        """
        generate schedule result report
        
        Args:
            output_directory: output directory, if None, use current directory
        """
        # if output directory is not specified, use current directory
        if output_directory is None:
            output_directory = '.'
        
        # ensure directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # calculate various indicators
        overall_satisfaction = self.get_overall_satisfaction()
        satisfaction_trend = self.satisfaction_history
        
        # user satisfaction of each manager
        manager_satisfaction = {}
        for manager_id, scheduler in self.user_schedulers.items():
            manager_satisfaction[manager_id] = scheduler.get_overall_satisfaction()
        
        # create report data
        report_data = {
            'timestamp': timestamp,
            'overall_satisfaction': overall_satisfaction,
            'manager_satisfaction': manager_satisfaction,
            'satisfaction_history': self.satisfaction_history
        }
        
        # save report data
        report_path = os.path.join(output_directory, f'schedule_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2)
        
        # generate satisfaction chart
        satisfaction_path = os.path.join(output_directory, f'satisfaction_{timestamp}.png')
        self.visualize_satisfaction(satisfaction_path)
        
        # generate user allocation chart for each manager
        for manager_id, scheduler in self.user_schedulers.items():
            allocation_path = os.path.join(output_directory, f'allocation_{manager_id}_{timestamp}.png')
            scheduler.visualize_allocation(save_path=allocation_path)
        
        logger.info(f"report generated to directory: {output_directory}")
        return report_path

    def update_user_demands_for_timestep(self, cumulative_demands, timestep):
        """update user demands for specified time step"""
        try:
            logger.info(f"update user demands for time step {timestep}")
            
            # ensure demand data dimension is correct
            total_users = sum(len(manager.users) for manager in self.managers)
            if cumulative_demands.shape[0] != total_users:
                logger.warning(f"user number mismatch: demand data {cumulative_demands.shape[0]}, actual users {total_users}")
                return
            
            # update current time step's demand state
            if hasattr(self, 'current_timestep_demands'):
                self.current_timestep_demands = cumulative_demands
            else:
                self.current_timestep_demands = cumulative_demands
            
            # update user scheduler state for each manager
            current_user_index = 0
            for manager in self.managers:
                manager_users = len(manager.users)
                
                if current_user_index < cumulative_demands.shape[0]:
                    # get cumulative demands up to current time step
                    end_user_index = min(current_user_index + manager_users, cumulative_demands.shape[0])
                    manager_cumulative_demands = cumulative_demands[current_user_index:end_user_index, :timestep+1]
                    
                    scheduler = self.user_schedulers.get(manager.manager_id)
                    if scheduler and hasattr(scheduler, 'update_cumulative_demands'):
                        scheduler.update_cumulative_demands(manager_cumulative_demands, timestep)
                    
                    current_user_index = end_user_index
            
            logger.info(f"time step {timestep} user demands updated")
            
        except Exception as e:
            logger.error(f"error updating user demands: {e}")
            import traceback
            logger.error(traceback.format_exc())


class AggregatedResultDisaggregator:

    
    def __init__(self, time_horizon: int = 24, default_algorithm: str = 'proportional'):

        self.time_horizon = time_horizon
        self.default_algorithm = default_algorithm
        self.algorithm_cache = {}  # cache algorithm instances
        self.performance_history = []  # performance history
        
        # validate default algorithm exists
        available_algorithms = DisaggregationAlgorithmFactory.get_available_algorithms()
        if default_algorithm not in available_algorithms:
            logger.warning(f"default algorithm '{default_algorithm}' does not exist, use 'proportional' algorithm")
            self.default_algorithm = 'proportional'
        
        logger.info(f"initialize aggregated result disaggregator, time horizon: {time_horizon} hours, default algorithm: {self.default_algorithm}")
        logger.info(f"available algorithms: {available_algorithms}")
    
    def disaggregate(self, 
                     aggregated_result: Union[AggregatedFlexOffer, Any], 
                     original_data: List[Dict], 
                     weighting_method: Optional[str] = None,
                     time_step: int = 0) -> List[Dict]:

        if not original_data:
            logger.warning("没有原始数据进行分解")
            return []
        
        # determine the algorithm to use
        algorithm_name = weighting_method or self.default_algorithm
        
        # process old version algorithm name mapping
        algorithm_mapping = {
            'equal': 'average',
            'proportional': 'proportional',
            'priority': 'priority'
        }
        algorithm_name = algorithm_mapping.get(algorithm_name, algorithm_name)
        
        # get total energy
        total_energy = 0.0
        
        if hasattr(aggregated_result, 'total_energy'):
            # direct total_energy attribute
            total_energy = getattr(aggregated_result, 'total_energy', 0.0)
        elif hasattr(aggregated_result, 'total_energy_max'):
            # AggregatedFlexOffer's total_energy_max attribute
            total_energy = getattr(aggregated_result, 'total_energy_max', 0.0)
            logger.debug(f"get total energy from AggregatedFlexOffer: {total_energy}")
        elif hasattr(aggregated_result, 'aggregated_fo'):
            # try to get total energy from aggregated_fo
            agg_fo = getattr(aggregated_result, 'aggregated_fo', None)
            if agg_fo and hasattr(agg_fo, 'total_energy_max'):
                total_energy = getattr(agg_fo, 'total_energy_max', 0.0)
                logger.debug(f"get total energy from aggregated_fo: {total_energy}")
            elif agg_fo and hasattr(agg_fo, 'quantity'):
                total_energy = getattr(agg_fo, 'quantity', 0.0)
                logger.debug(f"get total energy from aggregated_fo.quantity: {total_energy}")
        
        # if still cannot get total energy, calculate from original data
        if total_energy <= 0:
            total_energy = sum(item.get('energy', 0) for item in original_data)
            logger.debug(f"calculate total energy from original data: {total_energy}")
        
        # check if total energy is 0, if so, special handling may be needed
        if total_energy <= 0:
            logger.warning(f"total energy is 0 or negative({total_energy}), cannot perform effective decomposition")
            # if average algorithm, return full zero allocation
            if algorithm_name == 'average':
                logger.info("use average algorithm, return full zero allocation")
                return [dict(item, allocated_energy=0.0, allocation_method='average', allocation_ratio=0.0) 
                        for item in original_data]
            # for other algorithms, return empty list
            return []
        
        # create decomposition request
        request = DisaggregationRequest(
            aggregated_result=aggregated_result,
            original_data=original_data,
            total_energy=total_energy,
            time_step=time_step,
            metadata={
                'time_horizon': self.time_horizon,
                'original_count': len(original_data)
            }
        )
        
        # get or create algorithm instance
        try:
            algorithm = self._get_algorithm(algorithm_name)
        except Exception as e:
            logger.error(f"get algorithm instance failed: {e}")
            # fallback to average algorithm
            if algorithm_name != 'average':
                logger.info("fallback to average algorithm")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # if average also fails, return empty list
                return []
        
        # perform decomposition
        try:
            result = algorithm.disaggregate(request)
            
            # record performance
            self._record_performance(algorithm_name, request, result)
            
            logger.info(f"decomposition completed, using algorithm: {algorithm_name},"
                       f"original data: {len(original_data)},"
                       f"decomposition result: {len(result.disaggregated_data)},"
                       f"total energy: {total_energy:.2f} → {result.total_allocated_energy:.2f}")
            
            return result.disaggregated_data
            
        except Exception as e:
            logger.error(f"decomposition failed, algorithm: {algorithm_name}, error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # fallback to average algorithm
            if algorithm_name != 'average':
                logger.info("fallback to average algorithm")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # if average algorithm also fails, return result with zero energy allocation
                logger.info("average algorithm also fails, return result with zero energy allocation")
                return [dict(item, allocated_energy=0.0, allocation_method='fallback', allocation_ratio=0.0) 
                        for item in original_data]
    
    def _get_algorithm(self, algorithm_name: str) -> DisaggregationAlgorithm:

        if algorithm_name not in self.algorithm_cache:
            try:
                self.algorithm_cache[algorithm_name] = DisaggregationAlgorithmFactory.create_algorithm(algorithm_name)
            except ValueError as e:
                logger.error(f"create algorithm failed: {e}")
                # fallback to default algorithm
                if algorithm_name != self.default_algorithm:
                    logger.info(f"fallback to default algorithm: {self.default_algorithm}")
                    return self._get_algorithm(self.default_algorithm)
                else:
                    raise
        
        return self.algorithm_cache[algorithm_name]
    
    def _record_performance(self, algorithm_name: str, request: DisaggregationRequest, result: DisaggregationResult):
        """record performance metrics"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm_name,
            'time_step': request.time_step,
            'original_count': len(request.original_data),
            'total_energy': request.total_energy,
            'allocated_energy': result.total_allocated_energy,
            'allocation_efficiency': result.total_allocated_energy / request.total_energy if request.total_energy > 0 else 0
        }
        self.performance_history.append(performance_record)
        
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """get performance summary"""
        if not self.performance_history:
            return {"message": "no performance record"}
        
        # group by algorithm
        algorithm_stats = {}
        for record in self.performance_history:
            alg = record['algorithm']
            if alg not in algorithm_stats:
                algorithm_stats[alg] = {
                    'count': 0,
                    'total_energy': 0,
                    'total_allocated': 0,
                    'efficiency_sum': 0
                }
            
            stats = algorithm_stats[alg]
            stats['count'] += 1
            stats['total_energy'] += record['total_energy']
            stats['total_allocated'] += record['allocated_energy']
            stats['efficiency_sum'] += record['allocation_efficiency']
        
        # calculate average
        summary = {}
        for alg, stats in algorithm_stats.items():
            summary[alg] = {
                'usage_count': stats['count'],
                'average_efficiency': stats['efficiency_sum'] / stats['count'],
                'total_energy_processed': stats['total_energy'],
                'total_energy_allocated': stats['total_allocated']
            }
        
        return {
            'total_operations': len(self.performance_history),
            'algorithm_performance': summary,
            'default_algorithm': self.default_algorithm,
            'cached_algorithms': list(self.algorithm_cache.keys())
        }
    
    def get_available_algorithms(self) -> List[str]:
        """get available algorithms"""
        return DisaggregationAlgorithmFactory.get_available_algorithms()
    
    def set_default_algorithm(self, algorithm_name: str):
        """set default algorithm"""
        available = self.get_available_algorithms()
        if algorithm_name not in available:
            raise ValueError(f"algorithm '{algorithm_name}' does not exist. available algorithms: {available}")
        
        self.default_algorithm = algorithm_name
        logger.info(f"default disaggregation algorithm set to: {algorithm_name}")
    
    def clear_cache(self):
        """clear algorithm cache"""
        self.algorithm_cache.clear()
        logger.info("algorithm cache cleared") 
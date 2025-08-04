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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FlexScheduler")

# Add project root directory to system path for importing original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import related modules
from fo_aggregate.manager import Manager, User, Device
from fo_aggregate.aggregator import AggregatedFlexOffer
from fo_trading.pool import TradingPool, WeatherModel, DemandModel, Trade

# ========== New Addition: FO Disaggregation Algorithm Data Structure ==========

@dataclass
class DisaggregationRequest:
    """FO disaggregation request data structure"""
    aggregated_result: Any  # Aggregated result (AggregatedResult or AggregatedFlexOffer)
    original_data: List[Dict]  # Original data list
    total_energy: float  # Total energy
    time_step: int  # Time step
    metadata: Dict[str, Any]  # Additional metadata
    
    def __post_init__(self):
        """Validate input data"""
        # Check original data list
        if self.original_data is None:
            logger.warning("Original data list is None, initializing as empty list")
            self.original_data = []
        
        # Check total energy
        if self.total_energy is None:
            logger.warning("Total energy is None, setting to 0")
            self.total_energy = 0.0
        elif self.total_energy < 0:
            logger.warning(f"Total energy is negative ({self.total_energy}), setting to 0")
            self.total_energy = 0.0
        
        # Check time step
        if self.time_step is None:
            logger.warning("Time step is None, setting to 0")
            self.time_step = 0
        elif self.time_step < 0:
            logger.warning(f"Time step is negative ({self.time_step}), setting to 0")
            self.time_step = 0
        
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class DisaggregationResult:
    """FO disaggregation result data structure"""
    disaggregated_data: List[Dict]  # Disaggregated data list
    algorithm_used: str  # Algorithm name used
    allocation_ratios: List[float]  # Allocation ratio list
    total_allocated_energy: float  # Total allocated energy
    metadata: Dict[str, Any]  # Metadata
    
    def __post_init__(self):
        """Validate result data"""
        if len(self.disaggregated_data) != len(self.allocation_ratios):
            raise ValueError("Disaggregated data and allocation ratios count mismatch")
        if self.total_allocated_energy < 0:
            raise ValueError("Total allocated energy cannot be negative")

# ========== New Addition: Disaggregation Algorithm Abstract Base Class ==========

class DisaggregationAlgorithm(ABC):
    """FO disaggregation algorithm abstract base class"""
    
    def __init__(self, algorithm_name: str):
        """
        Initialize disaggregation algorithm
        
        Args:
            algorithm_name: Algorithm name
        """
        self.algorithm_name = algorithm_name
        self.total_requests = 0
        self.total_energy_processed = 0.0
        self.performance_metrics = {}
    
    @abstractmethod
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        Abstract method to perform disaggregation operation
        
        Args:
            request: Disaggregation request
            
        Returns:
            DisaggregationResult: Disaggregation result
        """
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            "name": self.algorithm_name,
            "total_requests": self.total_requests,
            "total_energy_processed": self.total_energy_processed,
            "performance_metrics": self.performance_metrics
        }
    
    def _validate_request(self, request: DisaggregationRequest) -> bool:
        """Validate disaggregation request"""
        if not isinstance(request, DisaggregationRequest):
            raise ValueError("Invalid disaggregation request type")
        return True
    
    def _update_metrics(self, request: DisaggregationRequest, result: DisaggregationResult):
        """Update performance metrics"""
        self.total_requests += 1
        self.total_energy_processed += request.total_energy
        
        # Calculate allocation efficiency
        efficiency = result.total_allocated_energy / request.total_energy if request.total_energy > 0 else 0
        if 'allocation_efficiency' not in self.performance_metrics:
            self.performance_metrics['allocation_efficiency'] = []
        self.performance_metrics['allocation_efficiency'].append(efficiency)

# ========== New Addition: Average Disaggregation Algorithm Implementation ==========

class AverageDisaggregationAlgorithm(DisaggregationAlgorithm):
    """Average disaggregation algorithm: E_i = E/N"""
    
    def __init__(self):
        super().__init__("average")
        logger.info("Initializing average disaggregation algorithm")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        Perform average disaggregation: distribute total energy equally among all participants
        
        Args:
            request: Disaggregation request
            
        Returns:
            DisaggregationResult: Disaggregation result
        """
        self._validate_request(request)
        
        logger.info(f"Starting average disaggregation, original data count: {len(request.original_data)}, total energy: {request.total_energy:.2f}")
        
        # Calculate average allocation energy: E_i = E/N
        num_participants = len(request.original_data)
        
        # Check participant count to avoid division by zero
        if num_participants == 0:
            logger.error("Cannot perform average disaggregation: no participants")
            # Return empty result
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
        
        # Create disaggregation result
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # Copy original data
            new_item = copy.deepcopy(item)
            
            # Allocate average energy
            new_item['allocated_energy'] = average_energy
            new_item['allocation_method'] = 'average'
            # Avoid division by zero
            original_energy = item.get('energy', 0)
            new_item['allocation_ratio'] = average_energy / original_energy if original_energy > 0 else 1.0
            
            disaggregated_data.append(new_item)
            # Avoid division by zero
            allocation_ratio = average_energy / request.total_energy if request.total_energy > 0 else 1.0 / num_participants
            allocation_ratios.append(allocation_ratio)
        
        # Create result object
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
        
        # Update performance metrics
        self._update_metrics(request, result)
        
        logger.info(f"Average disaggregation completed, allocation per participant: {average_energy:.2f} kWh")
        return result

# ========== New Addition: Proportional Disaggregation Algorithm Implementation ==========

class ProportionalDisaggregationAlgorithm(DisaggregationAlgorithm):
    """Proportional disaggregation algorithm: E_i = (w_i/W) * E"""
    
    def __init__(self, weight_key: str = 'energy'):
        """
        Initialize proportional disaggregation algorithm
        
        Args:
            weight_key: Key name used for weight calculation, defaults to 'energy'
        """
        super().__init__("proportional")
        self.weight_key = weight_key
        logger.info(f"Initializing proportional disaggregation algorithm, weight key: {weight_key}")
    
    def disaggregate(self, request: DisaggregationRequest) -> DisaggregationResult:
        """
        Perform proportional disaggregation: allocate energy based on weights
        
        Args:
            request: Disaggregation request
            
        Returns:
            DisaggregationResult: Disaggregation result
        """
        self._validate_request(request)
        
        logger.info(f"Starting proportional disaggregation, original data count: {len(request.original_data)}, total energy: {request.total_energy:.2f}")
        
        # Calculate total weight: W = Σw_i
        total_weight = sum(item.get(self.weight_key, 1.0) for item in request.original_data)
        
        if total_weight <= 0:
            logger.warning("Total weight is zero, falling back to average allocation")
            # If total weight is zero, fall back to average allocation
            average_algo = AverageDisaggregationAlgorithm()
            return average_algo.disaggregate(request)
        
        # Create disaggregation result
        disaggregated_data = []
        allocation_ratios = []
        
        for i, item in enumerate(request.original_data):
            # Copy original data
            new_item = copy.deepcopy(item)
            
            # Calculate weight ratio: w_i/W
            weight = item.get(self.weight_key, 1.0)
            weight_ratio = weight / total_weight
            
            # Allocate energy proportionally: E_i = (w_i/W) * E
            allocated_energy = weight_ratio * request.total_energy
            
            new_item['allocated_energy'] = allocated_energy
            new_item['allocation_method'] = 'proportional'
            new_item['weight_used'] = weight
            new_item['weight_ratio'] = weight_ratio
            new_item['allocation_ratio'] = allocated_energy / item.get('energy', 1.0) if item.get('energy', 0) > 0 else 1.0
            
            disaggregated_data.append(new_item)
            allocation_ratios.append(weight_ratio)
        
        # Create result object
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
        
        # Update performance metrics
        self._update_metrics(request, result)
        
        logger.info(f"Proportional disaggregation completed, total weight: {total_weight:.2f}")
        return result

# ========== New Addition: Disaggregation Algorithm Factory ==========

class DisaggregationAlgorithmFactory:
    """FO disaggregation algorithm factory"""
    
    _algorithms = {}
    _initialized = False
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type, **kwargs):
        """
        Register disaggregation algorithm
        
        Args:
            name: Algorithm name
            algorithm_class: Algorithm class
            **kwargs: Algorithm initialization parameters
        """
        cls._algorithms[name] = {
            'class': algorithm_class,
            'kwargs': kwargs
        }
        logger.info(f"Registered FO disaggregation algorithm: {name}")
    
    @classmethod
    def create_algorithm(cls, name: str, **override_kwargs) -> DisaggregationAlgorithm:
        """
        Create disaggregation algorithm instance
        
        Args:
            name: Algorithm name
            **override_kwargs: Override default parameters
            
        Returns:
            DisaggregationAlgorithm: Algorithm instance
        """
        if name not in cls._algorithms:
            raise ValueError(f"Unknown disaggregation algorithm: {name}")
        
        algo_info = cls._algorithms[name]
        algo_class = algo_info['class']
        
        # Merge parameters
        kwargs = algo_info['kwargs'].copy()
        kwargs.update(override_kwargs)
        
        return algo_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms"""
        return list(cls._algorithms.keys())
    
    @classmethod
    def initialize_default_algorithms(cls):
        """Initialize default algorithms"""
        if cls._initialized:
            return
        
        # Register average disaggregation algorithm
        cls.register_algorithm("average", AverageDisaggregationAlgorithm)
        
        # Register proportional disaggregation algorithm
        cls.register_algorithm("proportional", ProportionalDisaggregationAlgorithm, weight_key='energy')
        cls.register_algorithm("equal_proportion", ProportionalDisaggregationAlgorithm, weight_key='energy')
        
        # For compatibility, register original method names
        cls.register_algorithm("equal", AverageDisaggregationAlgorithm)
        cls.register_algorithm("priority", ProportionalDisaggregationAlgorithm, weight_key='priority')
        
        cls._initialized = True
        logger.info("Default FO disaggregation algorithms initialized")

# Initialize default algorithms
DisaggregationAlgorithmFactory.initialize_default_algorithms()

# Define a simple FlexOffer class as a transition
class FlexOffer:
    """Simplified FlexOffer class for compatibility with existing code"""
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

# Define a simple FlexOfferManager class as a transition
class FlexOfferManager:
    """Simplified FlexOfferManager class for compatibility with existing code"""
    def __init__(self, manager_id=None, location=None):
        self.manager_id = manager_id
        self.location = location
        self.offers = []
    
    def add_offer(self, offer):
        self.offers.append(offer)

class FlexOfferDisaggregator:
    """Aggregated FlexOffer disaggregator, breaks down aggregated FlexOffer back to original FlexOffers"""
    
    def __init__(self, time_horizon: int = 24):
        """
        Initialize disaggregator
        
        Args:
            time_horizon: Time horizon
        """
        self.time_horizon = time_horizon
    
    def disaggregate(self, 
                     aggregated_offer: FlexOffer, 
                     original_offers: List[FlexOffer]) -> List[FlexOffer]:
        """
        Disaggregate an aggregated FlexOffer
        
        Args:
            aggregated_offer: Aggregated FlexOffer
            original_offers: Original FlexOffer list
            
        Returns:
            List[FlexOffer]: Disaggregated FlexOffer list
        """
        if not original_offers:
            raise ValueError("No original offers to disaggregate to")
        
        # Create disaggregated FlexOffer list
        disaggregated_offers = []
        
        # Total power and baseline
        total_min_power = np.zeros(self.time_horizon)
        total_max_power = np.zeros(self.time_horizon)
        total_baseline = np.zeros(self.time_horizon)
        
        for fo in original_offers:
            total_min_power += fo.power_profile[:, 0]
            total_max_power += fo.power_profile[:, 1]
            total_baseline += fo.baseline_profile
        
        # Calculate actual power of aggregated FlexOffer
        aggregated_power = aggregated_offer.baseline_profile
        
        # Allocate power
        for fo in original_offers:
            # Create new FlexOffer
            new_fo = FlexOffer(
                resource_id=fo.resource_id,
                resource_type=fo.resource_type,
                location=fo.location,
                time_horizon=fo.time_horizon,
                time_interval=fo.time_interval
            )
            
            # Copy power range
            new_fo.set_power_profile(fo.power_profile.copy())
            
            # Allocate power based on original contribution ratio
            baseline_ratio = np.zeros(self.time_horizon)
            for t in range(self.time_horizon):
                if total_baseline[t] > 0:
                    baseline_ratio[t] = fo.baseline_profile[t] / total_baseline[t]
                else:
                    baseline_ratio[t] = 1.0 / len(original_offers)
            
            # Calculate new baseline
            new_baseline = aggregated_power * baseline_ratio
            new_fo.set_baseline_profile(new_baseline)
            
            # Set reliability
            new_fo.set_reliability(fo.reliability)
            
            disaggregated_offers.append(new_fo)
        
        return disaggregated_offers


class UserScheduler:
    """User scheduler, allocates energy resources based on user demands"""
    
    def __init__(self, 
                 num_users: int = 20,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1):
        """
        Initialize scheduler
        
        Args:
            num_users: Number of users
            time_horizon: Time horizon (hours)
            time_steps_per_hour: Number of time steps per hour
        """
        self.num_users = num_users
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        
        # User demands
        self.user_demands = np.zeros((num_users, self.total_steps))
        
        # User allocations
        self.user_allocations = np.zeros((num_users, self.total_steps))
        
        # User energy sources
        self.user_sources = {}
        
        # User configurations (can specify user priorities, preferences, etc.)
        self.user_configs = [
            {'id': i, 'priority': random.uniform(0, 1), 'preferences': {}} 
            for i in range(num_users)
        ]
        
        logger.info(f"Initializing user scheduler, users: {num_users}, time horizon: {time_horizon} hours")
    
    def set_user_demands(self, demands: np.ndarray):
        """
        Set user demands
        
        Args:
            demands: User demands, dimension [num_users, total_steps]
        """
        assert demands.shape == (self.num_users, self.total_steps), f"Demand dimensions mismatch: {demands.shape} vs {(self.num_users, self.total_steps)}"
        self.user_demands = demands
        # Remove duplicate log output, unified output from upper ScheduleManager
    
    def get_user_demand(self, user_id: int, step: int) -> float:
        """
        Get demand for a specific user at a specific time
        
        Args:
            user_id: User ID
            step: Time step
            
        Returns:
            float: User demand
        """
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(f"User ID {user_id} out of range [0, {self.num_users-1}]")
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"Time step {step} out of range [0, {self.total_steps-1}]")
        
        return self.user_demands[user_id, step]
    
    def schedule(self, 
                energy_resources: List[Dict], 
                step: int,
                method: str = 'priority') -> Dict[int, List[Dict]]:
        """
        Allocate energy resources based on user demands
        
        Args:
            energy_resources: List of energy resources, each dictionary contains energy configuration
            step: Current time step
            method: Allocation method, options: 'priority', 'fairness', 'cost'
            
        Returns:
            Dict[int, List[Dict]]: User allocation results, keys are user IDs, values are allocated resources
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"Time step {step} out of range [0, {self.total_steps-1}]")
        
        # Get user demands for current time step
        current_demands = self.user_demands[:, step].copy()
        
        # Sort users according to the chosen method
        if method == 'priority':
            # Sort by priority (users with higher priority are allocated first)
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: self.user_configs[i]['priority'],
                reverse=True
            )
        elif method == 'fairness':
            # Sort by historical satisfaction rate (users with lower satisfaction are allocated first)
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
            # Sort by demand (users with larger demands are allocated first, assuming economies of scale)
            sorted_user_indices = sorted(
                range(self.num_users), 
                key=lambda i: current_demands[i],
                reverse=True
            )
        else:
            # Default sort by user ID
            sorted_user_indices = list(range(self.num_users))
        
        # Filter users with demands
        sorted_user_indices = [i for i in sorted_user_indices if current_demands[i] > 0]
        
        # Calculate total available energy
        total_available_energy = sum(item.get('allocated_energy', 0) for item in energy_resources)
        
        # Calculate total current demand
        total_demand = sum(current_demands)
        
        logger.info(f"Time step {step}: Users={len(sorted_user_indices)}, "
                   f"Total demand={total_demand:.2f} kWh, Available energy={total_available_energy:.2f} kWh")
        
        # Allocation results
        allocations = {user_id: [] for user_id in range(self.num_users)}
        
        # Prioritize energy resource allocation
        for user_id in sorted_user_indices:
            user_demand = current_demands[user_id]
            
            # Skip if user has no demand
            if user_demand <= 0:
                continue
            
            # Allocate resources for user
            remaining_demand = user_demand
            
            for resource in energy_resources:
                # Check if resource has energy left
                available_energy = resource.get('allocated_energy', 0)
                if available_energy <= 0:
                    continue
                
                # Allocation amount = min(user's remaining demand, available energy)
                allocation_amount = min(remaining_demand, available_energy)
                
                if allocation_amount > 0:
                    # Update resource's available energy
                    resource['allocated_energy'] -= allocation_amount
                    
                    # Update user's remaining demand
                    remaining_demand -= allocation_amount
                    
                    # Record allocation result
                    allocation = {
                        'resource_id': resource.get('resource_id', ''),
                        'energy_type': resource.get('energy_type', ''),
                        'amount': allocation_amount,
                        'price': resource.get('price', 0.0)
                    }
                    
                    allocations[user_id].append(allocation)
                    
                    # Update user allocation record
                    self.user_allocations[user_id, step] += allocation_amount
                
                # If user demand is satisfied, end loop
                if remaining_demand <= 0:
                    break
        
        # Update user energy source records
        self.user_sources[step] = allocations
        
        # Calculate energy resource utilization rate
        total_allocated = sum(self.user_allocations[:, step])
        utilization_rate = total_allocated / total_available_energy if total_available_energy > 0 else 0.0
        
        logger.info(f"Time step {step}: Allocation complete, Total allocated={total_allocated:.2f} kWh, "
                   f"Resource utilization={utilization_rate*100:.2f}%")
        
        return allocations
    
    def get_user_satisfaction(self, step: int) -> np.ndarray:
        """
        Get user satisfaction (demand fulfillment rate)
        
        Args:
            step: Time step
            
        Returns:
            np.ndarray: User satisfaction, range [0,1]
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"Time step {step} out of range [0, {self.total_steps-1}]")
        
        # Calculate user satisfaction
        satisfaction = np.zeros(self.num_users)
        for user_id in range(self.num_users):
            user_demand = self.user_demands[user_id, step]
            if user_demand > 0:
                satisfaction[user_id] = min(1.0, self.user_allocations[user_id, step] / user_demand)
            else:
                satisfaction[user_id] = 1.0  # No demand, default to satisfied
        
        return satisfaction
    
    def get_overall_satisfaction(self) -> float:
        """
        Get overall satisfaction (total demand fulfillment rate)
        
        Returns:
            float: Overall satisfaction, range [0,1]
        """
        total_demand = np.sum(self.user_demands)
        total_allocation = np.sum(self.user_allocations)
        
        if total_demand > 0:
            return float(min(1.0, total_allocation / total_demand))
        else:
            return 1.0
    
    def visualize_allocation(self, step: Optional[int] = None, save_path: Optional[str] = None):
        """
        Visualize user energy allocation
        
        Args:
            step: Time step, if None shows total allocation for all time steps
            save_path: Save path, if None displays the chart
        """
        plt.figure(figsize=(12, 6))
        
        if step is not None:
            if step < 0 or step >= self.total_steps:
                raise ValueError(f"Time step {step} out of range [0, {self.total_steps-1}]")
            
            # Show allocation for specific time step
            demands = self.user_demands[:, step]
            allocations = self.user_allocations[:, step]
            
            # Calculate satisfaction
            satisfaction = self.get_user_satisfaction(step)
            
            # Set x-axis and chart
            x = np.arange(self.num_users)
            width = 0.4
            
            # Plot demands and allocations
            plt.bar(x - width/2, demands, width, label='Demand')
            plt.bar(x + width/2, allocations, width, label='Allocation')
            
            # Plot satisfaction line
            plt.plot(x, satisfaction, 'r-', label='Satisfaction')
            
            plt.xlabel('User ID')
            plt.ylabel('Energy (kWh)')
            plt.title(f'User Energy Allocation for Time Step {step}')
            plt.xticks(x)
            plt.legend()
            
        else:
            # Show total allocation for all time steps
            total_demands = np.sum(self.user_demands, axis=1)
            total_allocations = np.sum(self.user_allocations, axis=1)
            
            # Calculate overall satisfaction
            satisfaction = []
            for user_id in range(self.num_users):
                if total_demands[user_id] > 0:
                    satisfaction.append(min(1.0, total_allocations[user_id] / total_demands[user_id]))
                else:
                    satisfaction.append(1.0)
            
            # Set x-axis and chart
            x = np.arange(self.num_users)
            width = 0.4
            
            # Plot total demands and allocations
            plt.bar(x - width/2, total_demands, width, label='Total Demand')
            plt.bar(x + width/2, total_allocations, width, label='Total Allocation')
            
            # Plot satisfaction line
            plt.plot(x, satisfaction, 'r-', label='Overall Satisfaction')
            
            plt.xlabel('User ID')
            plt.ylabel('Energy (kWh)')
            plt.title('Total User Energy Allocation for All Time')
            plt.xticks(x)
            plt.legend()
        
        # Save or display chart
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def update_cumulative_demands(self, cumulative_demands, timestep):
        """Update cumulative demand status"""
        try:
            if cumulative_demands.shape[0] == self.num_users:
                # Update cumulative demand to current time step
                if hasattr(self, 'cumulative_user_demands'):
                    # Update existing cumulative demand
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                else:
                    # Initialize cumulative demand
                    self.cumulative_user_demands = np.zeros((self.num_users, self.total_steps))
                    if cumulative_demands.shape[1] > timestep:
                        self.cumulative_user_demands[:, :timestep+1] = cumulative_demands
                
                logger.debug(f"UserScheduler cumulative demand updated to time step {timestep}")  # Changed to DEBUG level to avoid duplication
            else:
                logger.warning(f"Cumulative demand data dimension mismatch: expected {self.num_users} users, actual {cumulative_demands.shape[0]} users")
        except Exception as e:
            logger.error(f"Error updating cumulative demand: {e}")


class ScheduleManager:
    """Schedule manager, coordinates energy resource disaggregation and user scheduling"""
    
    def __init__(self, 
                 managers: List[Manager],
                 trading_pool: TradingPool,
                 time_horizon: int = 24,
                 time_steps_per_hour: int = 1,
                 disaggregation_algorithm: str = 'proportional'):
        """
        Initialize schedule manager
        
        Args:
            managers: List of managers
            trading_pool: Trading pool
            time_horizon: Time horizon (hours)
            time_steps_per_hour: Number of time steps per hour
            disaggregation_algorithm: Disaggregation algorithm, options: 'average', 'proportional', 'equal_proportion'
        """
        self.managers = managers
        self.trading_pool = trading_pool
        self.time_horizon = time_horizon
        self.time_steps_per_hour = time_steps_per_hour
        self.total_steps = time_horizon * time_steps_per_hour
        self.disaggregation_algorithm = disaggregation_algorithm
        
        # Create disaggregator (using new algorithm architecture)
        self.disaggregator = AggregatedResultDisaggregator(
            time_horizon=time_horizon,
            default_algorithm=disaggregation_algorithm
        )
        
        # Create user schedulers (one per manager, based on actual user count)
        self.user_schedulers = {}
        for manager in managers:
            actual_users = len(manager.users)  # Use actual user count
            scheduler = UserScheduler(
                num_users=actual_users,
                time_horizon=time_horizon,
                time_steps_per_hour=time_steps_per_hour
            )
            self.user_schedulers[manager.manager_id] = scheduler
            logger.info(f"Created user scheduler for Manager {manager.manager_id}, users: {actual_users}")
        
        # User demand data
        self.user_demands = None
        
        # Satisfaction history
        self.satisfaction_history = []
        
        # Trade history cache
        self.processed_trades = set()
        
        logger.info(f"Initializing schedule manager, managers: {len(managers)}, time horizon: {time_horizon} hours, disaggregation algorithm: {disaggregation_algorithm}")
        logger.info(f"Available disaggregation algorithms: {self.disaggregator.get_available_algorithms()}")
    
    def set_disaggregation_algorithm(self, algorithm_name: str):
        """
        Set disaggregation algorithm
        
        Args:
            algorithm_name: Algorithm name
        """
        self.disaggregation_algorithm = algorithm_name
        self.disaggregator.set_default_algorithm(algorithm_name)
        logger.info(f"Switched disaggregation algorithm to: {algorithm_name}")
    
    def get_disaggregation_performance(self) -> Dict[str, Any]:
        """Get disaggregation algorithm performance statistics"""
        return self.disaggregator.get_performance_summary()
    
    def set_user_demands(self, demands: np.ndarray):
        """
        Set user demands
        
        Args:
            demands: User demands, dimension [actual_total_users, total_steps]
        """
        # Calculate actual total users
        actual_total_users = sum(len(manager.users) for manager in self.managers)
        expected_shape = (actual_total_users, self.total_steps)
        
        logger.info(f"Expected demand dimensions: {expected_shape}, actual input dimensions: {demands.shape}")
        
        if demands.shape != expected_shape:
            logger.warning(f"Demand dimensions mismatch: {demands.shape} vs {expected_shape}, will attempt to adjust")
            
            # If user count doesn't match, adjust proportionally or truncate/pad
            if demands.shape[0] > actual_total_users:
                demands = demands[:actual_total_users, :]
                logger.info(f"Truncating demand data to first {actual_total_users} users")
            elif demands.shape[0] < actual_total_users:
                padding = np.zeros((actual_total_users - demands.shape[0], self.total_steps))
                demands = np.vstack([demands, padding])
                logger.info(f"Padding zero demand for missing {actual_total_users - demands.shape[0]} users")
                
            # If time step count doesn't match, adjust
            if demands.shape[1] > self.total_steps:
                demands = demands[:, :self.total_steps]
            elif demands.shape[1] < self.total_steps:
                padding = np.zeros((demands.shape[0], self.total_steps - demands.shape[1]))
                demands = np.hstack([demands, padding])
        
        self.user_demands = demands
        
        # Update each scheduler's user demands based on actual user distribution
        current_user_index = 0
        for i, manager in enumerate(self.managers):
            manager_users = len(manager.users)
            
            if current_user_index < demands.shape[0]:
                end_user_index = min(current_user_index + manager_users, demands.shape[0])
                actual_assigned_users = end_user_index - current_user_index
                
                # Get this Manager's user demands
                manager_demands = demands[current_user_index:end_user_index]
                
                # If demand data is insufficient, pad with zeros
                if manager_demands.shape[0] < manager_users:
                    padding_users = manager_users - manager_demands.shape[0]
                    padding = np.zeros((padding_users, self.total_steps))
                    manager_demands = np.vstack([manager_demands, padding])
                    logger.info(f"Manager {manager.manager_id}: Actually has {manager_users} users, assigned {actual_assigned_users} user demands, padded {padding_users} zero demands")
                
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    scheduler.set_user_demands(manager_demands)
                    avg_demand = np.mean(manager_demands)
                    total_demand = np.sum(manager_demands)
                    logger.info(f"Set demands for {manager_users} users of Manager {manager.manager_id} (user indices {current_user_index}-{end_user_index-1}), average demand: {avg_demand:.2f} kWh, total demand: {total_demand:.2f} kWh")
                
                current_user_index = end_user_index
            else:
                # If no more demand data, set zero demands for remaining managers
                scheduler = self.user_schedulers.get(manager.manager_id)
                if scheduler:
                    zero_demands = np.zeros((manager_users, self.total_steps))
                    scheduler.set_user_demands(zero_demands)
                    logger.warning(f"Setting zero demands for {manager_users} users of Manager {manager.manager_id} (insufficient data)")
        
        logger.info(f"User demand setting complete, total demand: {np.sum(demands):.2f} kWh")
    
    def process_trades(self, step: int) -> Dict:
        """
        Process trades and scheduling
        
        Args:
            step: Current time step
            
        Returns:
            Dict: Processing result
        """
        # Validate step range
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"Time step {step} out of range [0, {self.total_steps-1}]")
        
        # Get current weather data
        current_weather = self.trading_pool.weather_model.get_current_weather()
        
        # Get current trade history
        trade_history = self.trading_pool.trade_history
        
        # Group by buyer
        trades_by_buyer = {}
        
        # Get new trades
        new_trades = []
        for trade in trade_history:
            # Skip already processed trades
            if trade.trade_id in self.processed_trades:
                continue
            
            # Only process completed trades
            if trade.status != "completed":
                continue
            
            # Add to new trades list
            new_trades.append(trade)
            self.processed_trades.add(trade.trade_id)
            
            # Group by buyer ID
            buyer_id = trade.buyer_id
            if buyer_id not in trades_by_buyer:
                trades_by_buyer[buyer_id] = []
            
            trades_by_buyer[buyer_id].append(trade)
        
        logger.info(f"Time step {step}: Processing {len(new_trades)} new trades")
        
        # Process trades for each buyer
        all_disaggregated = {}
        
        for buyer_id, trades in trades_by_buyer.items():
            # Find manager corresponding to buyer
            buyer_manager = None
            for manager in self.managers:
                if manager.manager_id == buyer_id:
                    buyer_manager = manager
                    break
            
            if not buyer_manager:
                logger.warning(f"Buyer manager {buyer_id} not found, skipping trade processing")
                continue
            
            # Process each trade
            buyer_resources = []
            
            for trade in trades:
                # Get trade resources
                energy_type = trade.energy_type
                quantity = trade.quantity
                price = trade.price
                
                # Create resource object
                resource = {
                    'resource_id': trade.trade_id,
                    'energy_type': energy_type,
                    'allocated_energy': quantity,
                    'price': price,
                    'trade_time': trade.trade_time,
                    'seller_id': trade.seller_id
                }
                
                buyer_resources.append(resource)
            
            # Store buyer resources
            all_disaggregated[buyer_id] = buyer_resources
        
        # Schedule users for each buyer
        allocations = {}
        
        for buyer_id, resources in all_disaggregated.items():
            scheduler = self.user_schedulers.get(buyer_id)
            if scheduler and resources:
                # Perform user scheduling
                allocations[buyer_id] = scheduler.schedule(
                    energy_resources=resources,
                    step=step,
                    method='priority'  # Can choose different scheduling methods as needed
                )
        
        # Calculate satisfaction
        satisfaction = {}
        overall_satisfaction = 0.0
        
        for buyer_id, scheduler in self.user_schedulers.items():
            user_satisfaction = scheduler.get_user_satisfaction(step)
            satisfaction[buyer_id] = user_satisfaction
            overall_satisfaction += np.mean(user_satisfaction)
        
        if self.user_schedulers:
            overall_satisfaction /= len(self.user_schedulers)
        
        self.satisfaction_history.append(overall_satisfaction)
        
        # Return results
        return {
            'disaggregated_resources': all_disaggregated,
            'allocations': allocations,
            'satisfaction': satisfaction,
            'overall_satisfaction': overall_satisfaction
        }
    
    def get_satisfaction_history(self) -> List[float]:
        """
        Get satisfaction history
        
        Returns:
            List[float]: Satisfaction history
        """
        return self.satisfaction_history
    
    def get_overall_satisfaction(self) -> float:
        """
        Get overall satisfaction
        
        Returns:
            float: Overall satisfaction
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
        Visualize satisfaction history
        
        Args:
            save_path: Save path, if None displays the chart
        """
        if not self.satisfaction_history:
            logger.warning("No satisfaction history data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.satisfaction_history, 'b-')
        plt.xlabel('Time Step')
        plt.ylabel('Overall Satisfaction')
        plt.title('User Satisfaction History')
        plt.grid(True)
        
        # Add overall average satisfaction line
        avg_satisfaction = float(np.mean(self.satisfaction_history))
        plt.axhline(y=avg_satisfaction, color='r', linestyle='--', 
                   label=f'Average Satisfaction: {avg_satisfaction:.2f}')
        
        plt.legend()
        
        # Save or display chart
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_report(self, output_directory: Optional[str] = None):
        """
        Generate scheduling result report
        
        Args:
            output_directory: Output directory, if None uses current directory
        """
        # If output directory not specified, use current directory
        if output_directory is None:
            output_directory = '.'
        
        # Ensure directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate various metrics
        overall_satisfaction = self.get_overall_satisfaction()
        satisfaction_trend = self.satisfaction_history
        
        # User satisfaction for each manager
        manager_satisfaction = {}
        for manager_id, scheduler in self.user_schedulers.items():
            manager_satisfaction[manager_id] = scheduler.get_overall_satisfaction()
        
        # Create report data
        report_data = {
            'timestamp': timestamp,
            'overall_satisfaction': overall_satisfaction,
            'manager_satisfaction': manager_satisfaction,
            'satisfaction_history': self.satisfaction_history
        }
        
        # Save report data
        report_path = os.path.join(output_directory, f'schedule_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2)
        
        # Generate satisfaction chart
        satisfaction_path = os.path.join(output_directory, f'satisfaction_{timestamp}.png')
        self.visualize_satisfaction(satisfaction_path)
        
        # Generate user allocation chart for each manager
        for manager_id, scheduler in self.user_schedulers.items():
            allocation_path = os.path.join(output_directory, f'allocation_{manager_id}_{timestamp}.png')
            scheduler.visualize_allocation(save_path=allocation_path)
        
        logger.info(f"Report generated in directory: {output_directory}")
        return report_path

    def update_user_demands_for_timestep(self, cumulative_demands, timestep):
        """Update user demand status for specified time step"""
        try:
            logger.info(f"Updating user demand status for time step {timestep}")
            
            # Ensure demand data dimensions are correct  
            total_users = sum(len(manager.users) for manager in self.managers)
            if cumulative_demands.shape[0] != total_users:
                logger.warning(f"User count mismatch: demand data {cumulative_demands.shape[0]}, actual users {total_users}")
                return
            
            # Update demand status for current time step
            if hasattr(self, 'current_timestep_demands'):
                self.current_timestep_demands = cumulative_demands
            else:
                self.current_timestep_demands = cumulative_demands
            
            # Update user scheduler status for each Manager
            current_user_index = 0
            for manager in self.managers:
                manager_users = len(manager.users)
                
                if current_user_index < cumulative_demands.shape[0]:
                    # Get cumulative demand up to current time step
                    end_user_index = min(current_user_index + manager_users, cumulative_demands.shape[0])
                    manager_cumulative_demands = cumulative_demands[current_user_index:end_user_index, :timestep+1]
                    
                    scheduler = self.user_schedulers.get(manager.manager_id)
                    if scheduler and hasattr(scheduler, 'update_cumulative_demands'):
                        scheduler.update_cumulative_demands(manager_cumulative_demands, timestep)
                    
                    current_user_index = end_user_index
            
            logger.info(f"User demand status update completed for time step {timestep}")
            
        except Exception as e:
            logger.error(f"Error updating user demand status: {e}")
            import traceback
            logger.error(traceback.format_exc())


class AggregatedResultDisaggregator:
    """Aggregated result disaggregator, breaks down AggregatedResult into original energy configurations (refactored version)"""
    
    def __init__(self, time_horizon: int = 24, default_algorithm: str = 'proportional'):
        """
        Initialize disaggregator
        
        Args:
            time_horizon: Time horizon
            default_algorithm: Default disaggregation algorithm, options: 'average', 'proportional', 'equal_proportion'
        """
        self.time_horizon = time_horizon
        self.default_algorithm = default_algorithm
        self.algorithm_cache = {}  # Cache algorithm instances
        self.performance_history = []  # Performance history records
        
        # Validate if default algorithm exists
        available_algorithms = DisaggregationAlgorithmFactory.get_available_algorithms()
        if default_algorithm not in available_algorithms:
            logger.warning(f"Default algorithm '{default_algorithm}' does not exist, using 'proportional' algorithm")
            self.default_algorithm = 'proportional'
        
        logger.info(f"Initializing aggregated result disaggregator, time horizon: {time_horizon} hours, default algorithm: {self.default_algorithm}")
        logger.info(f"Available algorithms: {available_algorithms}")
    
    def disaggregate(self, 
                     aggregated_result: Union[AggregatedFlexOffer, Any], 
                     original_data: List[Dict], 
                     weighting_method: Optional[str] = None,
                     time_step: int = 0) -> List[Dict]:
        """
        Disaggregate aggregated result (refactored version)
        
        Args:
            aggregated_result: Aggregated result object
            original_data: Original data list, each dictionary contains energy configuration
            weighting_method: Weight allocation method, options: 'average', 'proportional', 'equal_proportion', etc.
            time_step: Current time step
            
        Returns:
            List[Dict]: Disaggregated energy configuration list
        """
        if not original_data:
            logger.warning("No original data for disaggregation")
            return []
        
        # Determine algorithm to use
        algorithm_name = weighting_method or self.default_algorithm
        
        # Handle old version algorithm name mapping
        algorithm_mapping = {
            'equal': 'average',
            'proportional': 'proportional',
            'priority': 'priority'
        }
        algorithm_name = algorithm_mapping.get(algorithm_name, algorithm_name)
        
        # Get total energy
        total_energy = 0.0
        
        if hasattr(aggregated_result, 'total_energy'):
            # Direct total_energy attribute
            total_energy = getattr(aggregated_result, 'total_energy', 0.0)
        elif hasattr(aggregated_result, 'total_energy_max'):
            # AggregatedFlexOffer's total_energy_max attribute
            total_energy = getattr(aggregated_result, 'total_energy_max', 0.0)
            logger.debug(f"Getting total energy from AggregatedFlexOffer: {total_energy}")
        elif hasattr(aggregated_result, 'aggregated_fo'):
            # Try to get from aggregated_fo
            agg_fo = getattr(aggregated_result, 'aggregated_fo', None)
            if agg_fo and hasattr(agg_fo, 'total_energy_max'):
                total_energy = getattr(agg_fo, 'total_energy_max', 0.0)
                logger.debug(f"Getting total energy from aggregated_fo: {total_energy}")
            elif agg_fo and hasattr(agg_fo, 'quantity'):
                total_energy = getattr(agg_fo, 'quantity', 0.0)
                logger.debug(f"Getting total energy from aggregated_fo.quantity: {total_energy}")
        
        # If still unable to get total energy, calculate from original data
        if total_energy <= 0:
            total_energy = sum(item.get('energy', 0) for item in original_data)
            logger.debug(f"Calculating total energy from original data: {total_energy}")
        
        # Check if total energy is zero, may need special handling
        if total_energy <= 0:
            logger.warning(f"Total energy is zero or negative ({total_energy}), cannot perform effective disaggregation")
            # If using average algorithm, can return all-zero allocation
            if algorithm_name == 'average':
                logger.info("Using average algorithm, returning all-zero allocation")
                return [dict(item, allocated_energy=0.0, allocation_method='average', allocation_ratio=0.0) 
                        for item in original_data]
            # For other algorithms, return empty list
            return []
        
        # Create disaggregation request
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
        
        # Get or create algorithm instance
        try:
            algorithm = self._get_algorithm(algorithm_name)
        except Exception as e:
            logger.error(f"Failed to get algorithm instance: {e}")
            # Fall back to average algorithm
            if algorithm_name != 'average':
                logger.info("Falling back to average algorithm")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # If average also fails, return empty list
                return []
        
        # Perform disaggregation
        try:
            result = algorithm.disaggregate(request)
            
            # Record performance
            self._record_performance(algorithm_name, request, result)
            
            logger.info(f"Disaggregation complete, using algorithm: {algorithm_name}, "
                       f"original data: {len(original_data)}, "
                       f"disaggregated results: {len(result.disaggregated_data)}, "
                       f"total energy: {total_energy:.2f} → {result.total_allocated_energy:.2f}")
            
            return result.disaggregated_data
            
        except Exception as e:
            logger.error(f"Disaggregation failed, algorithm: {algorithm_name}, error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to average allocation
            if algorithm_name != 'average':
                logger.info("Falling back to average allocation algorithm")
                return self.disaggregate(aggregated_result, original_data, 'average', time_step)
            else:
                # If average allocation also fails, return result with zero energy allocation
                logger.info("Average allocation also failed, returning zero energy allocation")
                return [dict(item, allocated_energy=0.0, allocation_method='fallback', allocation_ratio=0.0) 
                        for item in original_data]
    
    def _get_algorithm(self, algorithm_name: str) -> DisaggregationAlgorithm:
        """
        Get algorithm instance (with caching)
        
        Args:
            algorithm_name: Algorithm name
            
        Returns:
            DisaggregationAlgorithm: Algorithm instance
        """
        if algorithm_name not in self.algorithm_cache:
            try:
                self.algorithm_cache[algorithm_name] = DisaggregationAlgorithmFactory.create_algorithm(algorithm_name)
            except ValueError as e:
                logger.error(f"Failed to create algorithm: {e}")
                # Fall back to default algorithm
                if algorithm_name != self.default_algorithm:
                    logger.info(f"Falling back to default algorithm: {self.default_algorithm}")
                    return self._get_algorithm(self.default_algorithm)
                else:
                    raise
        
        return self.algorithm_cache[algorithm_name]
    
    def _record_performance(self, algorithm_name: str, request: DisaggregationRequest, result: DisaggregationResult):
        """Record performance metrics"""
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
        
        # Keep history record not exceeding 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {"message": "No performance records"}
        
        # Group statistics by algorithm
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
        
        # Calculate averages
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
        """Get list of available algorithms"""
        return DisaggregationAlgorithmFactory.get_available_algorithms()
    
    def set_default_algorithm(self, algorithm_name: str):
        """Set default algorithm"""
        available = self.get_available_algorithms()
        if algorithm_name not in available:
            raise ValueError(f"Algorithm '{algorithm_name}' does not exist. Available algorithms: {available}")
        
        self.default_algorithm = algorithm_name
        logger.info(f"Default disaggregation algorithm set to: {algorithm_name}")
    
    def clear_cache(self):
        """Clear algorithm cache"""
        self.algorithm_cache.clear()
        logger.info("Algorithm cache cleared") 
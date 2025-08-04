"""
Dynamic Observation Quality Management Module
Implements dynamic observation quality change mechanisms, including network condition simulation, quality assessment, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

class NetworkCondition(Enum):
    """Network condition enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class ObservationQualityMetrics:
    """Observation quality metrics"""
    accuracy: float = 1.0          # Accuracy [0,1]
    completeness: float = 1.0      # Completeness [0,1]
    timeliness: float = 1.0        # Timeliness [0,1]
    reliability: float = 1.0       # Reliability [0,1]
    consistency: float = 1.0       # Consistency [0,1]
    
    def overall_quality(self) -> float:
        """Calculate overall quality score"""
        return float(np.mean([
            self.accuracy,
            self.completeness,
            self.timeliness,
            self.reliability,
            self.consistency
        ]))

class DynamicObservationQuality:
    """Dynamic observation quality manager"""
    
    def __init__(self):
        # Network condition history
        self.network_history: List[NetworkCondition] = []
        
        # Communication quality between Managers
        self.communication_quality: Dict[str, float] = {}
        
        # Observation quality history
        self.quality_history: Dict[str, List[ObservationQualityMetrics]] = {}
        
        # Time steps
        self.current_step = 0
        
        # Quality change parameters
        self.quality_params = {
            'network_volatility': 0.1,      # Network condition volatility
            'degradation_rate': 0.02,       # Quality degradation rate
            'recovery_rate': 0.05,          # Quality recovery rate
            'baseline_quality': 0.85,       # Baseline quality
            'min_quality': 0.3,             # Minimum quality
            'max_quality': 1.0,             # Maximum quality
        }
    
    def update_network_condition(self) -> NetworkCondition:
        """
        Update network condition
        Simulate network condition changes based on Markov chain
        """
        # Get current network condition
        if len(self.network_history) == 0:
            current_condition = NetworkCondition.GOOD
        else:
            current_condition = self.network_history[-1]
        
        # Network condition transition probability matrix
        transition_probs = {
            NetworkCondition.EXCELLENT: {
                NetworkCondition.EXCELLENT: 0.7,
                NetworkCondition.GOOD: 0.25,
                NetworkCondition.FAIR: 0.05,
                NetworkCondition.POOR: 0.0,
                NetworkCondition.CRITICAL: 0.0
            },
            NetworkCondition.GOOD: {
                NetworkCondition.EXCELLENT: 0.1,
                NetworkCondition.GOOD: 0.6,
                NetworkCondition.FAIR: 0.25,
                NetworkCondition.POOR: 0.05,
                NetworkCondition.CRITICAL: 0.0
            },
            NetworkCondition.FAIR: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.2,
                NetworkCondition.FAIR: 0.5,
                NetworkCondition.POOR: 0.25,
                NetworkCondition.CRITICAL: 0.05
            },
            NetworkCondition.POOR: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.05,
                NetworkCondition.FAIR: 0.25,
                NetworkCondition.POOR: 0.6,
                NetworkCondition.CRITICAL: 0.1
            },
            NetworkCondition.CRITICAL: {
                NetworkCondition.EXCELLENT: 0.0,
                NetworkCondition.GOOD: 0.0,
                NetworkCondition.FAIR: 0.1,
                NetworkCondition.POOR: 0.4,
                NetworkCondition.CRITICAL: 0.5
            }
        }
        
        # Choose next state based on transition probabilities
        probs = transition_probs[current_condition]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Use random number to select state
        random_value = np.random.random()
        cumulative_prob = 0.0
        next_condition = states[0]  # Default value
        
        for state, prob in zip(states, probabilities):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                next_condition = state
                break
        self.network_history.append(next_condition)
        
        # Limit history length
        if len(self.network_history) > 100:
            self.network_history = self.network_history[-100:]
        
        return next_condition
    
    def calculate_communication_quality(self, manager_i: str, manager_j: str) -> float:
        """
        Calculate communication quality between Managers
        Based on distance, network condition, load, etc.
        """
        # Get current network condition
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        
        # Network condition impact on communication quality
        network_impact = {
            NetworkCondition.EXCELLENT: 1.0,
            NetworkCondition.GOOD: 0.9,
            NetworkCondition.FAIR: 0.7,
            NetworkCondition.POOR: 0.5,
            NetworkCondition.CRITICAL: 0.3
        }
        
        # Base communication quality (simulating geographic distance and infrastructure between Managers)
        manager_ids = sorted([manager_i, manager_j])
        manager_pair_key = f"{manager_ids[0]}_{manager_ids[1]}"
        
        if manager_pair_key not in self.communication_quality:
            # Initialize communication quality (simulating distance based on Manager ID difference)
            id_diff = abs(int(manager_i.split('_')[-1]) - int(manager_j.split('_')[-1]))
            distance_factor = max(0.5, 1.0 - id_diff * 0.1)  # Lower quality for greater distance
            
            # Add randomness
            random_factor = np.random.uniform(0.8, 1.2)
            
            base_quality = distance_factor * random_factor
            self.communication_quality[manager_pair_key] = np.clip(base_quality, 0.3, 1.0)
        
        # Get base quality
        base_quality = self.communication_quality[manager_pair_key]
        
        # Apply network condition impact
        current_quality = base_quality * network_impact[current_network]
        
        # Add time variation (simulating network congestion)
        time_factor = 1.0 + 0.1 * math.sin(self.current_step * 0.1)  # Periodic variation
        
        final_quality = current_quality * time_factor
        
        return np.clip(final_quality, 0.1, 1.0)
    
    def calculate_observation_quality(self, manager_id: str, 
                                    other_manager_ids: List[str]) -> ObservationQualityMetrics:
        """
        Calculate observation quality metrics for a Manager
        """
        # Accuracy: based on network condition and noise level
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        network_accuracy = {
            NetworkCondition.EXCELLENT: 0.98,
            NetworkCondition.GOOD: 0.95,
            NetworkCondition.FAIR: 0.88,
            NetworkCondition.POOR: 0.75,
            NetworkCondition.CRITICAL: 0.60
        }
        accuracy = network_accuracy[current_network]
        
        # Completeness: based on communication quality with other Managers
        communication_qualities = []
        for other_id in other_manager_ids:
            if other_id != manager_id:
                comm_quality = self.calculate_communication_quality(manager_id, other_id)
                communication_qualities.append(comm_quality)
        
        if communication_qualities:
            completeness = np.mean(communication_qualities)
        else:
            completeness = 1.0
        
        # Timeliness: based on network delay and system load
        timeliness = network_accuracy[current_network] * np.random.uniform(0.9, 1.0)
        
        # Reliability: based on consistency of historical observation quality
        if manager_id in self.quality_history and len(self.quality_history[manager_id]) > 0:
            recent_qualities = [q.overall_quality() for q in self.quality_history[manager_id][-10:]]
            quality_variance = np.var(recent_qualities)
            reliability = max(0.5, 1.0 - quality_variance * 2)  # Lower reliability for higher variance
        else:
            reliability = 0.9
        
        # Consistency: based on temporal consistency of observations
        consistency = np.random.uniform(0.85, 0.98)  # Simulating data consistency
        
        # Apply random fluctuations
        volatility = self.quality_params['network_volatility']
        accuracy *= np.random.uniform(1 - volatility, 1 + volatility)
        completeness *= np.random.uniform(1 - volatility, 1 + volatility)
        timeliness *= np.random.uniform(1 - volatility, 1 + volatility)
        reliability *= np.random.uniform(1 - volatility, 1 + volatility)
        consistency *= np.random.uniform(1 - volatility, 1 + volatility)
        
        # Ensure within reasonable range
        quality_metrics = ObservationQualityMetrics(
            accuracy=np.clip(accuracy, 0.3, 1.0),
            completeness=np.clip(completeness, 0.3, 1.0),
            timeliness=np.clip(timeliness, 0.3, 1.0),
            reliability=np.clip(reliability, 0.3, 1.0),
            consistency=np.clip(consistency, 0.3, 1.0)
        )
        
        return quality_metrics
    
    def apply_quality_degradation(self, observation: np.ndarray, 
                                quality_metrics: ObservationQualityMetrics) -> np.ndarray:
        """
        Apply degradation effects to observations based on quality metrics
        """
        degraded_obs = observation.copy()
        
        # 1. Accuracy impact: add noise
        if quality_metrics.accuracy < 1.0:
            noise_std = (1.0 - quality_metrics.accuracy) * 0.2
            noise = np.random.normal(0, noise_std, size=observation.shape)
            degraded_obs += noise
        
        # 2. Completeness impact: randomly set parts of observation to zero
        if quality_metrics.completeness < 1.0:
            missing_prob = (1.0 - quality_metrics.completeness) * 0.3
            missing_mask = np.random.random(observation.shape) < missing_prob
            degraded_obs[missing_mask] = 0.0
        
        # 3. Timeliness impact: use historical observation values
        if quality_metrics.timeliness < 0.9:
            # Simplified handling, in real environment would use historical observations
            delay_factor = 1.0 - quality_metrics.timeliness
            degraded_obs *= (1.0 - delay_factor * 0.1)
        
        # 4. Reliability impact: add systematic bias
        if quality_metrics.reliability < 0.9:
            bias = (1.0 - quality_metrics.reliability) * 0.1
            degraded_obs += bias
        
        # 5. Consistency impact: add random perturbations
        if quality_metrics.consistency < 0.9:
            inconsistency = (1.0 - quality_metrics.consistency) * 0.15
            perturbation = np.random.uniform(-inconsistency, inconsistency, size=observation.shape)
            degraded_obs += perturbation
        
        return degraded_obs
    
    def update_quality_history(self, manager_id: str, quality_metrics: ObservationQualityMetrics):
        """Update observation quality history"""
        if manager_id not in self.quality_history:
            self.quality_history[manager_id] = []
        
        self.quality_history[manager_id].append(quality_metrics)
        
        # Limit history length
        if len(self.quality_history[manager_id]) > 50:
            self.quality_history[manager_id] = self.quality_history[manager_id][-50:]
    
    def step(self):
        """Execute one step update"""
        self.current_step += 1
        self.update_network_condition()
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality report"""
        current_network = self.network_history[-1] if self.network_history else NetworkCondition.GOOD
        
        # Calculate average quality
        average_qualities = {}
        for manager_id, qualities in self.quality_history.items():
            if qualities:
                avg_quality = np.mean([q.overall_quality() for q in qualities[-10:]])
                average_qualities[manager_id] = avg_quality
        
        return {
            'current_network_condition': current_network.value,
            'network_history_length': len(self.network_history),
            'average_manager_qualities': average_qualities,
            'communication_pairs': len(self.communication_quality),
            'current_step': self.current_step,
            'quality_parameters': self.quality_params
        }
    
    def reset(self):
        """Reset quality manager"""
        self.network_history.clear()
        self.communication_quality.clear()
        self.quality_history.clear()
        self.current_step = 0 
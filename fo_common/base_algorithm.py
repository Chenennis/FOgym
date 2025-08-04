"""
FlexOffer Multi-Agent Reinforcement Learning Algorithm Base Class

This module provides common base classes and interfaces for all multi-agent RL algorithms
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BaseMARL(ABC):
    """Multi-Agent Reinforcement Learning Algorithm Base Class"""
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 device: str = "cpu"):
        """
        Initialize base MARL algorithm
        
        Args:
            n_agents: Number of agents
            state_dim: State dimension
            action_dim: Action dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Hidden layer dimension
            max_action: Maximum action value
            gamma: Discount factor
            tau: Soft update coefficient
            batch_size: Batch size
            buffer_capacity: Buffer capacity
            device: Computation device
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.device = torch.device(device)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # FlexOffer specific parameters
        self.fo_generation_mode = True
        self.manager_coordination_weight = 0.1
        
        # Initialize components
        self._setup_networks()
        self._setup_optimizers()
        self._setup_replay_buffer()
    
    @abstractmethod
    def _setup_networks(self):
        """Setup network structures - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _setup_optimizers(self):
        """Setup optimizers - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _setup_replay_buffer(self):
        """Setup experience replay buffer - implemented by subclasses"""
        pass
    
    @abstractmethod
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select actions - implemented by subclasses"""
        pass
    
    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """Update algorithm - implemented by subclasses"""
        pass
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, 
                        rewards: np.ndarray, next_states: np.ndarray, 
                        dones: np.ndarray, **kwargs):
        """Store experience - common implementation"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones, **kwargs)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float = None):
        """Soft update target network - common implementation"""
        if tau is None:
            tau = self.tau
            
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def save_models(self, path: str):
        """Save models - common framework"""
        save_dict = {
            'training_step': self.training_step,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }
        self._add_algorithm_specific_save_data(save_dict)
        torch.save(save_dict, path)
    
    def load_models(self, path: str):
        """Load models - common framework"""
        checkpoint = torch.load(path, map_location=self.device)
        self.training_step = checkpoint.get('training_step', 0)
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self._load_algorithm_specific_data(checkpoint)
    
    @abstractmethod
    def _add_algorithm_specific_save_data(self, save_dict: Dict):
        """Add algorithm-specific save data"""
        pass
    
    @abstractmethod
    def _load_algorithm_specific_data(self, checkpoint: Dict):
        """Load algorithm-specific data"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics - common implementation"""
        return {
            'training_step': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'buffer_size': len(self.replay_buffer) if hasattr(self, 'replay_buffer') else 0
        }


class BaseReplayBuffer(ABC):
    """Experience Replay Buffer Base Class"""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.size = 0
    
    @abstractmethod
    def push(self, *args, **kwargs):
        """Store experience"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int):
        """Sample experience"""
        pass
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.size = 0


class BaseActorNetwork(nn.Module, ABC):
    """Actor Network Base Class"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        pass


class BaseCriticNetwork(nn.Module, ABC):
    """Critic Network Base Class"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        pass


class FlexOfferMixin:
    """FlexOffer specific functionality mixin class"""
    
    def apply_fo_constraints(self, actions: torch.Tensor, 
                           fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply FlexOffer constraints"""
        if fo_constraints is None:
            return actions
        
        # Simplified constraint application logic
        constrained_actions = torch.clamp(actions, -1.0, 1.0)
        return constrained_actions
    
    def compute_fo_constraint_loss(self, actions: torch.Tensor, 
                                  fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute FlexOffer constraint loss"""
        if fo_constraints is None:
            return torch.tensor(0.0, device=actions.device)
        
        # Constraint violation loss
        constraint_violations = torch.relu(torch.abs(actions) - 1.0)
        return constraint_violations.mean()
    
    def compute_device_coordination_loss(self, actions: torch.Tensor, 
                                       device_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute device coordination loss"""
        if device_states is None:
            return torch.tensor(0.0, device=actions.device)
        
        # Encourage moderate action variance (coordinated but not identical)
        action_variance = torch.var(actions, dim=-1).mean()
        target_variance = 0.5  # Target variance
        coordination_loss = torch.relu(action_variance - target_variance)
        return coordination_loss


class AlgorithmRegistry:
    """Algorithm Registry"""
    
    _algorithms = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: type):
        """Register algorithm"""
        cls._algorithms[name] = algorithm_class
        logger.info(f"Algorithm registered successfully: {name}")
    
    @classmethod
    def get(cls, name: str):
        """Get algorithm class"""
        if name not in cls._algorithms:
            raise ValueError(f"Unregistered algorithm: {name}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithms"""
        return list(cls._algorithms.keys())


# Algorithm factory function
def create_algorithm(algorithm_name: str, config: Dict[str, Any]) -> BaseMARL:
    """
    Create algorithm instance
    
    Args:
        algorithm_name: Algorithm name
        config: Configuration parameters
        
    Returns:
        Algorithm instance
    """
    algorithm_class = AlgorithmRegistry.get(algorithm_name)
    return algorithm_class(**config) 
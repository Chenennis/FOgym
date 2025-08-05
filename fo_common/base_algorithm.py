import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BaseMARL(ABC):
    """multi-agent reinforcement learning algorithm base class"""
    
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
        initialize base MARL algorithm
        
        Args:
            n_agents: number of agents
            state_dim: state dimension
            action_dim: action dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: hidden layer dimension
            max_action: maximum action value
            gamma: discount factor
            tau: soft update coefficient
            batch_size: batch size
            buffer_capacity: buffer capacity
            device: compute device
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
        
        # training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # FlexOffer specific parameters
        self.fo_generation_mode = True
        self.manager_coordination_weight = 0.1
        
        # initialize components
        self._setup_networks()
        self._setup_optimizers()
        self._setup_replay_buffer()
    
    @abstractmethod
    def _setup_networks(self):
        """set network structure - implemented by subclass"""
        pass
    
    @abstractmethod
    def _setup_optimizers(self):
        """set optimizer - implemented by subclass"""
        pass
    
    @abstractmethod
    def _setup_replay_buffer(self):
        """set experience replay buffer - implemented by subclass"""
        pass
    
    @abstractmethod
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """select actions - implemented by subclass"""
        pass
    
    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """update algorithm - implemented by subclass"""
        pass
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, 
                        rewards: np.ndarray, next_states: np.ndarray, 
                        dones: np.ndarray, **kwargs):
        """store experience - implemented by subclass"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones, **kwargs)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float = None):
        """soft update target network - implemented by subclass"""
        if tau is None:
            tau = self.tau
            
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def save_models(self, path: str):
        """save models - implemented by subclass"""
        save_dict = {
            'training_step': self.training_step,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }
        self._add_algorithm_specific_save_data(save_dict)
        torch.save(save_dict, path)
    
    def load_models(self, path: str):
        """load models - implemented by subclass"""
        checkpoint = torch.load(path, map_location=self.device)
        self.training_step = checkpoint.get('training_step', 0)
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self._load_algorithm_specific_data(checkpoint)
    
    @abstractmethod
    def _add_algorithm_specific_save_data(self, save_dict: Dict):
        """add algorithm specific save data"""
        pass
    
    @abstractmethod
    def _load_algorithm_specific_data(self, checkpoint: Dict):
        """load algorithm specific data"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics - implemented by subclass"""
        return {
            'training_step': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'buffer_size': len(self.replay_buffer) if hasattr(self, 'replay_buffer') else 0
        }


class BaseReplayBuffer(ABC):
    """experience replay buffer base class"""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.size = 0
    
    @abstractmethod
    def push(self, *args, **kwargs):
        """store experience"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int):
        """sample experience"""
        pass
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """clear buffer"""
        self.buffer.clear()
        self.size = 0


class BaseActorNetwork(nn.Module, ABC):
    """Actor network base class"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """forward propagation"""
        pass


class BaseCriticNetwork(nn.Module, ABC):
    """Critic network base class"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """forward propagation"""
        pass


class FlexOfferMixin:
    """FlexOffer specific feature mixin class"""
    
    def apply_fo_constraints(self, actions: torch.Tensor, 
                           fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """apply FlexOffer constraints"""
        if fo_constraints is None:
            return actions
        
        constrained_actions = torch.clamp(actions, -1.0, 1.0)
        return constrained_actions
    
    def compute_fo_constraint_loss(self, actions: torch.Tensor, 
                                  fo_constraints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """compute FlexOffer constraint loss"""
        if fo_constraints is None:
            return torch.tensor(0.0, device=actions.device)
        
        # constraint violation loss
        constraint_violations = torch.relu(torch.abs(actions) - 1.0)
        return constraint_violations.mean()
    
    def compute_device_coordination_loss(self, actions: torch.Tensor, 
                                       device_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """compute device coordination loss"""
        if device_states is None:
            return torch.tensor(0.0, device=actions.device)
        
        # encourage moderate action variance (coordination but not completely same)
        action_variance = torch.var(actions, dim=-1).mean()
        target_variance = 0.5  # target variance
        coordination_loss = torch.relu(action_variance - target_variance)
        return coordination_loss


class AlgorithmRegistry:
    """algorithm registry"""
    
    _algorithms = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: type):
        """register algorithm"""
        cls._algorithms[name] = algorithm_class
        logger.info(f"algorithm registered successfully: {name}")
    
    @classmethod
    def get(cls, name: str):
        """get algorithm class"""
        if name not in cls._algorithms:
            raise ValueError(f"unregistered algorithm: {name}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """list all registered algorithms"""
        return list(cls._algorithms.keys())


# 算法工厂函数
def create_algorithm(algorithm_name: str, config: Dict[str, Any]) -> BaseMARL:
    """
    create algorithm instance
    
    Args:
        algorithm_name: algorithm name
        config: configuration parameters
        
    Returns:
        algorithm instance
    """
    algorithm_class = AlgorithmRegistry.get(algorithm_name)
    return algorithm_class(**config) 
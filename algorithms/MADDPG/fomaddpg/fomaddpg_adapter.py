"""
FOMADDPG Adapter - 基于MADDPG的FlexOffer多智能体算法适配器

提供与FOMAPPO相同的接口，但内部使用MADDPG的off-policy学习机制。
支持与FO Pipeline的无缝集成。

Algorithm: FOMADDPG (FlexOffer Multi-Agent Deep Deterministic Policy Gradient)
Base: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
Key Features:
- Off-policy learning with replay buffer
- Continuous action spaces
- Actor-Critic architecture
- Multi-agent coordination
- FlexOffer constraint awareness

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import os

from .fomaddpg import FOMADDPG, map_normalized_fo_actions

logger = logging.getLogger(__name__)

class FOMAddpgArgs:
    """FOMADDPG parameter configuration class - inherit MADDPG parameters and add FlexOffer specific parameters"""
    
    def __init__(self, **kwargs):
        # ========== core MADDPG parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.buffer_capacity = kwargs.get('buffer_capacity', 100000)
        self.batch_size = kwargs.get('batch_size', 64)
        
        # learning rate parameters
        self.lr = kwargs.get('lr_actor', 1e-4)
        self.lr_actor = kwargs.get('lr_actor', 1e-4)
        self.critic_lr = kwargs.get('lr_critic', 1e-3)
        self.tau = kwargs.get('tau', 0.005)  # soft update parameter
        
        # DDPG specific parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.noise_scale = kwargs.get('noise_scale', 0.1)
        self.max_action = kwargs.get('max_action', 1.0)
        
        # network parameters
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.layer_N = kwargs.get('layer_N', 2)
        self.use_orthogonal = kwargs.get('use_orthogonal', True)
        self.gain = kwargs.get('gain', 0.01)
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)
        self.activation_id = kwargs.get('activation_id', 1)
        self.use_ReLU = kwargs.get('use_ReLU', False)
        
        # training options
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        
        # algorithm name
        self.algorithm_name = kwargs.get('algorithm_name', 'fomaddpg')
        
        # ========== FOMADDPG specific parameters ==========
        self.use_device_coordination = kwargs.get('use_device_coordination', True)
        self.device_coordination_weight = kwargs.get('device_coordination_weight', 0.1)
        self.fo_constraint_weight = kwargs.get('fo_constraint_weight', 0.2)
        self.use_manager_coordination = kwargs.get('use_manager_coordination', True)
        self.manager_coordination_weight = kwargs.get('manager_coordination_weight', 0.05)
        
        # network architecture specific parameters
        self.num_managers = kwargs.get('num_managers', 4)
        self.devices_per_manager = kwargs.get('devices_per_manager', 10)

class FOMAddpgAdapter:
   
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 device: str = "cpu",
                 observation_dims: Optional[Dict[str, int]] = None,
                 action_dims: Optional[Dict[str, int]] = None,
                 **kwargs):
        """
        initialize FOMADDPG adapter
        
        Args:
            state_dim: state dimension
            action_dim: action dimension
            num_agents: number of agents (number of Managers)
            episode_length: episode length
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: computing device
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.agent_ids = [f"manager_{i + 1}" for i in range(num_agents)]
        self.observation_dims = {
            mid: (observation_dims or {}).get(mid, state_dim)
            for mid in self.agent_ids
        }
        self.action_dims = {
            mid: (action_dims or {}).get(mid, action_dim)
            for mid in self.agent_ids
        }
        if (any(dim <= 0 or dim > state_dim for dim in self.observation_dims.values())
                or any(dim <= 0 or dim > action_dim for dim in self.action_dims.values())):
            raise ValueError("Manager dimensions must fit padded state/action widths")
        
        logger.info(f"🔧 initialize FOMADDPG adapter (based on MADDPG architecture)")
        logger.info(f"    parameters: {num_agents} Managers, {state_dim} state dimension, {action_dim} action dimension")
        
        # create parameter object
        self.args = FOMAddpgArgs(
            episode_length=episode_length,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            num_managers=num_agents,
            **kwargs
        )
        
        # initialize FOMADDPG algorithm
        self.fomaddpg = FOMADDPG(
            n_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=kwargs.get('hidden_dim', 256),
            max_action=kwargs.get('max_action', 1.0),
            gamma=kwargs.get('gamma', 0.99),
            tau=kwargs.get('tau', 0.005),
            noise_scale=kwargs.get('noise_scale', 0.1),
            buffer_capacity=kwargs.get('buffer_capacity', 100000),
            batch_size=kwargs.get('batch_size', 64),
            action_dims=[self.action_dims[mid] for mid in self.agent_ids],
            device=device
        )
        
        # training statistics
        self.training_iterations = 0
        self.total_episodes = 0
        
        # Manager statistics
        self.manager_stats = {}
        for i in range(num_agents):
            manager_id = f"manager_{i + 1}"
            self.manager_stats[manager_id] = {
                'total_reward': 0.0,
                'episode_count': 0,
                'avg_reward': 0.0,
                'best_reward': float('-inf'),
                'training_updates': 0
            }
        
        logger.info("✅ FOMADDPG adapter initialized")
        logger.info(f"    architecture: Off-policy MADDPG, experience replay buffer, continuous action space")
    
    def reset_buffers(self):
        """reset buffers - MADDPG uses experience replay, no episode-level reset"""
        # MADDPG uses experience replay buffer, no episode-level reset
        # here maintain interface compatibility, but in reality MADDPG's buffer is continuously accumulated
        logger.debug("FOMADDPG uses experience replay buffer, no episode-level reset")
        pass
    
    def _stack_observations(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        for i, mid in enumerate(self.agent_ids):
            value = np.asarray(obs[mid], dtype=np.float32).reshape(-1)
            if value.size != self.observation_dims[mid]:
                raise ValueError(
                    f"{mid} observation has {value.size} entries; "
                    f"expected {self.observation_dims[mid]}"
                )
            padded[i, :value.size] = value
        return padded

    def _stack_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        for i, mid in enumerate(self.agent_ids):
            value = np.asarray(actions[mid], dtype=np.float32).reshape(-1)
            if value.size != self.action_dims[mid]:
                raise ValueError(
                    f"{mid} action has {value.size} entries; "
                    f"expected {self.action_dims[mid]}"
                )
            padded[i, :value.size] = value
        return padded

    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        states = self._stack_observations(obs)
        # Do not replace a failed policy decision with untracked random actions.
        raw_actions = self.fomaddpg.select_actions(states, add_noise=not deterministic)
        actions, action_log_probs, values = {}, {}, {}
        for i, mid in enumerate(self.agent_ids):
            raw = raw_actions[i, :self.action_dims[mid]]
            action = self._map_action_to_fo_params(raw)
            actions[mid] = action
            action_log_probs[mid] = np.zeros_like(action)
            values[mid] = np.array([0.0])
        return actions, action_log_probs, values

    def collect_step(self, 
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     next_obs: Dict[str, np.ndarray],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """
        collect one step of experience data to experience replay buffer
        
        Args:
            obs: current observation
            next_obs: actual next observation returned by the environment
            actions: executed actions
            rewards: received rewards
            dones: whether to end
            infos: additional information
            action_log_probs: action log probabilities (MADDPG does not use)
            values: value function predictions (MADDPG does not use)
        """
        states = self._stack_observations(obs)
        next_states = self._stack_observations(next_obs)
        agent_actions = self._stack_actions(actions)
        agent_rewards = []
        agent_dones = []
        for mid in self.agent_ids:
            raw_reward = float(rewards[mid])
            if not np.isfinite(raw_reward):
                raise ValueError(f"{mid} received non-finite reward")
            # Fixed linear scaling preserves reward differences over time.
            # Keep manager statistics in the original environmental reward units.
            normalized_reward = raw_reward / 1000.0
            agent_rewards.append(normalized_reward)
            agent_dones.append(bool(dones[mid]))
            self.manager_stats[mid]['total_reward'] += raw_reward
            self.manager_stats[mid]['best_reward'] = max(
                self.manager_stats[mid]['best_reward'], raw_reward
            )
        self.fomaddpg.store_experience(
            states=states, actions=agent_actions,
            rewards=np.asarray(agent_rewards, dtype=np.float32),
            next_states=next_states,
            dones=np.asarray(agent_dones, dtype=bool),
        )

    def compute_returns(self):
        """compute returns - MADDPG does not need to compute returns like PPO"""
        # MADDPG is off-policy algorithm, does not need to compute episode-level returns
        # maintain interface compatibility
        pass
    
    def train_on_batch(self) -> Optional[Dict[str, Any]]:
        """Report only successful optimizer updates; propagate training errors."""
        if len(self.fomaddpg.replay_buffer) < self.fomaddpg.batch_size:
            return None

        try:
            update_info = self.fomaddpg.update()
        except Exception:
            logger.exception("FOMADDPG optimizer update failed")
            raise
        if not update_info:
            return None

        self.training_iterations += 1
        for manager_id in self.manager_stats:
            self.manager_stats[manager_id]['training_updates'] += 1
        return {
            'actor_loss': update_info['actor_loss'],
            'critic_loss': update_info['critic_loss'],
            'training_iterations': self.training_iterations,
            'buffer_size': len(self.fomaddpg.replay_buffer),
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.fomaddpg.replay_buffer) if hasattr(self.fomaddpg, 'replay_buffer') else 0,
            'algorithm': 'FOMADDPG'
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """get Manager reward summary"""
        summary = {}
        for manager_id, stats in self.manager_stats.items():
            if stats['episode_count'] > 0:
                stats['avg_reward'] = stats['total_reward'] / stats['episode_count']
            summary[manager_id] = stats.copy()
        return summary
    
    def save_models(self, save_path: str):
        """Save all agent models; report filesystem and serialization failures."""
        directory = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(directory, exist_ok=True)
        self.fomaddpg.save_models(save_path)
        logger.info("FOMADDPG models saved to %s", save_path)

    def load_models(self, load_path: str):
        """Restore all agent models, raising when any model is absent."""
        self.fomaddpg.load_models(load_path)
        logger.info("FOMADDPG models loaded from %s", load_path)

    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        """Use the exact differentiable action conversion used during training."""
        with torch.no_grad():
            tensor = torch.as_tensor(raw_action, dtype=torch.float32)
            return map_normalized_fo_actions(tensor).cpu().numpy()

    def _generate_default_fo_action(self) -> np.ndarray:
        default_action = np.zeros(self.action_dim)
        num_devices = self.action_dim // 5 if self.action_dim >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < self.action_dim:
                default_action[base_idx] = 0.0      # start_flex = 0
                default_action[base_idx + 1] = 0.0  # end_flex = 0  
                default_action[base_idx + 2] = 0.55 # energy_min_factor = 0.55
                default_action[base_idx + 3] = 1.5  # energy_max_factor = 1.5
                default_action[base_idx + 4] = 1.0  # priority_weight = 1.0
        
        return default_action 
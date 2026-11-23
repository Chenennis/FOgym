#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import os
from collections import defaultdict

# FO Framework imports
from .fosqddpg import FOSQDDPG, map_normalized_fo_actions

logger = logging.getLogger(__name__)


class FOSQDDPGAdapter:
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 num_agents: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_scale: float = 0.1,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 sample_size: int = 5,  # Shapley sampling size
                 policy_delay: int = 1,  # FOSQDDPG usually does not need delay update
                 device: str = "cpu",
                 observation_dims: Optional[Dict[str, int]] = None,
                 action_dims: Optional[Dict[str, int]] = None):
        
        # core parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.policy_delay = policy_delay
        
        # FOSQDDPG specific parameters
        self.max_action = max_action
        self.buffer_capacity = buffer_capacity
        
        # agent ID management
        self.agent_ids = [f"manager_{i+1}" for i in range(num_agents)]
        self.observation_dims = {
            agent_id: (observation_dims or {}).get(agent_id, state_dim)
            for agent_id in self.agent_ids
        }
        self.action_dims = {
            agent_id: (action_dims or {}).get(agent_id, action_dim)
            for agent_id in self.agent_ids
        }
        if (any(dim <= 0 or dim > state_dim for dim in self.observation_dims.values())
                or any(dim <= 0 or dim > action_dim for dim in self.action_dims.values())):
            raise ValueError("Manager dimensions must be positive and within padded dimensions")
        
        # training state
        self.training_iterations = 0
        self.total_iterations = 0
        self.current_episode = 0
        
        # initialize FOSQDDPG algorithm
        self.fosqddpg = FOSQDDPG(
            n_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            max_action=max_action,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            sample_size=sample_size,
            device=device,
            action_dims=[self.action_dims[agent_id] for agent_id in self.agent_ids]
        )
        
        # cache mechanism (compatible with FO Pipeline interface)
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
        
        # manager reward statistics
        self._manager_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # reward normalizer
        self.reward_normalizer = self._create_reward_normalizer()
        
        # compatibility parameter object
        class Args:
            def __init__(self, policy_delay, noise_scale, sample_size):
                self.policy_delay = policy_delay
                self.noise_scale = noise_scale
                self.sample_size = sample_size
        
        self.args = Args(policy_delay, noise_scale, sample_size)
        
        logger.info(f"FOSQDDPG adapter initialized: {num_agents} agents, "
                   f"state dimension={state_dim}, action dimension={action_dim}, "
                   f"Shapley sampling={sample_size}")
    
    def _create_reward_normalizer(self):
        """create reward normalizer"""
        return {
            'running_min': float('inf'),
            'running_max': float('-inf'),
            'running_mean': 0.0,
            'running_var': 1.0,
            'count': 0,
            'epsilon': 1e-8
        }
    
    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Apply a fixed scale so Bellman targets keep consistent reward units."""
        rewards = np.asarray(rewards, dtype=np.float32)
        if not np.isfinite(rewards).all():
            raise ValueError("SQDDPG received non-finite rewards")

        # Track raw reward statistics for diagnostics; do not use them to
        # alter the reward function as more experience is collected.
        stats = self.reward_normalizer
        for reward in rewards.flat:
            count = stats['count'] + 1
            delta = float(reward) - stats['running_mean']
            mean = stats['running_mean'] + delta / count
            delta2 = float(reward) - mean
            previous_m2 = stats['running_var'] * (count - 1) if count > 1 else 0.0
            stats['running_mean'] = mean
            stats['running_var'] = (previous_m2 + delta * delta2) / count
            stats['count'] = count
            stats['running_min'] = min(stats['running_min'], float(reward))
            stats['running_max'] = max(stats['running_max'], float(reward))
        return rewards / 1000.0
    
    def _stack_observations(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            values = np.asarray(observations[agent_id], dtype=np.float32).reshape(-1)
            expected = self.observation_dims[agent_id]
            if values.size != expected:
                raise ValueError(f"{agent_id} observation has {values.size} entries; expected {expected}")
            padded[i, :expected] = values
        return padded

    def _stack_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            values = np.asarray(actions[agent_id], dtype=np.float32).reshape(-1)
            expected = self.action_dims[agent_id]
            if values.size != expected:
                raise ValueError(f"{agent_id} action has {values.size} entries; expected {expected}")
            padded[i, :expected] = values
        return padded

    def select_actions(self, 
                      observations: Dict[str, np.ndarray], 
                      deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Optional[Dict], Optional[Dict]]:

        # convert to numpy array
        obs_array = self._stack_observations(observations)
        
        # The training loop adjusts adapter.noise_scale after warmup.
        # Keep the underlying action sampler in sync.
        self.fosqddpg.noise_scale = self.noise_scale
        raw_actions_array = self.fosqddpg.select_actions(obs_array, add_noise=not deterministic)
        
        # convert back to dictionary format and map to FlexOffer parameter range
        actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            raw_action = raw_actions_array[i, :self.action_dims[agent_id]]
            fo_action = self._map_action_to_fo_params(raw_action)
            actions[agent_id] = fo_action
            
            logger.debug(f"Manager {agent_id} FOSQDDPG FlexOffer action: {fo_action.shape} dimensions, "
                       f"first 5 parameters: {fo_action[:5]}, Shapley fairness weight applied")
        
        return actions, None, None
    
    def collect_step(self,
                    obs: Dict[str, np.ndarray],
                    actions: Dict[str, np.ndarray],
                    rewards: Dict[str, float],
                    dones: Dict[str, bool],
                    infos: Dict[str, Any],
                    timestep: int,
                    next_obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Store the actual transition generated by this environment step."""
        states = self._stack_observations(obs)
        next_states = self._stack_observations(next_obs)
        actions_array = self._stack_actions(actions)
        rewards_array = np.array([rewards[agent_id] for agent_id in self.agent_ids])
        dones_array = np.array([dones[agent_id] for agent_id in self.agent_ids])

        normalized_rewards = self._normalize_rewards(rewards_array)
        for i, agent_id in enumerate(self.agent_ids):
            self._manager_rewards[agent_id].append(rewards_array[i])

        # Store every step immediately, including the terminal transition.
        self.fosqddpg.store_experience(
            states.copy(), actions_array.copy(), normalized_rewards.copy(),
            next_states.copy(), dones_array.copy()
        )

        return {
            "states": states,
            "actions": actions_array,
            "rewards": normalized_rewards,
            "next_states": next_states,
            "dones": dones_array,
            "normalized_rewards": normalized_rewards
        }

    def train_on_batch(self) -> Optional[Dict[str, float]]:
        """
        execute one batch training
        """
        if len(self.fosqddpg.replay_buffer) < self.batch_size:
            return None
        
        try:
            # execute FOSQDDPG training
            training_info = self.fosqddpg.update()
            
            if training_info:
                self.training_iterations += 1
                self.total_iterations += 1
                
                # add FOSQDDPG specific statistics
                training_stats = training_info.copy()
                additional_stats = {
                    'total_iterations': self.total_iterations,
                    'buffer_size': len(self.fosqddpg.replay_buffer),
                    'current_noise_scale': self.noise_scale,
                    'sample_size': self.sample_size,
                    'reward_stats': {
                        'mean': self.reward_normalizer['running_mean'],
                        'std': np.sqrt(self.reward_normalizer['running_var']),
                        'count': self.reward_normalizer['count']
                    }
                }
                
                # merge statistics
                for key, value in additional_stats.items():
                    training_stats[key] = value
                
                return training_stats
            
        except Exception:
            logger.exception("FOSQDDPG optimizer update failed")
            raise
        
        return None
    
    def reset_episode(self):
        """reset episode state"""
        self.current_episode += 1
        
        # clear cache
        self._prev_states = None
        self._prev_actions = None
        self._prev_rewards = None
        self._prev_dones = None
    
    def reset_buffers(self):
        """reset experience buffer"""
        self.fosqddpg.replay_buffer.buffer.clear()
        logger.debug("FOSQDDPG experience buffer reset")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'training_iterations': self.training_iterations,
            'total_iterations': self.total_iterations, 
            'current_episode': self.current_episode,
            'buffer_size': len(self.fosqddpg.replay_buffer) if hasattr(self.fosqddpg, 'replay_buffer') else 0,
            'noise_scale': self.noise_scale,
            'sample_size': self.sample_size
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Dict[str, float]]:
        """get manager reward statistics summary"""
        summary = {}
        for agent_id in self.agent_ids:
            if agent_id in self._manager_rewards and self._manager_rewards[agent_id]:
                rewards = self._manager_rewards[agent_id]
                summary[agent_id] = {
                    "mean_reward": np.mean(rewards),
                    "total_reward": np.sum(rewards),
                    "episodes": len(rewards),
                    "max_reward": np.max(rewards),
                    "min_reward": np.min(rewards)
                }
            else:
                summary[agent_id] = {
                    "mean_reward": 0.0,
                    "total_reward": 0.0,
                    "episodes": 0,
                    "max_reward": 0.0,
                    "min_reward": 0.0
                }
        return summary
    
    def save_models(self, save_path: str):
        """save models"""
        os.makedirs(save_path, exist_ok=True)
        
        for i, policy in enumerate(self.fosqddpg.policies):
            agent_id = self.agent_ids[i]
            
            # save Actor and Critic networks
            actor_path = os.path.join(save_path, f"{agent_id}_actor.pt")
            critic_path = os.path.join(save_path, f"{agent_id}_critic.pt")
            
            torch.save(policy.actor.state_dict(), actor_path)
            torch.save(policy.critic.state_dict(), critic_path)
            
            logger.info(f"save {agent_id} model to {save_path}")
    
    def load_models(self, load_path: str):
        """load models"""
        for i, policy in enumerate(self.fosqddpg.policies):
            agent_id = self.agent_ids[i]
            
            # load Actor and Critic networks
            actor_path = os.path.join(load_path, f"{agent_id}_actor.pt")
            critic_path = os.path.join(load_path, f"{agent_id}_critic.pt")
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                policy.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                policy.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                logger.info(f"load {agent_id} model from {load_path}")
            else:
                logger.warning(f"model file not found for {agent_id}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        """Use the same physical FO action mapping as actor/target training."""
        with torch.no_grad():
            raw_tensor = torch.as_tensor(raw_action, dtype=torch.float32)
            return map_normalized_fo_actions(raw_tensor).cpu().numpy()

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
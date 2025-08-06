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

from .fomaddpg import FOMADDPG

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
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        actions = {}
        action_log_probs = {}  # MADDPG does not use, but maintain interface compatibility
        values = {}  # MADDPG does not use, but maintain interface compatibility
        
        manager_ids = list(obs.keys())
        
        # prepare states array format, MADDPG expects numpy array
        states = []
        for manager_id in manager_ids:
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                states.append(current_obs)
            else:
                states.append(np.array(current_obs))
        
        states = np.array(states)  # Shape: (num_agents, state_dim)
        
        # use FOMADDPG to select actions
        try:
            # call FOMADDPG's select_actions method
            agent_actions = self.fomaddpg.select_actions(states, add_noise=not deterministic)
            
            # convert to dictionary format and map to FlexOffer parameter range
            for i, manager_id in enumerate(manager_ids):
                raw_action = agent_actions[i]
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.zeros_like(fo_action)  # placeholder
                values[manager_id] = np.array([0.0])  # placeholder
                
                logger.debug(f"Manager {manager_id} MADDPG FlexOffer action: {fo_action.shape} dimension, "
                           f"first 5 parameters: {fo_action[:5]}")
                
        except Exception as e:
            logger.error(f"FOMADDPG action selection failed: {e}")
            for manager_id in manager_ids:
                actions[manager_id] = np.random.uniform(-1, 1, self.action_dim)
                action_log_probs[manager_id] = np.zeros(self.action_dim)
                values[manager_id] = np.array([0.0])
        
        return actions, action_log_probs, values
    
    def collect_step(self, 
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """
        collect one step of experience data to experience replay buffer
        
        Args:
            obs: current observation
            actions: executed actions
            rewards: received rewards
            dones: whether to end
            infos: additional information
            action_log_probs: action log probabilities (MADDPG does not use)
            values: value function predictions (MADDPG does not use)
        """
        # numerical stability: check and fix rewards
        manager_ids = list(obs.keys())
        
        # prepare MADDPG format data
        states = []
        agent_actions = []
        agent_rewards = []
        agent_dones = []
        
        for manager_id in manager_ids:
            # observation
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                states.append(current_obs)
            else:
                states.append(np.array(current_obs))
            
            # action
            action = actions[manager_id]
            if isinstance(action, np.ndarray):
                agent_actions.append(action)
            else:
                agent_actions.append(np.array(action))
            
            # reward 
            raw_reward = rewards[manager_id]
            if np.isnan(raw_reward) or np.isinf(raw_reward):
                logger.warning(f"Manager {manager_id} reward invalid ({raw_reward}), set to 0")
                raw_reward = 0.0
            
            # optimized reward scaling
            normalized_reward = np.clip(raw_reward, -50.0, 50.0) * 0.5  
            agent_rewards.append(normalized_reward)
            
            # done flag
            agent_dones.append(dones[manager_id])
            
            # update Manager statistics
            self.manager_stats[manager_id]['total_reward'] += normalized_reward
            if normalized_reward > self.manager_stats[manager_id]['best_reward']:
                self.manager_stats[manager_id]['best_reward'] = normalized_reward
        
        # convert to numpy array
        states = np.array(states)
        agent_actions = np.array(agent_actions)
        agent_rewards = np.array(agent_rewards)
        agent_dones = np.array(agent_dones)
        
        if hasattr(self, '_prev_states'):
            # if there are previous states, store them as next_states
            try:
                self.fomaddpg.store_experience(
                    states=self._prev_states,
                    actions=self._prev_actions,
                    rewards=self._prev_rewards,
                    next_states=states,
                    dones=self._prev_dones
                )
            except Exception as e:
                logger.warning(f"FOMADDPG experience storage failed: {e}")
        
        # save current state for next storage
        self._prev_states = states.copy()
        self._prev_actions = agent_actions.copy()
        self._prev_rewards = agent_rewards.copy()
        self._prev_dones = agent_dones.copy()
    
    def compute_returns(self):
        """compute returns - MADDPG does not need to compute returns like PPO"""
        # MADDPG is off-policy algorithm, does not need to compute episode-level returns
        # maintain interface compatibility
        pass
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        perform one MADDPG training update
        
        Returns:
            training information dictionary
        """
        try:
            # check if there is enough experience for training
            if len(self.fomaddpg.replay_buffer) < self.fomaddpg.batch_size:
                logger.debug(f"experience buffer insufficient ({len(self.fomaddpg.replay_buffer)}/{self.fomaddpg.batch_size}), skip training")
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
            
            # perform MADDPG update
            update_info = self.fomaddpg.update()
            
            if update_info:
                self.training_iterations += 1
                
                # update Manager statistics
                for manager_id in self.manager_stats:
                    self.manager_stats[manager_id]['training_updates'] += 1
                
                return {
                    'actor_loss': update_info.get('actor_loss', 0.0),
                    'critic_loss': update_info.get('critic_loss', 0.0),
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
            else:
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'training_iterations': self.training_iterations,
                    'buffer_size': len(self.fomaddpg.replay_buffer)
                }
                
        except Exception as e:
            logger.error(f"FOMADDPG training update failed: {e}")
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'training_iterations': self.training_iterations,
                'buffer_size': len(self.fomaddpg.replay_buffer) if hasattr(self.fomaddpg, 'replay_buffer') else 0,
                'error': str(e)
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
        """save models"""
        try:
            # ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # use FOMADDPG's save method
            self.fomaddpg.save_models(save_path)
            logger.info(f"FOMADDPG models saved to {save_path}")
        except Exception as e:
            logger.error(f"save FOMADDPG models failed: {e}")
    
    def load_models(self, load_path: str):
        """load models"""
        try:
            # use FOMADDPG's load method
            self.fomaddpg.load_models(load_path)
            logger.info(f"FOMADDPG models loaded from {load_path}")
        except Exception as e:
            logger.error(f"load FOMADDPG models failed: {e}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        fo_action = np.zeros_like(raw_action)
        
        num_devices = len(raw_action) // 5 if len(raw_action) >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < len(raw_action):
                # start_flex: [-1, 1] → [-1, 1] 
                fo_action[base_idx] = np.clip(raw_action[base_idx], -1.0, 1.0)
                
                # end_flex: [-1, 1] → [-1, 1] 
                fo_action[base_idx + 1] = np.clip(raw_action[base_idx + 1], -1.0, 1.0)
                
                # energy_min_factor: [-1, 1] → [0.1, 1.0]
                fo_action[base_idx + 2] = 0.1 + 0.45 * (raw_action[base_idx + 2] + 1.0)
                
                # energy_max_factor: [-1, 1] → [1.0, 2.0]  
                fo_action[base_idx + 3] = 1.0 + 0.5 * (raw_action[base_idx + 3] + 1.0)
                
                # priority_weight: [-1, 1] → [0.1, 2.0]
                fo_action[base_idx + 4] = 0.1 + 0.95 * (raw_action[base_idx + 4] + 1.0)
        
        return fo_action
    
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
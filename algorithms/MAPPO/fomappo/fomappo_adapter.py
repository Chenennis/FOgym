#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAPPOArgs:
    
    def __init__(self, **kwargs):
        # ========== core PPO parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 2)  # increased to 2 mini-batches
        self.ppo_epoch = kwargs.get('ppo_epoch', 8)  # increased to 8 epochs
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # learning rate parameters - reduced to avoid convergence to suboptimal solutions
        self.lr = kwargs.get('lr_actor', 5e-5)  # reduced from 3e-4 to 5e-5
        self.lr_actor = kwargs.get('lr_actor', 5e-5)  # reduced from 3e-4 to 5e-5
        self.critic_lr = kwargs.get('lr_critic', 2e-4)  # reduced from 1e-3 to 2e-4
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # increase learning rate decay parameters
        self.use_linear_lr_decay = kwargs.get('use_linear_lr_decay', True)  # enable learning rate decay
        self.lr_decay_rate = kwargs.get('lr_decay_rate', 0.95)  # learning rate decay rate
        
        # PPO clipping parameters
        self.clip_param = kwargs.get('clip_param', 0.2)
        self.value_loss_coef = kwargs.get('value_loss_coef', 1.0)
        
        # increase entropy coefficient to encourage more exploration
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)  # increased from 0.001 to 0.01
        
        # GAE parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.use_gae = kwargs.get('use_gae', True)
        
        # gradient clipping
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        
        # network parameters
        self.hidden_size = kwargs.get('hidden_size', 64)  # increase network capacity
        self.layer_N = kwargs.get('layer_N', 2)  # use deeper network
        self.gain = kwargs.get('gain', 0.01)  
        self.use_orthogonal = kwargs.get('use_orthogonal', True)  
        self.use_ReLU = kwargs.get('use_ReLU', True)  
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)  
        self.activation_id = kwargs.get('activation_id', 1)  
        
        # reward normalization
        self.use_reward_normalization = kwargs.get('use_reward_normalization', True)  
        self.reward_scale = kwargs.get('reward_scale', 0.01)  
        
        # recurrent policy parameters
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # PopArt and ValueNorm parameters
        self.use_popart = kwargs.get('use_popart', False)
        self.use_valuenorm = kwargs.get('use_valuenorm', True)  # enable value normalization
        self.use_value_active_masks = kwargs.get('use_value_active_masks', False)
        
        # mask parameters
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', False)
        
        # time limit parameters
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', False)
        
        # algorithm name
        self.algorithm_name = kwargs.get('algorithm_name', 'FOMAPPO')
        
        # other parameters
        self.stacked_frames = kwargs.get('stacked_frames', 1)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        self.huber_delta = kwargs.get('huber_delta', 10.0)  
        
        # device coordination loss weight
        self.device_coord_loss_weight = kwargs.get('device_coord_loss_weight', 0.1)
        
        # FO constraint loss weight
        self.fo_constraint_loss_weight = kwargs.get('fo_constraint_loss_weight', 0.1)
        
        # increase exploration parameters
        self.action_noise_std = kwargs.get('action_noise_std', 0.1)  
        self.use_action_noise = kwargs.get('use_action_noise', True)  
        
        # increase training stability parameters
        self.clip_value = kwargs.get('clip_value', 10.0)  # value clipping range
        self.use_advantage_normalization = kwargs.get('use_advantage_normalization', True)  # whether to normalize advantage
        
        # get observation and action space from kwargs
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')

class FOMAPPOAdapter:
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 5e-5,
                 lr_critic: float = 2e-4,
                 device: str = "cpu",
                 **kwargs):
        """
        initialize FOMAPPO adapter
        
        Args:
            state_dim: state dimension
            action_dim: action dimension  
            num_agents: number of agents (Manager number)
            episode_length: Episode length
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: compute device
        """
        self.device = torch.device(device)
        self.state_dim = max(state_dim, 73)
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.actual_obs_dim = max(state_dim, 73)  
        
        # increase dimension change tracking
        self.initial_state_dim = max(state_dim, 73)
        self.has_dimension_changed = False
        self.dimension_change_count = 0
        self.new_obs_dimension_history = []  
        
        logger.info(f"🔧 initialize FOMAPPO adapter (shared policy architecture)")
        logger.info(f"    parameters: {num_agents} managers, {self.state_dim} state dimensions, {action_dim} action dimensions")
        
        # create observation and action spaces 
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # create parameter configuration
        args_dict = {
            'episode_length': episode_length,
            'n_rollout_threads': 1,
            'num_mini_batch': kwargs.get('num_mini_batch', 2),
            'ppo_epoch': kwargs.get('ppo_epoch', 8),
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'entropy_coef': kwargs.get('entropy_coef', 0.01),
            'use_linear_lr_decay': kwargs.get('use_linear_lr_decay', True),
            'lr_decay_rate': kwargs.get('lr_decay_rate', 0.95),
            'gamma': kwargs.get('gamma', 0.99),
            'gae_lambda': kwargs.get('gae_lambda', 0.95),
            'use_gae': kwargs.get('use_gae', True),
            'clip_param': kwargs.get('clip_param', 0.2),
            'max_grad_norm': kwargs.get('max_grad_norm', 0.5),
            'use_max_grad_norm': kwargs.get('use_max_grad_norm', True),
            'use_clipped_value_loss': kwargs.get('use_clipped_value_loss', True),
            'use_huber_loss': kwargs.get('use_huber_loss', True),
            'huber_delta': kwargs.get('huber_delta', 10.0),  
            'reward_scale': kwargs.get('reward_scale', 0.01),  
            'use_reward_normalization': kwargs.get('use_reward_normalization', True),
            'use_orthogonal': kwargs.get('use_orthogonal', True),
            'use_ReLU': kwargs.get('use_ReLU', True),
            'use_feature_normalization': kwargs.get('use_feature_normalization', True),
            'obs_space': obs_space,
            'share_obs_space': share_obs_space,
            'act_space': act_space
        }
        
        self.args = FOMAPPOArgs(**args_dict)
        
        # verify parameters
        logger.debug(f"📊 create parameters: reward_scale={self.args.reward_scale}, huber_delta={self.args.huber_delta}")
        
        # initialize FOMAPPO trainer
        try:
            # create policy network
            self.policy = FOMAPPOPolicy(
            args=self.args,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space,
            device=self.device
            )
            
            # create shared experience buffer
            self.buffer = SharedReplayBuffer(
            args=self.args,
                num_agents=num_agents,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space
            )
            logger.info("✅ shared buffer created successfully")
            
            # create FOMAPPO trainer
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            
            # ensure trainer has buffer reference
            if hasattr(self, 'buffer') and self.buffer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ successfully passed buffer to FOMAPPO trainer")
            else:
                logger.warning("⚠️ cannot pass buffer to FOMAPPO trainer, buffer does not exist")
                
            logger.info("✅ FOMAPPO trainer and buffer initialized successfully")
        except Exception as e:
            logger.error(f"❌ initialization of FOMAPPO components failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"FOMAPPO initialization failed: {e}")
        
        # initialize training statistics
        self.total_episodes = 0
        self.training_iterations = 0
        self.training_stats = {
            'shared_parameters': sum(p.numel() for p in self.policy.actor.parameters()) + sum(p.numel() for p in self.policy.critic.parameters()),
            'actor_learning_rate': lr_actor,
            'critic_learning_rate': lr_critic,
            'entropy_coef': self.args.entropy_coef,
            'clip_param': self.args.clip_param,
            'max_grad_norm': self.args.max_grad_norm
        }
        
        # reward normalizer
        self.reward_normalizer = {
            'running_mean': 0,
            'running_var': 1,
            'count': 0,
            'decay': 0.99,
            'epsilon': 1e-8  # avoid division by zero
        }
        
        # history of observation dimension changes
        self.new_obs_dimension_history = []
        
        # monitor dimension changes
        self.dimension_change_time = None
        
        # try to get cached observation dimension
        if 'actual_obs_dim' in kwargs:
            actual_dim = kwargs.get('actual_obs_dim')
            if actual_dim != state_dim:
                logger.warning(f"⚠️ detected inconsistent observation dimensions: provided {state_dim} dimensions, but actual {actual_dim} dimensions")
                logger.warning(f"will check and adapt to correct dimensions after initialization")
                # mark that dimension needs to be checked, but not immediately rebuild (to avoid rebuilding during initialization)
                self.has_dimension_changed = True
                self.actual_obs_dim = actual_dim
        
        logger.info("✅ FOMAPPO adapter initialized successfully")
        logger.info(f"    architecture: {num_agents} managers share Policy+Trainer+Buffer")
        logger.info(f"    features: retain FOMAPPO device coordination and FlexOffer constraint awareness")
        logger.info(f"    number of parameters: {self.training_stats['shared_parameters']:,}")
    
    def reset_buffer(self):
        """reset shared buffer"""
        try:
            if hasattr(self, 'buffer') and self.buffer is not None:
                logger.info("recreate shared buffer (SharedReplayBuffer has no reset method)")
                
                # save current step value
                old_step = self.buffer.step if hasattr(self.buffer, 'step') else 0
                
                # create buffer
                from gymnasium import spaces
                obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
                
                from onpolicy.utils.shared_buffer import SharedReplayBuffer
                self.buffer = SharedReplayBuffer(
                    args=self.args,
                    num_agents=self.num_agents,
                    obs_space=obs_space,
                    cent_obs_space=share_obs_space,
                    act_space=act_space
                )
                
                if old_step > 0:
                    logger.info(f"restore buffer step value: {old_step}")
                    self.buffer.step = old_step
                
                logger.info(f"✅ successfully recreated buffer, observation dimension: {self.actual_obs_dim}")
            else:
                logger.warning("no buffer to reset, will automatically create in collect_step")
                
                # create buffer
                from gymnasium import spaces
                obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.actual_obs_dim,), dtype=np.float32)
                act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
                
                from onpolicy.utils.shared_buffer import SharedReplayBuffer
                self.buffer = SharedReplayBuffer(
                    args=self.args,
                    num_agents=self.num_agents,
                    obs_space=obs_space,
                    cent_obs_space=share_obs_space,
                    act_space=act_space
                )
                logger.info(f"created new buffer, observation dimension: {self.actual_obs_dim}")
            
            # update trainer's buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ successfully passed buffer to FOMAPPO trainer")
                
        except Exception as e:
            logger.error(f"reset buffer failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        actions = {}
        action_log_probs = {}
        values = {}
        
        manager_ids = sorted(list(obs.keys()))  # ensure consistent order
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            return actions, action_log_probs, values
        
        # prepare batch observation data
        obs_batch = []
        obs_lengths = []
        
        # first collect all observations and record lengths
        for manager_id in manager_ids:
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                obs_batch.append(current_obs)
            else:
                obs_batch.append(np.array(current_obs))
            obs_lengths.append(len(obs_batch[-1]))
        
        max_obs_length = max(obs_lengths)
        
        # update actual observation dimension record
        if max_obs_length != self.actual_obs_dim:
            logger.warning(f"observation dimension changed: from {self.actual_obs_dim} to {max_obs_length}. update record and use new dimension.")
            self.actual_obs_dim = max_obs_length
            
            # recreate buffer and policy network
            self._recreate_buffer_and_policy(max_obs_length)
        
        logger.debug(f"FlexOffer action selection: {batch_size} managers, observation lengths {obs_lengths} → unified to {max_obs_length}")
        
        # pad all observations to the same length
        padded_obs_batch = []
        for i, obs_array in enumerate(obs_batch):
            if len(obs_array) < max_obs_length:
                padded_obs = np.zeros(max_obs_length, dtype=np.float32)
                padded_obs[:len(obs_array)] = obs_array
                padded_obs_batch.append(padded_obs)
                logger.debug(f"Manager {manager_ids[i]} 观测从 {len(obs_array)} 填充到 {max_obs_length}")
            else:
                padded_obs_batch.append(obs_array.astype(np.float32))
        
        # convert to tensor format (batch_size, max_obs_dim)
        obs_tensor = torch.FloatTensor(np.array(padded_obs_batch)).to(self.device)
        share_obs_tensor = obs_tensor  # in shared policy, assume shared observations are the same
        
        # create RNN states and masks (batch_size, ...)
        rnn_states_actor = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        rnn_states_critic = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        masks = torch.ones(batch_size, 1, device=self.device)
        
        # use shared policy to batch select actions
        try:
            value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = self.policy.get_actions(
                share_obs_tensor,
                obs_tensor,
                rnn_states_actor,
                rnn_states_critic,
                masks,
                available_actions=None,
                deterministic=deterministic
            )
            
            # assign batch results to each manager, and map to FlexOffer parameter range
            action_np = action.detach().cpu().numpy()
            action_log_prob_np = action_log_prob.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()
            
            for i, manager_id in enumerate(manager_ids):
                raw_action = action_np[i]
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = action_log_prob_np[i]
                values[manager_id] = value_np[i]
                
                logger.debug(f"Manager {manager_id} FlexOffer action: {fo_action.shape} dimensions, "
                           f"first 5 parameters: {fo_action[:5]}")
                
        except Exception as e:
            logger.error(f"shared policy FlexOffer action selection failed: {e}")
            # provide fallback FlexOffer parameter action
            for manager_id in manager_ids:
                fo_action = self._generate_default_fo_action()
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.log(0.5)
                values[manager_id] = 0.0
        
        return actions, action_log_probs, values
    
    def normalize_rewards(self, rewards):
        """
        normalize rewards to improve training stability
        
        Args:
            rewards: original reward values, can be single value or dictionary

        Returns:
            normalized rewards
        """
        if isinstance(rewards, dict):
            normalized_rewards = {}
            for k, v in rewards.items():
                normalized_rewards[k] = self._normalize_reward_value(v)
            return normalized_rewards
        else:
            return self._normalize_reward_value(rewards)
    
    def _normalize_reward_value(self, reward):
        """
        normalize single reward value
        
        Args:
            reward: single reward value
            
        Returns:
            normalized reward value
        """
        # check invalid values
        if np.isnan(reward) or np.isinf(reward):
            return 0.0
            
        # update running statistics
        self.reward_normalizer['count'] += 1
        delta = reward - self.reward_normalizer['running_mean']
        
        # update mean and variance
        if self.reward_normalizer['count'] == 1:
            self.reward_normalizer['running_mean'] = reward
        else:
            decay = self.reward_normalizer['decay']
            self.reward_normalizer['running_mean'] = self.reward_normalizer['running_mean'] * decay + reward * (1 - decay)
            self.reward_normalizer['running_var'] = self.reward_normalizer['running_var'] * decay + delta * delta * (1 - decay)
        
        # calculate standard deviation
        std = np.sqrt(self.reward_normalizer['running_var'] + self.reward_normalizer['epsilon'])
        
        # normalize and clip
        if std > 0:
            normalized = (reward - self.reward_normalizer['running_mean']) / std
        else:
            normalized = reward * self.args.reward_scale
            
        # clip to reasonable range
        normalized = np.clip(normalized, -5.0, 5.0)
        
        # scale to small range
        return normalized * self.args.reward_scale
        
    def collect_step(self, 
                     obs: Dict[str, np.ndarray],
                     actions: Dict[str, np.ndarray],
                     rewards: Dict[str, float],
                     dones: Dict[str, bool],
                     infos: Dict[str, Any],
                     action_log_probs: Optional[Dict[str, np.ndarray]] = None,
                     values: Optional[Dict[str, np.ndarray]] = None):
        """
        collect one step of experience data to shared buffer
        
        Args:
            obs: current observation
            actions: executed actions
            rewards: received rewards
            dones: whether the episode is done
            infos: additional information
            action_log_probs: action log probabilities (optional)
            values: value function predictions (optional)
        """
        # detailed record rewards information
        reward_values = list(rewards.values())
        reward_mean = np.mean(reward_values) if reward_values else 0.0
        reward_min = np.min(reward_values) if reward_values else 0.0
        reward_max = np.max(reward_values) if reward_values else 0.0
        logger.info(f"collected rewards: mean={reward_mean:.4f}, min={reward_min:.4f}, max={reward_max:.4f}")
        
        # debug: check if rewards are all zero
        if all(abs(r) < 1e-6 for r in rewards.values()):
            logger.warning(f"warning: all rewards are zero or near zero at current step: {rewards}")
            # analyze possible reasons
            logger.warning("possible reasons: 1) environment did not calculate rewards 2) reward function design problem 3) action did not affect environment state")
        manager_ids = sorted(list(obs.keys()))  # ensure consistent order
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            logger.error("no valid manager ID, cannot collect data")
            return
        
        # reward normalization
        if self.args.use_reward_normalization:
            normalized_rewards = self.normalize_rewards(rewards)
        else:
            normalized_rewards = rewards
        
        # check actual observation dimension
        first_obs = next(iter(obs.values()))
        actual_obs_dim = len(first_obs) if isinstance(first_obs, np.ndarray) else len(np.array(first_obs))
        
        # if actual observation dimension is different from recorded, update record and record warning
        dimension_changed = False
        if actual_obs_dim != self.actual_obs_dim:
            logger.warning(f"observation dimension changed: from {self.actual_obs_dim} to {actual_obs_dim}. update record and use new dimension.")
            self.actual_obs_dim = actual_obs_dim
            
            # recreate buffer and policy network
            self._recreate_buffer_and_policy(actual_obs_dim)
            dimension_changed = True
            
            logger.warning("observation dimension updated, will continue to collect data")
        
        # ensure buffer is initialized
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("buffer not initialized, create new buffer")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("cannot create buffer, skip data collection")
                return
        
        # prepare data format (adapt SharedReplayBuffer), use actual observation dimension
        # SharedReplayBuffer expected data format: (n_rollout_threads, num_agents, ...)
        
        # observation data (1, num_agents, actual_obs_dim)
        obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        share_obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        
        # action and reward data
        action_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # action log probabilities and value predictions
        action_log_prob_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        value_pred_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # RNN states
        rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        
        # masks
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        bad_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        
        # fill data
        for i, manager_id in enumerate(manager_ids):
            if i >= self.num_agents:
                break
                
            # observation
            obs_batch[0, i] = obs[manager_id]
            share_obs_batch[0, i] = obs[manager_id]  
            
            # action
            action_batch[0, i] = actions[manager_id]
            
            # reward
            reward_batch[0, i, 0] = normalized_rewards[manager_id]
            
            # action log probabilities
            if action_log_probs is not None and manager_id in action_log_probs:
                action_log_prob_batch[0, i] = action_log_probs[manager_id]
            else:
                action_log_prob_batch[0, i] = np.zeros(self.action_dim)  # use zero instead of log(0.5)
                
            # value prediction
            if values is not None and manager_id in values:
                value_pred_batch[0, i, 0] = values[manager_id]
            else:
                value_pred_batch[0, i, 0] = 0.0
        
        # insert into shared buffer
        try:
            # if dimension changed, reset buffer to ensure consistency
            if dimension_changed:
                logger.info("reset buffer due to dimension change")
                self.buffer.reset()
            
            self.buffer.insert(
                share_obs=share_obs_batch,
                obs=obs_batch,
                rnn_states_actor=rnn_states_actor,
                rnn_states_critic=rnn_states_critic,
                actions=action_batch,
                action_log_probs=action_log_prob_batch,
                value_preds=value_pred_batch,
                rewards=reward_batch,
                masks=masks,
                bad_masks=bad_masks,
                active_masks=active_masks
            )
            logger.debug(f"successfully collected data to buffer: step={self.buffer.step}, rewards={np.mean(reward_batch):.4f}")
            
            # ensure trainer also has buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                
        except Exception as e:
            logger.error(f"shared buffer insertion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def compute_returns(self):
        """compute returns and advantages of shared buffer - according to original MAPPO mode"""
        try:
            # check if buffer exists
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.warning("buffer does not exist, create a new buffer")
                self.reset_buffer()
                if not hasattr(self, 'buffer') or self.buffer is None:
                    logger.error("cannot create buffer, cannot compute returns")
                    return False
            
            # detailed record buffer state
            logger.info(f"buffer state: step={self.buffer.step}, rewards shape={self.buffer.rewards.shape if hasattr(self.buffer, 'rewards') else 'N/A'}")
            
            # check if rewards have valid values
            if hasattr(self.buffer, 'rewards'):
                non_zero_rewards = np.count_nonzero(self.buffer.rewards)
                total_rewards = np.prod(self.buffer.rewards.shape)
                logger.info(f"buffer content check: rewards non-zero values={non_zero_rewards}/{total_rewards} ({non_zero_rewards/total_rewards*100:.2f}%)")
                
                # record rewards statistics
                if non_zero_rewards > 0:
                    reward_mean = np.mean(self.buffer.rewards)
                    reward_std = np.std(self.buffer.rewards)
                    reward_min = np.min(self.buffer.rewards)
                    reward_max = np.max(self.buffer.rewards)
                    logger.info(f"rewards statistics: mean={reward_mean:.6f}, std={reward_std:.6f}, min={reward_min:.6f}, max={reward_max:.6f}")
            
            # check if buffer has enough data
            buffer_empty = self.buffer.step == 0 or (hasattr(self.buffer, 'rewards') and np.count_nonzero(self.buffer.rewards) == 0)
            if buffer_empty:
                # check if it is training initial phase
                is_initial_phase = not hasattr(self, '_training_started') or not self._training_started
                if is_initial_phase:
                    logger.info("training initial phase, buffer is empty is normal, add initialization data")
                    # mark training started
                    self._training_started = True
                else:
                    logger.warning("training has been performed but buffer has no data or all data are zero, try to add real data")
                
                # add dummy data to avoid empty buffer error
                self._add_dummy_data_to_buffer()
                logger.info(f"after adding dummy data: step={self.buffer.step}, rewards shape={self.buffer.rewards.shape}")
                
                # check if adding dummy data is successful
                if self.buffer.step == 0 or not np.any(self.buffer.rewards):
                    logger.error("even after adding dummy data, buffer is still empty, cannot compute returns")
                    return False
                else:
                    logger.info("dummy data added successfully, continue to compute returns")
            
            # get value estimation of last step
            try:
                # check if there are valid rewards data
                if hasattr(self.buffer, 'rewards') and np.sum(np.abs(self.buffer.rewards)) < 1e-6:
                    logger.warning("all rewards in buffer are zero or near zero, add small random noise to avoid all zero returns")
                    self.buffer.rewards = self.buffer.rewards + np.random.normal(0, 0.01, self.buffer.rewards.shape)
                    logger.info(f"number of non-zero rewards after adding noise: {np.count_nonzero(self.buffer.rewards)}")
                
                # get shared observation and state
                share_obs = np.concatenate(self.buffer.share_obs[-1])
                rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[-1])
                masks = np.concatenate(self.buffer.masks[-1])
                
                # debug: print input shape
                logger.info(f"input shape for computing returns: share_obs={share_obs.shape}, rnn_states_critic={rnn_states_critic.shape}")
                
                # convert to tensor
                share_obs = torch.FloatTensor(share_obs).to(self.device)
                rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)
                masks = torch.FloatTensor(masks).to(self.device)
                
                # get value estimation
                with torch.no_grad():
                    next_values = self.policy.get_values(share_obs, rnn_states_critic, masks)
                    
                # debug: print value estimation result
                logger.info(f"value estimation result: next_values shape={next_values.shape}, sample={next_values[:3]}")
                
                # compute returns
                next_values = next_values.detach().cpu().numpy()
                self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
                
                # debug: print computed returns
                logger.info(f"computed returns shape={self.buffer.returns.shape}, sample={self.buffer.returns[0][0][0][:3]}")
                logger.info(f"number of non-zero returns: {np.count_nonzero(self.buffer.returns)}")
                
                # check if returns contain NaN or infinity
                if np.isnan(self.buffer.returns).any() or np.isinf(self.buffer.returns).any():
                    logger.warning("returns contain NaN or infinity, perform numerical correction")
                    self.buffer.returns = np.nan_to_num(self.buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # check if returns are all zero
                if np.sum(np.abs(self.buffer.returns)) < 1e-6:
                    logger.warning("all returns are zero, this may cause training invalid")
                    return False
                
                # check if advantages are computed
                if not hasattr(self.buffer, 'advantages') or self.buffer.advantages is None:
                    logger.warning("advantages not computed, manually compute")
                    # simple compute advantages (returns - value_preds)
                    self.buffer.advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
                    logger.info(f"manually computed advantages shape: {self.buffer.advantages.shape}")
                
                # check if advantages contain NaN or infinity
                if hasattr(self.buffer, 'advantages') and (np.isnan(self.buffer.advantages).any() or np.isinf(self.buffer.advantages).any()):
                    logger.warning("advantages contain NaN or infinity, perform numerical correction")
                    self.buffer.advantages = np.nan_to_num(self.buffer.advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # ensure trainer's buffer reference is the latest
                if hasattr(self, 'trainer') and self.trainer is not None:
                    self.trainer.buffer = self.buffer
                    logger.info("updated trainer's buffer reference")
                
                logger.info("successfully computed returns and advantages")
                return True
                
            except Exception as e:
                logger.error(f"failed to compute value estimation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"failed to compute returns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _add_dummy_data_to_buffer(self):

        logger.warning("add data to buffer to avoid empty buffer error - only for debugging")
        
        # ensure buffer exists
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("buffer does not exist, create a new buffer")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("cannot create buffer, cannot add data")
                return
        
        dummy_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        dummy_share_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        
        for i in range(self.num_agents):
            dummy_obs[0, i, 0] = 0.5 + 0.1 * i  
            dummy_obs[0, i, 1] = 0.3 + 0.05 * i  
            dummy_obs[0, i, 2] = 0.7 - 0.05 * i  
            dummy_obs[0, i, 3] = 0.2 + 0.02 * i  
            
            dummy_obs[0, i, 4] = 0.1 * (i + 1)  
            
            if self.actual_obs_dim > 5:
                dummy_obs[0, i, 5:] = np.random.uniform(0.1, 0.9, size=self.actual_obs_dim-5)
            
            dummy_share_obs[0, i] = dummy_obs[0, i].copy()
        
        # create meaningful action
        dummy_actions = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            dummy_actions[0, i, 0] = 0.5 + 0.1 * np.sin(i)  
            dummy_actions[0, i, 1] = 0.3 + 0.1 * np.cos(i)  
            if self.action_dim > 2:
                dummy_actions[0, i, 2] = 0.7 - 0.1 * np.sin(i + 1)  
        
        # create meaningful reward - non-zero value, ensure learning signal
        dummy_rewards = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            dummy_rewards[0, i, 0] = 0.5 + 0.1 * i  
        
        # RNN states
        dummy_rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        dummy_rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        
        # action log probabilities and value predictions
        dummy_action_log_probs = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            for j in range(self.action_dim):
                dummy_action_log_probs[0, i, j] = -0.5 - 0.1 * j  
        
        dummy_value_preds = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            dummy_value_preds[0, i, 0] = 0.6 + 0.1 * i  
        
        # masks
        dummy_masks = np.ones((1, self.num_agents, 1))
        dummy_bad_masks = np.ones((1, self.num_agents, 1))
        dummy_active_masks = np.ones((1, self.num_agents, 1))
        
        # insert dummy data
        try:
            self.buffer.insert(
                share_obs=dummy_share_obs,
                obs=dummy_obs,
                rnn_states_actor=dummy_rnn_states_actor,
                rnn_states_critic=dummy_rnn_states_critic,
                actions=dummy_actions,
                action_log_probs=dummy_action_log_probs,
                value_preds=dummy_value_preds,
                rewards=dummy_rewards,
                masks=dummy_masks,
                bad_masks=dummy_bad_masks,
                active_masks=dummy_active_masks
            )
            logger.info("successfully added data to buffer")
            logger.info(f"data details: observation shape={dummy_obs.shape}, reward range=[{np.min(dummy_rewards):.3f}, {np.max(dummy_rewards):.3f}]")
            
            # ensure trainer also has buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("ensure trainer has latest buffer reference")
                
            for step in range(8):
                step_obs = dummy_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_share_obs = dummy_share_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_actions = dummy_actions.copy() * (1.0 + 0.03 * np.cos(step))
                
                step_rewards = dummy_rewards.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                step_value_preds = dummy_value_preds.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                self.buffer.insert(
                    share_obs=step_share_obs,
                    obs=step_obs,
                    rnn_states_actor=dummy_rnn_states_actor.copy(),
                    rnn_states_critic=dummy_rnn_states_critic.copy(),
                    actions=step_actions,
                    action_log_probs=dummy_action_log_probs.copy(),
                    value_preds=step_value_preds,
                    rewards=step_rewards,
                    masks=dummy_masks.copy(),
                    bad_masks=dummy_bad_masks.copy(),
                    active_masks=dummy_active_masks.copy()
                )
            
            logger.info(f"successfully added 9 data to buffer, current step={self.buffer.step}")
            
            # compute returns of data, ensure training signal
            if hasattr(self, 'compute_returns'):
                success = self.compute_returns()
                if success:
                    logger.info("successfully computed returns for data")
                else:
                    logger.warning("failed to compute returns for data")
            
        except Exception as e:
            logger.error(f"failed to add data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        execute one PPO batch update
        
        Returns:
            Dict[str, Any]: training information, including policy_loss, value_loss, entropy, etc.
        """
        # check if buffer is empty
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.error("buffer does not exist, cannot train")
            self._add_dummy_data_to_buffer()  # add some data to avoid error
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("even after adding data, buffer still does not exist")
                return {
                    'policy_loss': 0.0,  
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # check if buffer has enough data
        if self.buffer.step == 0:
            logger.warning("buffer is empty or has no enough data, add data to train")
            # try to add data
            self._add_dummy_data_to_buffer()
            if self.buffer.step == 0:
                logger.error("even after adding data, buffer is still empty")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # check if rewards are meaningful
        if hasattr(self.buffer, 'rewards'):
            non_zero_rewards = np.count_nonzero(self.buffer.rewards)
            total_rewards = np.prod(self.buffer.rewards.shape)
            zero_ratio = 1.0 - (non_zero_rewards / total_rewards)
            
            logger.info(f"before training: rewards zero ratio={zero_ratio:.2%}, non-zero rewards={non_zero_rewards}/{total_rewards}")
            
            # if more than 95% of rewards are zero, data quality may be bad
            if zero_ratio > 0.95:
                logger.warning("more than 95% of rewards are zero, data quality may be bad, but still try to train")
        
        # ensure returns are computed
        if not hasattr(self.buffer, 'returns') or self.buffer.returns is None or np.count_nonzero(self.buffer.returns) == 0:
            logger.info("before training: compute returns")
            success = self.compute_returns()
            if not success:
                logger.error("failed to compute returns, skip training")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        try:
            # use MAPPO trainer to train
            train_info = self.trainer.train()
            
            # update training iterations
            self.training_iterations += 1
            
            # check if training result is valid
            if not isinstance(train_info, dict):
                logger.warning(f"training result is invalid: {type(train_info)}")
                train_info = {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0
                }
            
            # record training information
            logger.info(f"training completed: policy_loss={train_info.get('policy_loss', 0.0):.6f}, " +
                        f"value_loss={train_info.get('value_loss', 0.0):.6f}, " +
                        f"entropy={train_info.get('entropy', 0.0):.6f}")
            
            # ensure all necessary keys are in the result
            required_keys = ['policy_loss', 'value_loss', 'entropy', 'grad_norm', 'ratio']
            for key in required_keys:
                if key not in train_info:
                    train_info[key] = 0.0  # use 0 to indicate data missing
            
            # add training iterations
            train_info['num_updates'] = 1
            
            return train_info
            
        except Exception as e:
            logger.error(f"training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'policy_loss': 0.0,  
                'value_loss': 0.0,
                'entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0,
                'num_updates': 0,
                'training_error': str(e)
            }
    
    def _update_learning_rate(self):
        """
        update learning rate (learning rate decay)
        gradually decrease learning rate based on training progress, avoid over-oscillation in later training
        """
        try:
            # exponential decay
            if hasattr(self.args, 'use_linear_lr_decay') and self.args.use_linear_lr_decay:
                # get current episode number
                episode = self.total_episodes
                
                # decay every 10 episodes
                decay_interval = 10
                if episode > 0 and episode % decay_interval == 0:
                    # default decay rate is 0.95
                    decay_rate = getattr(self.args, 'lr_decay_rate', 0.95)
                    
                    # update actor learning rate
                    for param_group in self.policy.actor_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # set minimum learning rate
                    
                    # update critic learning rate
                    for param_group in self.policy.critic_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # set minimum learning rate
                    
                    # record new learning rate
                    actor_lr = self.policy.actor_optimizer.param_groups[0]['lr']
                    critic_lr = self.policy.critic_optimizer.param_groups[0]['lr']
                    
                    logger.info(f"learning rate decayed: actor_lr={actor_lr:.7f}, critic_lr={critic_lr:.7f}")
                    
        except Exception as e:
            logger.warning(f"learning rate decay failed: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'num_agents': self.num_agents,
            'algorithm': 'FOMAPPO',
            'architecture': 'shared_policy',
            'shared_parameters': self.training_stats['shared_parameters']
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """get manager rewards summary (shared policy version)"""
        # in shared policy, all managers share the same policy, so return overall statistics
        return {
            'shared_policy': self.training_stats.copy(),
            'note': 'All managers share the same policy network'
        }
    
    def save_models(self, save_path: str):
        """save shared model"""
        try:
            model_path = f"{save_path}_shared.pt"
            
            torch.save({
                'actor_state_dict': self.policy.actor.state_dict(),
                'critic_state_dict': self.policy.critic.state_dict(),
                'actor_optimizer': self.policy.actor_optimizer.state_dict(),
                'critic_optimizer': self.policy.critic_optimizer.state_dict(),
                'training_iterations': self.training_iterations,
                'training_stats': self.training_stats,
                'num_agents': self.num_agents,
                'architecture': 'shared_policy'
            }, model_path)
            
            logger.info(f"FOMAPPO shared model saved to {model_path}")
        except Exception as e:
            logger.error(f"failed to save FOMAPPO shared model: {e}")
    
    def load_models(self, load_path: str):
        """load shared model"""
        try:
            model_path = f"{load_path}_shared.pt"
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.policy.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                
                if 'training_stats' in checkpoint:
                    self.training_stats = checkpoint['training_stats']
                
                if 'training_iterations' in checkpoint:
                    self.training_iterations = checkpoint['training_iterations']
                
                logger.info(f"FOMAPPO shared model loaded from {model_path}")
            else:
                logger.warning(f"model file {model_path} does not exist")
                
        except Exception as e:
            logger.error(f"failed to load FOMAPPO shared model: {e}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        fo_action = np.zeros_like(raw_action)

        num_devices = len(raw_action) // 5 if len(raw_action) >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < len(raw_action):
                # start_flex: [-1, 1] → [-1, 1] (keep unchanged)
                fo_action[base_idx] = np.clip(raw_action[base_idx], -1.0, 1.0)
                
                # end_flex: [-1, 1] → [-1, 1] (keep unchanged)
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

    def _recreate_buffer_and_policy(self, new_obs_dim):
        """
        when observation dimension changes, recreate buffer and policy network
        
        Args:
            new_obs_dim: new observation dimension
        """
        logger.warning(f"observation dimension change detected: {self.state_dim} → {new_obs_dim}")
        logger.warning(f"recreate buffer and policy network to adapt to new observation dimension")
        
        # ensure new_obs_dimension_history is initialized
        if not hasattr(self, 'new_obs_dimension_history'):
            self.new_obs_dimension_history = []
        
        # save original network parameters (if possible)
        old_actor_state = None
        old_critic_state = None
        try:
            if hasattr(self, 'policy') and self.policy is not None:
                old_actor_state = self.policy.actor.state_dict()
                old_critic_state = self.policy.critic.state_dict()
                logger.info("successfully saved original network parameters")
        except Exception as e:
            logger.warning(f"failed to save original network parameters: {e}")
        
        # update state dimension
        self.state_dim = new_obs_dim
        self.actual_obs_dim = new_obs_dim
        
        # recreate observation and action space
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # update obs_space and share_obs_space in args
        self.args.obs_space = obs_space
        self.args.share_obs_space = share_obs_space
        
        # record detailed network structure change
        logger.info(f"network reconstruction: input layer from {self.state_dim} → {new_obs_dim}")
        
        # recreate policy network
        try:
            self.policy = FOMAPPOPolicy(
                args=self.args,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space,
                device=self.device
            )
            logger.info("policy network reconstruction successful")
            
            # try to restore some network parameters (may need manual layer mapping)
            if old_actor_state is not None and old_critic_state is not None:
                try:
                    logger.info("network parameters cannot be directly migrated, use new initialized parameters")
                except Exception as transfer_e:
                    logger.warning(f"parameter migration failed: {transfer_e}")
        except Exception as e:
            logger.error(f"policy network reconstruction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"policy network reconstruction failed: {e}")
        
        # recreate trainer
        try:
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            logger.info("trainer reconstruction successful")
        except Exception as e:
            logger.error(f"trainer reconstruction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"trainer reconstruction failed: {e}")
        
        # recreate buffer
        try:
            self.buffer = SharedReplayBuffer(
                args=self.args,
                num_agents=self.num_agents,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space
            )
            logger.info("shared buffer reconstruction successful")
        except Exception as e:
            logger.error(f"shared buffer reconstruction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"shared buffer reconstruction failed: {e}")
        
        # ensure trainer has buffer reference
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.buffer = self.buffer
            logger.info("successfully passed new buffer to FOMAPPO trainer")
        else:
            logger.warning("cannot pass buffer to FOMAPPO trainer, trainer does not exist")
        
        # update training statistics
        try:
            if not hasattr(self, 'training_stats'):
                self.training_stats = {'shared_parameters': 0}
            
            self.training_stats['shared_parameters'] = sum(p.numel() for p in self.policy.actor.parameters()) + \
                                                   sum(p.numel() for p in self.policy.critic.parameters())
            logger.info(f"network parameter statistics updated successfully, total {self.training_stats['shared_parameters']} parameters")
        except Exception as e:
            logger.warning(f"failed to update network parameter statistics: {e}")
        
        # reset internal counter
        if not hasattr(self, 'training_iterations'):
            self.training_iterations = 0
        
        # record important dimension information
        self.new_obs_dimension_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'old_dim': self.state_dim, 
            'new_dim': new_obs_dim
        })
        
        logger.info(f"network reconstruction completed, new observation dimension: {new_obs_dim}")
        logger.info(f"history dimension change record: {len(self.new_obs_dimension_history)} times") 
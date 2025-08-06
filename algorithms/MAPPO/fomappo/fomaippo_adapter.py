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

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAIPPOArgs:
    """FOMAIPPO parameter configuration class - inherits MAPPO parameters and adds FlexOffer specific parameters"""
    
    def __init__(self, **kwargs):
        # ========== core PPO parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 1)
        self.ppo_epoch = kwargs.get('ppo_epoch', 4)
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # learning rate parameters
        self.lr = kwargs.get('lr_actor', 1e-4)  # reduce actor learning rate
        self.lr_actor = kwargs.get('lr_actor', 1e-4)  # reduce actor learning rate
        self.critic_lr = kwargs.get('lr_critic', 5e-4)  # reduce critic learning rate
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # PPO specific parameters
        self.clip_param = kwargs.get('clip_param', 0.1)  # reduce clip range for stability
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)  # reduce value loss weight
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.2)  # stronger gradient clipping
        self.huber_delta = kwargs.get('huber_delta', 1.0)  # reduce huber delta
        
        # GAE parameters
        self.use_gae = kwargs.get('use_gae', True)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        
        # this attribute is used to control whether to consider time limits in return calculation
        # in FlexOffer system, each episode has a clear time limit (24 hours), so set to True
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', True)
        
        # network parameters
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.layer_N = kwargs.get('layer_N', 2)
        self.use_orthogonal = kwargs.get('use_orthogonal', True)
        self.gain = kwargs.get('gain', 0.01)
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)
        self.activation_id = kwargs.get('activation_id', 1)
        self.use_ReLU = kwargs.get('use_ReLU', False)  # use Tanh activation function (False) or ReLU (True)
        self.stacked_frames = kwargs.get('stacked_frames', 1)  # stacked frames
        self.use_stacked_frames = kwargs.get('use_stacked_frames', False)  # whether to use stacked frames
        
        # RNN parameters
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # training options
        self.use_centralized_V = kwargs.get('use_centralized_V', True)
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        self.use_popart = kwargs.get('use_popart', False)
        self.use_valuenorm = kwargs.get('use_valuenorm', True)
        self.use_value_active_masks = kwargs.get('use_value_active_masks', True)
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', True)
        
        # algorithm name (for compatibility)
        self.algorithm_name = kwargs.get('algorithm_name', 'fomaippo')
        
        # policy sharing options
        self.share_policy = kwargs.get('share_policy', False)  # FOMAIPPO uses independent policy, set to False
        
        # ========== FOMAIPPO specific parameters ==========
        self.use_device_coordination = kwargs.get('use_device_coordination', True)
        self.device_coordination_weight = kwargs.get('device_coordination_weight', 0.1)
        self.fo_constraint_weight = kwargs.get('fo_constraint_weight', 0.2)
        self.use_manager_coordination = kwargs.get('use_manager_coordination', True)
        self.manager_coordination_weight = kwargs.get('manager_coordination_weight', 0.05)
        
        # network architecture specific parameters
        self.num_managers = kwargs.get('num_managers', 4)
        self.devices_per_manager = kwargs.get('devices_per_manager', 10)
        
        # get observation and action space from kwargs
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')

class FOMAIPPOAdapter:
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 episode_length: int = 24,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 5e-4,
                 device: str = "cpu",
                 **kwargs):
        """
        initialize FOMAIPPO adapter
        
        Args:
            state_dim: state dimension
            action_dim: action dimension  
            num_agents: number of agents (Manager number)
            episode_length: Episode length
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: computing device
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        
        logger.info(f"🔧 initialize FOMAIPPO adapter (separated policy architecture)")
        logger.info(f"    parameters: {num_agents} managers, {state_dim} state dimensions, {action_dim} action dimensions")
        
        # create observation and action space (compatible with original MAPPO format)
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # create parameter configuration
        self.args = FOMAIPPOArgs(
            episode_length=episode_length,
            n_rollout_threads=1,
            num_mini_batch=1,
            ppo_epoch=4,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            num_agents=num_agents,
            obs_space=obs_space,
            share_obs_space=share_obs_space,
            act_space=act_space,
            **kwargs
        )
        
        # ========== create independent Policy, Trainer, Buffer ==========
        # reference original MAPPO separated architecture
        
        # 1. create independent Policy for each Manager
        self.policies = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # use FOMAPPO policy (keep special features)
            policy = FOMAPPOPolicy(
                args=self.args,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space,
                device=self.device
            )
            
            self.policies.append(policy)
            logger.info(f"   ✅ create {manager_id} independent FOMAIPPO policy")
        
        # 2. create independent Trainer for each Manager
        self.trainers = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # use FOMAPPO trainer (keep special features)
            trainer = FOMAPPO(
                args=self.args,
                policy=self.policies[agent_id],
                device=self.device
            )
            
            self.trainers.append(trainer)
            logger.info(f"   ✅ create {manager_id} independent FOMAIPPO trainer")
        
        # 3. create independent Buffer for each Manager
        self.buffers = []
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            
            # use separated Buffer (original MAPPO separated architecture)
            buffer = SeparatedReplayBuffer(
                args=self.args,
                obs_space=obs_space,
                share_obs_space=share_obs_space,
                act_space=act_space
            )
            
            self.buffers.append(buffer)
            logger.info(f"   ✅ create {manager_id} independent SeparatedReplayBuffer")
        
        # training statistics
        self.total_episodes = 0
        self.training_iterations = 0
        
        # track training statistics for each Manager
        self.manager_stats = {}
        for agent_id in range(self.num_agents):
            manager_id = f"manager_{agent_id + 1}"
            self.manager_stats[manager_id] = {
                'episodes': 0,
                'total_reward': 0.0,
                'best_reward': float('-inf'),
                'avg_loss': 0.0,
                'training_updates': 0
            }
        
        logger.info("✅ FOMAIPPO adapter initialized")
        logger.info(f"    architecture: {num_agents} independent managers, each with independent Policy+Trainer+Buffer")
        logger.info(f"    features: keep FOMAPPO device coordination and FlexOffer constraint awareness")
    
    def reset_buffers(self):
        """reset all Manager's buffer"""
        for agent_id in range(self.num_agents):
            self.buffers[agent_id].step = 0
        logger.debug("all Manager's buffer has been reset")
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        actions = {}
        action_log_probs = {}
        values = {}
        
        manager_ids = list(obs.keys())
        
        for i, manager_id in enumerate(manager_ids):
            if i >= len(self.policies):
                logger.warning(f"Manager {manager_id} exceeds policy number, skip")
                continue
                
            policy = self.policies[i]
            current_obs = obs[manager_id]
            
            # ensure observation format is correct
            if isinstance(current_obs, np.ndarray):
                if len(current_obs.shape) == 1:
                    obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor(current_obs).to(self.device)
            else:
                obs_tensor = torch.FloatTensor([current_obs]).to(self.device)
            
            share_obs_tensor = obs_tensor  # in separated policy, assume shared observations are the same
            
            # create RNN state and mask
            batch_size = obs_tensor.shape[0]
            rnn_states_actor = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            rnn_states_critic = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            masks = torch.ones(batch_size, 1, device=self.device)
            
            # use policy to select action
            try:
                value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = policy.get_actions(
                    share_obs_tensor,
                    obs_tensor,
                    rnn_states_actor,
                    rnn_states_critic,
                    masks,
                    available_actions=None,
                    deterministic=deterministic
                )
                
                # convert to numpy format and map to FlexOffer parameter range
                raw_action = action.detach().cpu().numpy().squeeze()
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = action_log_prob.detach().cpu().numpy().squeeze()
                values[manager_id] = value.detach().cpu().numpy().squeeze()
                
                logger.debug(f"Manager {manager_id} FlexOffer independent action: {fo_action.shape} dimensions, "
                           f"first 5 parameters: {fo_action[:5]}")
                
            except Exception as e:
                logger.error(f"Manager {manager_id} FlexOffer action selection failed: {e}")
                # provide backup FlexOffer parameter action
                fo_action = self._generate_default_fo_action()
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.log(0.5)
                values[manager_id] = 0.0
        
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
        collect one step of experience data into each buffer
        
        Args:
            obs: current observation
            actions: executed actions
            rewards: received rewards
            dones: whether to end
            infos: additional information
            action_log_probs: action log probabilities (optional)
            values: value function prediction (optional)
        """
        manager_ids = list(obs.keys())
        
        for i, manager_id in enumerate(manager_ids):
            if i >= len(self.buffers):
                continue
                
            buffer = self.buffers[i]
            
            # prepare data format (adapt SeparatedReplayBuffer)
            current_obs = obs[manager_id].reshape(1, -1)  # (1, obs_dim)
            share_obs = current_obs  # in separated policy, assume shared observations are the same
            
            action = actions[manager_id].reshape(1, -1)  # (1, action_dim)
            
            # reward clipping and normalization
            raw_reward = rewards[manager_id]
            
            # check if reward is valid
            if np.isnan(raw_reward) or np.isinf(raw_reward):
                logger.warning(f"Manager {manager_id} reward is invalid ({raw_reward}), set to 0")
                raw_reward = 0.0
            
            # clip reward to prevent extreme values
            clipped_reward = np.clip(raw_reward, -10.0, 10.0)
            
            # slight reward scaling
            normalized_reward = clipped_reward * 0.1  # scale to smaller range
            
            reward = np.array([[normalized_reward]], dtype=np.float32)  # (1, 1)
            
            # RNN state 
            rnn_states_actor = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            
            # mask
            mask = np.array([[1.0]], dtype=np.float32)  # not end is 1
            bad_mask = np.array([[1.0]], dtype=np.float32)
            active_mask = np.array([[1.0]], dtype=np.float32)
            
            # action log probabilities and value prediction
            if action_log_probs is not None and manager_id in action_log_probs:
                action_log_prob = action_log_probs[manager_id].reshape(1, -1)
            else:
                action_log_prob = np.array([[np.log(0.5)]], dtype=np.float32)
                
            if values is not None and manager_id in values:
                value_pred = np.array([[values[manager_id]]], dtype=np.float32)
            else:
                value_pred = np.array([[0.0]], dtype=np.float32)
            
            # insert into buffer
            try:
                buffer.insert(
                    share_obs=share_obs,
                    obs=current_obs,
                    rnn_states=rnn_states_actor,
                    rnn_states_critic=rnn_states_critic,
                    actions=action,
                    action_log_probs=action_log_prob,
                    value_preds=value_pred,
                    rewards=reward,
                    masks=mask,
                    bad_masks=bad_mask,
                    active_masks=active_mask
                )
            except Exception as e:
                logger.error(f"Manager {manager_id} buffer insertion failed: {e}")
    
    def compute_returns(self):
        """compute returns and advantages for all Managers"""
        for agent_id in range(self.num_agents):
            buffer = self.buffers[agent_id]
            
            try:
                # use last value_pred as next_value, if invalid, use 0
                if hasattr(buffer, 'value_preds') and len(buffer.value_preds) > 0:
                    last_value = buffer.value_preds[-1]
                    
                    # check if last_value is valid
                    if isinstance(last_value, np.ndarray):
                        if np.isnan(last_value).any() or np.isinf(last_value).any():
                            next_value = np.zeros_like(last_value)
                        else:
                            next_value = last_value
                    else:
                        next_value = np.zeros((1, 1), dtype=np.float32)
                else:
                    next_value = np.zeros((1, 1), dtype=np.float32)
                
                # compute GAE
                buffer.compute_returns(
                    next_value=next_value,
                    value_normalizer=self.trainers[agent_id].value_normalizer
                )
                
                # verify calculation results
                if hasattr(buffer, 'returns'):
                    returns_has_nan = np.isnan(buffer.returns).any() if isinstance(buffer.returns, np.ndarray) else torch.isnan(buffer.returns).any()
                    if returns_has_nan:
                        logger.warning(f"Manager {agent_id} GAE calculation contains NaN, use safe value instead")
                        if isinstance(buffer.returns, np.ndarray):
                            buffer.returns = np.nan_to_num(buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                        else:
                            buffer.returns = torch.nan_to_num(buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
            except Exception as e:
                logger.error(f"Manager {agent_id} GAE calculation failed: {e}")
                # provide backup: if GAE calculation fails completely, create safe returns
                try:
                    if hasattr(buffer, 'rewards'):
                        # use accumulated reward as returns
                        buffer.returns = np.cumsum(buffer.rewards, axis=0)
                        logger.warning(f"Manager {agent_id} use accumulated reward as returns")
                except Exception as backup_error:
                    logger.error(f"Manager {agent_id} backup returns creation also failed: {backup_error}")
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        train all Managers once
        
        Returns:
            training information dictionary
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_ratio = 0.0
        update_count = 0
        
        for agent_id in range(self.num_agents):
            trainer = self.trainers[agent_id]
            buffer = self.buffers[agent_id]
            
            try:
                # check buffer data quality
                returns = buffer.returns[:-1]
                value_preds = buffer.value_preds[:-1]
                
                # check data quality of returns and value_preds
                returns_has_nan = np.isnan(returns).any() if isinstance(returns, np.ndarray) else torch.isnan(returns).any()
                value_preds_has_nan = np.isnan(value_preds).any() if isinstance(value_preds, np.ndarray) else torch.isnan(value_preds).any()
                
                if returns_has_nan or value_preds_has_nan:
                    logger.warning(f"Manager {agent_id} buffer data quality problem:")
                    logger.warning(f"  returns has NaN: {returns_has_nan}")
                    logger.warning(f"  value_preds has NaN: {value_preds_has_nan}")
                    
                    # replace NaN/Inf with safe values
                    if isinstance(returns, np.ndarray):
                        returns = np.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if isinstance(value_preds, np.ndarray):
                        value_preds = np.nan_to_num(value_preds, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        value_preds = torch.nan_to_num(value_preds, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    logger.warning(f"  Manager {agent_id} NaN/Inf data has been fixed")
                
                # compute advantages
                advantages = returns - value_preds
                
                # final safe check: if advantages still have problems, provide backup
                if isinstance(advantages, np.ndarray):
                    if np.isnan(advantages).any() or np.isinf(advantages).any():
                        logger.warning(f"Manager {agent_id} advantages still contains NaN/Inf, use zero advantages")
                        advantages = np.zeros_like(advantages)
                    # convert to torch tensor
                    advantages = torch.from_numpy(advantages).float()
                elif torch.is_tensor(advantages):
                    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                        logger.warning(f"Manager {agent_id} advantages still contains NaN/Inf, use zero advantages")
                        advantages = torch.zeros_like(advantages)
                else:
                    logger.warning(f"Manager {agent_id} advantages data type unknown ({type(advantages)}), use zero advantages")
                    advantages = torch.zeros((len(buffer.returns)-1, 1), dtype=torch.float32)
                
                # safe standardization of advantages (ensure it is a torch tensor)
                if not torch.is_tensor(advantages):
                    advantages = torch.from_numpy(advantages).float()
                
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                
                if adv_std > 1e-8 and not torch.isnan(adv_std) and not torch.isinf(adv_std):
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    # if standard deviation is too small or invalid, skip standardization
                    logger.warning(f"Manager {agent_id} advantages standard deviation is invalid ({adv_std}), skip standardization")
                    advantages = advantages - adv_mean
                
                # clip advantages to prevent extreme values
                advantages = torch.clamp(advantages, -10.0, 10.0)
                
                # execute PPO update
                train_info = trainer.train(buffer)
                
                # accumulate training statistics
                if isinstance(train_info, dict):
                    total_policy_loss += train_info.get('policy_loss', 0.0)
                    total_value_loss += train_info.get('value_loss', 0.0)
                    total_entropy += train_info.get('dist_entropy', 0.0)
                    total_ratio += train_info.get('ratio', 1.0)
                    update_count += 1
                    
                    # update Manager statistics
                    manager_id = f"manager_{agent_id + 1}"
                    self.manager_stats[manager_id]['training_updates'] += 1
                    self.manager_stats[manager_id]['avg_loss'] = train_info.get('policy_loss', 0.0)
                
                # reset Buffer
                buffer.after_update()
                
            except Exception as e:
                logger.error(f"Manager {agent_id} training failed: {e}")
                continue
        
        # compute average training statistics
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy = total_entropy / update_count
            avg_ratio = total_ratio / update_count
        else:
            avg_policy_loss = avg_value_loss = avg_entropy = avg_ratio = 0.0
        
        self.training_iterations += 1
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'ratio': avg_ratio,
            'training_iterations': self.training_iterations,
            'updated_managers': update_count
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'num_managers': self.num_agents,
            'algorithm': 'FOMAIPPO',
            'architecture': 'separated_policy'
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """get Manager reward summary"""
        return self.manager_stats.copy()
    
    def save_models(self, save_path: str):
        """save all Manager's models"""
        try:
            for agent_id in range(self.num_agents):
                manager_id = f"manager_{agent_id + 1}"
                model_path = f"{save_path}_{manager_id}.pt"
                
                torch.save({
                    'actor_state_dict': self.policies[agent_id].actor.state_dict(),
                    'critic_state_dict': self.policies[agent_id].critic.state_dict(),
                    'actor_optimizer': self.policies[agent_id].actor_optimizer.state_dict(),
                    'critic_optimizer': self.policies[agent_id].critic_optimizer.state_dict(),
                    'training_iterations': self.training_iterations,
                    'manager_stats': self.manager_stats[manager_id]
                }, model_path)
                
            logger.info(f"FOMAIPPO models have been saved to {save_path}_manager_*.pt")
        except Exception as e:
            logger.error(f"save FOMAIPPO models failed: {e}")
    
    def load_models(self, load_path: str):
        """load all Manager's models"""
        try:
            for agent_id in range(self.num_agents):
                manager_id = f"manager_{agent_id + 1}"
                model_path = f"{load_path}_{manager_id}.pt"
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.policies[agent_id].actor.load_state_dict(checkpoint['actor_state_dict'])
                    self.policies[agent_id].critic.load_state_dict(checkpoint['critic_state_dict'])
                    self.policies[agent_id].actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.policies[agent_id].critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                    
                    if 'manager_stats' in checkpoint:
                        self.manager_stats[manager_id] = checkpoint['manager_stats']
                    
                    logger.info(f"loaded {manager_id} model")
                else:
                    logger.warning(f"model file {model_path} does not exist")
                    
            logger.info(f"FOMAIPPO models have been loaded from {load_path}_manager_*.pt")
        except Exception as e:
            logger.error(f"load FOMAIPPO models failed: {e}")
    
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
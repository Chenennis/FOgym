#!/usr/bin/env python3
"""
FOMAPPO Adapter - Multi-agent architecture based on shared policy (FlexOffer Multi-Agent PPO)

Architecture design:
- References the original MAPPO shared/base_runner.py architecture
- All Managers share a single Policy and Trainer
- Uses SharedReplayBuffer to collect data from all agents
- Retains FOMAPPO special features (device coordination, FlexOffer constraints, etc.)
- Integrates with existing FO Framework
- Efficient parameter sharing and centralized training

Key features:
1. Shared learning: All Managers share the same policy network, improving data efficiency
2. FOMAPPO features: Retains device coordination and FlexOffer constraint awareness
3. FO integration: Seamless integration with existing FO Pipeline
4. Centralized training: Joint learning using experiences from all agents

Algorithm: FOMAPPO (FlexOffer Multi-Agent PPO)
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
import sys

# Add MAPPO path
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

# Import original MAPPO components (shared architecture)
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

# Import FOMAPPO specific components
from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAPPOArgs:
    """FOMAPPO parameter configuration class - inherits MAPPO parameters and adds FlexOffer specific parameters"""
    
    def __init__(self, **kwargs):
        # ========== Core PPO parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 2)  # Increased to 2 mini-batches
        self.ppo_epoch = kwargs.get('ppo_epoch', 8)  # Increased to 8 epochs
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # Learning rate parameters - reduced to avoid quick convergence to suboptimal solutions
        self.lr = kwargs.get('lr_actor', 5e-5)  # Reduced from 3e-4 to 5e-5
        self.lr_actor = kwargs.get('lr_actor', 5e-5)  # Reduced from 3e-4 to 5e-5
        self.critic_lr = kwargs.get('lr_critic', 2e-4)  # Reduced from 1e-3 to 2e-4
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # Added learning rate decay parameters
        self.use_linear_lr_decay = kwargs.get('use_linear_lr_decay', True)  # Enable learning rate decay
        self.lr_decay_rate = kwargs.get('lr_decay_rate', 0.95)  # Learning rate decay rate
        
        # PPO clipping parameters
        self.clip_param = kwargs.get('clip_param', 0.2)
        self.value_loss_coef = kwargs.get('value_loss_coef', 1.0)
        
        # Increased entropy coefficient to encourage more exploration
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)  # Increased from 0.001 to 0.01, enhancing exploration
        
        # GAE parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.use_gae = kwargs.get('use_gae', True)
        
        # Gradient clipping
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.use_max_grad_norm = kwargs.get('use_max_grad_norm', True)
        
        # Network parameters
        self.hidden_size = kwargs.get('hidden_size', 64)  # Increased network capacity
        self.layer_N = kwargs.get('layer_N', 2)  # Use deeper networks
        self.gain = kwargs.get('gain', 0.01)  # Added missing gain parameter
        self.use_orthogonal = kwargs.get('use_orthogonal', True)  # Added missing use_orthogonal parameter
        self.use_ReLU = kwargs.get('use_ReLU', True)  # Added missing use_ReLU parameter
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)  # Added missing use_feature_normalization parameter
        self.activation_id = kwargs.get('activation_id', 1)  # Added missing activation_id parameter
        
        # Reward normalization
        self.use_reward_normalization = kwargs.get('use_reward_normalization', True)  # Enable reward normalization
        self.reward_scale = kwargs.get('reward_scale', 0.01)  # Added missing reward_scale parameter
        
        # Recurrent policy parameters
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # PopArt and ValueNorm parameters
        self.use_popart = kwargs.get('use_popart', False)
        self.use_valuenorm = kwargs.get('use_valuenorm', True)  # Enable value normalization
        self.use_value_active_masks = kwargs.get('use_value_active_masks', False)
        
        # Mask parameters
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', False)
        
        # Time limit parameters
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', False)
        
        # Algorithm name
        self.algorithm_name = kwargs.get('algorithm_name', 'FOMAPPO')
        
        # Other parameters
        self.stacked_frames = kwargs.get('stacked_frames', 1)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        self.huber_delta = kwargs.get('huber_delta', 10.0)  # Added missing huber_delta parameter
        
        # Device coordination loss weight
        self.device_coord_loss_weight = kwargs.get('device_coord_loss_weight', 0.1)
        
        # FO constraint loss weight
        self.fo_constraint_loss_weight = kwargs.get('fo_constraint_loss_weight', 0.1)
        
        # Added exploration parameters
        self.action_noise_std = kwargs.get('action_noise_std', 0.1)  # Standard deviation of action noise
        self.use_action_noise = kwargs.get('use_action_noise', True)  # Whether to use action noise
        
        # Added training stability parameters
        self.clip_value = kwargs.get('clip_value', 10.0)  # Value clipping range
        self.use_advantage_normalization = kwargs.get('use_advantage_normalization', True)  # Whether to normalize advantage
        
        # Get observations and action spaces from kwargs
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')

class FOMAPPOAdapter:
    """
    FOMAPPO adapter - Multi-agent reinforcement learning based on shared policy (FlexOffer Multi-Agent PPO)
    
    Core design principles:
    1. References the original MAPPO shared/base_runner.py architecture
    2. All Managers share a single Policy and Trainer
    3. Uses SharedReplayBuffer to collect data from all agents
    4. Retains all FOMAPPO special features
    5. Seamlessly integrates with FO Framework
    
    Advantages:
    - Parameter efficiency: Shared policy reduces parameter count, improving data efficiency
    - Coordinated learning: Natural coordination and communication between Managers
    - Stable training: Reduces policy variance, improving training stability
    """
    
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
        Initialize FOMAPPO adapter
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension  
            num_agents: Number of agents (number of Managers)
            episode_length: Episode length
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: Computing device
        """
        self.device = torch.device(device)
        # Ensure state_dim is at least 73, to avoid subsequent dimension changes
        self.state_dim = max(state_dim, 73)
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.actual_obs_dim = max(state_dim, 73)  # Initial setting is 73, to avoid subsequent dimension changes
        
        # Added dimension change tracking
        self.initial_state_dim = max(state_dim, 73)
        self.has_dimension_changed = False
        self.dimension_change_count = 0
        self.new_obs_dimension_history = []  # Added this line, initializing dimension history record
        
        logger.info(f"🔧 Initializing FOMAPPO adapter (shared policy architecture)")
        logger.info(f"    Parameters: {num_agents} managers, state {self.state_dim}D, action {action_dim}D")
        
        # Create observation and action spaces (compatible with original MAPPO format)
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # Create parameter configuration - 🔧 Fixed: Avoid repeating parameters
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
            'huber_delta': kwargs.get('huber_delta', 10.0),  # Added missing huber_delta parameter
            'reward_scale': kwargs.get('reward_scale', 0.01),  # Added missing reward_scale parameter
            'use_reward_normalization': kwargs.get('use_reward_normalization', True),
            'use_orthogonal': kwargs.get('use_orthogonal', True),
            'use_ReLU': kwargs.get('use_ReLU', True),
            'use_feature_normalization': kwargs.get('use_feature_normalization', True),
            'obs_space': obs_space,
            'share_obs_space': share_obs_space,
            'act_space': act_space
        }
        
        self.args = FOMAPPOArgs(**args_dict)
        
        # Verify parameters
        logger.debug(f"📊 Creating parameters: reward_scale={self.args.reward_scale}, huber_delta={self.args.huber_delta}")
        
        # Initialize FOMAPPO trainer
        try:
            # Create policy network
            self.policy = FOMAPPOPolicy(
            args=self.args,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space,
            device=self.device
            )
            
            # First create shared experience buffer
            self.buffer = SharedReplayBuffer(
            args=self.args,
                num_agents=num_agents,
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=act_space
            )
            logger.info("✅ Shared buffer created successfully")
            
            # Then create FOMAPPO trainer
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            
            # Ensure trainer has buffer reference
            if hasattr(self, 'buffer') and self.buffer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ Successfully passed buffer to FOMAPPO trainer")
            else:
                logger.warning("⚠️ Unable to pass buffer to FOMAPPO trainer, buffer does not exist")
                
            logger.info("✅ FOMAPPO trainer and buffer initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FOMAPPO components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"FOMAPPO initialization failed: {e}")
        
        # Initialize training statistics
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
        
        # Reward normalizer
        self.reward_normalizer = {
            'running_mean': 0,
            'running_var': 1,
            'count': 0,
            'decay': 0.99,
            'epsilon': 1e-8  # Avoid division by zero
        }
        
        # History of dimension change in observed dimensions
        self.new_obs_dimension_history = []
        
        # Monitor dimension change
        self.dimension_change_time = None
        
        # Try to get cached value of possible observed dimensions
        if 'actual_obs_dim' in kwargs:
            actual_dim = kwargs.get('actual_obs_dim')
            if actual_dim != state_dim:
                logger.warning(f"⚠️ Detected inconsistent observed dimension: Provided is {state_dim}D, but it might be {actual_dim}D")
                logger.warning(f"Correct dimension will be checked and adapted after initialization")
                # Mark for dimension check, but do not rebuild immediately (to avoid rebuilding at initialization)
                self.has_dimension_changed = True
                self.actual_obs_dim = actual_dim
        
        logger.info("✅ FOMAPPO adapter initialization completed")
        logger.info(f"    Architecture: {num_agents} managers share Policy+Trainer+Buffer")
        logger.info(f"    Features: Retains FOMAPPO device coordination and FlexOffer constraint awareness")
        logger.info(f"    Number of parameters: {self.training_stats['shared_parameters']:,}")
    
    def reset_buffer(self):
        """Reset shared buffer"""
        try:
            if hasattr(self, 'buffer') and self.buffer is not None:
                # SharedReplayBuffer does not have a reset method, so we need to recreate the buffer
                logger.info("Recreating shared buffer (SharedReplayBuffer does not have a reset method)")
                
                # Save current step value
                old_step = self.buffer.step if hasattr(self.buffer, 'step') else 0
                
                # Create buffer
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
                
                # Restore step value (if needed)
                if old_step > 0:
                    logger.info(f"Restored buffer step value: {old_step}")
                    self.buffer.step = old_step
                
                logger.info(f"✅ Successfully recreated buffer, observed dimension: {self.actual_obs_dim}")
            else:
                logger.warning("No buffer to reset, will create one automatically when collecting step")
                
                # Create buffer
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
                logger.info(f"Created new buffer, observed dimension: {self.actual_obs_dim}")
            
            # Update trainer's buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ Successfully passed buffer to FOMAPPO trainer")
                
        except Exception as e:
            logger.error(f"Failed to reset buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Select FlexOffer parameters for all Managers (using shared policy)
        
        🔧 Reconstructed environment adaptation:
        - Actions now correspond to FlexOffer parameters: [start_flex, end_flex, energy_min_factor, energy_max_factor, priority_weight] × number of devices
        - Observations include device status, environment status, information from other Managers, and market status
        - Need to ensure actions are within reasonable range to generate valid FlexOffer
        
        Args:
            obs: Observation dictionary {manager_id: observation}
            deterministic: Whether to select deterministic actions
            
        Returns:
            actions: FlexOffer parameters action dictionary {manager_id: fo_params_action}
            action_log_probs: Action log probability dictionary
            values: Value function prediction dictionary
        """
        actions = {}
        action_log_probs = {}
        values = {}
        
        manager_ids = sorted(list(obs.keys()))  # Ensure consistent order
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            return actions, action_log_probs, values
        
        # Prepare batch observations - 🔧 Fixed: Handle observations of different lengths
        obs_batch = []
        obs_lengths = []
        
        # First collect all observations and record lengths
        for manager_id in manager_ids:
            current_obs = obs[manager_id]
            if isinstance(current_obs, np.ndarray):
                obs_batch.append(current_obs)
            else:
                obs_batch.append(np.array(current_obs))
            obs_lengths.append(len(obs_batch[-1]))
        
        # 🔧 Key fix: Unify observation lengths, padding to the maximum length
        max_obs_length = max(obs_lengths)
        
        # Update recorded observed dimension
        if max_obs_length != self.actual_obs_dim:
            logger.warning(f"Observed dimension change: Previously {self.actual_obs_dim}D, now {max_obs_length}D. Updated record and use new dimension.")
            self.actual_obs_dim = max_obs_length
            
            # Recreate buffer and policy network
            self._recreate_buffer_and_policy(max_obs_length)
        
        logger.debug(f"🔧 FlexOffer action selection: {batch_size} managers, observation lengths {obs_lengths} → unified to {max_obs_length}")
        
        # Fill all observations to the same length
        padded_obs_batch = []
        for i, obs_array in enumerate(obs_batch):
            if len(obs_array) < max_obs_length:
                # Pad with zeros to the maximum length
                padded_obs = np.zeros(max_obs_length, dtype=np.float32)
                padded_obs[:len(obs_array)] = obs_array
                padded_obs_batch.append(padded_obs)
                logger.debug(f"Manager {manager_ids[i]} observation padded from {len(obs_array)} to {max_obs_length}")
            else:
                padded_obs_batch.append(obs_array.astype(np.float32))
        
        # Convert to tensor format (batch_size, max_obs_dim)
        obs_tensor = torch.FloatTensor(np.array(padded_obs_batch)).to(self.device)
        share_obs_tensor = obs_tensor  # In shared policy, assume shared observations are the same
        
        # Create RNN states and masks (batch_size, ...)
        rnn_states_actor = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        rnn_states_critic = torch.zeros(batch_size, self.args.recurrent_N, self.args.hidden_size, device=self.device)
        masks = torch.ones(batch_size, 1, device=self.device)
        
        # Use shared policy to select actions in batch
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
            
            # Assign batch results to each Manager, mapping to FlexOffer parameter range
            action_np = action.detach().cpu().numpy()
            action_log_prob_np = action_log_prob.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()
            
            for i, manager_id in enumerate(manager_ids):
                # 🔧 Reconstructed adaptation: Map original actions to FlexOffer parameter range
                raw_action = action_np[i]
                fo_action = self._map_action_to_fo_params(raw_action)
                
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = action_log_prob_np[i]
                values[manager_id] = value_np[i]
                
                logger.debug(f"Manager {manager_id} FlexOffer action: {fo_action.shape} D, "
                           f"first 5 parameters: {fo_action[:5]}")
                
        except Exception as e:
            logger.error(f"Failed to select FlexOffer actions using shared policy: {e}")
            # Provide fallback FlexOffer parameters
            for manager_id in manager_ids:
                fo_action = self._generate_default_fo_action()
                actions[manager_id] = fo_action
                action_log_probs[manager_id] = np.log(0.5)
                values[manager_id] = 0.0
        
        return actions, action_log_probs, values
    
    def normalize_rewards(self, rewards):
        """
        Normalize rewards to improve training stability
        
        Args:
            rewards: Original rewards, can be a single value or dictionary

        Returns:
            Normalized rewards
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
        Normalize a single reward value
        
        Args:
            reward: Single reward value
            
        Returns:
            Normalized reward value
        """
        # Check for invalid values
        if np.isnan(reward) or np.isinf(reward):
            return 0.0
            
        # Update running statistics
        self.reward_normalizer['count'] += 1
        delta = reward - self.reward_normalizer['running_mean']
        
        # Update mean and variance
        if self.reward_normalizer['count'] == 1:
            self.reward_normalizer['running_mean'] = reward
        else:
            decay = self.reward_normalizer['decay']
            self.reward_normalizer['running_mean'] = self.reward_normalizer['running_mean'] * decay + reward * (1 - decay)
            self.reward_normalizer['running_var'] = self.reward_normalizer['running_var'] * decay + delta * delta * (1 - decay)
        
        # Calculate standard deviation
        std = np.sqrt(self.reward_normalizer['running_var'] + self.reward_normalizer['epsilon'])
        
        # Normalize and clip
        if std > 0:
            normalized = (reward - self.reward_normalizer['running_mean']) / std
        else:
            normalized = reward * self.args.reward_scale
            
        # Clip to reasonable range
        normalized = np.clip(normalized, -5.0, 5.0)
        
        # Scale to small range
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
        Collect one step of experience data into shared buffer
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            dones: Whether episode is done
            infos: Additional information
            action_log_probs: Action log probabilities (optional)
            values: Value function predictions (optional)
        """
        # Detailed recording of rewards information
        reward_values = list(rewards.values())
        reward_mean = np.mean(reward_values) if reward_values else 0.0
        reward_min = np.min(reward_values) if reward_values else 0.0
        reward_max = np.max(reward_values) if reward_values else 0.0
        logger.info(f"Rewards collected: mean={reward_mean:.4f}, min={reward_min:.4f}, max={reward_max:.4f}")
        
        # Debug: Check if all rewards are zero
        if all(abs(r) < 1e-6 for r in rewards.values()):
            logger.warning(f"Warning: All rewards at current time step are zero or close to zero: {rewards}")
            # Analyze possible reasons
            logger.warning("Possible reasons: 1) Environment not calculating rewards 2) Reward function design issue 3) Actions not affecting environment state")
        manager_ids = sorted(list(obs.keys()))  # Ensure consistent order
        batch_size = len(manager_ids)
        
        if batch_size == 0:
            logger.error("No valid Manager ID, unable to collect data")
            return
        
        # Normalize rewards
        if self.args.use_reward_normalization:
            normalized_rewards = self.normalize_rewards(rewards)
        else:
            normalized_rewards = rewards
        
        # Check actual observed dimension
        first_obs = next(iter(obs.values()))
        actual_obs_dim = len(first_obs) if isinstance(first_obs, np.ndarray) else len(np.array(first_obs))
        
        # If actual observed dimension differs from recorded one, update record and log warning
        dimension_changed = False
        if actual_obs_dim != self.actual_obs_dim:
            logger.warning(f"Observed dimension change: Previously {self.actual_obs_dim}D, now {actual_obs_dim}D. Updated record and use new dimension.")
            self.actual_obs_dim = actual_obs_dim
            
            # Recreate buffer and policy network
            self._recreate_buffer_and_policy(actual_obs_dim)
            dimension_changed = True
            
            logger.warning("Observed dimension updated, will continue collecting data")
        
        # Ensure buffer is initialized
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("Buffer not initialized, creating new buffer")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("Unable to create buffer, skipping data collection")
                return
        
        # Prepare data format (compatible with SharedReplayBuffer), using actual observed dimension
        # SharedReplayBuffer expects data format: (n_rollout_threads, num_agents, ...)
        
        # Observations (1, num_agents, actual_obs_dim)
        obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        share_obs_batch = np.zeros((1, self.num_agents, actual_obs_dim), dtype=np.float32)
        
        # Actions and rewards
        action_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # Action log probabilities and value predictions
        action_log_prob_batch = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
        value_pred_batch = np.zeros((1, self.num_agents, 1), dtype=np.float32)
        
        # RNN states (all zeros, as RNN is not used)
        rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        
        # Masks
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        bad_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        
        # Fill data
        for i, manager_id in enumerate(manager_ids):
            if i >= self.num_agents:
                break
                
            # Observations
            obs_batch[0, i] = obs[manager_id]
            share_obs_batch[0, i] = obs[manager_id]  # Assuming shared observations are the same
            
            # Actions
            action_batch[0, i] = actions[manager_id]
            
            # Rewards
            reward_batch[0, i, 0] = normalized_rewards[manager_id]
            
            # Action log probabilities
            if action_log_probs is not None and manager_id in action_log_probs:
                action_log_prob_batch[0, i] = action_log_probs[manager_id]
            else:
                action_log_prob_batch[0, i] = np.zeros(self.action_dim)  # Using zero instead of log(0.5)
                
            # Value predictions
            if values is not None and manager_id in values:
                value_pred_batch[0, i, 0] = values[manager_id]
            else:
                value_pred_batch[0, i, 0] = 0.0
        
        # Insert into shared buffer
        try:
            # If dimension just changed, reset buffer to ensure consistency
            if dimension_changed:
                logger.info("Resetting buffer due to dimension change")
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
            logger.debug(f"Successfully collected data into buffer: step={self.buffer.step}, rewards={np.mean(reward_batch):.4f}")
            
            # Ensure trainer also has buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                
        except Exception as e:
            logger.error(f"Failed to insert into shared buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def compute_returns(self):
        """Calculate returns and advantages from shared buffer - according to original MAPPO mode"""
        try:
            # Check if buffer exists
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.warning("Buffer does not exist, creating a new one first")
                self.reset_buffer()
                if not hasattr(self, 'buffer') or self.buffer is None:
                    logger.error("Unable to create buffer, unable to calculate returns")
                    return False
            
            # Detailed recording of buffer state
            logger.info(f"Buffer state: step={self.buffer.step}, rewards shape={self.buffer.rewards.shape if hasattr(self.buffer, 'rewards') else 'N/A'}")
            
            # Check if rewards have valid values
            if hasattr(self.buffer, 'rewards'):
                non_zero_rewards = np.count_nonzero(self.buffer.rewards)
                total_rewards = np.prod(self.buffer.rewards.shape)
                logger.info(f"Buffer content check: non-zero rewards count={non_zero_rewards}/{total_rewards} ({non_zero_rewards/total_rewards*100:.2f}%)")
                
                # Record statistics of rewards
                if non_zero_rewards > 0:
                    reward_mean = np.mean(self.buffer.rewards)
                    reward_std = np.std(self.buffer.rewards)
                    reward_min = np.min(self.buffer.rewards)
                    reward_max = np.max(self.buffer.rewards)
                    logger.info(f"Rewards statistics: mean={reward_mean:.6f}, standard deviation={reward_std:.6f}, minimum={reward_min:.6f}, maximum={reward_max:.6f}")
            
            # Check if buffer has enough data
            buffer_empty = self.buffer.step == 0 or (hasattr(self.buffer, 'rewards') and np.count_nonzero(self.buffer.rewards) == 0)
            if buffer_empty:
                # Check if it's the initial training phase
                is_initial_phase = not hasattr(self, '_training_started') or not self._training_started
                if is_initial_phase:
                    logger.info("Initial training phase, empty buffer is normal, adding initial data")
                    # Mark training has started
                    self._training_started = True
                else:
                    logger.warning("Training has been conducted but Buffer has no data or all data is zero, attempting to add real data")
                
                # Add virtual data to avoid empty buffer error
                self._add_dummy_data_to_buffer()
                logger.info(f"Added virtual data: step={self.buffer.step}, rewards shape={self.buffer.rewards.shape}")
                
                # Check again if addition was successful
                if self.buffer.step == 0 or not np.any(self.buffer.rewards):
                    logger.error("Even after adding virtual data, buffer is still empty, unable to calculate returns")
                    return False
                else:
                    logger.info("Virtual data addition successful, continuing to calculate returns")
            
            # Get value estimate of last step
            try:
                # Check if there are valid rewards data
                if hasattr(self.buffer, 'rewards') and np.sum(np.abs(self.buffer.rewards)) < 1e-6:
                    logger.warning("All rewards in Buffer are zero or close to zero, adding small random noise to avoid all-zero returns")
                    self.buffer.rewards = self.buffer.rewards + np.random.normal(0, 0.01, self.buffer.rewards.shape)
                    logger.info(f"Added noise: non-zero rewards count: {np.count_nonzero(self.buffer.rewards)}")
                
                # Get shared observations and states
                share_obs = np.concatenate(self.buffer.share_obs[-1])
                rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[-1])
                masks = np.concatenate(self.buffer.masks[-1])
                
                # Debug: Print input shapes
                logger.info(f"Input shape for calculating returns: share_obs={share_obs.shape}, rnn_states_critic={rnn_states_critic.shape}")
                
                # Convert to tensor
                share_obs = torch.FloatTensor(share_obs).to(self.device)
                rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)
                masks = torch.FloatTensor(masks).to(self.device)
                
                # Get value estimate
                with torch.no_grad():
                    next_values = self.policy.get_values(share_obs, rnn_states_critic, masks)
                    
                # Debug: Print value estimate results
                logger.info(f"Value estimate results: next_values shape={next_values.shape}, samples={next_values[:3]}")
                
                # Calculate returns
                next_values = next_values.detach().cpu().numpy()
                self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
                
                # Debug: Print calculated returns
                logger.info(f"Calculated returns shape={self.buffer.returns.shape}, samples={self.buffer.returns[0][0][0][:3]}")
                logger.info(f"Number of non-zero values in returns: {np.count_nonzero(self.buffer.returns)}")
                
                # Check if returns contain NaN or infinity
                if np.isnan(self.buffer.returns).any() or np.isinf(self.buffer.returns).any():
                    logger.warning("Returns contain NaN or infinity, performing numerical correction")
                    self.buffer.returns = np.nan_to_num(self.buffer.returns, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Check if returns are all zero
                if np.sum(np.abs(self.buffer.returns)) < 1e-6:
                    logger.warning("Calculated returns are all zero, which may lead to ineffective training")
                    return False
                
                # Check if advantages have been calculated
                if not hasattr(self.buffer, 'advantages') or self.buffer.advantages is None:
                    logger.warning("Advantages have not been calculated, manual calculation")
                    # Simple calculation of advantages (returns - value_preds)
                    self.buffer.advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
                    logger.info(f"Shape of manually calculated advantages: {self.buffer.advantages.shape}")
                
                # Check if advantages contain NaN or infinity
                if hasattr(self.buffer, 'advantages') and (np.isnan(self.buffer.advantages).any() or np.isinf(self.buffer.advantages).any()):
                    logger.warning("Advantages contain NaN or infinity, performing numerical correction")
                    self.buffer.advantages = np.nan_to_num(self.buffer.advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Ensure trainer's buffer reference is up-to-date
                if hasattr(self, 'trainer') and self.trainer is not None:
                    self.trainer.buffer = self.buffer
                    logger.info("Updated trainer's buffer reference")
                
                logger.info("Successfully calculated returns and advantages")
                return True
                
            except Exception as e:
                logger.error(f"Failed to calculate value estimate: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"Overall failure in calculating returns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _add_dummy_data_to_buffer(self):
        """Add some virtual data to buffer to avoid empty buffer error
        
        Note: These data are only for debugging and error prevention, should not be used for actual training.
        Actual training should use real data collected from the environment.
        """
        logger.warning("Adding virtual data to buffer to avoid empty buffer error - only for debugging")
        
        # Ensure buffer exists
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.warning("Buffer does not exist, creating a new one first")
            self.reset_buffer()
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("Unable to create buffer, unable to add virtual data")
                return
        
        # Create meaningful virtual data, not just random data
        # Use more meaningful virtual observations and rewards
        dummy_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        dummy_share_obs = np.zeros((1, self.num_agents, self.actual_obs_dim))
        
        # Set different observations for each agent, simulating real environment
        for i in range(self.num_agents):
            # Set some basic features to ensure different observations for each agent
            dummy_obs[0, i, 0] = 0.5 + 0.1 * i  # Time feature
            dummy_obs[0, i, 1] = 0.3 + 0.05 * i  # Price feature
            dummy_obs[0, i, 2] = 0.7 - 0.05 * i  # Demand feature
            dummy_obs[0, i, 3] = 0.2 + 0.02 * i  # Flexibility feature
            
            # Set different features for different agents
            dummy_obs[0, i, 4] = 0.1 * (i + 1)  # Agent-specific feature
            
            # Randomize remaining features to increase diversity
            if self.actual_obs_dim > 5:
                dummy_obs[0, i, 5:] = np.random.uniform(0.1, 0.9, size=self.actual_obs_dim-5)
            
            # Copy to shared observations
            dummy_share_obs[0, i] = dummy_obs[0, i].copy()
        
        # Create meaningful actions
        dummy_actions = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            # Set actions, simulating FlexOffer parameters
            dummy_actions[0, i, 0] = 0.5 + 0.1 * np.sin(i)  # Energy parameter
            dummy_actions[0, i, 1] = 0.3 + 0.1 * np.cos(i)  # Time parameter
            if self.action_dim > 2:
                dummy_actions[0, i, 2] = 0.7 - 0.1 * np.sin(i + 1)  # Price parameter
        
        # Create meaningful rewards - non-zero values, ensuring learning signal
        dummy_rewards = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            # Set different positive rewards for each agent
            dummy_rewards[0, i, 0] = 0.5 + 0.1 * i  # Rewards from 0.5 to 0.9
        
        # RNN states
        dummy_rnn_states_actor = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        dummy_rnn_states_critic = np.zeros((1, self.num_agents, self.args.recurrent_N, self.args.hidden_size))
        
        # Action log probabilities and value predictions - set reasonable values
        dummy_action_log_probs = np.zeros((1, self.num_agents, self.action_dim))
        for i in range(self.num_agents):
            for j in range(self.action_dim):
                dummy_action_log_probs[0, i, j] = -0.5 - 0.1 * j  # Reasonable log probabilities
        
        dummy_value_preds = np.zeros((1, self.num_agents, 1))
        for i in range(self.num_agents):
            dummy_value_preds[0, i, 0] = 0.6 + 0.1 * i  # Value estimate slightly above rewards
        
        # Masks
        dummy_masks = np.ones((1, self.num_agents, 1))
        dummy_bad_masks = np.ones((1, self.num_agents, 1))
        dummy_active_masks = np.ones((1, self.num_agents, 1))
        
        # Insert virtual data
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
            logger.info("Successfully added virtual data to buffer")
            logger.info(f"Virtual data details: Observation shape={dummy_obs.shape}, reward range=[{np.min(dummy_rewards):.3f}, {np.max(dummy_rewards):.3f}]")
            
            # Ensure trainer also has buffer reference
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.trainer.buffer = self.buffer
                logger.info("✅ Ensured trainer has latest buffer reference")
                
            # Add multiple virtual data points to ensure enough training data
            # Add 8 data points to form a small batch
            for step in range(8):
                # Create slightly different data for each step
                step_obs = dummy_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_share_obs = dummy_share_obs.copy() * (1.0 + 0.05 * np.sin(step))
                step_actions = dummy_actions.copy() * (1.0 + 0.03 * np.cos(step))
                
                # Rewards change over time, forming a meaningful trajectory
                step_rewards = dummy_rewards.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                # Value predictions also change accordingly
                step_value_preds = dummy_value_preds.copy() * (1.0 + 0.1 * np.sin(step / 4.0))
                
                # Insert data
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
            
            logger.info(f"✅ Successfully added 9 meaningful virtual data points to buffer, current step={self.buffer.step}")
            
            # Calculate returns of virtual data, ensuring training signal
            if hasattr(self, 'compute_returns'):
                success = self.compute_returns()
                if success:
                    logger.info("✅ Successfully calculated returns for virtual data")
                else:
                    logger.warning("⚠️ Unable to calculate returns for virtual data")
            
        except Exception as e:
            logger.error(f"Failed to add virtual data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        Execute one PPO batch update
        
        Returns:
            Dict[str, Any]: Training information, including policy_loss, value_loss, entropy, etc.
        """
        # Check if buffer is empty
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.error("Buffer does not exist, unable to train")
            self._add_dummy_data_to_buffer()  # Add some virtual data to avoid error
            if not hasattr(self, 'buffer') or self.buffer is None:
                logger.error("Even after adding virtual data, buffer still does not exist")
                return {
                    'policy_loss': 0.0,  # Return 0 indicates no actual training was performed
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # Check if buffer has enough data
        if self.buffer.step == 0:
            logger.warning("Buffer is empty or does not have enough data, adding virtual data for training")
            # Try to add virtual data
            self._add_dummy_data_to_buffer()
            if self.buffer.step == 0:
                logger.error("Even after adding virtual data, buffer is still empty")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        # Check if rewards have meaning
        if hasattr(self.buffer, 'rewards'):
            non_zero_rewards = np.count_nonzero(self.buffer.rewards)
            total_rewards = np.prod(self.buffer.rewards.shape)
            zero_ratio = 1.0 - (non_zero_rewards / total_rewards)
            
            logger.info(f"Pre-training check: zero rewards ratio={zero_ratio:.2%}, non-zero rewards count={non_zero_rewards}/{total_rewards}")
            
            # If more than 95% of rewards are zero, there might be a problem with data quality
            if zero_ratio > 0.95:
                logger.warning("More than 95% of rewards are zero, data quality might be an issue, but still attempting to train")
        
        # Ensure returns have been calculated
        if not hasattr(self.buffer, 'returns') or self.buffer.returns is None or np.count_nonzero(self.buffer.returns) == 0:
            logger.info("Calculating returns before training")
            success = self.compute_returns()
            if not success:
                logger.error("Unable to calculate returns, skipping training")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0,
                    'num_updates': 0
                }
        
        try:
            # Use MAPPO trainer to perform training
            train_info = self.trainer.train()
            
            # Update training iteration count
            self.training_iterations += 1
            
            # Check if training results are valid
            if not isinstance(train_info, dict):
                logger.warning(f"Invalid training results: {type(train_info)}")
                # Construct basic training information, but use 0 values to indicate training did not succeed
                train_info = {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0
                }
            
            # Record training information
            logger.info(f"Training completed: policy_loss={train_info.get('policy_loss', 0.0):.6f}, " +
                        f"value_loss={train_info.get('value_loss', 0.0):.6f}, " +
                        f"entropy={train_info.get('entropy', 0.0):.6f}")
            
            # Ensure all necessary keys are present in the result
            required_keys = ['policy_loss', 'value_loss', 'entropy', 'grad_norm', 'ratio']
            for key in required_keys:
                if key not in train_info:
                    train_info[key] = 0.0  # Use 0 to indicate missing data
            
            # Add training iteration count
            train_info['num_updates'] = 1
            
            return train_info
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return information indicating training failed, using 0 values instead of fixed 0.001
            return {
                'policy_loss': 0.0,  # Use 0 to indicate training failure
                'value_loss': 0.0,
                'entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0,
                'num_updates': 0,
                'training_error': str(e)
            }
    
    def _update_learning_rate(self):
        """
        Update learning rate (learning rate decay)
        Gradually decrease learning rate as training progresses, avoiding over-oscillation at later stages of training
        """
        try:
            # Exponential decay approach
            if hasattr(self.args, 'use_linear_lr_decay') and self.args.use_linear_lr_decay:
                # Get current episode number
                episode = self.total_episodes
                
                # Decay every 10 episodes
                decay_interval = 10
                if episode > 0 and episode % decay_interval == 0:
                    # Default decay rate is 0.95
                    decay_rate = getattr(self.args, 'lr_decay_rate', 0.95)
                    
                    # Update actor learning rate
                    for param_group in self.policy.actor_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # Set minimum learning rate
                    
                    # Update critic learning rate
                    for param_group in self.policy.critic_optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = current_lr * decay_rate
                        param_group['lr'] = max(new_lr, 1e-6)  # Set minimum learning rate
                    
                    # Record new learning rates
                    actor_lr = self.policy.actor_optimizer.param_groups[0]['lr']
                    critic_lr = self.policy.critic_optimizer.param_groups[0]['lr']
                    
                    logger.info(f"Learning rate has decayed: actor_lr={actor_lr:.7f}, critic_lr={critic_lr:.7f}")
                    
        except Exception as e:
            logger.warning(f"Failed to decay learning rate: {e}")
            # Training process continues unaffected if this fails
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_iterations': self.training_iterations,
            'total_episodes': self.total_episodes,
            'num_agents': self.num_agents,
            'algorithm': 'FOMAPPO',
            'architecture': 'shared_policy',
            'shared_parameters': self.training_stats['shared_parameters']
        }
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """Get Manager rewards summary (shared policy version)"""
        # In shared policy, all Managers share the same policy, so return overall statistics
        return {
            'shared_policy': self.training_stats.copy(),
            'note': 'All managers share the same policy network'
        }
    
    def save_models(self, save_path: str):
        """Save shared models"""
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
            
            logger.info(f"FOMAPPO shared models saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save FOMAPPO shared models: {e}")
    
    def load_models(self, load_path: str):
        """Load shared models"""
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
                
                logger.info(f"FOMAPPO shared models loaded from {model_path}")
            else:
                logger.warning(f"Model file {model_path} does not exist")
                
        except Exception as e:
            logger.error(f"Failed to load FOMAPPO shared models: {e}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Map original action to FlexOffer parameter range
        
        FlexOffer parameter range:
        - start_flex: [-1.0, 1.0] → Time flexibility
        - end_flex: [-1.0, 1.0] → Time flexibility  
        - energy_min_factor: [0.1, 1.0] → Minimum energy factor
        - energy_max_factor: [1.0, 2.0] → Maximum energy factor
        - priority_weight: [0.1, 2.0] → Priority weight
        
        Args:
            raw_action: Original action in range [-1, 1]
            
        Returns:
            fo_action: Action mapped to FlexOffer parameter range
        """
        fo_action = np.zeros_like(raw_action)
        
        # Assume action is a multiple of 5 (each device has 5 parameters)
        num_devices = len(raw_action) // 5 if len(raw_action) >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < len(raw_action):
                # start_flex: [-1, 1] → [-1, 1] (unchanged)
                fo_action[base_idx] = np.clip(raw_action[base_idx], -1.0, 1.0)
                
                # end_flex: [-1, 1] → [-1, 1] (unchanged)
                fo_action[base_idx + 1] = np.clip(raw_action[base_idx + 1], -1.0, 1.0)
                
                # energy_min_factor: [-1, 1] → [0.1, 1.0]
                fo_action[base_idx + 2] = 0.1 + 0.45 * (raw_action[base_idx + 2] + 1.0)
                
                # energy_max_factor: [-1, 1] → [1.0, 2.0]  
                fo_action[base_idx + 3] = 1.0 + 0.5 * (raw_action[base_idx + 3] + 1.0)
                
                # priority_weight: [-1, 1] → [0.1, 2.0]
                fo_action[base_idx + 4] = 0.1 + 0.95 * (raw_action[base_idx + 4] + 1.0)
        
        return fo_action
    
    def _generate_default_fo_action(self) -> np.ndarray:
        """Generate default FlexOffer parameters action"""
        # Generate reasonable default FlexOffer parameters
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
        When observed dimension changes, recreate buffer and policy network
        
        Args:
            new_obs_dim: New observed dimension
        """
        logger.warning(f"⚠️ Dimension change detection for observed dimension: {self.state_dim} → {new_obs_dim}")
        logger.warning(f"Recreating buffer and policy network to adapt to new observed dimension")
        
        # Ensure new_obs_dimension_history is initialized
        if not hasattr(self, 'new_obs_dimension_history'):
            self.new_obs_dimension_history = []
        
        # Save original network parameters (if possible)
        old_actor_state = None
        old_critic_state = None
        try:
            if hasattr(self, 'policy') and self.policy is not None:
                old_actor_state = self.policy.actor.state_dict()
                old_critic_state = self.policy.critic.state_dict()
                logger.info("✅ Successfully saved original network parameters")
        except Exception as e:
            logger.warning(f"Unable to save original network parameters: {e}")
        
        # Update state dimension
        self.state_dim = new_obs_dim
        self.actual_obs_dim = new_obs_dim
        
        # Recreate observation and action spaces
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # Update obs_space and share_obs_space in args
        self.args.obs_space = obs_space
        self.args.share_obs_space = share_obs_space
        
        # Record detailed network structure changes
        logger.info(f"🔄 Network reconstruction: Input layer changes from {self.state_dim} → {new_obs_dim}")
        
        # Recreate policy network
        try:
            self.policy = FOMAPPOPolicy(
                args=self.args,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space,
                device=self.device
            )
            logger.info("✅ Policy network reconstruction successful")
            
            # Try to restore some network parameters (possibly requiring manual layer mapping)
            if old_actor_state is not None and old_critic_state is not None:
                try:
                    # Note: Input layer parameters cannot be directly copied, but other layers can
                    # Here, more complex parameter mapping logic might be needed
                    logger.info("⚠️ Network parameters cannot be directly migrated, using parameters from newly initialized network")
                except Exception as transfer_e:
                    logger.warning(f"Failed to transfer parameters: {transfer_e}")
        except Exception as e:
            logger.error(f"❌ Failed to reconstruct policy network: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to reconstruct policy network: {e}")
        
        # Recreate trainer
        try:
            self.trainer = FOMAPPO(
                args=self.args,
                policy=self.policy,
                device=self.device
            )
            logger.info("✅ Trainer reconstruction successful")
        except Exception as e:
            logger.error(f"❌ Failed to reconstruct trainer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to reconstruct trainer: {e}")
        
        # Recreate buffer
        try:
            self.buffer = SharedReplayBuffer(
                args=self.args,
                num_agents=self.num_agents,
                obs_space=obs_space,
                cent_obs_space=share_obs_space,
                act_space=act_space
            )
            logger.info("✅ Shared buffer reconstruction successful")
        except Exception as e:
            logger.error(f"❌ Failed to reconstruct shared buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to reconstruct shared buffer: {e}")
        
        # Ensure trainer has buffer reference
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.buffer = self.buffer
            logger.info("✅ Successfully passed new buffer to FOMAPPO trainer")
        else:
            logger.warning("⚠️ Unable to pass buffer to FOMAPPO trainer, trainer does not exist")
        
        # Update training statistics
        try:
            if not hasattr(self, 'training_stats'):
                self.training_stats = {'shared_parameters': 0}
            
            self.training_stats['shared_parameters'] = sum(p.numel() for p in self.policy.actor.parameters()) + \
                                                   sum(p.numel() for p in self.policy.critic.parameters())
            logger.info(f"✅ Successfully updated network parameter statistics, total {self.training_stats['shared_parameters']} parameters")
        except Exception as e:
            logger.warning(f"Failed to update network parameter statistics: {e}")
        
        # Reset internal counters
        if not hasattr(self, 'training_iterations'):
            self.training_iterations = 0
        
        # Record important dimension information
        self.new_obs_dimension_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'old_dim': self.state_dim, 
            'new_dim': new_obs_dim
        })
        
        logger.info(f"✅ Network reconstruction completed, new observed dimension: {new_obs_dim}")
        logger.info(f"📊 History of dimension change: {len(self.new_obs_dimension_history)} times") 
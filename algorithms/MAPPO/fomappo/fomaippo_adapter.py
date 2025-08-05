#!/usr/bin/env python3
"""
FOMAIPPO Adapter - Independent Agent architecture based on separated policies (FlexOffer Multi-Agent Independent PPO)

Architecture design:
- References the original MAPPO separated/base_runner.py architecture
- Creates independent Policy, Trainer, and Buffer for each Manager
- Retains FOMAPPO special features (device coordination, FlexOffer constraints, etc.)
- Integrates with existing FO Framework
- Resolves policy conflict issues, enabling independent learning

Key features:
1. Independent learning: Each Manager has its own policy network, avoiding policy conflicts
2. FOMAPPO features: Retains device coordination and FlexOffer constraint awareness
3. FO integration: Seamless integration with existing FO Pipeline
4. Universal configuration: Multi-agent settings configured uniformly in FO Framework

Algorithm: FOMAIPPO (FlexOffer Multi-Agent Independent PPO)
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

# Import original MAPPO components (separated architecture)
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss

# Import FOMAPPO specific components
from .fomappo_policy import FOMAPPOPolicy
from .fomappo import FOMAPPO

logger = logging.getLogger(__name__)

class FOMAIPPOArgs:
    """FOMAIPPO parameter configuration class - inherits MAPPO parameters and adds FlexOffer specific parameters"""
    
    def __init__(self, **kwargs):
        # ========== Core PPO parameters ==========
        self.episode_length = kwargs.get('episode_length', 24)
        self.n_rollout_threads = kwargs.get('n_rollout_threads', 1)
        self.num_mini_batch = kwargs.get('num_mini_batch', 1)
        self.ppo_epoch = kwargs.get('ppo_epoch', 4)
        self.data_chunk_length = kwargs.get('data_chunk_length', 10)
        
        # Learning rate parameters - 🔧 Reduced learning rates for better numerical stability
        self.lr = kwargs.get('lr_actor', 1e-4)  # Reduced actor learning rate
        self.lr_actor = kwargs.get('lr_actor', 1e-4)  # Reduced actor learning rate
        self.critic_lr = kwargs.get('lr_critic', 5e-4)  # Reduced critic learning rate
        self.opti_eps = kwargs.get('opti_eps', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0)
        
        # PPO specific parameters - 🔧 Enhanced numerical stability
        self.clip_param = kwargs.get('clip_param', 0.1)  # Reduced clip range for better stability
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)  # Reduced value loss weight
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.2)  # Stronger gradient clipping
        self.huber_delta = kwargs.get('huber_delta', 1.0)  # Reduced huber delta
        
        # GAE parameters
        self.use_gae = kwargs.get('use_gae', True)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        
        # 🔧 Important fix: Add missing use_proper_time_limits attribute
        # This attribute controls whether time limits are considered when calculating returns
        # In the FlexOffer system, each episode has a clear time limit (24 hours), so set to True
        self.use_proper_time_limits = kwargs.get('use_proper_time_limits', True)
        
        # Network parameters
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.layer_N = kwargs.get('layer_N', 2)
        self.use_orthogonal = kwargs.get('use_orthogonal', True)
        self.gain = kwargs.get('gain', 0.01)
        self.use_feature_normalization = kwargs.get('use_feature_normalization', True)
        self.activation_id = kwargs.get('activation_id', 1)
        self.use_ReLU = kwargs.get('use_ReLU', False)  # 🔧 Fix: Use Tanh activation function (False) or ReLU (True)
        self.stacked_frames = kwargs.get('stacked_frames', 1)  # Number of stacked frames
        self.use_stacked_frames = kwargs.get('use_stacked_frames', False)  # Whether to use stacked frames
        
        # RNN parameters
        self.use_recurrent_policy = kwargs.get('use_recurrent_policy', False)
        self.use_naive_recurrent_policy = kwargs.get('use_naive_recurrent_policy', False)
        self.recurrent_N = kwargs.get('recurrent_N', 1)
        
        # Training options
        self.use_centralized_V = kwargs.get('use_centralized_V', True)
        self.use_clipped_value_loss = kwargs.get('use_clipped_value_loss', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', True)
        
        # Normalization options
        self.use_valuenorm = kwargs.get('use_valuenorm', False)  # Value normalization
        self.use_popart = kwargs.get('use_popart', False)  # PopArt normalization
        self.use_reward_normalization = kwargs.get('use_reward_normalization', True)  # Reward normalization
        self.use_advantage_normalization = kwargs.get('use_advantage_normalization', True)  # Advantage normalization
        self.reward_scale = kwargs.get('reward_scale', 0.01)  # Scale rewards to avoid numerical instability
        
        # Mask options
        self.use_policy_active_masks = kwargs.get('use_policy_active_masks', False)
        self.use_value_active_masks = kwargs.get('use_value_active_masks', False)
        
        # Learning rate decay
        self.use_linear_lr_decay = kwargs.get('use_linear_lr_decay', False)
        self.lr_decay_rate = kwargs.get('lr_decay_rate', 0.95)
        
        # FlexOffer specific parameters
        self.device_coord_loss_weight = kwargs.get('device_coord_loss_weight', 0.1)
        self.fo_constraint_loss_weight = kwargs.get('fo_constraint_loss_weight', 0.1)
        
        # Exploration parameters
        self.action_noise_std = kwargs.get('action_noise_std', 0.1)
        self.use_action_noise = kwargs.get('use_action_noise', True)
        
        # Get observations and action spaces from kwargs
        self.obs_space = kwargs.get('obs_space')
        self.share_obs_space = kwargs.get('share_obs_space')
        self.act_space = kwargs.get('act_space')
        
        # Algorithm name
        self.algorithm_name = kwargs.get('algorithm_name', 'FOMAIPPO')

class FOMAIPPOAdapter:
    """
    FOMAIPPO adapter - Multi-agent reinforcement learning with independent policies (FlexOffer Multi-Agent Independent PPO)
    
    Core design principles:
    1. References the original MAPPO separated/base_runner.py architecture
    2. Creates independent Policy, Trainer, and Buffer for each Manager
    3. Retains FOMAPPO special features (device coordination, FlexOffer constraints)
    4. Integrates with FO Framework
    
    Advantages:
    - Policy specialization: Each Manager can develop its own specialized policy
    - Avoids policy conflicts: Independent learning avoids conflicts between Managers
    - Handles heterogeneous agents: Different Managers can have different device compositions
    """
    
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
        Initialize FOMAIPPO adapter
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            num_agents: Number of agents (Managers)
            episode_length: Episode length
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            device: Computing device
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.episode_length = episode_length
        
        # Manager IDs
        if 'manager_ids' in kwargs:
            self.manager_ids = kwargs.get('manager_ids')
        else:
            self.manager_ids = [f"manager_{i}" for i in range(num_agents)]
        
        logger.info(f"🔧 Initializing FOMAIPPO adapter (independent policies)")
        logger.info(f"    Parameters: {len(self.manager_ids)} managers, state {state_dim}D, action {action_dim}D")
        
        # Create observation and action spaces (compatible with original MAPPO format)
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # Create parameter configuration
        args_dict = {
            'episode_length': episode_length,
            'n_rollout_threads': 1,
            'num_mini_batch': 1,
            'ppo_epoch': 4,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'obs_space': obs_space,
            'share_obs_space': share_obs_space,
            'act_space': act_space,
            'use_centralized_V': True,
            'use_clipped_value_loss': True,
            'use_huber_loss': True,
            'use_gae': True,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'use_proper_time_limits': True,
            'use_valuenorm': False,
            'use_feature_normalization': True,
            'hidden_size': 256,
            'layer_N': 2,
            'use_orthogonal': True,
            'gain': 0.01,
            'use_ReLU': False,
            'stacked_frames': 1,
            'use_stacked_frames': False,
            'use_recurrent_policy': False,
            'recurrent_N': 1,
            'use_naive_recurrent_policy': False,
            'use_reward_normalization': True,
            'use_advantage_normalization': True,
            'reward_scale': 0.01,
            'use_policy_active_masks': False,
            'use_value_active_masks': False,
            'use_linear_lr_decay': False,
            'lr_decay_rate': 0.95,
            'device_coord_loss_weight': 0.1,
            'fo_constraint_loss_weight': 0.1,
            'action_noise_std': 0.1,
            'use_action_noise': True
        }
        
        # Create args object
        self.args = FOMAIPPOArgs(**args_dict)
        
        # Dictionary to store per-agent components
        self.policies = {}
        self.trainers = {}
        self.buffers = {}
        
        # Create policy, trainer, and buffer for each agent
        for manager_id in self.manager_ids:
            try:
                # Create policy
                self.policies[manager_id] = FOMAPPOPolicy(
                    args=self.args,
                    obs_space=obs_space,
                    cent_obs_space=share_obs_space,
                    act_space=act_space,
                    device=self.device
                )
                
                # Create trainer
                self.trainers[manager_id] = FOMAPPO(
                    args=self.args,
                    policy=self.policies[manager_id],
                    device=self.device
                )
                
                # Create buffer
                self.buffers[manager_id] = SeparatedReplayBuffer(
                    args=self.args,
                    obs_space=obs_space,
                    share_obs_space=share_obs_space,
                    act_space=act_space
                )
                
                # Ensure trainer has buffer reference
                self.trainers[manager_id].buffer = self.buffers[manager_id]
                
            except Exception as e:
                logger.error(f"Failed to create FOMAIPPO components for {manager_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Initialize training statistics
        self.total_episodes = 0
        self.training_iterations = 0
        self.training_stats = {
            'episodes': 0,
            'updates': 0,
            'rewards': {}
        }
        
        # Initialize per-agent statistics
        for manager_id in self.manager_ids:
            self.training_stats['rewards'][manager_id] = []
        
        logger.info("✅ FOMAIPPO adapter initialization completed")
        logger.info(f"    Architecture: {len(self.manager_ids)} managers with independent policies")
    
    def reset_buffers(self):
        """Reset all agent buffers"""
        for manager_id, buffer in self.buffers.items():
            buffer.reset()
    
    def select_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Select FlexOffer parameter generation actions for all Managers (using independent policies)
        
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
        
        # Process each manager independently
        for manager_id, observation in obs.items():
            if manager_id not in self.policies:
                logger.warning(f"No policy for manager {manager_id}, skipping action selection")
                continue
                
            # Get policy for this manager
            policy = self.policies[manager_id]
            
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Create RNN states and masks (not used in non-recurrent mode)
            rnn_states_actor = torch.zeros(1, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            rnn_states_critic = torch.zeros(1, self.args.recurrent_N, self.args.hidden_size, device=self.device)
            masks = torch.ones(1, 1, device=self.device)
            
            # Select action using this manager's policy
            with torch.no_grad():
                value, action, action_log_prob, rnn_states_actor_new, rnn_states_critic_new = policy.get_actions(
                    obs_tensor,
                    obs_tensor,  # Use same observation for centralized critic
                    rnn_states_actor,
                    rnn_states_critic,
                    masks,
                    deterministic=deterministic
                )
            
            # Convert to numpy
            action_np = action.squeeze(0).cpu().numpy()
            action_log_prob_np = action_log_prob.squeeze(0).cpu().numpy()
            value_np = value.squeeze(0).cpu().numpy()
            
            # Map raw action to FlexOffer parameters
            fo_action = self._map_action_to_fo_params(action_np)
            
            # Store results
            actions[manager_id] = fo_action
            action_log_probs[manager_id] = action_log_prob_np
            values[manager_id] = value_np.item()
            
            logger.debug(f"Manager {manager_id} FlexOffer action: {fo_action.shape}D, first 5 parameters: {fo_action[:5]}")
        
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
        Collect one step of experience data into each agent's buffer
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            dones: Whether episode is done
            infos: Additional information
            action_log_probs: Action log probabilities (optional)
            values: Value function predictions (optional)
        """
        # Process each manager independently
        for manager_id in self.manager_ids:
            if manager_id not in obs or manager_id not in self.buffers:
                continue
                
            # Get buffer for this manager
            buffer = self.buffers[manager_id]
            
            # Get data for this manager
            manager_obs = obs[manager_id]
            manager_action = actions.get(manager_id, np.zeros(self.action_dim))
            manager_reward = rewards.get(manager_id, 0.0)
            manager_done = dones.get(manager_id, False)
            
            # Get log prob and value if available
            manager_action_log_prob = action_log_probs.get(manager_id, np.zeros(self.action_dim))
            manager_value = values.get(manager_id, 0.0)
            
            # Prepare data for buffer
            share_obs = manager_obs.reshape(1, -1)  # Use same observation for centralized critic
            obs_array = manager_obs.reshape(1, -1)
            action_array = manager_action.reshape(1, -1)
            reward_array = np.array([[manager_reward]])
            done_array = np.array([[manager_done]])
            action_log_prob_array = manager_action_log_prob.reshape(1, -1)
            value_array = np.array([[manager_value]])
            
            # RNN states (not used in non-recurrent mode)
            rnn_states_actor = np.zeros((1, self.args.recurrent_N, self.args.hidden_size))
            rnn_states_critic = np.zeros((1, self.args.recurrent_N, self.args.hidden_size))
            
            # Masks
            masks = np.ones((1, 1))
            bad_masks = np.ones((1, 1))
            active_masks = np.ones((1, 1))
            
            # Insert into buffer
            buffer.insert(
                share_obs=share_obs,
                obs=obs_array,
                rnn_states_actor=rnn_states_actor,
                rnn_states_critic=rnn_states_critic,
                actions=action_array,
                action_log_probs=action_log_prob_array,
                value_preds=value_array,
                rewards=reward_array,
                masks=masks,
                bad_masks=bad_masks,
                active_masks=active_masks
            )
            
            # Update training statistics
            if manager_id in self.training_stats['rewards']:
                self.training_stats['rewards'][manager_id].append(manager_reward)
    
    def compute_returns(self):
        """Calculate returns for all agents"""
        success = True
        
        for manager_id, trainer in self.trainers.items():
            if manager_id not in self.buffers:
                continue
                
            buffer = self.buffers[manager_id]
            
            try:
                # Get last observation
                if buffer.step > 0:
                    # Get last value prediction
                    share_obs = buffer.share_obs[-1]
                    rnn_states_critic = buffer.rnn_states_critic[-1]
                    masks = buffer.masks[-1]
                    
                    # Convert to tensor
                    share_obs = torch.FloatTensor(share_obs).to(self.device)
                    rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)
                    masks = torch.FloatTensor(masks).to(self.device)
                    
                    # Get value prediction
                    with torch.no_grad():
                        policy = self.policies[manager_id]
                        next_values = policy.get_values(
                            share_obs,
                            rnn_states_critic,
                            masks
                        )
                    
                    # Calculate returns
                    next_values = next_values.detach().cpu().numpy()
                    buffer.compute_returns(next_values, trainer.value_normalizer)
                else:
                    logger.warning(f"Buffer for {manager_id} is empty (step=0)")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error computing returns for {manager_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                success = False
        
        return success
    
    def train_on_batch(self) -> Dict[str, Any]:
        """
        Train all agents on their collected experiences
        
        Returns:
            Dict[str, Any]: Training information for each agent
        """
        train_info = {}
        
        # Train each agent independently
        for manager_id, trainer in self.trainers.items():
            if manager_id not in self.buffers:
                continue
                
            buffer = self.buffers[manager_id]
            
            try:
                # Check if buffer has data
                if buffer.step == 0:
                    logger.warning(f"Buffer for {manager_id} is empty, skipping training")
                    continue
                
                # Train on collected data
                agent_train_info = trainer.train()
                
                # Store training info
                train_info[manager_id] = agent_train_info
                
            except Exception as e:
                logger.error(f"Error training {manager_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                train_info[manager_id] = {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'dist_entropy': 0.0
                }
        
        # Update training statistics
        self.training_iterations += 1
        self.training_stats['updates'] += 1
        
        return train_info
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.training_stats.copy()
    
    def get_manager_rewards_summary(self) -> Dict[str, Any]:
        """Get Manager rewards summary"""
        return {manager_id: np.mean(rewards) if rewards else 0.0 
                for manager_id, rewards in self.training_stats['rewards'].items()}
    
    def save_models(self, save_path: str):
        """Save all agent models"""
        for manager_id, policy in self.policies.items():
            try:
                model_path = f"{save_path}_{manager_id}.pt"
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'training_iterations': self.training_iterations
                }, model_path)
                logger.info(f"Saved model for {manager_id} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model for {manager_id}: {e}")
    
    def load_models(self, load_path: str):
        """Load all agent models"""
        for manager_id, policy in self.policies.items():
            try:
                model_path = f"{load_path}_{manager_id}.pt"
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    policy.load_state_dict(checkpoint['policy_state_dict'])
                    if 'training_iterations' in checkpoint:
                        self.training_iterations = checkpoint['training_iterations']
                    logger.info(f"Loaded model for {manager_id} from {model_path}")
                else:
                    logger.warning(f"Model file {model_path} does not exist")
            except Exception as e:
                logger.error(f"Failed to load model for {manager_id}: {e}")
    
    def _map_action_to_fo_params(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Map original action to FlexOffer parameter range
        
        FlexOffer parameter ranges:
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
                fo_action[base_idx + 2] = 0.1 + 0.9 * (raw_action[base_idx + 2] + 1) / 2
                
                # energy_max_factor: [-1, 1] → [1.0, 2.0]
                fo_action[base_idx + 3] = 1.0 + (raw_action[base_idx + 3] + 1) / 2
                
                # priority_weight: [-1, 1] → [0.1, 2.0]
                fo_action[base_idx + 4] = 0.1 + 1.9 * (raw_action[base_idx + 4] + 1) / 2
        
        return fo_action
    
    def _generate_default_fo_action(self) -> np.ndarray:
        """Generate default FlexOffer parameters action"""
        default_action = np.zeros(self.action_dim)
        num_devices = self.action_dim // 5 if self.action_dim >= 5 else 1
        
        for i in range(num_devices):
            base_idx = i * 5
            if base_idx + 4 < self.action_dim:
                # Default values for each parameter
                default_action[base_idx] = 0.0      # start_flex: neutral flexibility
                default_action[base_idx + 1] = 0.0  # end_flex: neutral flexibility
                default_action[base_idx + 2] = 0.5  # energy_min_factor: 0.55 (middle of range)
                default_action[base_idx + 3] = 0.0  # energy_max_factor: 1.5 (middle of range)
                default_action[base_idx + 4] = 0.0  # priority_weight: 1.05 (middle of range)
        
        return default_action 
import torch
import numpy as np
import sys
import os

# Add onpolicy module path (fixed version)
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/

# 🔧 Critical fix: Add the parent directory containing onpolicy, not the onpolicy directory itself
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

# Now we can safely import onpolicy modules
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class FOMAPPOPolicy:
    """
    FlexOffer Multi-Agent PPO Policy class
    
    A policy network specifically designed for the FlexOffer system, supporting:
    - Manager-level observations and actions
    - Device-level state awareness
    - Integration of FlexOffer constraints
    - Multi-agent collaboration mechanisms
    
    Args:
        args: Parameter configuration
        obs_space: Observation space
        cent_obs_space: Centralized observation space (for critic)
        action_space: Action space
        device: Computation device
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        # FlexOffer specific parameters
        self.num_managers = getattr(args, 'num_managers', 4)
        self.devices_per_manager = getattr(args, 'devices_per_manager', 10)
        self.use_device_attention = getattr(args, 'use_device_attention', True)
        self.use_manager_coordination = getattr(args, 'use_manager_coordination', True)

        # Create actor and critic networks
        self.actor = FOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = FOCritic(args, self.share_obs_space, self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """Learning rate decay"""
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, device_states=None, fo_constraints=None):
        """
        Calculate actions and value function predictions
        
        Args:
            cent_obs: Centralized observations (for critic)
            obs: Local observations (for actor)
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            masks: RNN state reset masks
            available_actions: Available actions mask
            deterministic: Whether to use deterministic actions
            device_states: Device state information
            fo_constraints: FlexOffer constraint information
            
        Returns:
            values: Value function predictions
            actions: Selected actions
            action_log_probs: Action log probabilities
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
        """
        # Process FlexOffer specific information
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        
        # Actor forward pass
        actions, action_log_probs, rnn_states_actor = self.actor(enhanced_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        # Critic forward pass
        values, rnn_states_critic = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        
        # Apply FlexOffer constraints
        if fo_constraints is not None:
            actions = self._apply_fo_constraints(actions, fo_constraints)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, device_states=None, fo_constraints=None):
        """Get value function predictions"""
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        values, _ = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, device_states=None, fo_constraints=None):
        """
        Evaluate action log probabilities, entropy, and value function
        
        Used for calculating gradients during actor updates
        """
        # Enhance observations
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        enhanced_cent_obs = self._enhance_centralized_observation(cent_obs, device_states, fo_constraints)
        
        # Actor evaluation
        action_log_probs, dist_entropy = self.actor.evaluate_actions(enhanced_obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        # Critic evaluation
        values, _ = self.critic(enhanced_cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, 
            device_states=None, fo_constraints=None):
        """Calculate actions only (for inference)"""
        enhanced_obs = self._enhance_observation(obs, device_states, fo_constraints)
        actions, _, rnn_states_actor = self.actor(enhanced_obs, rnn_states_actor, masks, 
                                                  available_actions, deterministic)
        
        # Apply FlexOffer constraints
        if fo_constraints is not None:
            actions = self._apply_fo_constraints(actions, fo_constraints)
            
        return actions, rnn_states_actor

    def _enhance_observation(self, obs, device_states=None, fo_constraints=None):
        """Enhance observation information, integrating device states and FlexOffer constraints"""
        if device_states is None and fo_constraints is None:
            return obs
        
        enhanced_obs = obs
        
        # Add device state information
        if device_states is not None:
            if isinstance(device_states, np.ndarray):
                device_features = torch.FloatTensor(device_states).to(self.device)
            else:
                device_features = device_states
            enhanced_obs = torch.cat([enhanced_obs, device_features], dim=-1)
        
        # Add FlexOffer constraint information
        if fo_constraints is not None:
            if isinstance(fo_constraints, np.ndarray):
                constraint_features = torch.FloatTensor(fo_constraints).to(self.device)
            else:
                constraint_features = fo_constraints
            enhanced_obs = torch.cat([enhanced_obs, constraint_features], dim=-1)
        
        return enhanced_obs

    def _enhance_centralized_observation(self, cent_obs, device_states=None, fo_constraints=None):
        """Enhance centralized observation information"""
        if device_states is None and fo_constraints is None:
            return cent_obs
        
        enhanced_cent_obs = cent_obs
        
        # Add global device state information
        if device_states is not None:
            if isinstance(device_states, np.ndarray):
                global_device_features = torch.FloatTensor(device_states).to(self.device)
            else:
                global_device_features = device_states
            
            # For centralized observations, we may need to aggregate device states from all Managers
            if len(global_device_features.shape) > 2:
                global_device_features = global_device_features.mean(dim=1)  # Aggregate Manager dimension
            
            enhanced_cent_obs = torch.cat([enhanced_cent_obs, global_device_features], dim=-1)
        
        # Add global FlexOffer constraint information
        if fo_constraints is not None:
            if isinstance(fo_constraints, np.ndarray):
                global_constraint_features = torch.FloatTensor(fo_constraints).to(self.device)
            else:
                global_constraint_features = fo_constraints
            
            if len(global_constraint_features.shape) > 2:
                global_constraint_features = global_constraint_features.mean(dim=1)
            
            enhanced_cent_obs = torch.cat([enhanced_cent_obs, global_constraint_features], dim=-1)
        
        return enhanced_cent_obs

    def _apply_fo_constraints(self, actions, fo_constraints):
        """Apply FlexOffer constraints to actions"""
        if fo_constraints is None:
            return actions
        
        # Simplified implementation: limit actions to constraint range
        # In a real application, more complex constraint handling logic should be implemented here
        constrained_actions = torch.clamp(actions, 0.0, 1.0)
        
        return constrained_actions


class FOActor(R_Actor):
    """FlexOffer specialized Actor network"""
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(FOActor, self).__init__(args, obs_space, action_space, device)
        
        # FlexOffer specific network layers
        self.device_attention_dim = getattr(args, 'device_attention_dim', 64)
        self.use_device_attention = getattr(args, 'use_device_attention', True)
        
        if self.use_device_attention:
            # Device attention mechanism
            self.device_attention = torch.nn.MultiheadAttention(
                embed_dim=self.device_attention_dim,
                num_heads=4,
                batch_first=True
            ).to(device)


class FOCritic(R_Critic):
    """FlexOffer specialized Critic network"""
    
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(FOCritic, self).__init__(args, cent_obs_space, device)
        
        # FlexOffer specific network layers
        self.manager_coordination_dim = getattr(args, 'manager_coordination_dim', 128)
        self.use_manager_coordination = getattr(args, 'use_manager_coordination', True)
        
        if self.use_manager_coordination:
            # Manager coordination mechanism
            self.manager_coordination = torch.nn.MultiheadAttention(
                embed_dim=self.manager_coordination_dim,
                num_heads=4,
                batch_first=True
            ).to(device) 
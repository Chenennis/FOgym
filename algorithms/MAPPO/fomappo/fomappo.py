import numpy as np
import torch
import torch.nn as nn
import sys
import os
import logging
import traceback

# Get logger
logger = logging.getLogger(__name__)

# Add onpolicy module path (fixed version)
current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/

# 🔧 Critical fix: Add the parent directory containing onpolicy, not the onpolicy directory itself
if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

# Now we can safely import onpolicy modules
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, check
from onpolicy.utils.valuenorm import ValueNorm

class FOMAPPO():
    """
    FlexOffer Multi-Agent PPO (FOMAPPO) Algorithm
    
    A multi-agent PPO algorithm specifically designed for the FlexOffer system.
    Supports collaborative learning at the Manager level and precise control at the device level.
    
    Key features:
    - Device-level state transition modeling
    - Inter-Manager collaboration mechanism
    - FlexOffer constraint-aware reward design
    - Distributed training and centralized execution
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.args = args  # Save args reference

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        # FlexOffer specific parameters
        self._use_device_coordination = getattr(args, 'use_device_coordination', True)
        self._device_coordination_weight = getattr(args, 'device_coordination_weight', 0.1)
        self._fo_constraint_weight = getattr(args, 'fo_constraint_weight', 0.2)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            
        # Initialize buffer attribute
        self.buffer = None
        
        # Get buffer from FOMAPPOAdapter
        try:
            # Try to get buffer from adapter
            from onpolicy.utils.shared_buffer import SharedReplayBuffer
            
            # If args has necessary space information, create a default buffer
            if hasattr(args, 'obs_space') and hasattr(args, 'share_obs_space') and hasattr(args, 'act_space'):
                self.buffer = SharedReplayBuffer(
                    args=args,
                    num_agents=getattr(args, 'num_agents', 4),
                    obs_space=args.obs_space,
                    cent_obs_space=args.share_obs_space,
                    act_space=args.act_space
                )
                logger.info("✅ Successfully created default buffer in FOMAPPO")
            else:
                logger.warning("⚠️ Cannot create default buffer in FOMAPPO, missing necessary space information")
        except Exception as e:
            logger.error(f"❌ Error creating buffer in FOMAPPO: {e}")
            logger.debug(traceback.format_exc())
            
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        
        Args:
            values: Current value function predictions
            value_preds_batch: Old value predictions
            return_batch: Discounted returns
            active_masks_batch: Masks indicating which agents are active
            
        Returns:
            value_loss: Value function loss
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            value_normalizer = self.value_normalizer
            value_pred_clipped = value_normalizer.denormalize(value_pred_clipped)
            values = value_normalizer.denormalize(values)
            return_batch = value_normalizer.denormalize(return_batch)

        if self._use_huber_loss:
            value_loss = huber_loss(values, return_batch, self.huber_delta)
        else:
            value_loss = mse_loss(values, return_batch)

        if self._use_clipped_value_loss:
            value_loss_clipped = huber_loss(value_pred_clipped, return_batch, self.huber_delta) if self._use_huber_loss else mse_loss(value_pred_clipped, return_batch)
            value_loss = torch.max(value_loss, value_loss_clipped)

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def cal_device_coordination_loss(self, actions_batch, device_states_batch=None):
        """
        Calculate device coordination loss to encourage coordination between devices.
        
        Args:
            actions_batch: Batch of actions
            device_states_batch: Batch of device states (optional)
            
        Returns:
            coordination_loss: Device coordination loss
        """
        # Simple implementation: encourage moderate diversity in actions
        if not self._use_device_coordination:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate action variance as a proxy for coordination
        action_variance = torch.var(actions_batch, dim=1).mean()
        
        # We want moderate variance (not too high, not too low)
        target_variance = 0.5
        coordination_loss = torch.abs(action_variance - target_variance)
        
        return coordination_loss

    def cal_fo_constraint_loss(self, actions_batch, fo_constraints_batch=None):
        """
        Calculate FlexOffer constraint loss to ensure actions respect FlexOffer constraints.
        
        Args:
            actions_batch: Batch of actions
            fo_constraints_batch: Batch of FlexOffer constraints (optional)
            
        Returns:
            constraint_loss: FlexOffer constraint loss
        """
        # Simple implementation: penalize extreme actions
        constraint_violation = torch.relu(torch.abs(actions_batch) - 0.9).mean()
        
        return constraint_violation

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        
        Args:
            sample: Dictionary containing sampled experiences
            update_actor: Whether to update actor network
            
        Returns:
            value_loss: Value function loss
            critic_grad_norm: Critic gradient norm
            policy_loss: Policy loss
            dist_entropy: Action distribution entropy
            actor_grad_norm: Actor gradient norm
            imp_weights: Importance weights
        """
        share_obs_batch = sample["share_obs"]
        obs_batch = sample["obs"]
        actions_batch = sample["actions"]
        value_preds_batch = sample["value_preds"]
        return_batch = sample["returns"]
        masks_batch = sample["masks"]
        active_masks_batch = sample["active_masks"]
        old_action_log_probs_batch = sample["action_log_probs"]
        adv_targ = sample["advantages"]
        available_actions_batch = sample["available_actions"] if "available_actions" in sample else None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # Value loss
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # Policy loss
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # FlexOffer specific losses
        device_coordination_loss = self.cal_device_coordination_loss(actions_batch)
        fo_constraint_loss = self.cal_fo_constraint_loss(actions_batch)
        
        # Add FlexOffer specific losses to policy loss
        policy_loss = policy_loss + self._device_coordination_weight * device_coordination_loss + self._fo_constraint_weight * fo_constraint_loss

        # Update actor
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # Update critic
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio

    def train(self):
        """
        Perform a training update using minibatch GD.
        
        Returns:
            train_info: Dictionary containing training statistics
        """
        if self.buffer is None:
            logger.error("❌ Buffer is None in FOMAPPO.train(), cannot train")
            return {}
            
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(self.buffer.returns[:-1])

        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.feed_forward_generator(self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        """Prepare for training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Prepare for rollout mode"""
        self.policy.actor.eval()
        self.policy.critic.eval() 
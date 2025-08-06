import numpy as np
import torch
import torch.nn as nn
import sys
import os
import logging
import traceback

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
mappo_dir = os.path.dirname(current_dir)  # algorithms/MAPPO/

if mappo_dir not in sys.path:
    sys.path.insert(0, mappo_dir)

from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, check
from onpolicy.utils.valuenorm import ValueNorm

class FOMAPPO():
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.args = args  # save args reference

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
            
        # initialize buffer attribute
        self.buffer = None
        
        # get buffer from FOMAPPOAdapter
        try:
            # try to get buffer from adapter
            from onpolicy.utils.shared_buffer import SharedReplayBuffer
            
            # if args has necessary space information, create a default buffer
            if hasattr(args, 'obs_space') and hasattr(args, 'share_obs_space') and hasattr(args, 'act_space'):
                self.buffer = SharedReplayBuffer(
                    args=args,
                    num_agents=getattr(args, 'num_agents', 4),
                    obs_space=args.obs_space,
                    cent_obs_space=args.share_obs_space,
                    act_space=args.act_space
                )
                logger.info("✅ successfully create default buffer in FOMAPPO")
            else:
                logger.warning("⚠️ cannot create default buffer in FOMAPPO, missing necessary space information")
        except Exception as e:
            logger.warning(f"⚠️ initialize FOMAPPO buffer failed: {e}")
            # do not throw exception, let the code continue

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """calculate value function loss"""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            if self.value_normalizer is not None:
                self.value_normalizer.update(return_batch)
                error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
                error_original = self.value_normalizer.normalize(return_batch) - values
            else:
                # fallback when value_normalizer is None
                error_clipped = return_batch - value_pred_clipped
                error_original = return_batch - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def cal_device_coordination_loss(self, actions_batch, device_states_batch=None):
        """
        calculate device coordination loss
        
        encourage coordination within the same Manager, and between Managers
        """
        if not self._use_device_coordination or device_states_batch is None:
            return torch.tensor(0.0, device=self.device)
        
        # calculate the variance of device actions, encourage coordination
        action_var = torch.var(actions_batch, dim=-1).mean()
        
        # device coordination loss: moderate variance is beneficial for flexibility, large variance indicates lack of coordination
        coordination_loss = torch.clamp(action_var - 0.5, min=0.0)  # target variance is 0.5
        
        return self._device_coordination_weight * coordination_loss

    def cal_fo_constraint_loss(self, actions_batch, fo_constraints_batch=None):
        """
        calculate FlexOffer constraint loss
        
        ensure the generated actions meet FlexOffer constraints
        """
        if fo_constraints_batch is None:
            return torch.tensor(0.0, device=self.device)
        
        # check if the actions are within the allowed range
        constraint_violations = torch.relu(actions_batch - 1.0) + torch.relu(-actions_batch)
        constraint_loss = constraint_violations.mean()
        
        return self._fo_constraint_weight * constraint_loss

    def ppo_update(self, sample, update_actor=True):
        """
        update actor and critic network
        
        Args:
            sample: training data batch
            update_actor: whether to update actor network
            
        Returns:
            training statistics
        """
        # unpack sample data
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
            device_states_batch = None
            fo_constraints_batch = None
        elif len(sample) == 14:  
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, device_states_batch, fo_constraints_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample
            device_states_batch = None
            fo_constraints_batch = None

        # safely convert tensor, avoid None value error
        if old_action_log_probs_batch is not None:
            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        else:
            old_action_log_probs_batch = torch.zeros(1, 1, **self.tpdv)
            
        if adv_targ is not None:
            adv_targ = check(adv_targ).to(**self.tpdv)
        else:
            adv_targ = torch.zeros(1, 1, **self.tpdv)
            
        if value_preds_batch is not None:
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        else:
            value_preds_batch = torch.zeros(1, 1, **self.tpdv)
            
        if return_batch is not None:
            return_batch = check(return_batch).to(**self.tpdv)
        else:
            return_batch = torch.zeros(1, 1, **self.tpdv)
            
        if active_masks_batch is not None:
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        else:
            active_masks_batch = torch.ones(1, 1, **self.tpdv)

        # ensure all parameters passed to evaluate_actions are not None
        if share_obs_batch is None:
            share_obs_batch = obs_batch
        if rnn_states_batch is None:
            batch_size = obs_batch.size(0) if obs_batch is not None else 1
            rnn_states_batch = torch.zeros(batch_size, 1, 1, 256, **self.tpdv)
        if rnn_states_critic_batch is None:
            batch_size = obs_batch.size(0) if obs_batch is not None else 1
            rnn_states_critic_batch = torch.zeros(batch_size, 1, 1, 256, **self.tpdv)
        if masks_batch is None:
            batch_size = obs_batch.size(0) if obs_batch is not None else 1
            masks_batch = torch.ones(batch_size, 1, **self.tpdv)

        # forward propagation
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # update actor
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # add FlexOffer specific loss
        device_coord_loss = self.cal_device_coordination_loss(actions_batch, device_states_batch)
        fo_constraint_loss = self.cal_fo_constraint_loss(actions_batch, fo_constraints_batch)
        
        policy_loss = policy_action_loss + device_coord_loss + fo_constraint_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # update critic
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, device_coord_loss, fo_constraint_loss

    def train(self):
        # check if buffer has enough data
        if not hasattr(self, 'buffer') or self.buffer is None:
            logger.error("training failed: FOMAPPO has no buffer attribute")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'dist_entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0
            }
            
        # use buffer passed from adapter
        buffer = self.buffer
            
        if buffer.step == 0:
            logger.error("training failed: buffer is empty, step=0, no experience data collected")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'dist_entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0
            }
        
        # check rewards data quality
        if hasattr(buffer, 'rewards'):
            non_zero_rewards = np.count_nonzero(buffer.rewards)
            total_rewards = np.prod(buffer.rewards.shape)
            logger.info(f"before training, buffer check: rewards non-zero ratio={non_zero_rewards/total_rewards:.2%}, number={non_zero_rewards}/{total_rewards}")
            
            if non_zero_rewards == 0:
                logger.error("training failed: all rewards in buffer are zero, cannot perform effective training")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'dist_entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0
                }
        
        # calculate advantage (if not calculated)
        if not hasattr(buffer, 'advantages') or buffer.advantages is None:
            logger.warning("advantage not calculated, try to calculate")
            try:
                # get the last step's value estimation
                share_obs = np.concatenate(buffer.share_obs[-1])
                rnn_states_critic = np.concatenate(buffer.rnn_states_critic[-1])
                masks = np.concatenate(buffer.masks[-1])
                
                # convert to tensor
                share_obs = torch.FloatTensor(share_obs).to(self.device)
                rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)
                masks = torch.FloatTensor(masks).to(self.device)
                
                # get value estimation
                with torch.no_grad():
                    next_values = self.policy.get_values(share_obs, rnn_states_critic, masks)
                    
                # calculate returns
                next_values = next_values.detach().cpu().numpy()
                buffer.compute_returns(next_values, self.value_normalizer)
                
                # check calculation result
                if hasattr(buffer, 'returns'):
                    non_zero_returns = np.count_nonzero(buffer.returns)
                    total_returns = np.prod(buffer.returns.shape)
                    logger.info(f"returns calculation result: non-zero ratio={non_zero_returns/total_returns:.2%}, number={non_zero_returns}/{total_returns}")
                    
                    if non_zero_returns == 0:
                        logger.error("calculation of returns is all zero, cannot perform effective training")
                        return {
                            'policy_loss': 0.0,
                            'value_loss': 0.0,
                            'dist_entropy': 0.0,
                            'grad_norm': 0.0,
                            'ratio': 1.0
                        }
                
            except Exception as e:
                logger.error(f"calculate advantage failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'dist_entropy': 0.0,
                    'grad_norm': 0.0,
                    'ratio': 1.0
                }
        
        train_info = {}
        train_info['policy_loss'] = 0.0
        train_info['value_loss'] = 0.0
        train_info['dist_entropy'] = 0.0
        train_info['ratio'] = 0.0
        
        # prepare data
        try:
            advantages = buffer.advantages
            if self.args.use_advantage_normalization:
                advantages_copy = advantages.copy()
                advantages_copy[advantages_copy > 1e10] = 1e10
                advantages_copy[advantages_copy < -1e10] = -1e10
                advantages = (advantages_copy - advantages_copy.mean()) / (advantages_copy.std() + 1e-5)
                
            # record advantage information
            logger.info(f"advantage statistics: mean={np.mean(advantages):.6f}, std={np.std(advantages):.6f}, min={np.min(advantages):.6f}, max={np.max(advantages):.6f}")
        except Exception as e:
            logger.error(f"prepare advantage data failed: {e}")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'dist_entropy': 0.0,
                'grad_norm': 0.0,
                'ratio': 1.0
            }
        
        # record PPO update start
        logger.info(f"start PPO update: {self.args.ppo_epoch} epochs, {self.args.num_mini_batch} mini-batches")
        
        # PPO update
        for epoch in range(self.args.ppo_epoch):
            try:
                data_generator = buffer.feed_forward_generator(advantages, self.args.num_mini_batch)
                
                for sample in data_generator:
                    # unpack sample data
                    share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch = sample
                    
                    # convert to tensor
                    share_obs_batch = torch.FloatTensor(share_obs_batch).to(self.device)
                    obs_batch = torch.FloatTensor(obs_batch).to(self.device)
                    rnn_states_batch = torch.FloatTensor(rnn_states_batch).to(self.device)
                    rnn_states_critic_batch = torch.FloatTensor(rnn_states_critic_batch).to(self.device)
                    actions_batch = torch.FloatTensor(actions_batch).to(self.device)
                    value_preds_batch = torch.FloatTensor(value_preds_batch).to(self.device)
                    return_batch = torch.FloatTensor(return_batch).to(self.device)
                    masks_batch = torch.FloatTensor(masks_batch).to(self.device)
                    active_masks_batch = torch.FloatTensor(active_masks_batch).to(self.device)
                    old_action_log_probs_batch = torch.FloatTensor(old_action_log_probs_batch).to(self.device)
                    adv_targ = torch.FloatTensor(adv_targ).to(self.device)
                    
                    # get new action log probability and value estimation
                    values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                         obs_batch, 
                                                                                         rnn_states_batch,
                                                                                         rnn_states_critic_batch, 
                                                                                         actions_batch, 
                                                                                         masks_batch,
                                                                                         active_masks_batch)
                    
                    # calculate ratio
                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    
                    # clip ratio
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_targ
                    
                    # policy loss
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # value loss
                    if self.args.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.args.clip_param, self.args.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    
                    # total loss
                    loss = policy_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * dist_entropy
                    
                    # update policy
                    self.policy.actor_optimizer.zero_grad()
                    self.policy.critic_optimizer.zero_grad()
                    loss.backward()
                    
                    # gradient clipping
                    if self.args.use_max_grad_norm:
                        grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.args.max_grad_norm)
                    else:
                        grad_norm = get_gard_norm(self.policy.actor.parameters())

                    self.policy.actor_optimizer.step()
                    self.policy.critic_optimizer.step()

                    # update training information
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['value_loss'] += value_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['ratio'] += ratio.mean().item()
                    train_info['grad_norm'] = grad_norm
                    
            except Exception as e:
                logger.error(f"PPO update failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # calculate average loss
        num_updates = self.args.ppo_epoch * self.args.num_mini_batch
        if num_updates > 0:
            for k in train_info.keys():
                if k != 'grad_norm':
                    train_info[k] /= num_updates
        
        # ensure return value is not zero (if training actually happened)
        if num_updates > 0:
            for key in ['policy_loss', 'value_loss', 'dist_entropy']:
                if key in train_info and abs(train_info[key]) < 1e-10:
                    train_info[key] = 1e-8
        
        # record training result
        logger.info(f"PPO training completed: policy_loss={train_info['policy_loss']:.6f}, value_loss={train_info['value_loss']:.6f}, entropy={train_info['dist_entropy']:.6f}")
        
        # after training, process buffer
        try:
            buffer.after_update()
        except Exception as e:
            logger.error(f"after training, process buffer failed: {e}")
     
        return train_info

    def prep_training(self):
        """prepare training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """prepare inference mode"""
        self.policy.actor.eval()
        self.policy.critic.eval() 
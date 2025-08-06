#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPLossComputer:
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # loss function weights
        self.uncertainty_weight = 0.1        # uncertainty loss weight
        self.collaboration_weight = 0.05     # collaboration consistency loss weight
        self.information_quality_weight = 0.03  # information quality loss weight
        self.exploration_weight = 0.02       # exploration encouragement loss weight
        
        # parameters
        self.clip_param = 0.2               # PPO clipping parameter
        self.entropy_coef = 0.01            # entropy coefficient
        self.value_loss_coef = 0.5          # value loss coefficient
        
    def compute_ppo_loss(self, action_log_probs, old_action_log_probs, advantages, 
                         values, returns, active_masks=None):
        # importance sampling ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        
        # PPO clipping target
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # policy loss
        if active_masks is not None:
            policy_loss = -(torch.min(surr1, surr2) * active_masks).sum() / active_masks.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # value function loss
        value_loss = F.mse_loss(values, returns)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'ratio_mean': ratio.mean(),
            'ratio_std': ratio.std()
        }
    
    def compute_uncertainty_loss(self, private_features, public_features, others_features):
        """
        compute information uncertainty loss
        
        in partially observable environments, encourage agents to maintain appropriate caution on uncertain information
        
        Args:
            private_features: private information features
            public_features: public information features  
            others_features: other information features
            
        Returns:
            torch.Tensor: uncertainty loss
        """
        if not self.config.enable_observation_noise:
            return torch.tensor(0.0, device=self.device)
        
        # calculate the reliability weights of different information sources
        private_reliability = 1.0  # private information is the most reliable
        public_reliability = 0.8   # public information is relatively reliable
        others_reliability = 0.5 if self.config.enable_other_manager_info else 0.0  # other information is not very reliable
        
        # adjust the reliability according to the noise level
        noise_factor = 1.0 - self.config.noise_level
        others_reliability *= noise_factor
        
        # calculate the variance of information features (uncertainty indicator)
        private_var = torch.var(private_features, dim=-1).mean() if private_features is not None else 0.0
        public_var = torch.var(public_features, dim=-1).mean() if public_features is not None else 0.0
        others_var = torch.var(others_features, dim=-1).mean() if others_features is not None else 0.0
        
        # weighted uncertainty
        weighted_uncertainty = (
            private_var * (1.0 - private_reliability) +
            public_var * (1.0 - public_reliability) +
            others_var * (1.0 - others_reliability)
        )
        
        return weighted_uncertainty
    
    def compute_collaboration_loss(self, actions, others_actions=None):
        """
        compute collaboration consistency loss
        
        encourage collaboration between Managers, especially when there is other information
        
        Args:
            actions: current Manager's actions
            others_actions: other Manager's actions (if available)
            
        Returns:
            torch.Tensor: collaboration loss
        """
        if not self.config.enable_other_manager_info or others_actions is None:
            return torch.tensor(0.0, device=self.device)
        
        # action consistency loss: encourage similar action strategies
        action_consistency = torch.norm(actions - others_actions, dim=-1).mean()
        
        # avoid excessive consistency (maintain diversity)
        diversity_threshold = 0.5
        consistency_loss = torch.relu(diversity_threshold - action_consistency)
        
        return consistency_loss
    
    def compute_information_quality_loss(self, predicted_others_info, actual_others_info=None,
                                         information_attention_weights=None):
        """
        compute information quality awareness loss
        
        encourage agents to correctly evaluate and utilize information of different quality
        
        Args:
            predicted_others_info: predicted other information
            actual_others_info: actual other information (if available)
            information_attention_weights: information attention weights
            
        Returns:
            torch.Tensor: information quality loss
        """
        if not self.config.enable_other_manager_info:
            return torch.tensor(0.0, device=self.device)
        
        # if there is actual other information, calculate the prediction error
        if actual_others_info is not None:
            prediction_error = F.mse_loss(predicted_others_info, actual_others_info)
            
            # adjust the expected error according to the noise level
            expected_error = self.config.noise_level ** 2
            quality_loss = torch.abs(prediction_error - expected_error)
        else:
            if information_attention_weights is not None:
                # encourage attention weights to be consistent with information reliability
                expected_weights = torch.tensor([0.6, 0.3, 0.1], device=self.device)  # private > public > other
                quality_loss = F.mse_loss(information_attention_weights, expected_weights)
            else:
                quality_loss = torch.tensor(0.0, device=self.device)
        
        return quality_loss
    
    def compute_exploration_loss(self, action_distributions, exploration_bonus=None):
        """
        compute exploration encouragement loss
        
        in partially observable environments, appropriate exploration is particularly important
        
        Args:
            action_distributions: action distribution
            exploration_bonus: exploration reward (optional)
            
        Returns:
            torch.Tensor: exploration loss
        """
        # calculate the entropy of action distribution
        if hasattr(action_distributions, 'entropy'):
            entropy = action_distributions.entropy().mean()
        else:
            # for continuous actions, assume normal distribution
            entropy = 0.5 * torch.log(2 * np.pi * np.e * torch.var(action_distributions))
        
        # exploration loss: negative entropy (encourage exploration)
        exploration_loss = -entropy
        
        # if there is exploration reward, consider it
        if exploration_bonus is not None:
            exploration_loss -= exploration_bonus.mean()
        
        return exploration_loss
    
    def compute_total_loss(self, action_log_probs, old_action_log_probs, advantages, 
                           values, returns, private_features=None, public_features=None, 
                           others_features=None, others_actions=None, active_masks=None):
        """
        compute total loss
        
        Args:
            action_log_probs: current policy's action log probabilities
            old_action_log_probs: old policy's action log probabilities
            advantages: advantage function values
            values: value function prediction
            returns: return
            private_features: private information features
            public_features: public information features
            others_features: other information features
            others_actions: other actions
            active_masks: active masks
            
        Returns:
            dict: detailed dictionary containing all losses
        """
        # basic PPO loss
        ppo_losses = self.compute_ppo_loss(
            action_log_probs, old_action_log_probs, advantages, values, returns, active_masks
        )
        
        # Dec-POMDP specific losses
        uncertainty_loss = self.compute_uncertainty_loss(
            private_features, public_features, others_features
        )
        
        collaboration_loss = self.compute_collaboration_loss(
            None, others_actions  
        )
        
        # calculate total loss
        total_policy_loss = (
            ppo_losses['policy_loss'] +
            self.uncertainty_weight * uncertainty_loss +
            self.collaboration_weight * collaboration_loss
        )
        
        total_value_loss = self.value_loss_coef * ppo_losses['value_loss']
        total_loss = total_policy_loss + total_value_loss
        
        return {
            'total_loss': total_loss,
            'total_policy_loss': total_policy_loss,
            'total_value_loss': total_value_loss,
            'ppo_policy_loss': ppo_losses['policy_loss'],
            'ppo_value_loss': ppo_losses['value_loss'],
            'uncertainty_loss': uncertainty_loss,
            'collaboration_loss': collaboration_loss,
            'ratio_mean': ppo_losses['ratio_mean'],
            'ratio_std': ppo_losses['ratio_std']
        }

class DecPOMDPTrainer:
    
    def __init__(self, policy, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.policy = policy
        self.config = dec_pomdp_config
        self.device = device
        
        # create loss computer
        self.loss_computer = DecPOMDPLossComputer(dec_pomdp_config, device)
        
        # training statistics
        self.training_stats = {
            'total_updates': 0,
            'loss_history': [],
            'uncertainty_history': [],
            'collaboration_history': []
        }
    
    def update_policy(self, samples, update_actor=True):
        """
        update policy
        
        Args:
            samples: training samples
            update_actor: whether to update actor
            
        Returns:
            dict: training statistics
        """
        # parse sample data
        observations = samples.get('observations')
        actions = samples.get('actions')
        old_action_log_probs = samples.get('old_action_log_probs')
        advantages = samples.get('advantages')
        values = samples.get('values')
        returns = samples.get('returns')
        
        # parse Dec-POMDP specific information
        private_features = samples.get('private_features')
        public_features = samples.get('public_features')
        others_features = samples.get('others_features')
        others_actions = samples.get('others_actions')
        
        # forward propagation
        current_policy_outputs = self.policy.evaluate_actions(
            observations, actions
        )
        
        old_policy_outputs = {
            'action_log_probs': old_action_log_probs
        }
        
        # calculate loss
        loss_dict = self.loss_computer.compute_total_loss(
            current_policy_outputs.get('action_log_probs'),
            old_policy_outputs.get('action_log_probs'),
            advantages,
            values,
            returns,
            private_features,
            public_features,
            others_features,
            others_actions
        )
        
        # update network
        if update_actor:
            self.policy.actor_optimizer.zero_grad()
            loss_dict['total_policy_loss'].backward(retain_graph=True)
            self.policy.actor_optimizer.step()
        
        self.policy.critic_optimizer.zero_grad()
        loss_dict['total_value_loss'].backward()
        self.policy.critic_optimizer.step()
        
        # update training statistics
        self.training_stats['total_updates'] += 1
        self.training_stats['loss_history'].append(loss_dict['total_loss'].item())
        self.training_stats['uncertainty_history'].append(loss_dict['uncertainty_loss'].item())
        self.training_stats['collaboration_history'].append(loss_dict['collaboration_loss'].item())
        
        return loss_dict
    
    def get_training_stats(self):
        """get training statistics"""
        return self.training_stats.copy() 
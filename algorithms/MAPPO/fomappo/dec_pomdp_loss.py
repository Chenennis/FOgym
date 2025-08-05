#!/usr/bin/env python3
"""
Dec-POMDP Specific Loss Functions

Loss functions specially designed for Dec-POMDP environments, considering:
1. Uncertainty from partial observability
2. Impact of information asymmetry
3. Reliability of information from other agents
4. Balance between cooperation and competition
5. Information quality-aware reward design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPLossComputer:
    """
    Dec-POMDP Loss Function Calculator
    
    Integrates multiple loss functions:
    - Basic PPO loss (policy loss + value loss)
    - Information uncertainty loss
    - Collaborative consistency loss
    - Information quality-aware loss
    - Exploration encouragement loss
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Loss function weights
        self.uncertainty_weight = 0.1        # Uncertainty loss weight
        self.collaboration_weight = 0.05     # Collaboration consistency loss weight
        self.information_quality_weight = 0.03  # Information quality loss weight
        self.exploration_weight = 0.02       # Exploration encouragement loss weight
        
        # Parameters
        self.clip_param = 0.2               # PPO clipping parameter
        self.entropy_coef = 0.01            # Entropy coefficient
        self.value_loss_coef = 0.5          # Value loss coefficient
        
    def compute_ppo_loss(self, action_log_probs, old_action_log_probs, advantages, 
                         values, returns, active_masks=None):
        """
        Calculate basic PPO loss
        
        Args:
            action_log_probs: Action log probabilities of current policy
            old_action_log_probs: Action log probabilities of old policy
            advantages: Advantage function values
            values: Value function predictions
            returns: Returns
            active_masks: Active masks
            
        Returns:
            dict: Dictionary containing various loss components
        """
        # Importance sampling ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        
        # PPO clipping objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Policy loss
        if active_masks is not None:
            policy_loss = -(torch.min(surr1, surr2) * active_masks).sum() / active_masks.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values, returns)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'ratio_mean': ratio.mean(),
            'ratio_std': ratio.std()
        }
    
    def compute_uncertainty_loss(self, private_features, public_features, others_features):
        """
        Calculate information uncertainty loss
        
        Encourages agents to maintain appropriate caution regarding uncertain information in partially observable environments
        
        Args:
            private_features: Private information features
            public_features: Public information features
            others_features: Other agents' information features
            
        Returns:
            uncertainty_loss: Information uncertainty loss
        """
        # Calculate uncertainty based on feature variance
        if others_features is None:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate variance of other agents' features as a measure of uncertainty
        if len(others_features.shape) > 2:
            others_variance = torch.var(others_features, dim=1).mean()
        else:
            others_variance = torch.var(others_features, dim=0).mean()
        
        # Calculate variance of private features
        if len(private_features.shape) > 2:
            private_variance = torch.var(private_features, dim=1).mean()
        else:
            private_variance = torch.var(private_features, dim=0).mean()
        
        # Uncertainty loss: penalize high variance in private features and low variance in others' features
        # This encourages confidence in private information and caution with others' information
        uncertainty_loss = private_variance + 1.0 / (others_variance + 1e-8)
        
        return uncertainty_loss
    
    def compute_collaboration_loss(self, actions, others_actions=None):
        """
        Calculate collaboration consistency loss
        
        Encourages consistent and coordinated actions among agents
        
        Args:
            actions: Agent's actions
            others_actions: Other agents' actions
            
        Returns:
            collaboration_loss: Collaboration consistency loss
        """
        if others_actions is None:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate action diversity
        if len(actions.shape) > 2:
            # Batch of actions for multiple agents
            action_diversity = torch.var(actions, dim=1).mean()
        else:
            # Single agent actions
            action_diversity = torch.var(actions, dim=0).mean()
        
        # Calculate coordination loss
        if len(others_actions.shape) > 2:
            # Calculate difference between agent's actions and others' actions
            action_diff = torch.mean(torch.abs(actions.unsqueeze(1) - others_actions), dim=2).mean()
        else:
            # Simple case
            action_diff = torch.mean(torch.abs(actions - others_actions))
        
        # Balance between diversity (for exploration) and coordination (for collaboration)
        # We want moderate diversity but also coordination with others
        target_diversity = 0.5  # Target diversity level
        collaboration_loss = torch.abs(action_diversity - target_diversity) + 0.5 * action_diff
        
        return collaboration_loss
    
    def compute_information_quality_loss(self, predicted_others_info, actual_others_info=None,
                                         information_attention_weights=None):
        """
        Calculate information quality-aware loss
        
        Encourages accurate modeling of other agents' information and appropriate attention allocation
        
        Args:
            predicted_others_info: Predicted information about other agents
            actual_others_info: Actual information about other agents
            information_attention_weights: Attention weights for different information sources
            
        Returns:
            information_quality_loss: Information quality loss
        """
        if actual_others_info is None or predicted_others_info is None:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate prediction error
        prediction_error = F.mse_loss(predicted_others_info, actual_others_info)
        
        # If attention weights are provided, calculate attention allocation loss
        attention_loss = torch.tensor(0.0, device=self.device)
        if information_attention_weights is not None:
            # Encourage sparse attention (focus on important information)
            attention_entropy = -(information_attention_weights * 
                                torch.log(information_attention_weights + 1e-8)).sum(dim=-1).mean()
            
            # Penalize uniform attention
            uniform_attention = torch.ones_like(information_attention_weights) / information_attention_weights.shape[-1]
            uniform_distance = F.kl_div(
                F.log_softmax(information_attention_weights, dim=-1),
                uniform_attention,
                reduction='batchmean'
            )
            
            # We want low entropy (focused attention) but also distance from uniform
            attention_loss = attention_entropy - 0.1 * uniform_distance
        
        # Combine prediction error and attention loss
        information_quality_loss = prediction_error + 0.1 * attention_loss
        
        return information_quality_loss
    
    def compute_exploration_loss(self, action_distributions, exploration_bonus=None):
        """
        Calculate exploration encouragement loss
        
        Encourages appropriate exploration in Dec-POMDP settings
        
        Args:
            action_distributions: Action probability distributions
            exploration_bonus: Optional exploration bonus based on state visitation
            
        Returns:
            exploration_loss: Exploration encouragement loss
        """
        # Calculate entropy of action distributions
        if hasattr(action_distributions, 'entropy'):
            # If distribution object with entropy method is provided
            entropy = action_distributions.entropy().mean()
        else:
            # If logits or probabilities are provided
            if len(action_distributions.shape) > 2:
                # Batch of distributions
                entropy = -(F.softmax(action_distributions, dim=-1) * 
                          F.log_softmax(action_distributions, dim=-1)).sum(dim=-1).mean()
            else:
                # Single distribution
                entropy = -(F.softmax(action_distributions, dim=-1) * 
                          F.log_softmax(action_distributions, dim=-1)).sum(dim=-1)
        
        # Apply exploration bonus if provided
        if exploration_bonus is not None:
            exploration_loss = -entropy - exploration_bonus.mean()
        else:
            exploration_loss = -entropy
        
        return exploration_loss
    
    def compute_total_loss(self, action_log_probs, old_action_log_probs, advantages, 
                           values, returns, private_features=None, public_features=None, 
                           others_features=None, others_actions=None, active_masks=None):
        """
        Calculate total loss for Dec-POMDP setting
        
        Combines all loss components with appropriate weights
        
        Args:
            action_log_probs: Current policy action log probabilities
            old_action_log_probs: Old policy action log probabilities
            advantages: Advantage function values
            values: Value function predictions
            returns: Returns
            private_features: Private information features
            public_features: Public information features
            others_features: Other agents' information features
            others_actions: Other agents' actions
            active_masks: Active masks
            
        Returns:
            dict: Dictionary containing all loss components and total loss
        """
        # Compute basic PPO loss
        ppo_loss_dict = self.compute_ppo_loss(
            action_log_probs, old_action_log_probs, advantages, values, returns, active_masks
        )
        
        # Extract policy and value losses
        policy_loss = ppo_loss_dict['policy_loss']
        value_loss = ppo_loss_dict['value_loss']
        
        # Initialize additional losses
        uncertainty_loss = torch.tensor(0.0, device=self.device)
        collaboration_loss = torch.tensor(0.0, device=self.device)
        information_quality_loss = torch.tensor(0.0, device=self.device)
        exploration_loss = torch.tensor(0.0, device=self.device)
        
        # Compute additional losses if features are provided
        if private_features is not None and public_features is not None and others_features is not None:
            uncertainty_loss = self.compute_uncertainty_loss(
                private_features, public_features, others_features
            )
        
        if others_actions is not None:
            collaboration_loss = self.compute_collaboration_loss(
                action_log_probs.exp(), others_actions
            )
        
        # Compute total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss +
            self.uncertainty_weight * uncertainty_loss +
            self.collaboration_weight * collaboration_loss +
            self.information_quality_weight * information_quality_loss +
            self.exploration_weight * exploration_loss
        )
        
        # Return all loss components
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'uncertainty_loss': uncertainty_loss,
            'collaboration_loss': collaboration_loss,
            'information_quality_loss': information_quality_loss,
            'exploration_loss': exploration_loss
        }

class DecPOMDPTrainer:
    """
    Dec-POMDP Trainer
    
    Integrates Dec-POMDP loss functions with policy optimization
    """
    
    def __init__(self, policy, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.policy = policy
        self.config = dec_pomdp_config
        self.device = device
        
        # Create loss computer
        self.loss_computer = DecPOMDPLossComputer(dec_pomdp_config, device)
        
        # Optimizer parameters
        self.max_grad_norm = 0.5
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'uncertainty_loss': [],
            'collaboration_loss': [],
            'total_loss': []
        }
    
    def update_policy(self, samples, update_actor=True):
        """
        Update policy using Dec-POMDP loss functions
        
        Args:
            samples: Dictionary containing sampled experiences
            update_actor: Whether to update actor network
            
        Returns:
            dict: Dictionary containing training statistics
        """
        # Extract sample data
        share_obs_batch = samples['share_obs']
        obs_batch = samples['obs']
        actions_batch = samples['actions']
        value_preds_batch = samples['value_preds']
        return_batch = samples['returns']
        old_action_log_probs_batch = samples['action_log_probs']
        adv_targ = samples['advantages']
        active_masks_batch = samples.get('active_masks', None)
        
        # Extract Dec-POMDP specific data if available
        private_features = samples.get('private_features', None)
        public_features = samples.get('public_features', None)
        others_features = samples.get('others_features', None)
        others_actions = samples.get('others_actions', None)
        
        # Convert to tensors
        share_obs_batch = torch.FloatTensor(share_obs_batch).to(self.device)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        actions_batch = torch.FloatTensor(actions_batch).to(self.device)
        value_preds_batch = torch.FloatTensor(value_preds_batch).to(self.device)
        return_batch = torch.FloatTensor(return_batch).to(self.device)
        old_action_log_probs_batch = torch.FloatTensor(old_action_log_probs_batch).to(self.device)
        adv_targ = torch.FloatTensor(adv_targ).to(self.device)
        
        if active_masks_batch is not None:
            active_masks_batch = torch.FloatTensor(active_masks_batch).to(self.device)
        
        # Evaluate actions
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, actions_batch
        )
        
        # Compute Dec-POMDP losses
        loss_dict = self.loss_computer.compute_total_loss(
            action_log_probs, old_action_log_probs_batch, adv_targ,
            values, return_batch, private_features, public_features,
            others_features, others_actions, active_masks_batch
        )
        
        # Extract losses
        total_loss = loss_dict['total_loss']
        policy_loss = loss_dict['policy_loss']
        value_loss = loss_dict['value_loss']
        uncertainty_loss = loss_dict['uncertainty_loss']
        collaboration_loss = loss_dict['collaboration_loss']
        
        # Update actor
        if update_actor:
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        
        # Update training statistics
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['uncertainty_loss'].append(uncertainty_loss.item())
        self.training_stats['collaboration_loss'].append(collaboration_loss.item())
        self.training_stats['total_loss'].append(total_loss.item())
        
        return loss_dict
    
    def get_training_stats(self):
        """
        Get training statistics
        
        Returns:
            dict: Dictionary containing training statistics
        """
        return {k: np.mean(v) if v else 0.0 for k, v in self.training_stats.items()} 
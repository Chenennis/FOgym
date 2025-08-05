#!/usr/bin/env python3
"""
Dec-POMDP特定损失函数

专门为Dec-POMDP环境设计的损失函数，考虑：
1. 部分可观测性带来的不确定性
2. 信息不对称的影响
3. 他者信息的可靠性
4. 协作与竞争的平衡
5. 信息质量感知的奖励设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPLossComputer:
    """
    Dec-POMDP损失函数计算器
    
    集成多种损失函数：
    - 基础PPO损失（策略损失 + 价值损失）
    - 信息不确定性损失
    - 协作一致性损失
    - 信息质量感知损失
    - 探索鼓励损失
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # 损失函数权重
        self.uncertainty_weight = 0.1        # 不确定性损失权重
        self.collaboration_weight = 0.05     # 协作一致性损失权重
        self.information_quality_weight = 0.03  # 信息质量损失权重
        self.exploration_weight = 0.02       # 探索鼓励损失权重
        
        # 参数
        self.clip_param = 0.2               # PPO裁剪参数
        self.entropy_coef = 0.01            # 熵系数
        self.value_loss_coef = 0.5          # 价值损失系数
        
    def compute_ppo_loss(self, action_log_probs, old_action_log_probs, advantages, 
                         values, returns, active_masks=None):
        """
        计算基础PPO损失
        
        Args:
            action_log_probs: 当前策略的动作对数概率
            old_action_log_probs: 旧策略的动作对数概率
            advantages: 优势函数值
            values: 价值函数预测
            returns: 回报
            active_masks: 激活掩码
            
        Returns:
            dict: 包含各项损失的字典
        """
        # 重要性采样比率
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # 策略损失
        if active_masks is not None:
            policy_loss = -(torch.min(surr1, surr2) * active_masks).sum() / active_masks.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        value_loss = F.mse_loss(values, returns)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'ratio_mean': ratio.mean(),
            'ratio_std': ratio.std()
        }
    
    def compute_uncertainty_loss(self, private_features, public_features, others_features):
        """
        计算信息不确定性损失
        
        在部分可观测环境中，鼓励智能体对不确定信息保持适当的谨慎
        
        Args:
            private_features: 私有信息特征
            public_features: 公共信息特征  
            others_features: 他者信息特征
            
        Returns:
            torch.Tensor: 不确定性损失
        """
        if not self.config.enable_observation_noise:
            return torch.tensor(0.0, device=self.device)
        
        # 计算不同信息源的可靠性权重
        private_reliability = 1.0  # 私有信息最可靠
        public_reliability = 0.8   # 公共信息较可靠
        others_reliability = 0.5 if self.config.enable_other_manager_info else 0.0  # 他者信息不太可靠
        
        # 根据噪声水平调整可靠性
        noise_factor = 1.0 - self.config.noise_level
        others_reliability *= noise_factor
        
        # 计算信息特征的方差（不确定性指标）
        private_var = torch.var(private_features, dim=-1).mean() if private_features is not None else 0.0
        public_var = torch.var(public_features, dim=-1).mean() if public_features is not None else 0.0
        others_var = torch.var(others_features, dim=-1).mean() if others_features is not None else 0.0
        
        # 加权不确定性
        weighted_uncertainty = (
            private_var * (1.0 - private_reliability) +
            public_var * (1.0 - public_reliability) +
            others_var * (1.0 - others_reliability)
        )
        
        return weighted_uncertainty
    
    def compute_collaboration_loss(self, actions, others_actions=None):
        """
        计算协作一致性损失
        
        鼓励Manager之间的协作，特别是在有他者信息时
        
        Args:
            actions: 当前Manager的动作
            others_actions: 其他Manager的动作（如果可获得）
            
        Returns:
            torch.Tensor: 协作损失
        """
        if not self.config.enable_other_manager_info or others_actions is None:
            return torch.tensor(0.0, device=self.device)
        
        # 动作一致性损失：鼓励相似的动作策略
        action_consistency = torch.norm(actions - others_actions, dim=-1).mean()
        
        # 避免过度一致（保持多样性）
        diversity_threshold = 0.5
        consistency_loss = torch.relu(diversity_threshold - action_consistency)
        
        return consistency_loss
    
    def compute_information_quality_loss(self, predicted_others_info, actual_others_info=None,
                                         information_attention_weights=None):
        """
        计算信息质量感知损失
        
        鼓励智能体正确评估和利用不同质量的信息
        
        Args:
            predicted_others_info: 预测的他者信息
            actual_others_info: 实际的他者信息（如果可获得）
            information_attention_weights: 信息注意力权重
            
        Returns:
            torch.Tensor: 信息质量损失
        """
        if not self.config.enable_other_manager_info:
            return torch.tensor(0.0, device=self.device)
        
        # 如果有实际他者信息，计算预测误差
        if actual_others_info is not None:
            prediction_error = F.mse_loss(predicted_others_info, actual_others_info)
            
            # 根据噪声水平调整期望误差
            expected_error = self.config.noise_level ** 2
            quality_loss = torch.abs(prediction_error - expected_error)
        else:
            # 没有真实标签时，鼓励注意力权重的合理性
            if information_attention_weights is not None:
                # 鼓励注意力权重与信息可靠性相符
                expected_weights = torch.tensor([0.6, 0.3, 0.1], device=self.device)  # 私有>公共>他者
                quality_loss = F.mse_loss(information_attention_weights, expected_weights)
            else:
                quality_loss = torch.tensor(0.0, device=self.device)
        
        return quality_loss
    
    def compute_exploration_loss(self, action_distributions, exploration_bonus=None):
        """
        计算探索鼓励损失
        
        在部分可观测环境中，适当的探索特别重要
        
        Args:
            action_distributions: 动作分布
            exploration_bonus: 探索奖励（可选）
            
        Returns:
            torch.Tensor: 探索损失
        """
        # 计算动作分布的熵
        if hasattr(action_distributions, 'entropy'):
            entropy = action_distributions.entropy().mean()
        else:
            # 对于连续动作，假设正态分布
            entropy = 0.5 * torch.log(2 * np.pi * np.e * torch.var(action_distributions))
        
        # 探索损失：负熵（鼓励探索）
        exploration_loss = -entropy
        
        # 如果有探索奖励，加入考虑
        if exploration_bonus is not None:
            exploration_loss -= exploration_bonus.mean()
        
        return exploration_loss
    
    def compute_total_loss(self, action_log_probs, old_action_log_probs, advantages, 
                           values, returns, private_features=None, public_features=None, 
                           others_features=None, others_actions=None, active_masks=None):
        """
        计算总损失
        
        Args:
            action_log_probs: 当前策略的动作对数概率
            old_action_log_probs: 旧策略的动作对数概率
            advantages: 优势函数值
            values: 价值函数预测
            returns: 回报
            private_features: 私有信息特征
            public_features: 公共信息特征
            others_features: 他者信息特征
            others_actions: 他者动作
            active_masks: 激活掩码
            
        Returns:
            dict: 包含所有损失的详细字典
        """
        # 基础PPO损失
        ppo_losses = self.compute_ppo_loss(
            action_log_probs, old_action_log_probs, advantages, values, returns, active_masks
        )
        
        # Dec-POMDP特定损失
        uncertainty_loss = self.compute_uncertainty_loss(
            private_features, public_features, others_features
        )
        
        collaboration_loss = self.compute_collaboration_loss(
            None, others_actions  # 简化版本，暂时不使用动作
        )
        
        # 计算总损失
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
    """
    Dec-POMDP训练器
    
    集成Dec-POMDP损失函数的训练器
    """
    
    def __init__(self, policy, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.policy = policy
        self.config = dec_pomdp_config
        self.device = device
        
        # 创建损失计算器
        self.loss_computer = DecPOMDPLossComputer(dec_pomdp_config, device)
        
        # 训练统计
        self.training_stats = {
            'total_updates': 0,
            'loss_history': [],
            'uncertainty_history': [],
            'collaboration_history': []
        }
    
    def update_policy(self, samples, update_actor=True):
        """
        更新策略
        
        Args:
            samples: 训练样本
            update_actor: 是否更新actor
            
        Returns:
            dict: 训练统计信息
        """
        # 解析样本数据
        observations = samples.get('observations')
        actions = samples.get('actions')
        old_action_log_probs = samples.get('old_action_log_probs')
        advantages = samples.get('advantages')
        values = samples.get('values')
        returns = samples.get('returns')
        
        # 解析Dec-POMDP特定信息
        private_features = samples.get('private_features')
        public_features = samples.get('public_features')
        others_features = samples.get('others_features')
        others_actions = samples.get('others_actions')
        
        # 前向传播
        current_policy_outputs = self.policy.evaluate_actions(
            observations, actions
        )
        
        old_policy_outputs = {
            'action_log_probs': old_action_log_probs
        }
        
        # 计算损失
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
        
        # 更新网络
        if update_actor:
            self.policy.actor_optimizer.zero_grad()
            loss_dict['total_policy_loss'].backward(retain_graph=True)
            self.policy.actor_optimizer.step()
        
        self.policy.critic_optimizer.zero_grad()
        loss_dict['total_value_loss'].backward()
        self.policy.critic_optimizer.step()
        
        # 更新训练统计
        self.training_stats['total_updates'] += 1
        self.training_stats['loss_history'].append(loss_dict['total_loss'].item())
        self.training_stats['uncertainty_history'].append(loss_dict['uncertainty_loss'].item())
        self.training_stats['collaboration_history'].append(loss_dict['collaboration_loss'].item())
        
        return loss_dict
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return self.training_stats.copy() 
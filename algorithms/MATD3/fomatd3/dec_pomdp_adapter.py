#!/usr/bin/env python3
"""
FOMATD3 Dec-POMDP观测空间适配器

为FOMATD3算法提供Dec-POMDP观测空间的处理能力。
针对Twin Delayed DDPG算法的特点进行优化。

核心功能：
1. 观测空间分层解析（复用Dec-POMDP架构）
2. TD3特定的观测处理优化
3. 双Critic网络的观测增强
4. 目标策略平滑化的观测一致性
5. FlexOffer约束的观测集成
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMAtd3DecPOMDPAdapter:
    """
    FOMATD3 Dec-POMDP观测空间适配器
    
    专门为FOMATD3算法设计的Dec-POMDP观测处理器，
    支持Twin Delayed DDPG和FlexOffer约束的观测空间管理。
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP观测空间维度
        self.private_dim = 39      # 私有观测基础维度
        self.public_dim = 18       # 公共观测维度
        self.others_dim = 15       # 他者观测维度
        self.total_obs_dim = 72    # 总观测维度
        
        # TD3特定参数
        self.twin_critic_mode = True
        self.target_smoothing_factor = 0.8  # 目标策略平滑化因子
        self.delay_update_steps = 2         # 延迟更新步数
        self.observation_history_length = 3 # 观测历史长度（TD3优化）
        
        # FlexOffer约束集成
        self.fo_constraint_dim = 36        # FlexOffer约束维度
        self.fo_satisfaction_weight = 0.2  # FlexOffer满意度权重
        
        # 观测处理缓存
        self._observation_cache = {}
        self._history_buffer = {}
        
        # 初始化观测历史缓存
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """初始化观测历史缓冲区"""
        for manager_id in [f"manager_{i}" for i in range(4)]:
            self._history_buffer[manager_id] = {
                'private': [],
                'public': [],
                'others': [],
                'full_obs': []
            }
    
    def adapt_observation_for_fomatd3(self, 
                                     observation: np.ndarray, 
                                     manager_id: str,
                                     fo_constraints: Optional[np.ndarray] = None,
                                     fo_satisfaction: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        为FOMATD3适配观测空间
        
        Args:
            observation: 原始观测 [obs_dim]
            manager_id: Manager ID (e.g., "manager_0")
            fo_constraints: FlexOffer约束 [constraint_dim]
            fo_satisfaction: FlexOffer满意度标量
            
        Returns:
            适配后的分层观测字典
        """
        # 解析Dec-POMDP观测
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        # FlexOffer约束集成（始终确保40维）
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints(private_obs, fo_constraints, fo_satisfaction)
        else:
            # 如果没有FlexOffer约束，填充到40维
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # TD3特定观测增强
        enhanced_private = self._enhance_private_obs_for_td3(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_td3(public_obs)
        enhanced_others = self._enhance_others_obs_for_td3(others_obs, manager_id)
        
        # 观测噪声处理
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private')
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public')
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others')
        
        # 转换为张量
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device)
        }
        
        # 更新历史缓存
        self._update_history_buffer(manager_id, adapted_obs)
        
        return adapted_obs
    
    def _parse_dec_pomdp_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析Dec-POMDP观测结构"""
        if len(observation) < self.total_obs_dim:
            # 填充不足的维度
            observation = np.pad(observation, (0, self.total_obs_dim - len(observation)))
        elif len(observation) > self.total_obs_dim:
            # 截断多余的维度
            observation = observation[:self.total_obs_dim]
        
        # 分层解析
        private_start = 0
        private_end = self.private_dim
        public_start = private_end
        public_end = public_start + self.public_dim
        others_start = public_end
        others_end = others_start + self.others_dim
        
        private_obs = observation[private_start:private_end]
        public_obs = observation[public_start:public_end]
        others_obs = observation[others_start:others_end]
        
        return private_obs, public_obs, others_obs
    
    def _integrate_fo_constraints(self, 
                                 private_obs: np.ndarray, 
                                 fo_constraints: np.ndarray,
                                 fo_satisfaction: Optional[float] = None) -> np.ndarray:
        """集成FlexOffer约束到私有观测"""
        # FlexOffer约束特征提取
        constraint_features = self._extract_fo_constraint_features(fo_constraints)
        
        # FlexOffer满意度处理
        satisfaction_feature = fo_satisfaction if fo_satisfaction is not None else 0.8
        
        # 添加趋势信息（TD3优化：更关注长期趋势）
        constraint_trend = np.mean(constraint_features) - np.mean(private_obs[:10])  # 前10维比较
        
        # 扩展私有观测：39 + 1(趋势) = 40维
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40]  # 确保维度一致
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """提取FlexOffer约束特征"""
        if len(fo_constraints) == 0:
            return np.zeros(5)  # 默认约束特征
        
        # 统计特征
        constraint_features = np.array([
            np.mean(fo_constraints),     # 平均约束值
            np.std(fo_constraints),      # 约束方差
            np.min(fo_constraints),      # 最小约束
            np.max(fo_constraints),      # 最大约束
            np.sum(fo_constraints > 0.5) / len(fo_constraints)  # 激活比例
        ])
        
        return constraint_features
    
    def _enhance_private_obs_for_td3(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """为TD3增强私有观测"""
        # 获取历史观测用于平滑化
        history = self._history_buffer[manager_id]['private']
        
        if len(history) > 0:
            # TD3目标策略平滑化：结合历史信息
            recent_history = history[-2:] if len(history) >= 2 else history
            if recent_history:
                avg_history = np.mean(recent_history, axis=0)
                # 平滑当前观测
                smoothed_obs = (self.target_smoothing_factor * private_obs + 
                               (1 - self.target_smoothing_factor) * avg_history)
                return smoothed_obs
        
        return private_obs
    
    def _enhance_public_obs_for_td3(self, public_obs: np.ndarray) -> np.ndarray:
        """为TD3增强公共观测"""
        # TD3双Critic特性：增加观测鲁棒性
        # 添加小量噪声提高泛化能力
        if self.twin_critic_mode:
            noise_scale = 0.02  # 较小的噪声
            noise = np.random.normal(0, noise_scale, public_obs.shape)
            robust_obs = public_obs + noise
            return robust_obs
        
        return public_obs
    
    def _enhance_others_obs_for_td3(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """为TD3增强他者观测"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        # TD3延迟更新特性：保守的他者信息处理
        conservative_weight = 0.7  # 比DDPG更保守
        
        # 获取历史他者观测
        history = self._history_buffer[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.delay_update_steps:], axis=0) if len(history) >= self.delay_update_steps else np.mean(history, axis=0)
            # 保守更新
            conservative_obs = conservative_weight * others_obs + (1 - conservative_weight) * recent_avg
            return conservative_obs
        
        return others_obs * conservative_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default') -> np.ndarray:
        """添加观测噪声"""
        if not self.config.enable_observation_noise:
            return observation
        
        # TD3特定噪声设置
        noise_scales = {
            'private': self.config.noise_level * 0.8,  # 私有观测噪声较小
            'public': self.config.noise_level * 0.5,   # 公共观测噪声更小
            'others': self.config.noise_level * 1.2    # 他者观测噪声较大
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # 生成噪声
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # 添加噪声
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _update_history_buffer(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor]):
        """更新观测历史缓冲区"""
        history = self._history_buffer[manager_id]
        
        # 转换为numpy并添加到历史
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # 维持历史长度
        for key in history:
            if len(history[key]) > self.observation_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """获取适配后的观测维度信息"""
        return {
            'private_dim': 40,  # 39 + 1(趋势)
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  # 73
            'history_length': self.observation_history_length,
            'fo_constraint_dim': self.fo_constraint_dim
        }
    
    def get_td3_specific_info(self) -> Dict[str, Any]:
        """获取TD3特定的适配信息"""
        return {
            'twin_critic_mode': self.twin_critic_mode,
            'target_smoothing_factor': self.target_smoothing_factor,
            'delay_update_steps': self.delay_update_steps,
            'observation_history_length': self.observation_history_length,
            'conservative_weight': 0.7,
            'fo_integration': True
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """重置观测历史"""
        if manager_id is not None:
            self._init_history_buffers()
        else:
            if manager_id in self._history_buffer:
                for key in self._history_buffer[manager_id]:
                    self._history_buffer[manager_id][key].clear()
    
    def get_smoothed_observation(self, 
                                manager_id: str, 
                                current_obs: Dict[str, torch.Tensor],
                                smoothing_factor: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """获取平滑化观测（TD3目标策略平滑化）"""
        if smoothing_factor is None:
            smoothing_factor = self.target_smoothing_factor
        
        history = self._history_buffer[manager_id]
        
        if len(history['full_obs']) == 0:
            return current_obs
        
        # 获取最近的观测
        recent_obs = history['full_obs'][-1]
        
        # 平滑化处理
        smoothed_obs = {}
        for key in current_obs:
            if key in ['private', 'public', 'others']:
                current_tensor = current_obs[key]
                if len(history[key]) > 0:
                    recent_tensor = torch.FloatTensor(history[key][-1]).to(self.device)
                    smoothed_tensor = (smoothing_factor * current_tensor + 
                                     (1 - smoothing_factor) * recent_tensor)
                    smoothed_obs[key] = smoothed_tensor
                else:
                    smoothed_obs[key] = current_tensor
            else:
                smoothed_obs[key] = current_obs[key]
        
        return smoothed_obs 
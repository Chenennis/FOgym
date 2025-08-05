#!/usr/bin/env python3
"""
FOMADDPG Dec-POMDP观测空间适配器

为FOMADDPG算法提供Dec-POMDP观测空间的处理能力。
针对Actor-Critic确定性策略梯度算法的特点进行优化。

核心功能：
1. 观测空间分层解析（复用FOMAPPO架构）
2. 确定性策略的观测处理优化
3. 连续动作空间的观测增强
4. Multi-Agent协作信息的DDPG适配
5. 目标网络更新的观测一致性处理
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class FOMaddpgDecPOMDPAdapter:
    """
    FOMADDPG Dec-POMDP观测空间适配器
    
    专门为FOMADDPG算法设计的Dec-POMDP观测处理器，
    支持确定性策略梯度和多智能体协作的观测空间管理。
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP观测空间维度（与FOMAPPO保持一致）
        self.private_dim = 39  # 私有信息层维度
        self.public_dim = 18   # 公共信息层维度  
        self.others_dim = 15   # 有限他者信息层维度
        
        # 总观测维度
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72维
        
        # FOMADDPG特定的观测处理权重
        self.private_weight = 1.0   # 私有信息完全可信
        self.public_weight = 1.0    # 公共信息完全可信
        self.others_weight = 0.7    # 他者信息在DDPG中稍微提高可信度（原0.8→0.7）
        
        # 确定性策略特定参数
        self.deterministic_mode = True  # DDPG使用确定性策略
        self.action_smoothing_factor = 0.95  # 动作平滑因子
        
        # 历史观测缓存（用于目标网络更新的一致性）
        self.observation_history = {}
        self.max_history_len = 5  # DDPG通常需要较短的历史
        
        # 多智能体协作特定缓存
        self.global_observation_cache = None
        self.local_observation_cache = {}
        
    def parse_observation(self, observation: Union[np.ndarray, torch.Tensor], 
                         manager_id: str) -> Dict[str, torch.Tensor]:
        """
        解析Dec-POMDP观测空间的分层结构
        
        Args:
            observation: 完整观测向量 (72维)
            manager_id: Manager标识
            
        Returns:
            Dict containing:
                - 'private': 私有信息层 (39维)
                - 'public': 公共信息层 (18维)  
                - 'others': 有限他者信息层 (15维)
        """
        # 转换为torch tensor
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        # 处理批次维度
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        if observation.shape[-1] != self.total_obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.total_obs_dim}, 实际{observation.shape[-1]}")
        
        # 分离三层观测信息
        private_obs = observation[..., :self.private_dim]
        public_obs = observation[..., self.private_dim:self.private_dim + self.public_dim]
        others_obs = observation[..., self.private_dim + self.public_dim:]
        
        return {
            'private': private_obs,
            'public': public_obs, 
            'others': others_obs
        }
    
    def enhance_private_observation(self, private_obs: torch.Tensor, 
                                  manager_id: str) -> torch.Tensor:
        """
        增强私有观测信息 - 针对确定性策略优化
        
        DDPG特点：
        1. 确定性策略需要更稳定的观测
        2. 减少观测噪声的影响
        3. 保持时序一致性
        """
        enhanced_private = private_obs.clone()
        
        # 对于确定性策略，使用更平滑的历史信息处理
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-2:]  # 只使用最近2步
            if len(recent_obs) >= 2:
                # 计算趋势，但使用更平滑的方式
                prev_private = recent_obs[-1][:self.private_dim] if recent_obs[-1].shape[0] >= self.private_dim else torch.zeros_like(enhanced_private[0])
                trend = enhanced_private[0] - prev_private
                trend_norm = torch.norm(trend).item()
                
                # 使用指数平滑
                smoothed_trend = min(1.0, trend_norm) * self.action_smoothing_factor
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[smoothed_trend]]).to(self.device)], dim=-1)
            else:
                enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        else:
            enhanced_private = torch.cat([enhanced_private, torch.tensor([[0.0]]).to(self.device)], dim=-1)
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: torch.Tensor) -> torch.Tensor:
        """
        处理公共观测信息 - DDPG优化版本
        
        DDPG特点：
        1. 公共信息对所有智能体保持一致
        2. 支持集中式训练的信息共享
        3. 确保目标网络更新时的一致性
        """
        # 缓存全局观测信息，确保多智能体一致性
        processed_public = public_obs.clone()
        
        # 验证公共信息的合理性
        if torch.any(torch.isnan(processed_public)) or torch.any(torch.isinf(processed_public)):
            print(f"警告: 公共观测信息包含无效值")
            processed_public = torch.nan_to_num(processed_public, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新全局观测缓存
        self.global_observation_cache = processed_public.clone()
        
        return processed_public
    
    def process_others_observation(self, others_obs: torch.Tensor, 
                                 manager_id: str) -> torch.Tensor:
        """
        处理有限他者观测信息 - DDPG多智能体优化
        
        DDPG特点：
        1. 在集中式训练中，他者信息更重要
        2. 确定性策略对噪声更敏感
        3. 需要平衡探索和利用
        """
        if not self.config.enable_other_manager_info:
            # 如果禁用他者信息，返回零向量
            return torch.zeros_like(others_obs)
        
        processed_others = others_obs.clone()
        
        # DDPG特定的噪声处理（更温和）
        if self.config.enable_observation_noise:
            noise_level = self.config.noise_level * 0.7  # DDPG使用更小的噪声
            noise = torch.randn_like(processed_others) * noise_level
            processed_others += noise
        
        # 应用信息质量权重
        processed_others *= self.others_weight
        
        # DDPG中的信息丢失处理（更保守）
        if self.config.enable_observation_noise and hasattr(self.config, 'enable_info_missing'):
            if getattr(self.config, 'enable_info_missing', False):
                loss_prob = self.config.noise_level * 0.3  # 更低的信息丢失概率
                loss_mask = torch.rand_like(processed_others) > loss_prob
                processed_others *= loss_mask.float()
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: torch.Tensor, 
                              public_obs: torch.Tensor, 
                              others_obs: torch.Tensor,
                              manager_id: str,
                              enhanced: bool = True) -> torch.Tensor:
        """
        重构完整观测向量 - DDPG优化版本
        
        Args:
            private_obs: 处理后的私有观测
            public_obs: 处理后的公共观测
            others_obs: 处理后的他者观测
            manager_id: Manager标识
            enhanced: 是否使用增强模式
            
        Returns:
            重构的完整观测向量（适合DDPG使用）
        """
        if enhanced:
            # DDPG增强模式：添加确定性策略相关的特征
            
            # 计算私有-公共信息的相关性（使用cosine similarity）
            private_flat = private_obs.view(private_obs.shape[0], -1)
            public_flat = public_obs.view(public_obs.shape[0], -1)
            
            # 确保维度匹配
            min_dim = min(private_flat.shape[1], public_flat.shape[1])
            private_flat = private_flat[:, :min_dim]
            public_flat = public_flat[:, :min_dim]
            
            # 计算cosine similarity
            private_norm = torch.norm(private_flat, dim=1, keepdim=True) + 1e-8
            public_norm = torch.norm(public_flat, dim=1, keepdim=True) + 1e-8
            
            private_public_corr = torch.sum(private_flat * public_flat, dim=1) / (private_norm.squeeze() * public_norm.squeeze())
            private_public_corr = torch.tanh(private_public_corr).unsqueeze(1)  # 标准化到[-1,1]
            
            # 计算观测稳定性指标（DDPG特有）
            if manager_id in self.observation_history and len(self.observation_history[manager_id]) > 0:
                prev_obs = self.observation_history[manager_id][-1]
                current_obs = torch.cat([private_obs, public_obs, others_obs], dim=-1)
                
                # 确保维度匹配
                if prev_obs.shape[1] == current_obs.shape[1]:
                    stability = 1.0 - torch.norm(current_obs - prev_obs, dim=1).mean().item()
                    stability = max(0.0, min(1.0, stability))  # 限制在[0,1]
                else:
                    stability = 0.5  # 默认中等稳定性
            else:
                stability = 0.5
            
            stability_feature = torch.tensor([[stability]], device=self.device).expand(private_obs.shape[0], 1)
            
            # 添加DDPG特定的交互特征
            interaction_features = torch.cat([private_public_corr, stability_feature], dim=1)
            
            # 重构观测：私有 + 公共 + 他者 + 交互
            reconstructed = torch.cat([private_obs, public_obs, others_obs, interaction_features], dim=-1)
        else:
            # 基础模式：简单拼接
            reconstructed = torch.cat([private_obs, public_obs, others_obs], dim=-1)
        
        # 缓存观测用于稳定性计算
        self.local_observation_cache[manager_id] = reconstructed.clone()
        
        return reconstructed
    
    def adapt_observation_for_fomaddpg(self, observation: Union[np.ndarray, torch.Tensor], 
                                     manager_id: str) -> Dict[str, torch.Tensor]:
        """
        为FOMADDPG算法适配观测
        
        Args:
            observation: 原始观测
            manager_id: Manager标识
            
        Returns:
            适配后的观测字典，包含各层信息和融合后的观测
        """
        # 1. 解析观测
        parsed_obs = self.parse_observation(observation, manager_id)
        
        # 2. 增强各层观测
        enhanced_private = self.enhance_private_observation(parsed_obs['private'], manager_id)
        processed_public = self.process_public_observation(parsed_obs['public'])
        processed_others = self.process_others_observation(parsed_obs['others'], manager_id)
        
        # 3. 重构完整观测
        fused_observation = self.reconstruct_observation(
            enhanced_private, processed_public, processed_others, manager_id, enhanced=True
        )
        
        # 4. 更新历史记录
        self._update_observation_history(manager_id, fused_observation)
        
        return {
            'private': enhanced_private,
            'public': processed_public,
            'others': processed_others,
            'fused': fused_observation,
            'raw_parsed': parsed_obs
        }
    
    def _update_observation_history(self, manager_id: str, observation: torch.Tensor):
        """更新观测历史"""
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.clone())
        
        # 保持历史长度限制
        if len(self.observation_history[manager_id]) > self.max_history_len:
            self.observation_history[manager_id].pop(0)
    
    def get_observation_stats(self, manager_id: str) -> Dict[str, float]:
        """获取观测统计信息"""
        if manager_id not in self.observation_history or len(self.observation_history[manager_id]) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'stability': 0.0,
                'history_length': 0
            }
        
        recent_obs = torch.stack(self.observation_history[manager_id])
        
        return {
            'mean': recent_obs.mean().item(),
            'std': recent_obs.std().item(),
            'stability': 1.0 - recent_obs.std().item() if recent_obs.std().item() < 1.0 else 0.0,
            'history_length': len(self.observation_history[manager_id])
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """重置观测历史"""
        if manager_id is None:
            self.observation_history.clear()
            self.local_observation_cache.clear()
            self.global_observation_cache = None
        else:
            if manager_id in self.observation_history:
                del self.observation_history[manager_id]
            if manager_id in self.local_observation_cache:
                del self.local_observation_cache[manager_id]
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """获取适配后的观测维度信息"""
        return {
            'private_dim': self.private_dim + 1,  # +1 for trend
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'interaction_dim': 2,  # private_public_corr + stability
            'total_enhanced_dim': self.private_dim + 1 + self.public_dim + self.others_dim + 2
        }
    
    def enable_deterministic_mode(self, deterministic: bool = True):
        """启用/禁用确定性模式"""
        self.deterministic_mode = deterministic
        if deterministic:
            self.action_smoothing_factor = 0.98  # 更平滑
            self.others_weight = 0.8  # 提高他者信息权重
        else:
            self.action_smoothing_factor = 0.95  # 正常
            self.others_weight = 0.7  # 正常权重 
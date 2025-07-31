#!/usr/bin/env python3
"""
Dec-POMDP观测空间适配器

为FOMAPPO算法提供Dec-POMDP观测空间的处理能力。
支持三层观测信息的分离、处理和重组。

核心功能：
1. 观测空间分层解析
2. 私有信息处理
3. 公共信息标准化  
4. 有限他者信息噪声处理
5. 观测空间重组和增强
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fo_common.dec_pomdp_config import DecPOMDPConfig

class DecPOMDPObservationAdapter:
    """
    Dec-POMDP观测空间适配器
    
    处理分层观测信息，为FOMAPPO算法提供结构化的观测输入
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP观测空间维度（基于已实现的架构）
        self.private_dim = 39  # 私有信息层维度
        self.public_dim = 18   # 公共信息层维度  
        self.others_dim = 15   # 有限他者信息层维度
        
        # 总观测维度
        self.total_obs_dim = self.private_dim + self.public_dim + self.others_dim  # 72维
        
        # 观测处理权重
        self.private_weight = 1.0  # 私有信息完全可信
        self.public_weight = 1.0   # 公共信息完全可信
        self.others_weight = 0.8   # 他者信息部分可信（可配置噪声）
        
        # 历史观测缓存
        self.observation_history = {}
        self.max_history_len = 10
        
    def parse_observation(self, observation: np.ndarray, manager_id: str) -> Dict[str, np.ndarray]:
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
        if len(observation) != self.total_obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.total_obs_dim}, 实际{len(observation)}")
        
        # 分离三层观测信息
        private_obs = observation[:self.private_dim]
        public_obs = observation[self.private_dim:self.private_dim + self.public_dim]
        others_obs = observation[self.private_dim + self.public_dim:]
        
        return {
            'private': private_obs,
            'public': public_obs, 
            'others': others_obs
        }
    
    def enhance_private_observation(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """
        增强私有观测信息
        
        私有信息处理原则：
        1. 保持完整性（无噪声）
        2. 标准化处理
        3. 添加时序特征
        """
        enhanced_private = private_obs.copy()
        
        # 确保私有观测标准化
        # 已在Manager中进行标准化，这里保持原样
        
        # 添加历史趋势信息（如果有历史）
        if manager_id in self.observation_history:
            recent_obs = self.observation_history[manager_id][-3:]  # 最近3步
            if len(recent_obs) >= 2:
                # 计算私有信息趋势
                trend = recent_obs[-1][:self.private_dim] - recent_obs[-2][:self.private_dim]
                trend_norm = np.linalg.norm(trend)
                
                # 添加趋势强度作为私有信息增强
                enhanced_private = np.append(enhanced_private, min(1.0, trend_norm))
            else:
                enhanced_private = np.append(enhanced_private, 0.0)
        else:
            enhanced_private = np.append(enhanced_private, 0.0)
        
        return enhanced_private
    
    def process_public_observation(self, public_obs: np.ndarray) -> np.ndarray:
        """
        处理公共观测信息
        
        公共信息处理原则：
        1. 所有Manager观测相同（无噪声）
        2. 标准化和归一化
        3. 确保信息一致性
        """
        # 公共信息保持原样，已经在环境中标准化
        processed_public = public_obs.copy()
        
        # 验证公共信息的合理性
        if np.any(np.isnan(processed_public)) or np.any(np.isinf(processed_public)):
            print(f"警告: 公共观测信息包含无效值")
            processed_public = np.nan_to_num(processed_public, nan=0.0, posinf=0.0, neginf=0.0)
        
        return processed_public
    
    def process_others_observation(self, others_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """
        处理有限他者观测信息
        
        他者信息处理原则：
        1. 应用噪声和不确定性
        2. 信息质量降级
        3. 部分信息丢失模拟
        """
        if not self.config.enable_other_manager_info:
            # 如果禁用他者信息，返回零向量
            return np.zeros_like(others_obs)
        
        processed_others = others_obs.copy()
        
        # 应用观测噪声（如果启用）
        if self.config.enable_observation_noise:
            noise_level = self.config.noise_level
            noise = np.random.normal(0, noise_level, processed_others.shape)
            processed_others += noise
        
        # 应用信息质量权重
        processed_others *= self.others_weight
        
        # 模拟信息丢失（随机将部分信息置零）
        if self.config.enable_observation_noise:
            loss_prob = self.config.noise_level * 0.5  # 信息丢失概率
            loss_mask = np.random.random(processed_others.shape) > loss_prob
            processed_others *= loss_mask
        
        return processed_others
    
    def reconstruct_observation(self, 
                              private_obs: np.ndarray, 
                              public_obs: np.ndarray, 
                              others_obs: np.ndarray,
                              enhanced: bool = True) -> np.ndarray:
        """
        重构完整观测向量
        
        Args:
            private_obs: 处理后的私有观测
            public_obs: 处理后的公共观测
            others_obs: 处理后的他者观测
            enhanced: 是否使用增强模式
            
        Returns:
            重构的完整观测向量
        """
        if enhanced:
            # 增强模式：添加层间交互信息
            
            # 计算私有-公共信息的相关性
            private_public_corr = np.dot(private_obs[:min(len(private_obs), len(public_obs))], 
                                       public_obs[:min(len(private_obs), len(public_obs))])
            private_public_corr = np.tanh(private_public_corr)  # 标准化到[-1,1]
            
            # 计算私有-他者信息的相关性
            private_others_corr = np.dot(private_obs[:min(len(private_obs), len(others_obs))], 
                                       others_obs[:min(len(private_obs), len(others_obs))])
            private_others_corr = np.tanh(private_others_corr)  # 标准化到[-1,1]
            
            # 添加交互特征
            interaction_features = np.array([private_public_corr, private_others_corr])
            
            # 重构观测：私有 + 公共 + 他者 + 交互
            reconstructed = np.concatenate([private_obs, public_obs, others_obs, interaction_features])
        else:
            # 标准模式：直接拼接
            reconstructed = np.concatenate([private_obs, public_obs, others_obs])
        
        return reconstructed
    
    def adapt_observation_for_fomappo(self, observation: np.ndarray, manager_id: str) -> Dict[str, torch.Tensor]:
        """
        为FOMAPPO算法适配观测信息
        
        Args:
            observation: 原始观测向量
            manager_id: Manager标识
            
        Returns:
            适配后的观测字典，包含不同层次的信息
        """
        # 解析分层观测
        parsed_obs = self.parse_observation(observation, manager_id)
        
        # 处理各层观测
        enhanced_private = self.enhance_private_observation(parsed_obs['private'], manager_id)
        processed_public = self.process_public_observation(parsed_obs['public'])
        processed_others = self.process_others_observation(parsed_obs['others'], manager_id)
        
        # 重构完整观测
        reconstructed_obs = self.reconstruct_observation(
            enhanced_private, processed_public, processed_others, enhanced=True
        )
        
        # 更新观测历史
        self._update_observation_history(manager_id, observation)
        
        # 转换为PyTorch张量
        adapted_obs = {
            'full_observation': torch.FloatTensor(reconstructed_obs).to(self.device),
            'private_features': torch.FloatTensor(enhanced_private).to(self.device),
            'public_features': torch.FloatTensor(processed_public).to(self.device),
            'others_features': torch.FloatTensor(processed_others).to(self.device),
            'layer_weights': torch.FloatTensor([
                self.private_weight, self.public_weight, self.others_weight
            ]).to(self.device)
        }
        
        return adapted_obs
    
    def _update_observation_history(self, manager_id: str, observation: np.ndarray):
        """更新观测历史"""
        if manager_id not in self.observation_history:
            self.observation_history[manager_id] = []
        
        self.observation_history[manager_id].append(observation.copy())
        
        # 限制历史长度
        if len(self.observation_history[manager_id]) > self.max_history_len:
            self.observation_history[manager_id] = self.observation_history[manager_id][-self.max_history_len:]
    
    def get_observation_stats(self, manager_id: str) -> Dict[str, float]:
        """获取观测统计信息"""
        if manager_id not in self.observation_history or len(self.observation_history[manager_id]) == 0:
            return {}
        
        recent_obs = np.array(self.observation_history[manager_id])
        
        stats = {
            'mean': np.mean(recent_obs),
            'std': np.std(recent_obs),
            'min': np.min(recent_obs),
            'max': np.max(recent_obs),
            'history_length': len(self.observation_history[manager_id])
        }
        
        return stats
    
    def reset_history(self, manager_id: Optional[str] = None):
        """重置观测历史"""
        if manager_id is None:
            self.observation_history.clear()
        else:
            if manager_id in self.observation_history:
                del self.observation_history[manager_id]


class DecPOMDPAwareNetwork(nn.Module):
    """
    Dec-POMDP感知网络层
    
    专门为处理分层观测信息设计的神经网络层
    """
    
    def __init__(self, private_dim: int, public_dim: int, others_dim: int, 
                 hidden_dim: int = 128, output_dim: int = 64):
        super(DecPOMDPAwareNetwork, self).__init__()
        
        self.private_dim = private_dim
        self.public_dim = public_dim
        self.others_dim = others_dim
        
        # 分层处理网络
        self.private_net = nn.Sequential(
            nn.Linear(private_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.public_net = nn.Sequential(
            nn.Linear(public_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.others_net = nn.Sequential(
            nn.Linear(others_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )
        
        # 融合网络
        fusion_input_dim = hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 8
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim // 4, num_heads=4, batch_first=True)
        
    def forward(self, private_obs: torch.Tensor, public_obs: torch.Tensor, others_obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            private_obs: 私有观测 [batch_size, private_dim]
            public_obs: 公共观测 [batch_size, public_dim]
            others_obs: 他者观测 [batch_size, others_dim]
            
        Returns:
            融合后的特征表示 [batch_size, output_dim]
        """
        # 分层特征提取
        private_features = self.private_net(private_obs)
        public_features = self.public_net(public_obs)
        others_features = self.others_net(others_obs)
        
        # 应用注意力机制（可选）
        if private_features.dim() == 2:
            # 为注意力机制添加序列维度
            attention_input = torch.stack([private_features, public_features], dim=1)  # [batch_size, 2, hidden_dim//4]
            attended_features, _ = self.attention(attention_input, attention_input, attention_input)
            private_attended = attended_features[:, 0, :]  # [batch_size, hidden_dim//4]
            public_attended = attended_features[:, 1, :]   # [batch_size, hidden_dim//4]
        else:
            private_attended = private_features
            public_attended = public_features
        
        # 特征融合
        fused_features = torch.cat([private_attended, public_attended, others_features], dim=-1)
        output = self.fusion_net(fused_features)
        
        return output 
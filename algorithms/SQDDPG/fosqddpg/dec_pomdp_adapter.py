#!/usr/bin/env python3
"""
FOSQDDPG Dec-POMDP观测空间适配器

为FOSQDDPG算法提供Dec-POMDP观测空间的处理能力。
专门针对Shapley值公平分配和FlexOffer约束的观测空间管理。

核心功能：
1. 观测空间分层解析（复用Dec-POMDP架构）
2. Shapley值计算的观测增强
3. 公平性感知的观测处理
4. FlexOffer约束的观测集成
5. 协作信息的公平性权重分配
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

class FOSqddpgDecPOMDPAdapter:
    """
    FOSQDDPG Dec-POMDP观测空间适配器
    
    专门为FOSQDDPG算法设计的Dec-POMDP观测处理器，
    支持Shapley值公平分配和FlexOffer约束的观测空间管理。
    """
    
    def __init__(self, dec_pomdp_config: DecPOMDPConfig, device=torch.device("cpu")):
        self.config = dec_pomdp_config
        self.device = device
        
        # Dec-POMDP观测空间维度
        self.private_dim = 39      # 私有观测基础维度
        self.public_dim = 18       # 公共观测维度
        self.others_dim = 15       # 他者观测维度
        self.total_obs_dim = 72    # 总观测维度
        
        # FOSQDDPG特定参数
        self.shapley_mode = True
        self.fairness_weight = 0.3              # 公平性权重
        self.credit_assignment_factor = 0.2     # 信用分配因子
        self.coalition_history_length = 5       # 联盟历史长度
        
        # FlexOffer约束集成（增强版）
        self.fo_constraint_dim = 36             # FlexOffer约束维度
        self.fo_fairness_weight = 0.25          # FlexOffer公平性权重
        self.fo_shapley_integration = True      # Shapley值集成开关
        
        # 观测处理缓存
        self._observation_cache = {}
        self._coalition_history = {}
        self._fairness_scores = {}
        
        # 初始化历史缓存
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """初始化观测历史缓冲区"""
        for manager_id in [f"manager_{i}" for i in range(4)]:
            self._coalition_history[manager_id] = {
                'private': [],
                'public': [],
                'others': [],
                'full_obs': [],
                'shapley_values': [],
                'fairness_scores': []
            }
            self._fairness_scores[manager_id] = 1.0  # 初始公平性得分
    
    def adapt_observation_for_fosqddpg(self, 
                                      observation: np.ndarray, 
                                      manager_id: str,
                                      fo_constraints: Optional[np.ndarray] = None,
                                      fo_satisfaction: Optional[float] = None,
                                      coalition_info: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        为FOSQDDPG适配观测空间
        
        Args:
            observation: 原始观测 [obs_dim]
            manager_id: Manager ID (e.g., "manager_0")
            fo_constraints: FlexOffer约束 [constraint_dim]
            fo_satisfaction: FlexOffer满意度标量
            coalition_info: 联盟信息字典
            
        Returns:
            适配后的分层观测字典，包含Shapley值信息
        """
        # 解析Dec-POMDP观测
        private_obs, public_obs, others_obs = self._parse_dec_pomdp_observation(observation)
        
        # FlexOffer约束集成（Shapley值感知）
        if fo_constraints is not None:
            private_obs = self._integrate_fo_constraints_with_shapley(
                private_obs, fo_constraints, fo_satisfaction, manager_id
            )
        else:
            # 如果没有FlexOffer约束，填充到40维
            private_obs = np.pad(private_obs, (0, 1), mode='constant', constant_values=0.0)
        
        # FOSQDDPG特定观测增强
        enhanced_private = self._enhance_private_obs_for_fosqddpg(private_obs, manager_id)
        enhanced_public = self._enhance_public_obs_for_fosqddpg(public_obs, coalition_info)
        enhanced_others = self._enhance_others_obs_for_fosqddpg(others_obs, manager_id, coalition_info)
        
        # 观测噪声处理（公平性权重）
        if self.config.enable_observation_noise:
            enhanced_private = self._add_observation_noise(enhanced_private, noise_type='private', manager_id=manager_id)
            enhanced_public = self._add_observation_noise(enhanced_public, noise_type='public', manager_id=manager_id)
            enhanced_others = self._add_observation_noise(enhanced_others, noise_type='others', manager_id=manager_id)
        
        # 转换为张量
        adapted_obs = {
            'private': torch.FloatTensor(enhanced_private).to(self.device),
            'public': torch.FloatTensor(enhanced_public).to(self.device),
            'others': torch.FloatTensor(enhanced_others).to(self.device) if self.config.enable_other_manager_info else torch.zeros(self.others_dim).to(self.device),
            'full_obs': torch.FloatTensor(np.concatenate([enhanced_private, enhanced_public, enhanced_others])).to(self.device),
            'fairness_score': torch.FloatTensor([self._fairness_scores[manager_id]]).to(self.device),
            'shapley_weight': torch.FloatTensor([self._compute_shapley_weight(manager_id)]).to(self.device)
        }
        
        # 更新历史缓存（包含Shapley值信息）
        self._update_coalition_history(manager_id, adapted_obs, coalition_info)
        
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
    
    def _integrate_fo_constraints_with_shapley(self, 
                                              private_obs: np.ndarray, 
                                              fo_constraints: np.ndarray,
                                              fo_satisfaction: Optional[float] = None,
                                              manager_id: str = None) -> np.ndarray:
        """集成FlexOffer约束到私有观测（Shapley值感知）"""
        # FlexOffer约束特征提取
        constraint_features = self._extract_fo_constraint_features(fo_constraints)
        
        # FlexOffer满意度处理（公平性加权）
        satisfaction_feature = fo_satisfaction if fo_satisfaction is not None else 0.8
        fairness_score = self._fairness_scores.get(manager_id, 1.0)
        weighted_satisfaction = satisfaction_feature * fairness_score
        
        # Shapley值集成的约束趋势
        if self.fo_shapley_integration:
            shapley_weight = self._compute_shapley_weight(manager_id)
            constraint_trend = (np.mean(constraint_features) * shapley_weight - 
                              np.mean(private_obs[:10]) * (1 - shapley_weight))
        else:
            constraint_trend = np.mean(constraint_features) - np.mean(private_obs[:10])
        
        # 扩展私有观测：39 + 1(Shapley趋势) = 40维
        enhanced_private = np.concatenate([private_obs, [constraint_trend]])
        
        return enhanced_private[:40]  # 确保维度一致
    
    def _extract_fo_constraint_features(self, fo_constraints: np.ndarray) -> np.ndarray:
        """提取FlexOffer约束特征（公平性感知）"""
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
    
    def _enhance_private_obs_for_fosqddpg(self, private_obs: np.ndarray, manager_id: str) -> np.ndarray:
        """为FOSQDDPG增强私有观测（Shapley值集成）"""
        # 获取历史观测用于Shapley值计算
        history = self._coalition_history[manager_id]['private']
        
        if len(history) > 0:
            # Shapley值加权的历史信息
            shapley_values = self._coalition_history[manager_id]['shapley_values']
            if shapley_values:
                recent_shapley = np.mean(shapley_values[-3:]) if len(shapley_values) >= 3 else np.mean(shapley_values)
                recent_history = history[-2:] if len(history) >= 2 else history
                if recent_history:
                    avg_history = np.mean(recent_history, axis=0)
                    # Shapley值加权融合
                    shapley_enhanced_obs = (recent_shapley * private_obs + 
                                          (1 - recent_shapley) * avg_history)
                    return shapley_enhanced_obs
        
        return private_obs
    
    def _enhance_public_obs_for_fosqddpg(self, public_obs: np.ndarray, coalition_info: Optional[Dict] = None) -> np.ndarray:
        """为FOSQDDPG增强公共观测（联盟感知）"""
        # 联盟信息集成
        if coalition_info and self.shapley_mode:
            coalition_strength = coalition_info.get('coalition_strength', 1.0)
            coalition_fairness = coalition_info.get('fairness_index', 1.0)
            
            # 公共观测的联盟调整
            coalition_factor = 0.1 * coalition_strength * coalition_fairness
            enhanced_obs = public_obs * (1 + coalition_factor)
            return enhanced_obs
        
        return public_obs
    
    def _enhance_others_obs_for_fosqddpg(self, others_obs: np.ndarray, manager_id: str, coalition_info: Optional[Dict] = None) -> np.ndarray:
        """为FOSQDDPG增强他者观测（公平性加权）"""
        if not self.config.enable_other_manager_info:
            return np.zeros(self.others_dim)
        
        # 公平性权重应用
        fairness_weight = self.fairness_weight
        
        # 如果有联盟信息，调整公平性权重
        if coalition_info:
            member_fairness = coalition_info.get('member_fairness', {})
            if manager_id in member_fairness:
                individual_fairness = member_fairness[manager_id]
                fairness_weight *= individual_fairness
        
        # 获取历史他者观测
        history = self._coalition_history[manager_id]['others']
        
        if len(history) > 0:
            recent_avg = np.mean(history[-self.coalition_history_length:], axis=0) if len(history) >= self.coalition_history_length else np.mean(history, axis=0)
            # 公平性加权更新
            fair_weighted_obs = fairness_weight * others_obs + (1 - fairness_weight) * recent_avg
            return fair_weighted_obs
        
        return others_obs * fairness_weight
    
    def _add_observation_noise(self, observation: np.ndarray, noise_type: str = 'default', manager_id: str = None) -> np.ndarray:
        """添加观测噪声（公平性调整）"""
        if not self.config.enable_observation_noise:
            return observation
        
        # FOSQDDPG特定噪声设置（公平性调整）
        fairness_factor = self._fairness_scores.get(manager_id, 1.0) if manager_id else 1.0
        
        noise_scales = {
            'private': self.config.noise_level * 0.7 * fairness_factor,     # 私有观测噪声（公平性调整）
            'public': self.config.noise_level * 0.4,                        # 公共观测噪声较小
            'others': self.config.noise_level * 1.1 * (2 - fairness_factor) # 他者观测噪声（反向公平性调整）
        }
        
        noise_scale = noise_scales.get(noise_type, self.config.noise_level)
        
        # 生成噪声
        noise = np.random.normal(0, noise_scale, observation.shape)
        
        # 添加噪声
        noisy_obs = observation + noise
        
        return noisy_obs
    
    def _compute_shapley_weight(self, manager_id: str) -> float:
        """计算Shapley值权重"""
        history = self._coalition_history[manager_id]['shapley_values']
        if not history:
            return 0.25  # 默认Shapley权重
        
        # 使用最近的Shapley值
        recent_values = history[-3:] if len(history) >= 3 else history
        return np.mean(recent_values)
    
    def _update_coalition_history(self, manager_id: str, adapted_obs: Dict[str, torch.Tensor], coalition_info: Optional[Dict]):
        """更新联盟历史缓冲区"""
        history = self._coalition_history[manager_id]
        
        # 转换为numpy并添加到历史
        history['private'].append(adapted_obs['private'].cpu().numpy())
        history['public'].append(adapted_obs['public'].cpu().numpy())
        history['others'].append(adapted_obs['others'].cpu().numpy())
        history['full_obs'].append(adapted_obs['full_obs'].cpu().numpy())
        
        # 添加Shapley值和公平性信息
        if coalition_info:
            history['shapley_values'].append(coalition_info.get('shapley_value', 0.25))
            fairness_score = coalition_info.get('fairness_score', 1.0)
            history['fairness_scores'].append(fairness_score)
            # 更新当前公平性得分
            self._fairness_scores[manager_id] = fairness_score
        else:
            # 默认值
            history['shapley_values'].append(0.25)
            history['fairness_scores'].append(1.0)
        
        # 维持历史长度
        for key in history:
            if len(history[key]) > self.coalition_history_length:
                history[key].pop(0)
    
    def get_adapted_dimensions(self) -> Dict[str, int]:
        """获取适配后的观测维度信息"""
        return {
            'private_dim': 40,  # 39 + 1(Shapley趋势)
            'public_dim': self.public_dim,
            'others_dim': self.others_dim,
            'total_dim': 40 + self.public_dim + self.others_dim,  # 73
            'coalition_history_length': self.coalition_history_length,
            'fo_constraint_dim': self.fo_constraint_dim,
            'fairness_features': 2  # fairness_score + shapley_weight
        }
    
    def get_fosqddpg_specific_info(self) -> Dict[str, Any]:
        """获取FOSQDDPG特定的适配信息"""
        return {
            'shapley_mode': self.shapley_mode,
            'fairness_weight': self.fairness_weight,
            'credit_assignment_factor': self.credit_assignment_factor,
            'coalition_history_length': self.coalition_history_length,
            'fo_fairness_weight': self.fo_fairness_weight,
            'fo_shapley_integration': self.fo_shapley_integration,
            'fairness_scores': dict(self._fairness_scores)
        }
    
    def reset_history(self, manager_id: Optional[str] = None):
        """重置联盟历史"""
        if manager_id is None:
            self._init_history_buffers()
        else:
            if manager_id in self._coalition_history:
                for key in self._coalition_history[manager_id]:
                    self._coalition_history[manager_id][key].clear()
                self._fairness_scores[manager_id] = 1.0
    
    def update_fairness_scores(self, fairness_updates: Dict[str, float]):
        """更新公平性得分"""
        for manager_id, score in fairness_updates.items():
            if manager_id in self._fairness_scores:
                # 平滑更新公平性得分
                self._fairness_scores[manager_id] = (0.7 * self._fairness_scores[manager_id] + 
                                                   0.3 * score)
    
    def get_coalition_enhanced_observation(self, 
                                          manager_id: str, 
                                          current_obs: Dict[str, torch.Tensor],
                                          coalition_members: List[str],
                                          coalition_strength: float = 1.0) -> Dict[str, torch.Tensor]:
        """获取联盟增强观测（FOSQDDPG特色功能）"""
        enhanced_obs = current_obs.copy()
        
        # 联盟强度加权
        coalition_factor = coalition_strength * self.credit_assignment_factor
        
        # 增强私有观测（联盟影响）
        enhanced_obs['private'] = enhanced_obs['private'] * (1 + coalition_factor * 0.1)
        
        # 增强他者观测（联盟成员信息）
        if len(coalition_members) > 1:
            coalition_size_factor = len(coalition_members) / 4.0  # 最大4个Manager
            enhanced_obs['others'] = enhanced_obs['others'] * (1 + coalition_size_factor * 0.05)
        
        # 添加联盟特定信息
        enhanced_obs['coalition_strength'] = torch.FloatTensor([coalition_strength]).to(self.device)
        enhanced_obs['coalition_size'] = torch.FloatTensor([len(coalition_members)]).to(self.device)
        
        return enhanced_obs 
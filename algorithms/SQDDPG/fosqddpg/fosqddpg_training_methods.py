#!/usr/bin/env python3
"""
FOSQDDPG训练方法

提供标准化的FOSQDDPG训练实现
用于在FO Pipeline中集成FOSQDDPG算法 (Shapley Q-value Deep Deterministic Policy Gradient)
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
import random
import time

logger = logging.getLogger(__name__)

# 添加缺失的导入
try:
    from .fosqddpg_adapter import FOSQDDPGAdapter
    FOSQDDPG_ADAPTER_available = True
except ImportError:
    FOSQDDPGAdapter = None
    FOSQDDPG_ADAPTER_available = False

def train_fosqddpg_adapter(pipeline):
    """
    优化版FOSQDDPG训练方法 - 包含Shapley值公平信用分配和稳定性改进
    
    Args:
        pipeline: FO Pipeline实例
    
    Returns:
        包含训练结果的字典
    """
    logger.info("🚀 开始优化版FOSQDDPG训练（Shapley值信用分配）")
    logger.info(f"计划训练 {pipeline.num_episodes} 个episodes")
    
    # 强制检查num_episodes参数
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes参数无效，设置为默认值1")
        pipeline.num_episodes = 1
    
    # 记录最大允许的episodes数量
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # 设置一个安全上限
    logger.info(f"最大允许的episodes数量: {max_allowed_episodes}")
    
    # 更新实际运行的算法
    pipeline._update_actual_algorithm("FOSQDDPG_ADAPTER")
    
    # 1. 准备训练环境
    logger.info("正在准备FOSQDDPG训练环境...")
    
    # 创建FO环境
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # 复位环境状态
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # 初始化用户状态
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # 创建多智能体环境
    multi_env = None
    try:
        from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
        
        multi_env = MultiAgentFlexOfferEnv(
            data_dir="data",
            time_horizon=pipeline.time_horizon,
            time_step=pipeline.time_step,
            aggregation_method=pipeline.aggregation_method if hasattr(pipeline, 'aggregation_method') else "LP",
            trading_method=pipeline.trading_strategy if hasattr(pipeline, 'trading_strategy') else "pool",
            disaggregation_method=pipeline.disaggregation_method if hasattr(pipeline, 'disaggregation_method') else "proportional"
        )
        logger.info("✅ 成功创建multi_agent_env")
    except Exception as e:
        logger.error(f"❌ 创建multi_agent_env失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'创建环境失败: {e}'}
    
    # 2. 获取环境参数
    # 获取manager数量和ID
    num_managers = multi_env.get_manager_count()
    manager_ids = list(multi_env.manager_agents.keys())
    logger.info(f"🏗️ 环境配置: {num_managers} 个Manager: {manager_ids}")
    
    # 获取状态和动作空间维度
    try:
        sample_obs, _ = multi_env.reset()
        state_dim = len(sample_obs[manager_ids[0]])
        action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
        logger.info(f"📊 状态空间: {state_dim}维, 动作空间: {action_dim}维")
    except Exception as e:
        logger.error(f"❌ 获取观测和动作空间失败: {e}")
        return {'status': 'failed', 'error': f'获取环境参数失败: {e}'}
    
    # 3. 创建FOSQDDPG适配器
    try:
        if not FOSQDDPG_ADAPTER_available or FOSQDDPGAdapter is None:
            logger.error("❌ FOSQDDPGAdapter不可用")
            return {'status': 'failed', 'error': 'FOSQDDPGAdapter不可用'}
            
        # FOSQDDPG专用超参数
        # FOSQDDPG具有Shapley值信用分配等特点
        WARMUP_EPISODES = 5   # 前5个episode仅收集经验，不更新策略
        NOISE_DECAY = 0.99    # 噪声衰减率
        MIN_NOISE = 0.05      # 最小噪声比例
        INITIAL_NOISE = 0.3   # 初始噪声比例（略高于TD3，因为Shapley值需要探索）
        UPDATE_FREQ = 2       # 每隔2个时间步更新一次
        BATCH_SIZE = 128      # 批次大小
        SAMPLE_SIZE = 5       # Shapley值采样大小
        
        # 创建FOSQDDPG适配器，使用优化的超参数
        fosqddpg_adapter = FOSQDDPGAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            
            # 学习率设置
            lr_actor=5e-5,      # 低学习率以提高稳定性
            lr_critic=1e-4,     # 低学习率以提高稳定性
            hidden_dim=256,
            device=pipeline.device if hasattr(pipeline, 'device') else "cpu",
            
            # FOSQDDPG专用参数
            max_action=1.0,
            buffer_capacity=100000,
            batch_size=BATCH_SIZE,
            gamma=0.99,         # 折扣因子
            tau=0.005,          # 软更新参数
            noise_scale=INITIAL_NOISE,
            sample_size=SAMPLE_SIZE  # Shapley值采样大小
        )
        logger.info("✅ FOSQDDPG适配器初始化成功")
        logger.info(f"   使用FOSQDDPG特有参数: Shapley值信用分配, 采样大小({SAMPLE_SIZE}), 公平合作机制")
    except Exception as e:
        logger.error(f"❌ 创建FOSQDDPG适配器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'创建适配器失败: {e}'}
    
    # 4. 初始化训练历史记录
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    training_history = []
    
    # 5. 设置动态探索噪声
    current_noise_scale = INITIAL_NOISE
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 6. 开始训练循环
    logger.info(f"开始FOSQDDPG训练循环 ({pipeline.num_episodes}个episodes)...")
    
    # 训练循环 - 基于FOSQDDPG的Off-policy学习与Shapley值公平分配
    for episode in range(1, max_allowed_episodes + 1):
        if episode > pipeline.num_episodes:
            logger.warning(f"已达到指定的episodes数量 {pipeline.num_episodes}，终止训练")
            break
            
        logger.info(f"\n========== Episode {episode}/{pipeline.num_episodes} (FOSQDDPG) ==========")
        episode_start_time = datetime.now()
        
        # 重置环境
        obs, infos = multi_env.reset()
        fosqddpg_adapter.reset_episode()  # 重置episode状态
        
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        
        # 动态调整噪声比例
        if episode > WARMUP_EPISODES:
            current_noise_scale = max(MIN_NOISE, current_noise_scale * NOISE_DECAY)
            # 更新适配器的噪声参数
            fosqddpg_adapter.noise_scale = current_noise_scale
            logger.info(f"📉 噪声调整: {current_noise_scale:.4f}")
        
        # 每个episode运行步数
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"Episode {episode}, 时间步 {timestep}/{pipeline.steps_per_episode-1}")
            
            # 使用探索或利用策略
            use_noise = (episode <= WARMUP_EPISODES * 2 or random.random() < current_noise_scale)
            actions, action_log_probs, values = fosqddpg_adapter.select_actions(obs, deterministic=not use_noise)
            
            # 环境步进
            next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
            
            # 收集数据到经验回放缓冲区
            collect_info = fosqddpg_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                timestep=timestep
            )
            
            # 累积奖励
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # 更新观测
            obs = next_obs
            
            # 分批次更新，使用Shapley值公平信用分配
            if timestep % UPDATE_FREQ == 0 and episode > WARMUP_EPISODES:
                train_info = fosqddpg_adapter.train_on_batch()
                if isinstance(train_info, dict):
                    policy_loss = train_info.get('policy_loss', 0.0)
                    value_loss = train_info.get('value_loss', 0.0)
                    logger.debug(f"  ⚙️ 训练更新: Actor Loss: {policy_loss:.5f}, Critic Loss: {value_loss:.5f}")
            
            # 显示时间步奖励
            timestep_total = sum(rewards.values())
            logger.info(f"  时间步 {timestep} 总奖励: {timestep_total:.3f}")
        
        # Episode结束后的训练
        if episode > WARMUP_EPISODES:
            # 在episode结束后多进行几次更新
            for _ in range(5):
                update_info = fosqddpg_adapter.train_on_batch()
        
        # 记录episode奖励和统计
        episode_total_reward = sum(episode_rewards.values())
        logger.info(f"Episode {episode} 完成:")
        logger.info(f"  🎯 总奖励: {episode_total_reward:.3f}")
        
        # 如果有训练信息
        if 'update_info' in locals() and isinstance(update_info, dict):
            shapley_info = update_info.get('shapley_info', {})
            logger.info(f"  📈 训练损失: Actor {update_info.get('policy_loss', 0):.4f}, "
                       f"Critic {update_info.get('value_loss', 0):.4f}")
            logger.info(f"  🔍 Shapley值: {shapley_info}")
        
        # 显示每个Manager的奖励并记录到训练历史
        for manager_id, reward in episode_rewards.items():
            logger.info(f"  📊 {manager_id}: {reward:.3f}")
            training_episode_rewards[manager_id].append(reward)
            
            # 添加到训练历史
            training_data = {
                'algorithm': 'FOSQDDPG',
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': reward,
                'policy_loss': float(update_info.get('policy_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'value_loss': float(update_info.get('value_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'entropy': 0.0  # FOSQDDPG没有熵概念
            }
            training_history.append(training_data)
            
            # 记录训练损失
            if hasattr(pipeline, '_record_training_loss') and 'update_info' in locals():
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(update_info.get('policy_loss', 0.001)),
                    value_loss=float(update_info.get('value_loss', 0.001)),
                    entropy=0.0  # FOSQDDPG没有熵概念
                )
        
        # 记录总体奖励
        training_data_total = {
            'algorithm': 'FOSQDDPG',
            'manager_id': 'total',
            'episode': episode,
            'episode_reward': episode_total_reward,
            'policy_loss': float(update_info.get('policy_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'value_loss': float(update_info.get('value_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'entropy': 0.0
        }
        training_history.append(training_data_total)
        
        # 定期输出学习进度
        if (episode + 1) % 10 == 0:
            logger.info(f"\n========== FOSQDDPG训练进度: {episode+1}/{pipeline.num_episodes} episodes ==========")
            
            # 获取训练统计
            try:
                training_stats = fosqddpg_adapter.get_training_stats()
                manager_rewards = fosqddpg_adapter.get_manager_rewards_summary()
                
                if isinstance(manager_rewards, dict):
                    for manager_id, stats in manager_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            training_updates = stats.get('training_updates', 0)
                            logger.info(f"  🔥 {manager_id}: 累积奖励 {total_reward:.2f}, 最佳 {best_reward:.2f}, 更新 {training_updates} 次")
                        else:
                            logger.info(f"  🔥 {manager_id}: 累积奖励 {stats:.2f}")
                else:
                    logger.info(f"  🔥 管理者奖励: {manager_rewards}")
                
                if isinstance(training_stats, dict):
                    iterations = training_stats.get('training_iterations', 0)
                    logger.info(f"  🚀 总训练迭代: {iterations}")
                else:
                    logger.info(f"  🚀 训练统计: {training_stats}")
            except Exception as e:
                logger.warning(f"获取训练统计失败: {e}")
                logger.info("  🔥 训练进度: 正在学习中...")
            
            logger.info("=" * 70)
        
        # 定期保存模型
        if (episode + 1) % 20 == 0 or episode == pipeline.num_episodes:
            try:
                model_path = f"results/fosqddpg_adapter_ep{episode+1}"
                fosqddpg_adapter.save_models(model_path)
                logger.info(f"📀 模型已保存至: {model_path}")
                
                # 保存训练历史
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOSQDDPG_ADAPTER")
            except Exception as e:
                logger.error(f"保存模型失败: {e}")
        
        # 计算episode耗时
        episode_duration = datetime.now() - episode_start_time
        logger.info(f"Episode {episode} 耗时: {episode_duration}")
        
        # 显示总进度
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"已用时间: {total_elapsed}, 预计剩余: {estimated_remaining}")
    
    # 训练结束，保存最终模型
    try:
        save_path = f"results/fosqddpg_adapter_final"
        fosqddpg_adapter.save_models(save_path)
        logger.info(f"保存最终模型: {save_path}")
    except Exception as e:
        logger.error(f"保存最终模型失败: {e}")
    
    # 计算总训练时间
    total_training_time = datetime.now() - start_time
    logger.info(f"FOSQDDPG训练完成! 总训练时间: {total_training_time}")
    
    # 将训练历史整理为pipeline期望的格式
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},
            'episode_lengths': {},
            'training_loss': {},
            'training_metadata': {
                'algorithm': 'FOSQDDPG',
                'num_episodes': pipeline.num_episodes,
                'steps_per_episode': pipeline.steps_per_episode,
                'num_managers': num_managers,
                'shapley_sample_size': SAMPLE_SIZE  # FOSQDDPG特有参数
            }
        },
        'multi_agent_env': multi_env,
        'fosqddpg_adapter': fosqddpg_adapter
    }
    
    # 处理训练历史数据，按manager_id分组
    for item in training_history:
        manager_id = item.get('manager_id')
        if manager_id and manager_id != 'total':  # 排除总体记录
            if manager_id not in result['training_history']['episode_rewards']:
                result['training_history']['episode_rewards'][manager_id] = []
                result['training_history']['episode_lengths'][manager_id] = []
                result['training_history']['training_loss'][manager_id] = []
            
            # 添加奖励和长度
            result['training_history']['episode_rewards'][manager_id].append(item.get('episode_reward', 0.0))
            result['training_history']['episode_lengths'][manager_id].append(pipeline.steps_per_episode)
            
            # 添加训练损失
            loss_info = {
                'policy_loss': item.get('policy_loss', 0.001),
                'value_loss': item.get('value_loss', 0.001),
                'entropy': item.get('entropy', 0.0)  # FOSQDDPG没有熵
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"返回结果包含 {len(result['training_history']['episode_rewards'])} 个Manager的训练历史")
    return result 
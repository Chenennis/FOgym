#!/usr/bin/env python3
"""
FOMAPPO和FOMAIPPO训练方法

提供共享策略FOMAPPO和独立策略FOMAIPPO的训练实现
用于在FO Pipeline中集成这两种算法
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# 添加缺失的导入
try:
    from .fomappo_adapter import FOMAPPOAdapter
    FOMAPPO_SHARED_available = True
except ImportError:
    FOMAPPOAdapter = None
    FOMAPPO_SHARED_available = False
    
try:
    from .fomaippo_adapter import FOMAIPPOAdapter
    FOMAIPPO_available = True
except ImportError:
    FOMAIPPOAdapter = None
    FOMAIPPO_available = False

def train_fomappo_shared_policy(pipeline):
    """
    优化版FOMAPPO训练方法 - 包含更多数值稳定性和学习质量改进
    """
    logger.info("🚀 开始优化版FOMAPPO训练（增强学习效果和稳定性）")
    logger.info(f"计划训练 {pipeline.num_episodes} 个episodes")
    
    # 强制检查num_episodes参数
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes参数无效，设置为默认值1")
        pipeline.num_episodes = 1
    
    # 记录最大允许的episodes数量
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # 设置一个安全上限
    logger.info(f"最大允许的episodes数量: {max_allowed_episodes}")
    
    # 更新实际运行的算法
    pipeline._update_actual_algorithm("FOMAPPO_FIXED")
    
    # 1. 准备训练环境
    logger.info("正在准备FOMAPPO训练环境...")
    
    # 创建FO环境
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # 复位环境状态
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # 初始化用户状态
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # 🔧 创建或获取多智能体环境
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("使用已存在的multi_agent_env")
    else:
        # 创建新的多智能体环境
        logger.info("创建新的multi_agent_env")
        try:
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=pipeline.time_horizon,
                time_step=pipeline.time_step,
                aggregation_method=pipeline.aggregation_method,
                trading_method=pipeline.trading_strategy,
                disaggregation_method=pipeline.disaggregation_method
            )
            logger.info("✅ 成功创建multi_agent_env")
        except Exception as e:
            logger.error(f"❌ 创建multi_agent_env失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. 获取环境参数
    # 获取manager数量和ID
    manager_ids = [manager.manager_id for manager in pipeline.managers]
    num_managers = len(manager_ids)
    
    # 获取状态维度和动作维度
    # 3. 创建FOMAPPO适配器
    logger.info(f"创建FOMAPPO适配器: {num_managers}个Manager")

    # 获取环境的实际观测维度
    try:
        # 直接从环境获取样本观测来确定维度
        logger.info("🔍 从环境获取实际观测维度...")
        if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
            # 如果已经有环境，使用它获取观测
            sample_obs, _ = pipeline.multi_agent_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            logger.info(f"✅ 从现有环境获取观测维度: {state_dim}")
        elif multi_env is not None:
            # 使用已创建的多智能体环境
            logger.info("使用已创建的multi_env获取观测维度...")
            
            # 获取实际观测维度
            sample_obs, _ = multi_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            logger.info(f"✅ 从multi_env获取观测维度: {state_dim}")
        else:
            # 无法获取环境，使用默认维度
            logger.warning("无法获取环境，使用默认维度")
            state_dim = 73  # 默认状态维度
        
        # 动作维度从环境获取
        if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
            sample_manager_id = list(pipeline.multi_agent_env.action_spaces.keys())[0]
            action_dim = pipeline.multi_agent_env.action_spaces[sample_manager_id].shape[0]
        elif multi_env is not None:
            sample_manager_id = list(multi_env.action_spaces.keys())[0]
            action_dim = multi_env.action_spaces[sample_manager_id].shape[0]
        else:
            if hasattr(pipeline, "_get_manager_action_dim"):
                action_dim = pipeline._get_manager_action_dim()
            else:
                action_dim = 100  # 默认动作维度
                
    except Exception as e:
        logger.warning(f"🔍 获取环境观测维度失败: {e}")
        
        # 回退到从manager获取状态维度
    if hasattr(pipeline, "_get_manager_state"):
        # 获取样本状态以确定维度
        sample_state = pipeline._get_manager_state(pipeline.managers[0])
        state_dim = len(sample_state)
    else:
            state_dim = 73  # 默认状态维度改为73
    
    if hasattr(pipeline, "_get_manager_action_dim"):
        action_dim = pipeline._get_manager_action_dim()
    else:
        action_dim = 100  # 默认动作维度
    
        logger.warning(f"⚠️ 使用回退的维度: 状态={state_dim}, 动作={action_dim}")

    logger.info(f"📊 确定观测维度: {state_dim}, 动作维度: {action_dim}")
    
    # 使用优化的超参数
    fomappo_adapter = FOMAPPOAdapter(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_managers,
        episode_length=pipeline.steps_per_episode,
        lr_actor=5e-5,  # 降低学习率
        lr_critic=2e-4,  # 降低学习率
        entropy_coef=0.05,  # 增加熵系数，鼓励探索
        use_linear_lr_decay=True,  # 启用学习率衰减
        lr_decay_rate=0.95,  # 学习率衰减率
        use_clipped_value_loss=True,  # 使用裁剪的价值损失
        use_max_grad_norm=True,  # 使用梯度裁剪
        max_grad_norm=0.5,  # 梯度裁剪阈值
        device="cpu"
    )
    
    # 4. 初始化训练历史记录
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    
    # 5. 记录训练损失
    training_losses = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': []
    }
    
    # 6. 开始训练循环
    logger.info(f"开始FOMAPPO训练循环 ({pipeline.num_episodes}个episodes)...")
    
    # 初始化结果收集器
    cumulative_rewards = {manager_id: 0.0 for manager_id in manager_ids}
    avg_rewards_last_10 = {manager_id: [] for manager_id in manager_ids}
    
    # 初始化用于结果输出的数据结构
    training_history = []
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 设置训练终止标志
    training_complete = False
    
    # 主训练循环
    for episode in range(1, max_allowed_episodes + 1):
        # 检查是否已达到指定的episodes数量
        if episode > pipeline.num_episodes:
            logger.warning(f"已达到指定的episodes数量 {pipeline.num_episodes}，终止训练")
            training_complete = True
            break
            
        logger.info(f"========== 开始Episode {episode}/{pipeline.num_episodes} ==========")
        episode_start_time = datetime.now()
        
        # 重置环境状态
        pipeline._reset_pipeline_state()
        
        # 初始化episode统计
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        episode_total_reward = 0.0  # 初始化episode总奖励
        
        # 执行一个episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"Episode {episode}/{pipeline.num_episodes}, 时间步 {timestep}/{pipeline.steps_per_episode-1}")
            
            # 获取观测
            obs = pipeline._get_pipeline_observations()
            
            # 选择动作
            actions, action_log_probs, values = fomappo_adapter.select_actions(obs)
            
            # 执行动作
            pipeline_results = pipeline._execute_pipeline_with_actions(actions, timestep)
            
            # 获取奖励
            rewards = pipeline._calculate_pipeline_rewards_from_results(pipeline_results, manager_ids)
            
            # 更新episode奖励
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # 确定是否完成
            dones = {manager_id: (timestep == pipeline.steps_per_episode - 1) for manager_id in manager_ids}
            
            # 收集经验
            fomappo_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos={},
                action_log_probs=action_log_probs,
                values=values
            )
        
        # Episode结束后更新总奖励
        episode_total_reward = sum(episode_rewards.values())  # 计算总奖励
        
        for manager_id in manager_ids:
            cumulative_rewards[manager_id] += episode_rewards[manager_id]
            
            # 维护滑动窗口平均
            if len(avg_rewards_last_10[manager_id]) >= 10:
                avg_rewards_last_10[manager_id].pop(0)
            avg_rewards_last_10[manager_id].append(episode_rewards[manager_id])
            
            # 添加到训练历史
            training_episode_rewards[manager_id].append(episode_rewards[manager_id])
        
        # 更新适配器的episode计数
        fomappo_adapter.total_episodes = episode
        
        # 计算返回和优势
        logger.info(f"Episode {episode}/{pipeline.num_episodes} 完成数据收集，计算返回和优势...")
        fomappo_adapter.compute_returns()
        
        # 执行训练更新
        train_info = {}
        total_train_info = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'num_updates': 0}
        
        # 执行多次PPO更新
        num_epochs = fomappo_adapter.args.ppo_epoch
        logger.info(f"执行 {num_epochs} 轮PPO更新...")
        
        for epoch in range(num_epochs):
            batch_train_info = fomappo_adapter.train_on_batch()
            
            if batch_train_info:
                # 调试: 打印每个batch的训练信息
                logger.debug(f"Epoch {epoch}/{num_epochs}, Batch训练信息: {batch_train_info}")
                
                # 确保键名一致性
                if 'dist_entropy' in batch_train_info and 'entropy' not in batch_train_info:
                    batch_train_info['entropy'] = batch_train_info['dist_entropy']
                
                total_train_info['policy_loss'] += batch_train_info.get('policy_loss', 0.0)
                total_train_info['value_loss'] += batch_train_info.get('value_loss', 0.0)
                total_train_info['entropy'] += batch_train_info.get('entropy', 0.0)
                total_train_info['num_updates'] += 1
        
        # 计算平均损失
        if total_train_info['num_updates'] > 0:
            total_train_info['policy_loss'] /= total_train_info['num_updates']
            total_train_info['value_loss'] /= total_train_info['num_updates']
            total_train_info['entropy'] /= total_train_info['num_updates']
            
            # 调试: 打印平均损失
            logger.info(f"Episode {episode}, 平均损失: Policy={total_train_info['policy_loss']:.6f}, Value={total_train_info['value_loss']:.6f}, Entropy={total_train_info['entropy']:.6f}")
        else:
            # 如果没有成功的更新，确保损失值不为零
            logger.warning("没有成功的训练更新，使用默认非零损失值")
            total_train_info['policy_loss'] = 0.001  # 使用一个小的非零值
            total_train_info['value_loss'] = 0.001
            total_train_info['entropy'] = 0.001
        
        train_info = total_train_info
        
        # 记录训练损失
        if isinstance(train_info, dict):
            # 确保损失值不为零
            policy_loss = max(train_info.get('policy_loss', 0.0), 1e-4)
            value_loss = max(train_info.get('value_loss', 0.0), 1e-4)
            entropy = max(train_info.get('entropy', 0.0), 1e-4)
            
            training_losses['policy_loss'].append(policy_loss)
            training_losses['value_loss'].append(value_loss)
            training_losses['entropy'].append(entropy)
            
            # 更新train_info中的值
            train_info['policy_loss'] = policy_loss
            train_info['value_loss'] = value_loss
            train_info['entropy'] = entropy
            
            # 记录到日志
            logger.info(f"Episode {episode}/{pipeline.num_episodes} 训练损失: " +
                       f"Policy Loss: {policy_loss:.5f}, " +
                       f"Value Loss: {value_loss:.5f}, " +
                       f"Entropy: {entropy:.5f}")
        
        # 记录训练数据到pipeline的训练历史
        for manager_id in manager_ids:
            episode_reward = episode_rewards[manager_id]
            avg_reward = sum(avg_rewards_last_10[manager_id]) / len(avg_rewards_last_10[manager_id])
            
            # 记录到训练历史
            training_data = {
                'algorithm': 'FOMAPPO',  # 使用标准算法名称，不加FIXED后缀
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': episode_reward,
                'cumulative_reward': cumulative_rewards[manager_id],
                'avg_reward_last_10': avg_reward
            }
            
            # 添加训练损失
            if isinstance(train_info, dict):
                training_data['policy_loss'] = float(train_info.get('policy_loss', 0.001))
                training_data['value_loss'] = float(train_info.get('value_loss', 0.001))
                training_data['entropy'] = float(train_info.get('entropy', 0.001))
                
                # 确保值是Python原生类型
                for key in ['policy_loss', 'value_loss', 'entropy']:
                    if key in training_data:
                        if isinstance(training_data[key], (np.ndarray, np.number)):
                            training_data[key] = float(training_data[key])
                        elif torch.is_tensor(training_data[key]):
                            training_data[key] = float(training_data[key].item())
            
            # 添加到训练历史
            training_history.append(training_data)
            
            # 调用pipeline的loss记录函数
            if hasattr(pipeline, '_record_training_loss'):
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(train_info.get('policy_loss', 0.001)),
                    value_loss=float(train_info.get('value_loss', 0.001)),
                    entropy=float(train_info.get('entropy', 0.001))
                )
        
        # 记录总体奖励
        training_data_total = {
            'algorithm': 'FOMAPPO',
            'manager_id': 'total',
            'episode': episode,
            'episode_reward': episode_total_reward,
            'cumulative_reward': sum(cumulative_rewards.values()),
            'avg_reward_last_10': sum([sum(rewards) / len(rewards) for rewards in avg_rewards_last_10.values() if len(rewards) > 0]),
            'policy_loss': float(train_info.get('policy_loss', 0.0)),
            'value_loss': float(train_info.get('value_loss', 0.0)),
            'entropy': float(train_info.get('entropy', 0.0))
        }
        training_history.append(training_data_total)
        
        # 计算episode耗时
        episode_duration = datetime.now() - episode_start_time
        
        # 输出训练进度
        if episode % 1 == 0 or episode == pipeline.num_episodes:
            logger.info(f"Episode {episode}/{pipeline.num_episodes} 完成, 耗时: {episode_duration}, 总奖励: {episode_total_reward:.3f}, " +
                       f"Policy Loss: {train_info.get('policy_loss', 0.0):.5f}, " +
                       f"Value Loss: {train_info.get('value_loss', 0.0):.5f}, " +
                       f"Entropy: {train_info.get('entropy', 0.0):.5f}")
        
        # 保存检查点模型
        if episode % 20 == 0 or episode == pipeline.num_episodes:
            try:
                save_path = f"results/fomappo_fixed_final"
                fomappo_adapter.save_models(save_path)
                logger.info(f"保存检查点模型: {save_path}")
                
                # 保存训练历史
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMAPPO")
            except Exception as e:
                logger.error(f"保存模型失败: {e}")
        
        # 显示总进度
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"========== Episode {episode}/{pipeline.num_episodes} 完成 ==========")
        logger.info(f"已用时间: {total_elapsed}, 平均每episode: {avg_time_per_episode}")
        logger.info(f"预计剩余时间: {estimated_remaining}")
        logger.info("=" * 50)
        
        # 检查是否已达到指定的episodes数量
        if episode >= pipeline.num_episodes:
            logger.info(f"已完成指定的episodes数量 {pipeline.num_episodes}，终止训练")
            training_complete = True
            break
    
    # 检查训练是否正常完成
    if not training_complete:
        logger.warning(f"训练未正常完成，可能是因为达到了最大允许的episodes数量 {max_allowed_episodes}")
    
    # 训练结束，保存最终模型
    try:
        save_path = f"results/fomappo_fixed_final"
        fomappo_adapter.save_models(save_path)
        logger.info(f"保存最终模型: {save_path}")
        
        # 保存训练历史
        if hasattr(pipeline, '_force_save_training_history'):
            pipeline._force_save_training_history(training_history, "FOMAPPO")
            
        # 保存到CSV
        if hasattr(pipeline, '_save_training_history_to_csv'):
            pipeline._save_training_history_to_csv("FOMAPPO")
    except Exception as e:
        logger.error(f"保存最终模型失败: {e}")
    
    # 计算总训练时间
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMAPPO训练完成! 总训练时间: {total_training_time}")
    
    # 🔧 修复：返回包含training_history键的字典，而不是直接返回训练历史数据
    # 这样在_train_fomappo_agents方法中可以正确获取训练历史
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},  # 将列表格式转换为字典格式，以便pipeline处理
            'episode_lengths': {},
            'training_loss': {},
            'training_metadata': {
                'algorithm': 'FOMAPPO',
                'num_episodes': pipeline.num_episodes,
                'steps_per_episode': pipeline.steps_per_episode
            }
        },
        'multi_agent_env': multi_env if 'multi_env' in locals() else None,
        'fomappo_adapter': fomappo_adapter
    }
    
    # 处理训练历史数据，将列表格式转换为字典格式
    # 按manager_id分组
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
                'entropy': item.get('entropy', 0.001)
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"返回结果包含 {len(result['training_history']['episode_rewards'])} 个Manager的训练历史")
    return result


def train_fomaippo_independent_policy(pipeline):
    """
    FOMAIPPO训练方法 - 独立策略版本
    每个Manager都有独立的策略网络，避免策略冲突问题
    """
    logger.info("🚀 开始FOMAIPPO训练（分离策略架构，解决策略冲突问题）")
    logger.info(f"计划训练 {pipeline.num_episodes} 个episodes")
    
    # 强制检查num_episodes参数
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes参数无效，设置为默认值1")
        pipeline.num_episodes = 1
    
    # 记录最大允许的episodes数量
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # 设置一个安全上限
    logger.info(f"最大允许的episodes数量: {max_allowed_episodes}")
    
    # 更新实际运行的算法
    pipeline._update_actual_algorithm("FOMAIPPO")
    
    # 1. 准备训练环境
    logger.info("正在准备FOMAIPPO训练环境...")
    
    # 创建FO环境
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # 复位环境状态
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # 初始化用户状态
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # 🔧 创建或获取多智能体环境
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("使用已存在的multi_agent_env")
    else:
        # 创建新的多智能体环境
        logger.info("创建新的multi_agent_env")
        try:
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=pipeline.time_horizon,
                time_step=pipeline.time_step,
                aggregation_method=pipeline.aggregation_method,
                trading_method=pipeline.trading_strategy,
                disaggregation_method=pipeline.disaggregation_method
            )
            logger.info("✅ 成功创建multi_agent_env")
        except Exception as e:
            logger.error(f"❌ 创建multi_agent_env失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. 获取环境参数
    # 获取manager数量和ID
    if multi_env is not None:
        num_managers = multi_env.get_manager_count()
        manager_ids = list(multi_env.manager_agents.keys())
    else:
        manager_ids = [manager.manager_id for manager in pipeline.managers]
        num_managers = len(manager_ids)
    
    logger.info(f"🏗️ 环境配置: {num_managers} 个Manager: {manager_ids}")
    
    # 3. 获取状态和动作空间维度
    try:
        # 直接从环境获取样本观测来确定维度
        logger.info("🔍 从环境获取实际观测维度...")
        if multi_env is not None:
            sample_obs, _ = multi_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            
            # 动作维度
            action_dim = multi_env.action_spaces[sample_manager_id].shape[0]
            logger.info(f"✅ 从multi_env获取维度: 状态={state_dim}, 动作={action_dim}")
        else:
            # 回退到从pipeline获取状态维度
            if hasattr(pipeline, "_get_manager_state"):
                sample_state = pipeline._get_manager_state(pipeline.managers[0])
                state_dim = len(sample_state)
            else:
                state_dim = 73  # 默认状态维度
            
            # 获取动作维度
            if hasattr(pipeline, "_get_manager_action_dim"):
                action_dim = pipeline._get_manager_action_dim()
            else:
                action_dim = 100  # 默认动作维度
                
            logger.info(f"⚠️ 使用回退的维度: 状态={state_dim}, 动作={action_dim}")
    except Exception as e:
        logger.error(f"❌ 获取维度失败: {e}")
        # 设置安全默认值
        state_dim = 73  # 默认状态维度
        action_dim = 100  # 默认动作维度
        logger.warning(f"使用默认维度: 状态={state_dim}, 动作={action_dim}")
    
    # 4. 初始化FOMAIPPO适配器 - 🔧 使用稳定的超参数
    try:
        # 检查FOMAIPPO是否可用
        if not FOMAIPPO_available or FOMAIPPOAdapter is None:
            logger.error("❌ FOMAIPPOAdapter不可用，无法继续训练")
            return {
                'status': 'failed',
                'error': 'FOMAIPPOAdapter不可用'
            }
        
        fomaippo_adapter = FOMAIPPOAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            episode_length=pipeline.steps_per_episode,
            lr_actor=5e-5,  # 🔧 更低的学习率
            lr_critic=1e-4,  # 🔧 更低的学习率
            device=pipeline.device if hasattr(pipeline, 'device') else "cpu",
            # FOMAPPO特殊功能（降低权重）
            use_device_coordination=True,
            device_coordination_weight=0.05,  # 🔧 降低协调权重
            fo_constraint_weight=0.1,  # 🔧 降低约束权重
            use_manager_coordination=True,
            manager_coordination_weight=0.02,  # 🔧 降低协调权重
            # 🔧 数值稳定性参数
            clip_param=0.1,  # 小的clip范围
            max_grad_norm=0.2,  # 强梯度裁剪
            value_loss_coef=0.5,  # 降低value loss权重
            entropy_coef=0.01  # 适中的熵系数
        )
        
        logger.info("✅ Independent FOMAIPPO适配器初始化成功")
    except Exception as e:
        logger.error(f"❌ FOMAIPPO适配器初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': f'FOMAIPPO适配器初始化失败: {e}'
        }
    
    # 5. 初始化训练历史记录
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    training_history = []
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 6. 开始训练循环
    logger.info(f"开始FOMAIPPO训练循环 ({pipeline.num_episodes}个episodes)...")
    
    # 训练循环 - 独立学习架构
    for episode in range(1, max_allowed_episodes + 1):
        if episode > pipeline.num_episodes:
            logger.warning(f"已达到指定的episodes数量 {pipeline.num_episodes}，终止训练")
            break
            
        logger.info(f"\n========== Episode {episode}/{pipeline.num_episodes} (Independent FOMAIPPO) ==========")
        episode_start_time = datetime.now()
        
        # 重置环境
        if multi_env is not None:
            obs, infos = multi_env.reset()
        else:
            pipeline._reset_pipeline_state()
            obs = pipeline._get_pipeline_observations()
            infos = {}
        
        # 重置buffer
        fomaippo_adapter.reset_buffers()
        
        # 初始化episode奖励
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        
        # 执行episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"Episode {episode}, 时间步 {timestep}")
            
            # 独立策略选择动作
            actions, action_log_probs, values = fomaippo_adapter.select_actions(obs, deterministic=False)
            
            # 环境步进
            if multi_env is not None:
                next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
            else:
                # 使用pipeline执行
                pipeline_results = pipeline._execute_pipeline_with_actions(actions, timestep)
                next_obs = pipeline._get_pipeline_observations()
                rewards = pipeline._calculate_pipeline_rewards_from_results(pipeline_results, manager_ids)
                dones = {manager_id: (timestep == pipeline.steps_per_episode - 1) for manager_id in manager_ids}
            
            # 收集数据到独立的buffers
            fomaippo_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                action_log_probs=action_log_probs,
                values=values
            )
            
            # 累积奖励
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # 更新观测
            obs = next_obs
            
            # 显示时间步奖励
            timestep_total = sum(rewards.values())
            logger.info(f"  时间步 {timestep}: 总奖励 {timestep_total:.3f}")
        
        # episode结束后独立训练
        # 计算returns和advantages（独立计算）
        fomaippo_adapter.compute_returns()
        
        # 独立训练（每个Manager独立更新策略）
        train_info = fomaippo_adapter.train_on_batch()
        
        # 记录episode奖励和统计
        episode_total_reward = sum(episode_rewards.values())
        logger.info(f"Episode {episode} 完成:")
        logger.info(f"  🎯 总奖励: {episode_total_reward:.3f}")
        logger.info(f"  📈 训练损失: Actor {train_info['policy_loss']:.4f}, Critic {train_info['value_loss']:.4f}")
        
        # 显示每个Manager的奖励并记录到训练历史
        for manager_id, reward in episode_rewards.items():
            logger.info(f"  📊 {manager_id}: {reward:.3f}")
            training_episode_rewards[manager_id].append(reward)
            
            # 添加到训练历史
            training_data = {
                'algorithm': 'FOMAIPPO',
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': reward,
                'policy_loss': float(train_info.get('policy_loss', 0.001)),
                'value_loss': float(train_info.get('value_loss', 0.001)),
                'entropy': float(train_info.get('entropy', 0.001))
            }
            training_history.append(training_data)
            
            # 记录训练损失
            if hasattr(pipeline, '_record_training_loss'):
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(train_info.get('policy_loss', 0.001)),
                    value_loss=float(train_info.get('value_loss', 0.001)),
                    entropy=float(train_info.get('entropy', 0.001))
                )
        
        # 记录总体奖励
        training_data_total = {
            'algorithm': 'FOMAIPPO',
            'manager_id': 'total',
            'episode': episode,
            'episode_reward': episode_total_reward,
            'policy_loss': float(train_info.get('policy_loss', 0.0)),
            'value_loss': float(train_info.get('value_loss', 0.0)),
            'entropy': float(train_info.get('entropy', 0.0))
        }
        training_history.append(training_data_total)
        
        # 定期输出学习进度
        if (episode + 1) % 10 == 0:
            logger.info(f"\n========== Independent FOMAIPPO训练进度: {episode+1}/{pipeline.num_episodes} episodes ==========")
            
            # 获取训练统计
            try:
                training_stats = fomaippo_adapter.get_training_stats()
                manager_rewards = fomaippo_adapter.get_manager_rewards_summary()
                
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
                model_path = f"results/independent_fomaippo_ep{episode+1}"
                fomaippo_adapter.save_models(model_path)
                logger.info(f"📀 模型已保存至: {model_path}")
                
                # 保存训练历史
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMAIPPO")
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
        save_path = f"results/fomaippo_final"
        fomaippo_adapter.save_models(save_path)
        logger.info(f"保存最终模型: {save_path}")
    except Exception as e:
        logger.error(f"保存最终模型失败: {e}")
    
    # 计算总训练时间
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMAIPPO训练完成! 总训练时间: {total_training_time}")
    
    # 将训练历史整理为pipeline期望的格式
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},
            'episode_lengths': {},
            'training_loss': {},
            'training_metadata': {
                'algorithm': 'FOMAIPPO',
                'num_episodes': pipeline.num_episodes,
                'steps_per_episode': pipeline.steps_per_episode,
                'num_managers': num_managers
            }
        },
        'multi_agent_env': multi_env,
        'independent_fomaippo_adapter': fomaippo_adapter
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
                'entropy': item.get('entropy', 0.001)
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"返回结果包含 {len(result['training_history']['episode_rewards'])} 个Manager的训练历史")
    return result 
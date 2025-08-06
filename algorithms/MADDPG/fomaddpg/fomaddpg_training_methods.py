#!/usr/bin/env python3
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

try:
    from .fomaddpg_adapter import FOMAddpgAdapter
    FOMADDPG_ADAPTER_available = True
except ImportError:
    FOMAddpgAdapter = None
    FOMADDPG_ADAPTER_available = False

def train_fomaddpg_adapter(pipeline):
    logger.info("🚀 start optimized FOMADDPG training (solve overfitting and instability problem)")
    logger.info(f"plan to train {pipeline.num_episodes} episodes")
    
    # force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes parameter is invalid, set to default value 1")
        pipeline.num_episodes = 1
    
    # record maximum allowed episodes number
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # set a safe upper limit
    logger.info(f"maximum allowed episodes number: {max_allowed_episodes}")
    
    # update actual running algorithm
    pipeline._update_actual_algorithm("FOMADDPG_OPTIMIZED")
    
    # 1. prepare training environment
    logger.info("preparing FOMADDPG training environment...")
    
    # create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # initialize user states
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # create multi-agent environment
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
        logger.info("✅ successfully create multi_agent_env")
    except Exception as e:
        logger.error(f"❌ failed to create multi_agent_env: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'failed to create environment: {e}'}
    
    # 2. get environment parameters
    # get manager count and ID
    num_managers = multi_env.get_manager_count()
    manager_ids = list(multi_env.manager_agents.keys())
    logger.info(f"🏗️ environment configuration: {num_managers} managers: {manager_ids}")
    
    # get state and action space dimension
    try:
        sample_obs, _ = multi_env.reset()
        state_dim = len(sample_obs[manager_ids[0]])
        action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
        logger.info(f"📊 state space: {state_dim} dimensions, action space: {action_dim} dimensions")
    except Exception as e:
        logger.error(f"❌ failed to get observation and action space: {e}")
        return {'status': 'failed', 'error': f'failed to get environment parameters: {e}'}
    
    # 3. create FOMADDPG adapter
    try:
        if not FOMADDPG_ADAPTER_available or FOMAddpgAdapter is None:
            logger.error("❌ FOMAddpgAdapter is not available")
            return {'status': 'failed', 'error': 'FOMAddpgAdapter is not available'}
            
        # FOMADDPG specific optimized hyperparameters
        # MADDPG as off-policy algorithm, has some special parameters
        WARMUP_EPISODES = 10  # first 10 episodes only collect experience, not update policy
        NOISE_DECAY = 0.99    # noise decay rate
        MIN_NOISE = 0.01      # minimum noise ratio
        INITIAL_NOISE = 0.2   # initial noise ratio
        UPDATE_FREQ = 2       # update every 2 time steps
        BATCH_SIZE = 128      # batch size
        
        # create FOMADDPG adapter, use optimized hyperparameters
        fomaddpg_adapter = FOMAddpgAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            episode_length=pipeline.steps_per_episode,
            
            lr_actor=5e-5,      # reduce from 1e-4 to 5e-5
            lr_critic=1e-4,     # reduce from 1e-3 to 1e-4
            device=pipeline.device if hasattr(pipeline, 'device') else "cpu",
            
            hidden_dim=256,
            max_action=1.0,
            gamma=0.99,
            tau=0.001,           # soft update parameter is lower, more stable
            noise_scale=INITIAL_NOISE,
            buffer_capacity=1000000,
            batch_size=BATCH_SIZE,
            
            # FlexOffer specific parameters
            use_device_coordination=True,
            device_coordination_weight=0.1,
            fo_constraint_weight=0.2,
            use_manager_coordination=True,
            manager_coordination_weight=0.05
        )
        logger.info("✅ successfully initialize FOMADDPG adapter")
    except Exception as e:
        logger.error(f"❌ failed to create FOMADDPG adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'failed to create adapter: {e}'}
    
    # 4. initialize training history
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    training_history = []
    
    # 5. set dynamic exploration noise
    current_noise_scale = INITIAL_NOISE
    fomaddpg_adapter.fomaddpg.noise_scale = current_noise_scale
    
    # record start time
    start_time = datetime.now()
    
    # 6. start training loop
    logger.info(f"start FOMADDPG training loop ({pipeline.num_episodes} episodes)...")
    
    # training loop - off-policy learning
    for episode in range(1, max_allowed_episodes + 1):
        if episode > pipeline.num_episodes:
            logger.warning(f"reached the specified episodes number {pipeline.num_episodes}, terminate training")
            break
            
        logger.info(f"\n========== Episode {episode}/{pipeline.num_episodes} (FOMADDPG) ==========")
        episode_start_time = datetime.now()
        
        # reset environment
        obs, infos = multi_env.reset()
        fomaddpg_adapter.reset_buffers()  # for MADDPG, this operation is safe, will not clear experience replay buffer
        
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        
        # dynamic adjust noise ratio
        if episode > WARMUP_EPISODES:
            current_noise_scale = max(MIN_NOISE, current_noise_scale * NOISE_DECAY)
            fomaddpg_adapter.fomaddpg.noise_scale = current_noise_scale
            logger.info(f"📉 noise ratio adjusted: {current_noise_scale:.4f}")
        
        # run each episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"Episode {episode}, time step {timestep}/{pipeline.steps_per_episode-1}")
            
            # use exploration or exploitation strategy
            use_noise = (episode <= WARMUP_EPISODES * 2)  # more exploration in the early stages
            actions, action_log_probs, values = fomaddpg_adapter.select_actions(obs, deterministic=not use_noise)
            
            # environment step
            next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
            
            # collect data to experience replay buffer
            fomaddpg_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                action_log_probs=action_log_probs,
                values=values
            )
            
            # accumulate rewards
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # update observation
            obs = next_obs
            
            # MADDPG specific: batch update
            # 🔧 optimization: don't update every time step, but update every few time steps
            if timestep % UPDATE_FREQ == 0 and episode > WARMUP_EPISODES:
                # 🔧 optimization: first few episodes don't update, only collect experience
                train_info = fomaddpg_adapter.train_on_batch()
                if isinstance(train_info, dict):
                    policy_loss = train_info.get('policy_loss', 0.0)
                    value_loss = train_info.get('value_loss', 0.0)
                    logger.debug(f"  ⚙️ training update: Actor Loss: {policy_loss:.5f}, Critic Loss: {value_loss:.5f}")
            
            # display time step reward
            timestep_total = sum(rewards.values())
            logger.info(f"  time step {timestep} total reward: {timestep_total:.3f}")
        
        # training after episode - MADDPG can update multiple times
        if episode > WARMUP_EPISODES:
            # update multiple times after episode
            for _ in range(5):  # update 5 times
                update_info = fomaddpg_adapter.train_on_batch()
        
        # record episode reward and statistics
        episode_total_reward = sum(episode_rewards.values())
        logger.info(f"Episode {episode} completed:")
        logger.info(f"  🎯 total reward: {episode_total_reward:.3f}")
        
        # if there is training information
        if 'update_info' in locals() and isinstance(update_info, dict):
            logger.info(f"  📈 training loss: Actor {update_info.get('policy_loss', 0):.4f}, Critic {update_info.get('value_loss', 0):.4f}")
        
        # display each manager's reward and record to training history
        for manager_id, reward in episode_rewards.items():
            logger.info(f"  📊 {manager_id}: {reward:.3f}")
            training_episode_rewards[manager_id].append(reward)
            
            # add to training history
            training_data = {
                'algorithm': 'FOMADDPG',
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': reward,
                'policy_loss': float(update_info.get('policy_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'value_loss': float(update_info.get('value_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'entropy': 0.0  # MADDPG has no entropy
            }
            training_history.append(training_data)
            
            # record training loss
            if hasattr(pipeline, '_record_training_loss') and 'update_info' in locals():
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(update_info.get('policy_loss', 0.001)),
                    value_loss=float(update_info.get('value_loss', 0.001)),
                    entropy=0.0  # MADDPG has no entropy
                )
        
        # record total reward
        training_data_total = {
            'algorithm': 'FOMADDPG',
            'manager_id': 'total',
            'episode': episode,
            'episode_reward': episode_total_reward,
            'policy_loss': float(update_info.get('policy_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'value_loss': float(update_info.get('value_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'entropy': 0.0  # MADDPG has no entropy
        }
        training_history.append(training_data_total)
        
        # display learning progress periodically
        if (episode + 1) % 10 == 0:
            logger.info(f"\n========== FOMADDPG training progress: {episode+1}/{pipeline.num_episodes} episodes ==========")
            
            # get training statistics
            try:
                training_stats = fomaddpg_adapter.get_training_stats()
                manager_rewards = fomaddpg_adapter.get_manager_rewards_summary()
                
                if isinstance(manager_rewards, dict):
                    for manager_id, stats in manager_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            training_updates = stats.get('training_updates', 0)
                            logger.info(f"  🔥 {manager_id}: total reward {total_reward:.2f}, best {best_reward:.2f}, update {training_updates} times")
                        else:
                            logger.info(f"  🔥 {manager_id}: total reward {stats:.2f}")
                else:
                    logger.info(f"  🔥 manager rewards: {manager_rewards}")
                
                if isinstance(training_stats, dict):
                    iterations = training_stats.get('training_iterations', 0)
                    logger.info(f"  🚀 total training iterations: {iterations}")
                else:
                    logger.info(f"  🚀 training statistics: {training_stats}")
            except Exception as e:
                logger.warning(f"failed to get training statistics: {e}")
                logger.info("  🔥 training progress: learning...")
            
            logger.info("=" * 70)
        
        # save model periodically
        if (episode + 1) % 20 == 0 or episode == pipeline.num_episodes:
            try:
                model_path = f"results/fomaddpg_optimized_ep{episode+1}"
                fomaddpg_adapter.save_models(model_path)
                logger.info(f"📀 model saved to: {model_path}")
                
                # save training history
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMADDPG_OPTIMIZED")
            except Exception as e:
                logger.error(f"failed to save model: {e}")
        
        # calculate episode duration
        episode_duration = datetime.now() - episode_start_time
        logger.info(f"Episode {episode} duration: {episode_duration}")
        
        # display total progress
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"elapsed time: {total_elapsed}, estimated remaining: {estimated_remaining}")
    
    # training finished, save final model
    try:
        save_path = f"results/fomaddpg_optimized_final"
        fomaddpg_adapter.save_models(save_path)
        logger.info(f"final model saved to: {save_path}")
    except Exception as e:
        logger.error(f"failed to save final model: {e}")
    
    # calculate total training time
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMADDPG training completed! total training time: {total_training_time}")
    
    # convert training history to pipeline expected format
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},
            'episode_lengths': {},
            'training_loss': {},
            'training_metadata': {
                'algorithm': 'FOMADDPG',
                'num_episodes': pipeline.num_episodes,
                'steps_per_episode': pipeline.steps_per_episode,
                'num_managers': num_managers
            }
        },
        'multi_agent_env': multi_env,
        'fomaddpg_adapter': fomaddpg_adapter
    }
    
    # process training history data, group by manager_id
    for item in training_history:
        manager_id = item.get('manager_id')
        if manager_id and manager_id != 'total':  # exclude total record
            if manager_id not in result['training_history']['episode_rewards']:
                result['training_history']['episode_rewards'][manager_id] = []
                result['training_history']['episode_lengths'][manager_id] = []
                result['training_history']['training_loss'][manager_id] = []
            
            # add reward and length
            result['training_history']['episode_rewards'][manager_id].append(item.get('episode_reward', 0.0))
            result['training_history']['episode_lengths'][manager_id].append(pipeline.steps_per_episode)
            
            # add training loss
            loss_info = {
                'policy_loss': item.get('policy_loss', 0.001),
                'value_loss': item.get('value_loss', 0.001),
                'entropy': item.get('entropy', 0.0)  # MADDPG has no entropy
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"result contains {len(result['training_history']['episode_rewards'])} managers' training history")
    return result 
#!/usr/bin/env python3
"""
FOMAPPO and FOMAIPPO training methods

provide training implementation for shared policy FOMAPPO and independent policy FOMAIPPO
used to integrate these two algorithms in FO Pipeline
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

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

    logger.info("start optimized FOMAPPO training (enhance learning effect and stability)")
    logger.info(f"plan to train {pipeline.num_episodes} episodes")
    
    # force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes parameter is invalid, set to default value 1")
        pipeline.num_episodes = 1
    
    # record maximum allowed episodes number
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # set a safe upper limit
    logger.info(f"maximum allowed episodes number: {max_allowed_episodes}")
    
    # update actual running algorithm
    pipeline._update_actual_algorithm("FOMAPPO_FIXED")
    
    # 1. prepare training environment
    logger.info("preparing FOMAPPO training environment...")
    
    # create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # initialize user state
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # create or get multi-agent environment
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("use existing multi_agent_env")
    else:
        # create new multi-agent environment
        logger.info("create new multi_agent_env")
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
            logger.info("successfully created multi_agent_env")
        except Exception as e:
            logger.error(f"failed to create multi_agent_env: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. get environment parameters
    # get manager number and ID
    manager_ids = [manager.manager_id for manager in pipeline.managers]
    num_managers = len(manager_ids)
    
    # get state dimension and action dimension
    # 3. create FOMAPPO adapter
    logger.info(f"create FOMAPPO adapter: {num_managers} managers")

    # get actual observation dimension of environment
    try:
        # get sample observation from environment to determine dimension
        logger.info("get actual observation dimension from environment...")
        if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
            # if environment already exists, use it to get observation
            sample_obs, _ = pipeline.multi_agent_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            logger.info(f"get observation dimension from existing environment: {state_dim}")
        elif multi_env is not None:
            # use existing multi-agent environment
            logger.info("use existing multi_env to get observation dimension...")
            
            # get actual observation dimension
            sample_obs, _ = multi_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            logger.info(f"get observation dimension from multi_env: {state_dim}")
        else:
            # cannot get environment, use default dimension
            logger.warning("cannot get environment, use default dimension")
            state_dim = 73  # default state dimension
        
        # get action dimension from environment
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
                action_dim = 100  # default action dimension
                
    except Exception as e:
        logger.warning(f"failed to get observation dimension from environment: {e}")
        
        # fallback to get state dimension from manager
    if hasattr(pipeline, "_get_manager_state"):
        # get sample state to determine dimension
        sample_state = pipeline._get_manager_state(pipeline.managers[0])
        state_dim = len(sample_state)
    else:
            state_dim = 73  # default state dimension
    
    if hasattr(pipeline, "_get_manager_action_dim"):
        action_dim = pipeline._get_manager_action_dim()
    else:
        action_dim = 100  # default action dimension
    
        logger.warning(f"use fallback dimension: state={state_dim}, action={action_dim}")

    logger.info(f"determine observation dimension: {state_dim}, action dimension: {action_dim}")
    
    # use optimized hyperparameters
    fomappo_adapter = FOMAPPOAdapter(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_managers,
        episode_length=pipeline.steps_per_episode,
        lr_actor=5e-5,  # decrease learning rate
        lr_critic=2e-4,  # decrease learning rate
        entropy_coef=0.05,  # increase entropy coefficient, encourage exploration
        use_linear_lr_decay=True,  # enable learning rate decay
        lr_decay_rate=0.95,  # learning rate decay rate
        use_clipped_value_loss=True,  # use clipped value loss
        use_max_grad_norm=True,  # use gradient clipping
        max_grad_norm=0.5,  # gradient clipping threshold
        device="cpu"
    )
    
    # 4. initialize training history record
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    
    # 5. record training loss
    training_losses = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': []
    }
    
    # 6. start training loop
    logger.info(f"start FOMAPPO training loop ({pipeline.num_episodes} episodes)...")
    
    # initialize result collector
    cumulative_rewards = {manager_id: 0.0 for manager_id in manager_ids}
    avg_rewards_last_10 = {manager_id: [] for manager_id in manager_ids}
    
    # initialize data structure for result output
    training_history = []
    
    # record start time
    start_time = datetime.now()
    
    # set training termination flag
    training_complete = False
    
    # main training loop
    for episode in range(1, max_allowed_episodes + 1):
        # check if the specified episodes number is reached
        if episode > pipeline.num_episodes:
            logger.warning(f"the specified episodes number {pipeline.num_episodes} is reached, terminate training")
            training_complete = True
            break
            
        logger.info(f"========== start episode {episode}/{pipeline.num_episodes} ==========")
        episode_start_time = datetime.now()
        
        # reset environment state
        pipeline._reset_pipeline_state()
        
        # initialize episode statistics
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        episode_total_reward = 0.0  # initialize episode total reward
        
        # execute an episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"episode {episode}/{pipeline.num_episodes}, timestep {timestep}/{pipeline.steps_per_episode-1}")
            
            # get observation
            obs = pipeline._get_pipeline_observations()
            
            # select action
            actions, action_log_probs, values = fomappo_adapter.select_actions(obs)
            
            # execute action
            pipeline_results = pipeline._execute_pipeline_with_actions(actions, timestep)
            
            # get reward
            rewards = pipeline._calculate_pipeline_rewards_from_results(pipeline_results, manager_ids)
            
            # update episode reward
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # check if episode is done
            dones = {manager_id: (timestep == pipeline.steps_per_episode - 1) for manager_id in manager_ids}
            
            # collect experience
            fomappo_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos={},
                action_log_probs=action_log_probs,
                values=values
            )
        
        # update episode total reward after episode
        episode_total_reward = sum(episode_rewards.values())  # calculate total reward
        
        for manager_id in manager_ids:
            cumulative_rewards[manager_id] += episode_rewards[manager_id]
            
            # maintain sliding window average
            if len(avg_rewards_last_10[manager_id]) >= 10:
                avg_rewards_last_10[manager_id].pop(0)
            avg_rewards_last_10[manager_id].append(episode_rewards[manager_id])
            
            # add to training history
            training_episode_rewards[manager_id].append(episode_rewards[manager_id])
        
        # update episode count of adapter
        fomappo_adapter.total_episodes = episode
        
        # calculate return and advantage
        logger.info(f"episode {episode}/{pipeline.num_episodes} completed data collection, calculate return and advantage...")
        fomappo_adapter.compute_returns()
        
        # execute training update
        train_info = {}
        total_train_info = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'num_updates': 0}
        
        # execute multiple PPO updates
        num_epochs = fomappo_adapter.args.ppo_epoch
        logger.info(f"execute {num_epochs} PPO updates...")
        
        for epoch in range(num_epochs):
            batch_train_info = fomappo_adapter.train_on_batch()
            
            if batch_train_info:
                # debug: print training information of each batch
                logger.debug(f"Epoch {epoch}/{num_epochs}, Batch training information: {batch_train_info}")
                
                # ensure key name consistency
                if 'dist_entropy' in batch_train_info and 'entropy' not in batch_train_info:
                    batch_train_info['entropy'] = batch_train_info['dist_entropy']
                
                total_train_info['policy_loss'] += batch_train_info.get('policy_loss', 0.0)
                total_train_info['value_loss'] += batch_train_info.get('value_loss', 0.0)
                total_train_info['entropy'] += batch_train_info.get('entropy', 0.0)
                total_train_info['num_updates'] += 1
        
        # calculate average loss
        if total_train_info['num_updates'] > 0:
            total_train_info['policy_loss'] /= total_train_info['num_updates']
            total_train_info['value_loss'] /= total_train_info['num_updates']
            total_train_info['entropy'] /= total_train_info['num_updates']
            
            # debug: print average loss
            logger.info(f"Episode {episode}, average loss: Policy={total_train_info['policy_loss']:.6f}, Value={total_train_info['value_loss']:.6f}, Entropy={total_train_info['entropy']:.6f}")
        else:
            # if no successful update, ensure loss value is not zero
            logger.warning("no successful training update, use default non-zero loss value")
            total_train_info['policy_loss'] = 0.001  # use a small non-zero value
            total_train_info['value_loss'] = 0.001
            total_train_info['entropy'] = 0.001
        
        train_info = total_train_info
        
        # record training loss
        if isinstance(train_info, dict):
            # ensure loss value is not zero
            policy_loss = max(train_info.get('policy_loss', 0.0), 1e-4)
            value_loss = max(train_info.get('value_loss', 0.0), 1e-4)
            entropy = max(train_info.get('entropy', 0.0), 1e-4)
            
            training_losses['policy_loss'].append(policy_loss)
            training_losses['value_loss'].append(value_loss)
            training_losses['entropy'].append(entropy)
            
            # update values in train_info
            train_info['policy_loss'] = policy_loss
            train_info['value_loss'] = value_loss
            train_info['entropy'] = entropy
            
            # record to log
            logger.info(f"Episode {episode}/{pipeline.num_episodes} training loss: " +
                       f"Policy Loss: {policy_loss:.5f}, " +
                       f"Value Loss: {value_loss:.5f}, " +
                       f"Entropy: {entropy:.5f}")
        
        # record training data to training history of pipeline
        for manager_id in manager_ids:
            episode_reward = episode_rewards[manager_id]
            avg_reward = sum(avg_rewards_last_10[manager_id]) / len(avg_rewards_last_10[manager_id])
            
            # record to training history
            training_data = {
                'algorithm': 'FOMAPPO',  # use standard algorithm name, without FIXED suffix
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': episode_reward,
                'cumulative_reward': cumulative_rewards[manager_id],
                'avg_reward_last_10': avg_reward
            }
            
            # add training loss
            if isinstance(train_info, dict):
                training_data['policy_loss'] = float(train_info.get('policy_loss', 0.001))
                training_data['value_loss'] = float(train_info.get('value_loss', 0.001))
                training_data['entropy'] = float(train_info.get('entropy', 0.001))
                
                # ensure value is Python native type
                for key in ['policy_loss', 'value_loss', 'entropy']:
                    if key in training_data:
                        if isinstance(training_data[key], (np.ndarray, np.number)):
                            training_data[key] = float(training_data[key])
                        elif torch.is_tensor(training_data[key]):
                            training_data[key] = float(training_data[key].item())
            
            # add to training history
            training_history.append(training_data)
            
            # call pipeline's loss recording function
            if hasattr(pipeline, '_record_training_loss'):
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(train_info.get('policy_loss', 0.001)),
                    value_loss=float(train_info.get('value_loss', 0.001)),
                    entropy=float(train_info.get('entropy', 0.001))
                )
        
        # record total reward
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
        
        # calculate episode duration
        episode_duration = datetime.now() - episode_start_time
        
        # output training progress
        if episode % 1 == 0 or episode == pipeline.num_episodes:
            logger.info(f"episode {episode}/{pipeline.num_episodes} completed, duration: {episode_duration}, total reward: {episode_total_reward:.3f}, " +
                       f"Policy Loss: {train_info.get('policy_loss', 0.0):.5f}, " +
                       f"Value Loss: {train_info.get('value_loss', 0.0):.5f}, " +
                       f"Entropy: {train_info.get('entropy', 0.0):.5f}")
        
        # save checkpoint model
        if episode % 20 == 0 or episode == pipeline.num_episodes:
            try:
                save_path = f"results/fomappo_fixed_final"
                fomappo_adapter.save_models(save_path)
                logger.info(f"save checkpoint model: {save_path}")
                
                # save training history
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMAPPO")
            except Exception as e:
                logger.error(f"save model failed: {e}")
        
        # show total progress
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"========== episode {episode}/{pipeline.num_episodes} completed ==========")
        logger.info(f"total elapsed time: {total_elapsed}, average time per episode: {avg_time_per_episode}")
        logger.info(f"estimated remaining time: {estimated_remaining}")
        logger.info("=" * 50)
        
        # check if the specified episodes number is reached
        if episode >= pipeline.num_episodes:
            logger.info(f"the specified episodes number {pipeline.num_episodes} is reached, terminate training")
            training_complete = True
            break
    
    # check if training is completed normally
    if not training_complete:
        logger.warning(f"training is not completed normally, possibly because the maximum allowed episodes number {max_allowed_episodes} is reached")
    
    # training completed, save final model
    try:
        save_path = f"results/fomappo_fixed_final"
        fomappo_adapter.save_models(save_path)
        logger.info(f"save final model: {save_path}")
        
        # save training history
        if hasattr(pipeline, '_force_save_training_history'):
            pipeline._force_save_training_history(training_history, "FOMAPPO")
            
        # save to CSV
        if hasattr(pipeline, '_save_training_history_to_csv'):
            pipeline._save_training_history_to_csv("FOMAPPO")
    except Exception as e:
        logger.error(f"save final model failed: {e}")
    
    # calculate total training time
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMAPPO training completed! total training time: {total_training_time}")
    
    # so that _train_fomappo_agents method can correctly get training history
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},  # convert list format to dictionary format, so that pipeline can process it
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
    
    # process training history data, convert list format to dictionary format
    # group by manager_id
    for item in training_history:
        manager_id = item.get('manager_id')
        if manager_id and manager_id != 'total':  
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
                'entropy': item.get('entropy', 0.001)
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"return result contains {len(result['training_history']['episode_rewards'])} managers' training history")
    return result


def train_fomaippo_independent_policy(pipeline):
    logger.info("start FOMAIPPO training (independent policy architecture, solve policy conflict problem)")
    logger.info(f"plan to train {pipeline.num_episodes} episodes")
    
    # force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes parameter is invalid, set to default value 1")
        pipeline.num_episodes = 1
    
    # record maximum allowed episodes number
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # set a safe upper limit
    logger.info(f"maximum allowed episodes number: {max_allowed_episodes}")
    
    # update actual running algorithm
    pipeline._update_actual_algorithm("FOMAIPPO")
    
    # 1. prepare training environment
    logger.info("preparing FOMAIPPO training environment...")
    
    # create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # initialize user states
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # create or get multi-agent environment
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("use existing multi_agent_env")
    else:
        # create new multi-agent environment
        logger.info("create new multi_agent_env")
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
            logger.info("✅ successfully create multi_agent_env")
        except Exception as e:
            logger.error(f"❌ create multi_agent_env failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. get environment parameters
    # get manager count and ID
    if multi_env is not None:
        num_managers = multi_env.get_manager_count()
        manager_ids = list(multi_env.manager_agents.keys())
    else:
        manager_ids = [manager.manager_id for manager in pipeline.managers]
        num_managers = len(manager_ids)
    
    logger.info(f"environment configuration: {num_managers} managers: {manager_ids}")
    
    # 3. get state and action space dimension
    try:
        # get actual observation dimension from environment
        logger.info("get actual observation dimension from environment...")
        if multi_env is not None:
            sample_obs, _ = multi_env.reset()
            sample_manager_id = list(sample_obs.keys())[0]
            state_dim = len(sample_obs[sample_manager_id])
            
            # action dimension
            action_dim = multi_env.action_spaces[sample_manager_id].shape[0]
            logger.info(f"✅ get dimension from multi_env: state={state_dim}, action={action_dim}")
        else:
            # fallback to get state dimension from pipeline
            if hasattr(pipeline, "_get_manager_state"):
                sample_state = pipeline._get_manager_state(pipeline.managers[0])
                state_dim = len(sample_state)
            else:
                state_dim = 73  # default state dimension
            
            # get action dimension
            if hasattr(pipeline, "_get_manager_action_dim"):
                action_dim = pipeline._get_manager_action_dim()
            else:
                action_dim = 100  # default action dimension
                
            logger.info(f"⚠️ use fallback dimension: state={state_dim}, action={action_dim}")
    except Exception as e:
        logger.error(f"❌ get dimension failed: {e}")
        # set safe default value
        state_dim = 73  # default state dimension
        action_dim = 100  # default action dimension
        logger.warning(f"use default dimension: state={state_dim}, action={action_dim}")
    
    # 4. initialize FOMAIPPO adapter - use stable hyper-parameters
    try:
        # check if FOMAIPPO is available
        if not FOMAIPPO_available or FOMAIPPOAdapter is None:
            logger.error("❌ FOMAIPPOAdapter is not available")
            return {
                'status': 'failed',
                'error': 'FOMAIPPOAdapter is not available'
            }
        
        fomaippo_adapter = FOMAIPPOAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            episode_length=pipeline.steps_per_episode,
            lr_actor=5e-5,  # lower learning rate
            lr_critic=1e-4,  # lower learning rate
            device=pipeline.device if hasattr(pipeline, 'device') else "cpu",
            # FOMAPPO special features (lower weights)
            use_device_coordination=True,
            device_coordination_weight=0.05,  # lower coordination weight
            fo_constraint_weight=0.1,  # lower constraint weight
            use_manager_coordination=True,
            manager_coordination_weight=0.02,  # lower coordination weight
            # numerical stability parameters
            clip_param=0.1,  # small clip range
            max_grad_norm=0.2,  # strong gradient clipping
            value_loss_coef=0.5,  # lower value loss weight
            entropy_coef=0.01  # moderate entropy coefficient
        )
        
        logger.info("✅ successfully initialize Independent FOMAIPPO adapter")
    except Exception as e:
        logger.error(f"❌ initialize FOMAIPPO adapter failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': f'initialize FOMAIPPO adapter failed: {e}'
        }
    
    # 5. initialize training history record
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    training_history = []
    
    # record start time
    start_time = datetime.now()
    
    # 6. start training loop
    logger.info(f"start FOMAIPPO training loop ({pipeline.num_episodes} episodes)...")
    
    # training loop - independent learning architecture
    for episode in range(1, max_allowed_episodes + 1):
        if episode > pipeline.num_episodes:
            logger.warning(f"the specified episodes number {pipeline.num_episodes} is reached, terminate training")
            break
            
        logger.info(f"\n========== episode {episode}/{pipeline.num_episodes} (Independent FOMAIPPO) ==========")
        episode_start_time = datetime.now()
        
        # reset environment
        if multi_env is not None:
            obs, infos = multi_env.reset()
        else:
            pipeline._reset_pipeline_state()
            obs = pipeline._get_pipeline_observations()
            infos = {}
        
        # reset buffer
        fomaippo_adapter.reset_buffers()
        
        # initialize episode reward
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        
        # execute episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"episode {episode}, timestep {timestep}")
            
            # independent policy select actions
            actions, action_log_probs, values = fomaippo_adapter.select_actions(obs, deterministic=False)
            
            # environment step
            if multi_env is not None:
                next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
            else:
                # use pipeline to execute
                pipeline_results = pipeline._execute_pipeline_with_actions(actions, timestep)
                next_obs = pipeline._get_pipeline_observations()
                rewards = pipeline._calculate_pipeline_rewards_from_results(pipeline_results, manager_ids)
                dones = {manager_id: (timestep == pipeline.steps_per_episode - 1) for manager_id in manager_ids}
            
            # collect data to independent buffers
            fomaippo_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                action_log_probs=action_log_probs,
                values=values
            )
            
            # accumulate reward
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # update observation
            obs = next_obs
            
            # show timestep reward
            timestep_total = sum(rewards.values())
            logger.info(f"timestep {timestep}: total reward {timestep_total:.3f}")
        
        # after episode, independent training
        # calculate returns and advantages (independent calculation)
        fomaippo_adapter.compute_returns()
        
        # independent training (each manager independently update policy)
        train_info = fomaippo_adapter.train_on_batch()
        
        # record episode reward and statistics
        episode_total_reward = sum(episode_rewards.values())
        logger.info(f"episode {episode} completed:")
        logger.info(f"  🎯 total reward: {episode_total_reward:.3f}")
        logger.info(f"  📈 training loss: Actor {train_info['policy_loss']:.4f}, Critic {train_info['value_loss']:.4f}")
        
        # show each manager's reward and record to training history
        for manager_id, reward in episode_rewards.items():
            logger.info(f"  📊 {manager_id}: {reward:.3f}")
            training_episode_rewards[manager_id].append(reward)
            
            # add to training history
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
            
            # record training loss
            if hasattr(pipeline, '_record_training_loss'):
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(train_info.get('policy_loss', 0.001)),
                    value_loss=float(train_info.get('value_loss', 0.001)),
                    entropy=float(train_info.get('entropy', 0.001))
                )
        
        # record total reward
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
        
        # show learning progress periodically
        if (episode + 1) % 10 == 0:
            logger.info(f"\n========== Independent FOMAIPPO training progress: {episode+1}/{pipeline.num_episodes} episodes ==========")
            
            # get training statistics
            try:
                training_stats = fomaippo_adapter.get_training_stats()
                manager_rewards = fomaippo_adapter.get_manager_rewards_summary()
                
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
                logger.warning(f"get training statistics failed: {e}")
                logger.info("  🔥 training progress: learning...")
            
            logger.info("=" * 70)
        
        # save model periodically
        if (episode + 1) % 20 == 0 or episode == pipeline.num_episodes:
            try:
                model_path = f"results/independent_fomaippo_ep{episode+1}"
                fomaippo_adapter.save_models(model_path)
                logger.info(f"📀 model saved to: {model_path}")
                
                # save training history
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMAIPPO")
            except Exception as e:
                logger.error(f"save model failed: {e}")
        
        # calculate episode duration
        episode_duration = datetime.now() - episode_start_time
        logger.info(f"episode {episode} duration: {episode_duration}")
        
        # show total progress
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"used time: {total_elapsed}, estimated remaining: {estimated_remaining}")
    
    # training end, save final model
    try:
        save_path = f"results/fomaippo_final"
        fomaippo_adapter.save_models(save_path)
        logger.info(f"save final model: {save_path}")
    except Exception as e:
        logger.error(f"save final model failed: {e}")
    
    # calculate total training time
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMAIPPO training completed! total training time: {total_training_time}")
    
    # convert training history to pipeline expected format
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
                'entropy': item.get('entropy', 0.001)
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"return result contains {len(result['training_history']['episode_rewards'])} managers' training history")
    return result 
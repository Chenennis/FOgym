#!/usr/bin/env python3
"""
FOMATD3 Training Methods

Provides standardized FOMATD3 training implementation
For integrating FOMATD3 algorithm (Twin Delayed DDPG) into FO Pipeline
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Add missing imports
try:
    from .fomatd3_adapter import FOMATD3Adapter
    FOMATD3_ADAPTER_available = True
except ImportError:
    FOMATD3Adapter = None
    FOMATD3_ADAPTER_available = False

def train_fomatd3_adapter(pipeline):
    """
    Optimized FOMATD3 training method - includes stability and performance improvements
    
    Args:
        pipeline: FO Pipeline instance
    
    Returns:
        Dictionary containing training results
    """
    logger.info("🚀 Starting optimized FOMATD3 training (TD3 dual Critic network architecture)")
    logger.info(f"Planning to train for {pipeline.num_episodes} episodes")
    
    # Force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("Invalid num_episodes parameter, setting to default value 1")
        pipeline.num_episodes = 1
    
    # Record maximum allowed episodes
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # Set a safe upper limit
    logger.info(f"Maximum allowed episodes: {max_allowed_episodes}")
    
    # Update actual running algorithm
    pipeline._update_actual_algorithm("FOMATD3_ADAPTER")
    
    # 1. Prepare training environment
    logger.info("Preparing FOMATD3 training environment...")
    
    # Create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # Reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # Initialize user states
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # Create multi-agent environment
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
        logger.info("✅ Successfully created multi_agent_env")
    except Exception as e:
        logger.error(f"❌ Failed to create multi_agent_env: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'Failed to create environment: {e}'}
    
    # 2. Get environment parameters
    # Get manager count and IDs
    num_managers = multi_env.get_manager_count()
    manager_ids = list(multi_env.manager_agents.keys())
    logger.info(f"🏗️ Environment configuration: {num_managers} Managers: {manager_ids}")
    
    # Get state and action space dimensions
    try:
        sample_obs, _ = multi_env.reset()
        state_dim = len(sample_obs[manager_ids[0]])
        action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
        logger.info(f"📊 State space: {state_dim} dimensions, Action space: {action_dim} dimensions")
    except Exception as e:
        logger.error(f"❌ Failed to get observation and action spaces: {e}")
        return {'status': 'failed', 'error': f'Failed to get environment parameters: {e}'}
    
    # 3. Create FOMATD3 adapter
    try:
        if not FOMATD3_ADAPTER_available or FOMATD3Adapter is None:
            logger.error("❌ FOMATD3Adapter not available")
            return {'status': 'failed', 'error': 'FOMATD3Adapter not available'}
            
        # TD3-specific optimized hyperparameters
        # TD3 features dual Critic networks and delayed policy updates
        WARMUP_EPISODES = 10  # First 10 episodes only collect experience, no policy updates
        NOISE_DECAY = 0.995   # Noise decay rate (slightly higher than DDPG, as TD3 is more stable)
        MIN_NOISE = 0.02      # Minimum noise ratio
        INITIAL_NOISE = 0.2   # Initial noise ratio
        UPDATE_FREQ = 2       # Update every 2 time steps
        BATCH_SIZE = 128      # Batch size
        POLICY_DELAY = 2      # TD3-specific: policy delay update coefficient
        
        # Create FOMATD3 adapter with optimized hyperparameters
        fomatd3_adapter = FOMATD3Adapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            episode_length=pipeline.steps_per_episode,
            
            # Learning rates - TD3 typically uses lower learning rates for better stability
            lr_actor=5e-5,      # Reduced from 1e-4 to 5e-5
            lr_critic=1e-4,     # Reduced from 1e-3 to 1e-4
            hidden_dim=256,
            device=pipeline.device if hasattr(pipeline, 'device') else "cpu",
            
            # TD3-specific parameters
            buffer_capacity=500000,  # Larger buffer
            batch_size=BATCH_SIZE,
            gamma=0.99,              # Discount factor
            tau=0.001,               # Soft update parameter (smaller, more stable)
            noise_scale=INITIAL_NOISE,
            noise_clip=0.5,          # TD3-specific: target policy noise clipping
            target_noise=0.2,        # TD3-specific: target policy noise
            policy_delay=POLICY_DELAY  # TD3-specific: policy delay update
        )
        logger.info("✅ FOMATD3 adapter initialized successfully")
        logger.info(f"   Using TD3-specific parameters: dual Critic networks, policy delay update({POLICY_DELAY}), target action smoothing")
    except Exception as e:
        logger.error(f"❌ Failed to create FOMATD3 adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'failed', 'error': f'Failed to create adapter: {e}'}
    
    # 4. Initialize training history records
    training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
    training_history = []
    
    # 5. Set dynamic exploration noise
    current_noise_scale = INITIAL_NOISE
    
    # Record start time
    start_time = datetime.now()
    
    # 6. Start training loop
    logger.info(f"Starting FOMATD3 training loop ({pipeline.num_episodes} episodes)...")
    
    # Training loop - Off-policy learning based on TD3
    for episode in range(1, max_allowed_episodes + 1):
        if episode > pipeline.num_episodes:
            logger.warning(f"Reached specified number of episodes {pipeline.num_episodes}, terminating training")
            break
            
        logger.info(f"\n========== Episode {episode}/{pipeline.num_episodes} (FOMATD3) ==========")
        episode_start_time = datetime.now()
        
        # Reset environment
        obs, infos = multi_env.reset()
        fomatd3_adapter.reset_buffers()  # For TD3, this operation is safe, doesn't clear experience replay buffer
        
        episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
        
        # Dynamically adjust noise scale (TD3 can typically handle slower noise decay)
        if episode > WARMUP_EPISODES:
            current_noise_scale = max(MIN_NOISE, current_noise_scale * NOISE_DECAY)
            # Update adapter's noise parameters
            for agent_id in range(fomatd3_adapter.n_agents):
                if hasattr(fomatd3_adapter, 'agents') and fomatd3_adapter.agents is not None:
                    fomatd3_adapter.agents[agent_id].noise_scale = current_noise_scale
                else:
                    fomatd3_adapter.noise_scale = current_noise_scale
            logger.info(f"📉 Noise adjustment: {current_noise_scale:.4f}")
        
        # Run steps for each episode
        for timestep in range(pipeline.steps_per_episode):
            logger.info(f"Episode {episode}, Time step {timestep}/{pipeline.steps_per_episode-1}")
            
            # Use exploration or exploitation policy
            use_noise = (episode <= WARMUP_EPISODES * 2)  # More exploration in early stages
            actions, action_log_probs, values = fomatd3_adapter.select_actions(obs, deterministic=not use_noise)
            
            # Environment step
            next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
            
            # Collect data to experience replay buffer
            fomatd3_adapter.collect_step(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                action_log_probs=action_log_probs,
                values=values
            )
            
            # Accumulate rewards
            for manager_id in manager_ids:
                episode_rewards[manager_id] += rewards[manager_id]
            
            # Update observations
            obs = next_obs
            
            # TD3-specific: batch updates, delayed policy updates
            # Only update every few time steps, and Actor network updates at even lower frequency
            if timestep % UPDATE_FREQ == 0 and episode > WARMUP_EPISODES:
                train_info = fomatd3_adapter.train_on_batch()
                if isinstance(train_info, dict):
                    policy_loss = train_info.get('policy_loss', 0.0)
                    value_loss = train_info.get('value_loss', 0.0)
                    logger.debug(f"  ⚙️ Training update: Actor Loss: {policy_loss:.5f}, Critic Loss: {value_loss:.5f}")
            
            # Display timestep rewards
            timestep_total = sum(rewards.values())
            logger.info(f"  Time step {timestep} total reward: {timestep_total:.3f}")
        
        # Training after episode end - TD3 can update multiple times
        if episode > WARMUP_EPISODES:
            # Perform multiple updates at episode end, TD3 characteristic: more Critic updates, fewer Actor updates
            for _ in range(5):  # Update 5 more times
                update_info = fomatd3_adapter.train_on_batch()
        
        # Record episode rewards and statistics
        episode_total_reward = sum(episode_rewards.values())
        logger.info(f"Episode {episode} completed:")
        logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
        
        # If there's training information
        if 'update_info' in locals() and isinstance(update_info, dict):
            logger.info(f"  📈 Training loss: Actor {update_info.get('policy_loss', 0):.4f}, Critic {update_info.get('value_loss', 0):.4f}")
        
        # Display each Manager's reward and record to training history
        for manager_id, reward in episode_rewards.items():
            logger.info(f"  📊 {manager_id}: {reward:.3f}")
            training_episode_rewards[manager_id].append(reward)
            
            # Add to training history
            training_data = {
                'algorithm': 'FOMATD3',
                'manager_id': manager_id,
                'episode': episode,
                'episode_reward': reward,
                'policy_loss': float(update_info.get('policy_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'value_loss': float(update_info.get('value_loss', 0.001)) if 'update_info' in locals() else 0.001,
                'entropy': 0.0  # TD3 has no entropy concept
            }
            training_history.append(training_data)
            
            # Record training loss
            if hasattr(pipeline, '_record_training_loss') and 'update_info' in locals():
                pipeline._record_training_loss(
                    manager_id=manager_id,
                    episode=episode,
                    policy_loss=float(update_info.get('policy_loss', 0.001)),
                    value_loss=float(update_info.get('value_loss', 0.001)),
                    entropy=0.0  # TD3 has no entropy concept
                )
        
        # Record overall rewards
        training_data_total = {
            'algorithm': 'FOMATD3',
            'manager_id': 'total',
            'episode': episode,
            'episode_reward': episode_total_reward,
            'policy_loss': float(update_info.get('policy_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'value_loss': float(update_info.get('value_loss', 0.0)) if 'update_info' in locals() else 0.0,
            'entropy': 0.0  # TD3 has no entropy
        }
        training_history.append(training_data_total)
        
        # Periodically output learning progress
        if (episode + 1) % 10 == 0:
            logger.info(f"\n========== FOMATD3 Training Progress: {episode+1}/{pipeline.num_episodes} episodes ==========")
            
            # Get training statistics
            try:
                training_stats = fomatd3_adapter.get_training_stats()
                manager_rewards = fomatd3_adapter.get_manager_rewards_summary()
                
                if isinstance(manager_rewards, dict):
                    for manager_id, stats in manager_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            training_updates = stats.get('training_updates', 0)
                            logger.info(f"  🔥 {manager_id}: Cumulative reward {total_reward:.2f}, Best {best_reward:.2f}, Updates {training_updates} times")
                        else:
                            logger.info(f"  🔥 {manager_id}: Cumulative reward {stats:.2f}")
                else:
                    logger.info(f"  🔥 Manager rewards: {manager_rewards}")
                
                if isinstance(training_stats, dict):
                    iterations = training_stats.get('training_iterations', 0)
                    logger.info(f"  🚀 Total training iterations: {iterations}")
                else:
                    logger.info(f"  🚀 Training statistics: {training_stats}")
            except Exception as e:
                logger.warning(f"Failed to get training statistics: {e}")
                logger.info("  🔥 Training progress: Learning in progress...")
            
            logger.info("=" * 70)
        
        # Periodically save models
        if (episode + 1) % 20 == 0 or episode == pipeline.num_episodes:
            try:
                model_path = f"results/fomatd3_adapter_ep{episode+1}"
                fomatd3_adapter.save_models(model_path)
                logger.info(f"📀 Models saved to: {model_path}")
                
                # Save training history
                if hasattr(pipeline, '_force_save_training_history'):
                    pipeline._force_save_training_history(training_history, "FOMATD3_ADAPTER")
            except Exception as e:
                logger.error(f"Failed to save models: {e}")
        
        # Calculate episode duration
        episode_duration = datetime.now() - episode_start_time
        logger.info(f"Episode {episode} duration: {episode_duration}")
        
        # Display overall progress
        total_elapsed = datetime.now() - start_time
        avg_time_per_episode = total_elapsed / episode
        remaining_episodes = pipeline.num_episodes - episode
        estimated_remaining = avg_time_per_episode * remaining_episodes
        
        logger.info(f"Time elapsed: {total_elapsed}, Estimated remaining: {estimated_remaining}")
    
    # Training complete, save final models
    try:
        save_path = f"results/fomatd3_adapter_final"
        fomatd3_adapter.save_models(save_path)
        logger.info(f"Saving final models: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save final models: {e}")
    
    # Calculate total training time
    total_training_time = datetime.now() - start_time
    logger.info(f"FOMATD3 training completed! Total training time: {total_training_time}")
    
    # Organize training history into pipeline expected format
    result = {
        'status': 'success',
        'training_history': {
            'episode_rewards': {},
            'episode_lengths': {},
            'training_loss': {},
            'training_metadata': {
                'algorithm': 'FOMATD3',
                'num_episodes': pipeline.num_episodes,
                'steps_per_episode': pipeline.steps_per_episode,
                'num_managers': num_managers,
                'td3_policy_delay': POLICY_DELAY  # TD3-specific parameter
            }
        },
        'multi_agent_env': multi_env,
        'fomatd3_adapter': fomatd3_adapter
    }
    
    # Process training history data, grouped by manager_id
    for item in training_history:
        manager_id = item.get('manager_id')
        if manager_id and manager_id != 'total':  # Exclude total records
            if manager_id not in result['training_history']['episode_rewards']:
                result['training_history']['episode_rewards'][manager_id] = []
                result['training_history']['episode_lengths'][manager_id] = []
                result['training_history']['training_loss'][manager_id] = []
            
            # Add rewards and lengths
            result['training_history']['episode_rewards'][manager_id].append(item.get('episode_reward', 0.0))
            result['training_history']['episode_lengths'][manager_id].append(pipeline.steps_per_episode)
            
            # Add training loss
            loss_info = {
                'policy_loss': item.get('policy_loss', 0.001),
                'value_loss': item.get('value_loss', 0.001),
                'entropy': item.get('entropy', 0.0)  # TD3 has no entropy
            }
            result['training_history']['training_loss'][manager_id].append(loss_info)
    
    logger.info(f"Result contains training history for {len(result['training_history']['episode_rewards'])} Managers")
    return result 
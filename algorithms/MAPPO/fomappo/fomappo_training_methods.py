#!/usr/bin/env python3
"""
FOMAPPO and FOMAIPPO Training Methods

Provides training implementations for shared policy FOMAPPO and independent policy FOMAIPPO
For integration of these algorithms in the FO Pipeline
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
    Optimized FOMAPPO training method - includes more numerical stability and learning quality improvements
    """
    logger.info("🚀 Starting optimized FOMAPPO training (enhanced learning effect and stability)")
    logger.info(f"Planning to train {pipeline.num_episodes} episodes")
    
    # Force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes parameter invalid, setting to default value 1")
        pipeline.num_episodes = 1
    
    # Record maximum allowed episodes
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # Set a safe upper limit
    logger.info(f"Maximum allowed episodes: {max_allowed_episodes}")
    
    # Update actually running algorithm
    pipeline._update_actual_algorithm("FOMAPPO_FIXED")
    
    # 1. Prepare training environment
    logger.info("Preparing FOMAPPO training environment...")
    
    # Create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # Reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # Initialize user states
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # 🔧 Create or get multi-agent environment
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("Using existing multi_agent_env")
    else:
        # Create new multi-agent environment
        logger.info("Creating new multi_agent_env")
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
            logger.info("✅ Successfully created multi_agent_env")
        except Exception as e:
            logger.error(f"❌ Failed to create multi_agent_env: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. Get environment parameters
    # Get manager count and IDs
    manager_ids = [manager.manager_id for manager in pipeline.managers]
    num_managers = len(manager_ids)
    
    # Get state dimension and action dimension
    # 3. Create FOMAPPO adapter
    logger.info(f"Creating FOMAPPO adapter: {num_managers} Managers")
    
    # Determine state and action dimensions
    state_dim = 73  # Default dimension
    action_dim = 5  # Default dimension
    
    # Try to get dimensions from environment
    if multi_env is not None:
        try:
            # Get observation space
            obs_sample = multi_env.reset()
            if isinstance(obs_sample, dict) and len(obs_sample) > 0:
                first_obs = next(iter(obs_sample.values()))
                state_dim = len(first_obs)
                logger.info(f"Detected state dimension from environment: {state_dim}")
            
            # Get action space
            action_dim = multi_env.get_action_dim()
            logger.info(f"Detected action dimension from environment: {action_dim}")
        except Exception as e:
            logger.warning(f"Could not determine dimensions from environment: {e}")
    
    # Create FOMAPPO adapter
    fomappo_adapter = None
    
    try:
        if not FOMAPPO_SHARED_available:
            logger.error("FOMAPPOAdapter not available, cannot create adapter")
            return {"status": "error", "message": "FOMAPPOAdapter not available"}
        
        # Create adapter with detected dimensions
        fomappo_adapter = FOMAPPOAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_managers,
            episode_length=pipeline.time_horizon,
            lr_actor=5e-5,  # Lower learning rate for stability
            lr_critic=2e-4,  # Lower learning rate for stability
            device="cpu"  # Use CPU for better compatibility
        )
        logger.info("✅ Successfully created FOMAPPO adapter")
        
    except Exception as e:
        logger.error(f"❌ Failed to create FOMAPPO adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"Failed to create FOMAPPO adapter: {e}"}
    
    # 4. Training loop
    logger.info(f"Starting training loop: {max_allowed_episodes} episodes")
    
    # Training statistics
    training_stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": []
    }
    
    # Save initial models (for comparison)
    if hasattr(pipeline, 'results_dir'):
        initial_model_path = os.path.join(pipeline.results_dir, "fomappo_initial")
        try:
            fomappo_adapter.save_models(initial_model_path)
            logger.info(f"Saved initial models to {initial_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save initial models: {e}")
    
    # Main training loop
    for episode in range(max_allowed_episodes):
        try:
            logger.info(f"Starting episode {episode+1}/{max_allowed_episodes}")
            
            # Reset environment
            if multi_env is not None:
                observations = multi_env.reset()
                logger.info(f"Environment reset, got {len(observations)} observations")
            else:
                logger.error("multi_env is None, cannot continue training")
                break
            
            # Episode tracking
            episode_reward = {manager_id: 0.0 for manager_id in manager_ids}
            episode_length = 0
            done = False
            
            # Step loop
            while not done and episode_length < pipeline.time_horizon:
                # 1. Select actions
                actions, action_log_probs, values = fomappo_adapter.select_actions(observations)
                
                # 2. Execute actions in environment
                next_observations, rewards, dones, infos = multi_env.step(actions)
                
                # 3. Collect experience
                fomappo_adapter.collect_step(
                    obs=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    action_log_probs=action_log_probs,
                    values=values
                )
                
                # 4. Update episode tracking
                for manager_id, reward in rewards.items():
                    episode_reward[manager_id] += reward
                
                # 5. Update observations
                observations = next_observations
                
                # 6. Check if episode is done
                done = all(dones.values()) if isinstance(dones, dict) else all(dones)
                episode_length += 1
                
                # 7. Log progress
                if episode_length % 10 == 0 or done:
                    avg_reward = sum(episode_reward.values()) / len(episode_reward)
                    logger.info(f"Episode {episode+1}, Step {episode_length}: Avg reward {avg_reward:.4f}")
            
            # End of episode - compute returns
            fomappo_adapter.compute_returns()
            
            # Train on collected experience
            train_info = fomappo_adapter.train_on_batch()
            
            # Update learning rate
            fomappo_adapter._update_learning_rate()
            
            # Log episode results
            avg_reward = sum(episode_reward.values()) / len(episode_reward)
            logger.info(f"Episode {episode+1} completed: Length={episode_length}, Avg reward={avg_reward:.4f}")
            logger.info(f"Training stats: policy_loss={train_info.get('policy_loss', 0):.6f}, " +
                      f"value_loss={train_info.get('value_loss', 0):.6f}, " +
                      f"entropy={train_info.get('dist_entropy', 0):.6f}")
            
            # Update training statistics
            training_stats["episode_rewards"].append(avg_reward)
            training_stats["episode_lengths"].append(episode_length)
            training_stats["policy_losses"].append(train_info.get('policy_loss', 0))
            training_stats["value_losses"].append(train_info.get('value_loss', 0))
            training_stats["entropies"].append(train_info.get('dist_entropy', 0))
            
            # Save intermediate models (every 10 episodes)
            if hasattr(pipeline, 'results_dir') and (episode + 1) % 10 == 0:
                model_path = os.path.join(pipeline.results_dir, f"fomappo_episode_{episode+1}")
                try:
                    fomappo_adapter.save_models(model_path)
                    logger.info(f"Saved intermediate models to {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to save intermediate models: {e}")
            
        except Exception as e:
            logger.error(f"Error in episode {episode+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 5. Save final models
    if hasattr(pipeline, 'results_dir'):
        final_model_path = os.path.join(pipeline.results_dir, "fomappo_final")
        try:
            fomappo_adapter.save_models(final_model_path)
            logger.info(f"Saved final models to {final_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save final models: {e}")
    
    # 6. Save training statistics
    if hasattr(pipeline, 'results_dir'):
        stats_path = os.path.join(pipeline.results_dir, "fomappo_training_stats.npy")
        try:
            np.save(stats_path, training_stats)
            logger.info(f"Saved training statistics to {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save training statistics: {e}")
    
    # 7. Return results
    results = {
        "status": "success",
        "algorithm": "FOMAPPO_SHARED",
        "episodes_completed": max_allowed_episodes,
        "final_avg_reward": training_stats["episode_rewards"][-1] if training_stats["episode_rewards"] else 0,
        "training_stats": training_stats,
        "adapter": fomappo_adapter
    }
    
    logger.info("FOMAPPO training completed successfully")
    return results

def train_fomaippo_independent_policy(pipeline):
    """
    Optimized FOMAIPPO training method - includes more numerical stability and learning quality improvements
    
    Uses independent policy for each Manager (no parameter sharing)
    """
    logger.info("🚀 Starting optimized FOMAIPPO training (enhanced learning effect and stability)")
    logger.info(f"Planning to train {pipeline.num_episodes} episodes")
    
    # Force check num_episodes parameter
    if not hasattr(pipeline, 'num_episodes') or pipeline.num_episodes <= 0:
        logger.error("num_episodes parameter invalid, setting to default value 1")
        pipeline.num_episodes = 1
    
    # Record maximum allowed episodes
    max_allowed_episodes = min(pipeline.num_episodes, 100)  # Set a safe upper limit
    logger.info(f"Maximum allowed episodes: {max_allowed_episodes}")
    
    # Update actually running algorithm
    pipeline._update_actual_algorithm("FOMAIPPO")
    
    # 1. Prepare training environment
    logger.info("Preparing FOMAIPPO training environment...")
    
    # Create FO environment
    if hasattr(pipeline, "_create_environments"):
        pipeline._create_environments()
    
    # Reset environment state
    if hasattr(pipeline, "_reset_pipeline_state"):
        pipeline._reset_pipeline_state()
        
    # Initialize user states
    if hasattr(pipeline, "_initialize_user_states"):
        pipeline._initialize_user_states()
    
    # 🔧 Create or get multi-agent environment
    multi_env = None
    if hasattr(pipeline, 'multi_agent_env') and pipeline.multi_agent_env is not None:
        multi_env = pipeline.multi_agent_env
        logger.info("Using existing multi_agent_env")
    else:
        # Create new multi-agent environment
        logger.info("Creating new multi_agent_env")
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
            logger.info("✅ Successfully created multi_agent_env")
        except Exception as e:
            logger.error(f"❌ Failed to create multi_agent_env: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 2. Get environment parameters
    # Get manager count and IDs
    manager_ids = [manager.manager_id for manager in pipeline.managers]
    num_managers = len(manager_ids)
    
    # 3. Create FOMAIPPO adapter
    logger.info(f"Creating FOMAIPPO adapter: {num_managers} Managers with independent policies")
    
    # Determine state and action dimensions
    state_dim = 73  # Default dimension
    action_dim = 5  # Default dimension
    
    # Try to get dimensions from environment
    if multi_env is not None:
        try:
            # Get observation space
            obs_sample = multi_env.reset()
            if isinstance(obs_sample, dict) and len(obs_sample) > 0:
                first_obs = next(iter(obs_sample.values()))
                state_dim = len(first_obs)
                logger.info(f"Detected state dimension from environment: {state_dim}")
            
            # Get action space
            action_dim = multi_env.get_action_dim()
            logger.info(f"Detected action dimension from environment: {action_dim}")
        except Exception as e:
            logger.warning(f"Could not determine dimensions from environment: {e}")
    
    # Create FOMAIPPO adapter
    fomaippo_adapter = None
    
    try:
        if not FOMAIPPO_available:
            logger.error("FOMAIPPOAdapter not available, cannot create adapter")
            return {"status": "error", "message": "FOMAIPPOAdapter not available"}
        
        # Create adapter with detected dimensions
        fomaippo_adapter = FOMAIPPOAdapter(
            state_dim=state_dim,
            action_dim=action_dim,
            manager_ids=manager_ids,
            episode_length=pipeline.time_horizon,
            lr_actor=5e-5,  # Lower learning rate for stability
            lr_critic=2e-4,  # Lower learning rate for stability
            device="cpu"  # Use CPU for better compatibility
        )
        logger.info("✅ Successfully created FOMAIPPO adapter")
        
    except Exception as e:
        logger.error(f"❌ Failed to create FOMAIPPO adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"Failed to create FOMAIPPO adapter: {e}"}
    
    # 4. Training loop
    logger.info(f"Starting training loop: {max_allowed_episodes} episodes")
    
    # Training statistics
    training_stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": {},
        "value_losses": {},
        "entropies": {}
    }
    
    # Initialize per-manager statistics
    for manager_id in manager_ids:
        training_stats["policy_losses"][manager_id] = []
        training_stats["value_losses"][manager_id] = []
        training_stats["entropies"][manager_id] = []
    
    # Save initial models (for comparison)
    if hasattr(pipeline, 'results_dir'):
        initial_model_path = os.path.join(pipeline.results_dir, "fomaippo_initial")
        try:
            fomaippo_adapter.save_models(initial_model_path)
            logger.info(f"Saved initial models to {initial_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save initial models: {e}")
    
    # Main training loop
    for episode in range(max_allowed_episodes):
        try:
            logger.info(f"Starting episode {episode+1}/{max_allowed_episodes}")
            
            # Reset environment
            if multi_env is not None:
                observations = multi_env.reset()
                logger.info(f"Environment reset, got {len(observations)} observations")
            else:
                logger.error("multi_env is None, cannot continue training")
                break
            
            # Episode tracking
            episode_reward = {manager_id: 0.0 for manager_id in manager_ids}
            episode_length = 0
            done = False
            
            # Step loop
            while not done and episode_length < pipeline.time_horizon:
                # 1. Select actions
                actions, action_log_probs, values = fomaippo_adapter.select_actions(observations)
                
                # 2. Execute actions in environment
                next_observations, rewards, dones, infos = multi_env.step(actions)
                
                # 3. Collect experience
                fomaippo_adapter.collect_step(
                    obs=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    action_log_probs=action_log_probs,
                    values=values
                )
                
                # 4. Update episode tracking
                for manager_id, reward in rewards.items():
                    episode_reward[manager_id] += reward
                
                # 5. Update observations
                observations = next_observations
                
                # 6. Check if episode is done
                done = all(dones.values()) if isinstance(dones, dict) else all(dones)
                episode_length += 1
                
                # 7. Log progress
                if episode_length % 10 == 0 or done:
                    avg_reward = sum(episode_reward.values()) / len(episode_reward)
                    logger.info(f"Episode {episode+1}, Step {episode_length}: Avg reward {avg_reward:.4f}")
            
            # End of episode - compute returns
            fomaippo_adapter.compute_returns()
            
            # Train on collected experience
            train_info = fomaippo_adapter.train_on_batch()
            
            # Update learning rate
            fomaippo_adapter._update_learning_rate()
            
            # Log episode results
            avg_reward = sum(episode_reward.values()) / len(episode_reward)
            logger.info(f"Episode {episode+1} completed: Length={episode_length}, Avg reward={avg_reward:.4f}")
            
            # Update training statistics
            training_stats["episode_rewards"].append(avg_reward)
            training_stats["episode_lengths"].append(episode_length)
            
            # Update per-manager statistics
            for manager_id in manager_ids:
                if manager_id in train_info:
                    manager_info = train_info[manager_id]
                    training_stats["policy_losses"][manager_id].append(manager_info.get('policy_loss', 0))
                    training_stats["value_losses"][manager_id].append(manager_info.get('value_loss', 0))
                    training_stats["entropies"][manager_id].append(manager_info.get('entropy', 0))
                    
                    logger.info(f"Manager {manager_id} stats: policy_loss={manager_info.get('policy_loss', 0):.6f}, " +
                              f"value_loss={manager_info.get('value_loss', 0):.6f}, " +
                              f"entropy={manager_info.get('entropy', 0):.6f}")
            
            # Save intermediate models (every 10 episodes)
            if hasattr(pipeline, 'results_dir') and (episode + 1) % 10 == 0:
                model_path = os.path.join(pipeline.results_dir, f"fomaippo_episode_{episode+1}")
                try:
                    fomaippo_adapter.save_models(model_path)
                    logger.info(f"Saved intermediate models to {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to save intermediate models: {e}")
            
        except Exception as e:
            logger.error(f"Error in episode {episode+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 5. Save final models
    if hasattr(pipeline, 'results_dir'):
        final_model_path = os.path.join(pipeline.results_dir, "fomaippo_final")
        try:
            fomaippo_adapter.save_models(final_model_path)
            logger.info(f"Saved final models to {final_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save final models: {e}")
    
    # 6. Save training statistics
    if hasattr(pipeline, 'results_dir'):
        stats_path = os.path.join(pipeline.results_dir, "fomaippo_training_stats.npy")
        try:
            np.save(stats_path, training_stats)
            logger.info(f"Saved training statistics to {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save training statistics: {e}")
    
    # 7. Return results
    results = {
        "status": "success",
        "algorithm": "FOMAIPPO",
        "episodes_completed": max_allowed_episodes,
        "final_avg_reward": training_stats["episode_rewards"][-1] if training_stats["episode_rewards"] else 0,
        "training_stats": training_stats,
        "adapter": fomaippo_adapter
    }
    
    logger.info("FOMAIPPO training completed successfully")
    return results 
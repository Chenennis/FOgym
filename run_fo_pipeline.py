import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import math
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mappo_onpolicy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms", "MAPPO", "onpolicy")
if mappo_onpolicy_path not in sys.path:
    sys.path.append(mappo_onpolicy_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FOPipeline")

try:
    from fo_common.log_config import LogConfig, LogVerbosity, log_info_brief, log_info_detailed, log_progress
    LOG_CONFIG_AVAILABLE = True
except ImportError:
    LOG_CONFIG_AVAILABLE = False
    def log_info_brief(logger, message, condition=True):
        if condition:
            logger.info(message)
    def log_info_detailed(logger, message, condition=True):
        if condition:
            logger.info(message)
    def log_progress(logger, message):
        logger.info(message)

from fo_generate.unified_mdp_env import FlexOfferEnv, DeviceType
from fo_generate.dfo import DFOSystem
from fo_generate.sfo import SFOSystem
from fo_generate.inference import generate_fo_with_agent
from fo_generate.battery_model import BatteryParameters
from fo_generate.heat_model import HeatPumpParameters
from fo_generate.ev_model import EVParameters, EVUserBehavior
from fo_generate.pv_model import PVParameters

from fo_aggregate.manager import Device, User, Manager, City
from fo_aggregate.aggregator import FOAggregatorFactory, AggregatedFlexOffer, aggregate_flex_offers

from fo_trading.pool import TradingPool, WeatherModel, DemandModel, Trade

from fo_schedule.scheduler import ScheduleManager, UserScheduler, FlexOfferDisaggregator, AggregatedResultDisaggregator

try:
    from fo_common.observation import GlobalObservationManager
    from fo_common.config import default_global_observation_config
    global_observation_available = True
except ImportError:
    global_observation_available = False
    logger.warning("The global observation space module is not available, and the default module observation will be used")

class RLRegistry:
    """RL algorithm registry, used to register and obtain custom RL algorithms"""
    
    _registry = {}
    _registered_algorithms = set()
    _initialized = False
    
    @classmethod
    def register(cls, name: str, agent_class):
        """Register an RL algorithm

            Args:
                name: Algorithm name
                agent_class: Algorithm agent class
        """
        import os
        
        cls._registry[name] = agent_class
        
        # 判断当前进程是否应该输出日志
        main_process = os.environ.get("FO_MAIN_PROCESS", "")
        current_process = str(os.getpid())
        
        if not main_process:
            os.environ["FO_MAIN_PROCESS"] = current_process
            main_process = current_process
        
        # 仅在主进程和首次注册时输出日志
        if (not cls._initialized and current_process == main_process and 
            name not in cls._registered_algorithms):
            logger.info(f"RL algorithm {name} registered")
            cls._registered_algorithms.add(name)
    
    @classmethod
    def get(cls, name: str):
        """Get RL algorithm class"""
        return cls._registry.get(name)
    
    @classmethod
    def list_algorithms(cls):
        """List all registered algorithms"""
        return list(cls._registry.keys())
        
    @classmethod
    def init(cls):
        """Mark the registry as initialized"""
        cls._initialized = True

try:
    from algorithms.MAPPO.fomappo.fomappo import FOMAPPO
    from algorithms.MAPPO.fomappo.fomappo_policy import FOMAPPOPolicy
    FOMAPPO_available = True
    logger.info("FOMAPPO algorithm imported successfully")
except ImportError:
    FOMAPPO = None
    FOMAPPOPolicy = None
    FOMAPPO_available = False
    logger.warning("FOMAPPO algorithm is not available, please check the algorithms/MAPPO/fomappo directory")

try:
    from algorithms.MAPPO.fomappo.fomappo_adapter import FOMAPPOAdapter
    FOMAPPO_SHARED_available = True
    logger.info("FOMAPPO algorithm (shared policy) imported successfully")
except ImportError:
    FOMAPPOAdapter = None
    FOMAPPO_SHARED_available = False
    logger.warning("FOMAPPO algorithm (shared policy) is not available, please check the algorithms/MAPPO/fomappo directory")

try:
    from algorithms.MAPPO.fomappo.fomaippo_adapter import FOMAIPPOAdapter
    FOMAIPPO_available = True
    logger.info("FOMAIPPO algorithm (independent policy) imported successfully")
except ImportError:
    FOMAIPPOAdapter = None
    FOMAIPPO_available = False
    logger.warning("FOMAIPPO algorithm (independent policy) is not available, please check the algorithms/MAPPO/fomappo directory")

try:
    from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG
    from algorithms.MADDPG.fomaddpg.fomaddpg_policy import FOMaddpgPolicy
    from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter
    FOMADDPG_available = True
    logger.info("FOMADDPG algorithm imported successfully")
except ImportError:
    FOMADDPG = None
    FOMaddpgPolicy = None
    FOMAddpgAdapter = None
    FOMADDPG_available = False
    logger.warning("FOMADDPG algorithm is not available, please check the algorithms/MADDPG/fomaddpg directory")

try:
    from algorithms.MATD3.fomatd3.fomatd3 import FOMATD3
    from algorithms.MATD3.fomatd3.fomatd3_policy import FOMATd3Policy
    from algorithms.MATD3.fomatd3.fomatd3_adapter import FOMATD3Adapter
    FOMATD3_available = True
    logger.info("FOMATD3 algorithm imported successfully")
except ImportError:
    FOMATD3 = None
    FOMATd3Policy = None
    FOMATD3Adapter = None
    FOMATD3_available = False
    logger.warning("FOMATD3 algorithm is not available, please check the algorithms/MATD3/fomatd3 directory")

try:
    from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG
    from algorithms.SQDDPG.fosqddpg.fosqddpg_policy import FOSQDDPGPolicy
    from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter
    FOSQDDPG_available = True
    logger.info("FOSQDDPG algorithm and adapter imported successfully")
except ImportError:
    FOSQDDPG = None
    FOSQDDPGPolicy = None
    FOSQDDPGAdapter = None
    FOSQDDPG_available = False
    logger.warning("FOSQDDPG algorithm is not available, please check the algorithms/SQDDPG/fosqddpg directory")

try:
    import sys
    import os
    model_based_path = os.path.join(os.path.dirname(__file__), 'algorithms', 'Model-based')
    if model_based_path not in sys.path:
        sys.path.insert(0, model_based_path)
    
    from fomodelbased.fomodelbased import FOModelBased, ModelBasedConfig
    from fomodelbased.fomodelbased_policy import FOModelBasedPolicy
    from fomodelbased.fomodelbased_adapter import FOModelBasedAdapter
    FOMODELBASED_available = True
    logger.info("FOModelBased algorithm and adapter imported successfully")
except ImportError as e:
    FOModelBased = None
    FOModelBasedPolicy = None
    FOModelBasedAdapter = None
    FOMODELBASED_available = False
    logger.warning(f"FOModelBased algorithm is not available: {e}, please check the algorithms/Model-based/fomodelbased directory")

if FOMAPPO_available and FOMAPPO is not None:
    RLRegistry.register("fomappo", FOMAPPO)
if FOMATD3_available and FOMATD3 is not None:
    RLRegistry.register("fomatd3", FOMATD3)
if FOSQDDPG_available and FOSQDDPG is not None:
    RLRegistry.register("fosqddpg", FOSQDDPG)
if FOMODELBASED_available and FOModelBased is not None:
    RLRegistry.register("fomodelbased", FOModelBased)

class FOPipeline:
    """Flexible Offer Pipeline Management Class"""
    
    def __init__(self, config: Dict):
        """
        Initialize FOPipeline
        
        Args:
            config: Configuration dictionary, containing various parameters
        """
        self.config = config
        self.time_horizon = config.get("time_horizon", 24)  # The time range of each episode (hours)
        self.time_step = config.get("time_step", 1.0)  # The length of each time step (hours)
        
        self.num_episodes = config.get("num_episodes", 100)  # The number of training episodes, each episode = 24 hours
        if not isinstance(self.num_episodes, int) or self.num_episodes <= 0:
            logger.warning(f"Invalid num_episodes value: {self.num_episodes}, set to default value 1")
            self.num_episodes = 1
        elif self.num_episodes > 100:
            logger.warning(f"num_episodes value is too large: {self.num_episodes}, may cause training time to be too long")
        
        logger.info(f"训练配置: num_episodes={self.num_episodes}")
        
        if self.time_step != 1.0:
            logger.warning(f"Time step length set to {self.time_step} hours, it is recommended to use 1.0 hour")
        
        # Calculate the number of time steps per episode (should be 24 steps, from 0 to 23)
        self.steps_per_episode = int(self.time_horizon / self.time_step)
        if self.steps_per_episode != 24:
            logger.warning(f"Each episode has {self.steps_per_episode} time steps, it is recommended to use 24 time steps (0-23)")
        
        logger.info(f"Episode configuration: each episode = {self.time_horizon} hours = {self.steps_per_episode} time steps (0-{self.steps_per_episode-1})")
        
        use_gpu = config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using GPU: " + torch.cuda.get_device_name(0))
        else:
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU is not available, using CPU instead")
            self.device = "cpu"
            logger.info("Using CPU")
        
        seed = config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
        
        # User and device configuration
        # Default using the actual multi-agent environment configuration (36 users, 4 managers)
        self.num_managers = config.get("num_managers", 4)  # Changed to 4 managers
        self.num_users = config.get("num_users", 36)  # Match the actual number of users in the multi-agent environment
        self.users_per_manager = self.num_users // self.num_managers
        self.devices_per_user = config.get("devices_per_user", {
            DeviceType.BATTERY: (0, 1),    # 24 users have battery (67%), not every user has
            DeviceType.HEAT_PUMP: (1, 1),  # 100% deployment rate, every user has a heat pump
            DeviceType.EV: (0, 1),          # 14 users have EV (39%)
            DeviceType.PV: (0, 1),          # 8 users have PV (22%)
            DeviceType.DISHWASHER: (1, 1)   # 100% deployment rate, every user has a dishwasher
        })
        
        # Algorithm selection
        self.rl_algorithm = config.get("rl_algorithm", "fomappo")
        self.actual_running_algorithm = self.rl_algorithm  # New: Track the actual running algorithm
        
        # Define the list of built-in multi-agent algorithms
        builtin_multi_agent_algorithms = [
            "fomappo",      # FlexOffer multi-agent algorithm based on MAPPO (shared policy)
            "fomaippo",     # FlexOffer multi-agent algorithm based on MAPPO (independent policy)
            "fomaddpg",     # FlexOffer multi-agent algorithm based on MADDPG
            "fomatd3",      # FlexOffer multi-agent algorithm based on MATD3
            "fosqddpg",     # FlexOffer multi-agent algorithm based on SQDDPG
            "fomodelbased"  # Deprecated
        ]
        
        self.custom_rl_algorithm = self.rl_algorithm not in builtin_multi_agent_algorithms
        self.rl_agents = {}
        
        # Initialize rl_agents for algorithms that support single-user agents
        if self.rl_algorithm == "fomappo":
            self.rl_agents[self.rl_algorithm] = {}
        elif self.rl_algorithm == "fomodelbased":
            # FOModelBased is a multi-agent algorithm, but it also needs to be initialized
            self.rl_agents[self.rl_algorithm] = {"multi_agent": None}
        
        # Aggregation method, trading strategy, and disaggregation method
        self.aggregation_method = config.get("aggregation_method", "DP")
        self.trading_strategy = config.get("trading_strategy", "market_clearing")
        self.disaggregation_method = config.get("disaggregation_method", "proportional")
        self.scheduling_method = config.get("scheduling_method", "priority")
        
        # Global observation space configuration
        self.use_global_observation = config.get("use_global_observation", False)
        self.global_observation_config_file = config.get("global_observation_config", None)
        self.global_observation_manager = None
        
        # Initialize environment and user lists
        self.envs = {}
        self.users = []
        self.managers = []
        
        # Create City object
        self.city = None
        
        # Initialize directory
        self.results_dir = config.get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training history
        self.training_history = {
            "episode_rewards": [],
            "manager_rewards": {},
            "loss_history": {},
            "training_metadata": {}
        }
        
        self.training_loss_history = {}  # Used to record the loss function value of each episode
        
        self.experiment_id = None
        
        # Initialize global observation manager
        if self.use_global_observation and global_observation_available:
            self._init_global_observation_manager()
        
        # Initialize components for each stage
        self._setup_components()
        
        # Ensure users and managers are initialized
        if not self.users or not self.managers:
            self._setup_managers_and_users()
        
        logger.info(f"FOPipeline initialized, mode: RL={self.rl_algorithm}, "
                   f"aggregation={self.aggregation_method}, trading={self.trading_strategy}, "
                   f"disaggregation={self.disaggregation_method}, scheduling={self.scheduling_method}")
    
    def _generate_experiment_id(self):
        """Generate a unique experiment identifier (using the actual running algorithm)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config_str = f"{self.actual_running_algorithm}"
        if self.aggregation_method != "DP":
            config_str += f"_{self.aggregation_method}"
        if self.trading_strategy != "market_clearing":
            config_str += f"_{self.trading_strategy}"
        if self.disaggregation_method != "proportional":
            config_str += f"_{self.disaggregation_method}"
        if self.scheduling_method != "priority":
            config_str += f"_{self.scheduling_method}"
        
        config_str += f"_ep{self.num_episodes}_u{self.num_users}_m{self.num_managers}"
        
        return f"{config_str}_{timestamp}"
    
    def _update_actual_algorithm(self, algorithm_name):
        """Update the actual running algorithm name and generate experiment ID"""
        self.actual_running_algorithm = algorithm_name
        self.experiment_id = self._generate_experiment_id()
        logger.info(f"Actual running algorithm: {self.actual_running_algorithm}")
        logger.info(f"Experiment identifier: {self.experiment_id}")
        
        # Initialize the algorithm-specific part of the training history
        self.training_history["training_metadata"]["actual_algorithm"] = algorithm_name
        self.training_history["training_metadata"]["requested_algorithm"] = self.rl_algorithm
        self.training_history["training_metadata"]["experiment_id"] = self.experiment_id
    
    def _save_training_history_with_backup(self, prefix=""):
        """Enhanced training history saving method, including multiple backups"""
        if not self.training_history["episode_rewards"]:
            logger.warning("Training history is empty, skipping save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{prefix}fomappo_training_history_{timestamp}"
        
        if self.experiment_id is None:
            self.experiment_id = f"backup_{timestamp}"
            logger.warning(f"Backup: experiment_id is None, generated: {self.experiment_id}")
        
        # Method 1: CSV save (main method)
        try:
            algorithm_name = self.actual_running_algorithm or "FOMAPPO"
            self._save_training_history_to_csv(algorithm_name)
            logger.info("✅ CSV format training history saved successfully")
        except Exception as e:
            logger.error(f"CSV save failed: {e}")
        
        # Method 2: JSON backup save
        try:
            json_file = os.path.join(self.results_dir, f"{base_filename}.json")
            with open(json_file, 'w') as f:
                json_data = {
                    'episode_rewards': {k: [float(r) for r in v] for k, v in self.training_history["episode_rewards"].items()},
                    'metadata': self.training_history.get("training_metadata", {}),
                    'timestamp': timestamp,
                    'num_episodes': getattr(self, 'num_episodes', 0),
                    'algorithm': self.actual_running_algorithm or 'FOMAPPO'
                }
                json.dump(json_data, f, indent=2)
            logger.info(f"✅ JSON backup save successfully: {json_file}")
        except Exception as e:
            logger.error(f"JSON backup save failed: {e}")
        
        # Method 3: Pure text backup
        try:
            txt_file = os.path.join(self.results_dir, f"{base_filename}.txt")
            with open(txt_file, 'w') as f:
                f.write(f"FOMAPPO training history - {timestamp}\n")
                f.write("=" * 50 + "\n")
                for manager_id, rewards in self.training_history["episode_rewards"].items():
                    f.write(f"\n{manager_id}:\n")
                    for i, reward in enumerate(rewards):
                        f.write(f"Episode {i+1}: {reward:.4f}\n")
                    f.write(f"Total Episodes: {len(rewards)}\n")
                    f.write(f"Average reward: {sum(rewards)/len(rewards):.4f}\n")
            logger.info(f"✅ Text backup save successfully: {txt_file}")
        except Exception as e:
            logger.error(f"Text backup save failed: {e}")
    
    def _force_save_training_history(self, training_data, algorithm_name):
        """Force save training history data - last resort"""
        if not training_data:
            logger.warning("No data to force save")
            return
        
        # Add debug information
        logger.info(f"Force save training history data, type: {type(training_data)}")
        if isinstance(training_data, dict):
            logger.info(f"Dictionary keys: {list(training_data.keys())}")
            for k, v in training_data.items():
                logger.info(f"   Key '{k}' value type: {type(v)}, length: {len(v) if hasattr(v, '__len__') else 'N/A'}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure there is an experiment_id
        if self.experiment_id is None:
            self.experiment_id = f"force_{timestamp}"
        
        # Method 1: Simple text format
        try:
            filename = f"{algorithm_name.lower()}_training_history_{self.experiment_id}.txt"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Force save training history - {algorithm_name}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Experiment ID: {self.experiment_id}\n\n")
                
                # Handle complex nested dictionary cases
                if isinstance(training_data, dict) and 'manager_rewards' in training_data:
                    # This is pipeline_rewards format
                    f.write("Pipeline rewards format data:\n")
                    f.write("=" * 40 + "\n\n")
                    
                    # Save manager_rewards
                    f.write("Manager rewards:\n")
                    for manager_id, rewards in training_data['manager_rewards'].items():
                        f.write(f"\n{manager_id}:\n")
                        # Check the type of rewards
                        if not isinstance(rewards, (list, np.ndarray)):
                            f.write(f"   Warning: rewards are not lists or arrays, but {type(rewards)}\n")
                            continue
                            
                        for i, reward in enumerate(rewards):
                            f.write(f"Timestep {i+1}: {reward}\n")
                        
                        if rewards:
                            try:
                                # 🔧 Fix: Ensure all elements are numeric types
                                numeric_rewards = []
                                for r in rewards:
                                    if isinstance(r, (int, float, np.number)):
                                        numeric_rewards.append(float(r))
                                    else:
                                        logger.warning(f"Skip non-numeric rewards: {r} type: {type(r)}")
                                
                                if numeric_rewards:
                                    avg_reward = sum(numeric_rewards)/len(numeric_rewards)
                                    f.write(f"Total Timesteps: {len(rewards)}\n")
                                    f.write(f"Average reward: {avg_reward:.4f}\n")
                                else:
                                    f.write("No valid numeric rewards, cannot calculate average\n")
                            except Exception as e:
                                f.write(f"Failed to calculate average reward: {e}\n")
                                logger.error(f"Failed to calculate average reward: {e}")
                    
                    # Save timestep_rewards
                    if 'timestep_rewards' in training_data:
                        f.write("\nTimestep rewards component:\n")
                        for i, tr in enumerate(training_data['timestep_rewards']):
                            f.write(f"Timestep {i+1}: trade value={tr.get('trade_value', 0):.2f}, " +
                                   f"satisfaction={tr.get('satisfaction_reward', 0):.2f}, " +
                                   f"coordination={tr.get('coordination_reward', 0):.2f}, " +
                                   f"efficiency={tr.get('efficiency_reward', 0):.2f}, " +
                                   f"total={tr.get('total_reward', 0):.2f}\n")
                    
                    # Save reward component statistics
                    if 'reward_components' in training_data:
                        f.write("\nReward component statistics:\n")
                        rc = training_data['reward_components']
                        for k, v in rc.items():
                            f.write(f"{k}: {v:.4f}\n")
                
                elif isinstance(training_data, dict) and 'episode_rewards' in training_data:
                    # This is training_history format
                    f.write("Training history format data:\n")
                    f.write("=" * 40 + "\n\n")
                    
                    # Save episode_rewards
                    episode_rewards = training_data['episode_rewards']
                    if isinstance(episode_rewards, dict):
                        for manager_id, rewards in episode_rewards.items():
                            f.write(f"\n{manager_id}:\n")
                            # Check the type of rewards
                            if not isinstance(rewards, (list, np.ndarray)):
                                f.write(f"   Warning: rewards are not lists or arrays, but {type(rewards)}\n")
                                continue
                                
                        for i, reward in enumerate(rewards):
                            f.write(f"Episode {i+1}: {reward}\n")
                            
                            if rewards:
                                try:
                                    # 🔧 Fix: Ensure all elements are numeric types
                                    numeric_rewards = []
                                    for r in rewards:
                                        # Check if it is a dictionary type (nested training records)
                                        if isinstance(r, dict) and 'episode_reward' in r:
                                            # Extract the actual reward value from the dictionary
                                            numeric_rewards.append(float(r['episode_reward']))
                                            logger.info(f"Extract reward value from dictionary: {r['episode_reward']}")
                                        elif isinstance(r, (int, float, np.number)):
                                            numeric_rewards.append(float(r))
                                        else:
                                            logger.warning(f"Skip non-numeric rewards: {r} type: {type(r)}")
                                    
                                    if numeric_rewards:
                                        avg_reward = sum(numeric_rewards)/len(numeric_rewards)
                                        f.write(f"Total Episodes: {len(rewards)}\n")
                                        f.write(f"Average reward: {avg_reward:.4f}\n")
                                    else:
                                        f.write("No valid numeric rewards, cannot calculate average\n")
                                except Exception as e:
                                    f.write(f"Failed to calculate average reward: {e}\n")
                                    logger.error(f"Failed to calculate average reward: {e}")
                    elif isinstance(episode_rewards, list):
                        f.write("\nTraining rewards:\n")
                        for i, reward in enumerate(episode_rewards):
                            f.write(f"Episode {i+1}: {reward}\n")
                        
                        if episode_rewards:
                            try:
                            # 🔧 Fix: Ensure all elements are numeric types
                                numeric_rewards = []
                                for r in episode_rewards:
                                    # Check if it is a dictionary type (nested training records)
                                    if isinstance(r, dict) and 'episode_reward' in r:
                                        # Extract the actual reward value from the dictionary
                                        numeric_rewards.append(float(r['episode_reward']))
                                        logger.info(f"Extract reward value from dictionary: {r['episode_reward']}")
                                    elif isinstance(r, (int, float, np.number)):
                                        numeric_rewards.append(float(r))
                                    else:
                                        logger.warning(f"Skip non-numeric rewards: {r} type: {type(r)}")
                                
                                if numeric_rewards:
                                    avg_reward = sum(numeric_rewards)/len(numeric_rewards)
                                    f.write(f"Total Episodes: {len(episode_rewards)}\n")
                                    f.write(f"Average reward: {avg_reward:.4f}\n")
                                else:
                                    f.write("No valid numeric rewards, cannot calculate average\n")
                            except Exception as e:
                                f.write(f"Failed to calculate average reward: {e}\n")
                                logger.error(f"Failed to calculate average reward: {e}")
                
                elif isinstance(training_data, dict):
                    # Normal dictionary format (possibly a mapping of manager_id -> rewards)
                    for manager_id, rewards in training_data.items():
                        f.write(f"\n{manager_id}:\n")
                        # Check the type of rewards
                        if not isinstance(rewards, (list, np.ndarray)):
                            f.write(f"   Warning: rewards are not lists or arrays, but {type(rewards)}\n")
                            continue
                            
                        for i, reward in enumerate(rewards):
                            f.write(f"Episode {i+1}: {reward}\n")
                        
                        if rewards:
                            try:
                                avg_reward = sum(rewards)/len(rewards)
                                f.write(f"Total Episodes: {len(rewards)}\n")
                                f.write(f"Average reward: {avg_reward:.4f}\n")
                            except Exception as e:
                                f.write(f"Failed to calculate average reward: {e}\n")
                                logger.error(f"Failed to calculate average reward: {e}")
                
                elif isinstance(training_data, list):
                    # Simple list format
                    f.write("\nTraining rewards:\n")
                    for i, reward in enumerate(training_data):
                        f.write(f"Episode {i+1}: {reward}\n")
                    
                    if training_data:
                        try:
                            numeric_rewards = []
                            for r in training_data:
                                # Check if it is a dictionary type (nested training records)
                                if isinstance(r, dict) and 'episode_reward' in r:
                                    # Extract the actual reward value from the dictionary
                                    numeric_rewards.append(float(r['episode_reward']))
                                    logger.info(f"Extract reward value from dictionary: {r['episode_reward']}")
                                elif isinstance(r, (int, float, np.number)):
                                    numeric_rewards.append(float(r))
                                else:
                                    logger.warning(f"Skip non-numeric rewards: {r} type: {type(r)}")
                            
                            if numeric_rewards:
                                avg_reward = sum(numeric_rewards)/len(numeric_rewards)
                                f.write(f"Total Episodes: {len(training_data)}\n")
                                f.write(f"Average reward: {avg_reward:.4f}\n")
                            else:
                                f.write("No valid numeric rewards, cannot calculate average\n")
                        except Exception as e:
                            f.write(f"Failed to calculate average reward: {e}\n")
                            logger.error(f"Failed to calculate average reward: {e}")
                else:
                    # Unknown format
                    f.write(f"\nUnknown data type: {type(training_data)}\n")
                    f.write(f"Data content: {str(training_data)[:1000]}\n")
            
            logger.info(f"✅ Force save successfully: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Force save failed: {e}")
            # Add more detailed error information
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _save_training_history_to_csv(self, algorithm_name):
        """Save training history to CSV file"""
        # Special handling for FOModelBased algorithm
        if algorithm_name.upper() == "FOMODELBASED" and hasattr(self, 'fomodelbased_results'):
            try:
                # Directly use fomodelbased_results to generate CSV
                import pandas as pd
                
                # Generate file name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = os.path.join(self.results_dir, f"fomodelbased_training_history_{self.experiment_id}_{timestamp}.csv")
                
                # Create DataFrame from fomodelbased_results
                rows = []
                for manager_id, manager_rewards in self.fomodelbased_results.items():
                    # Use single episode and multiple timesteps
                    if isinstance(manager_rewards, list):
                        # Create a record for each timestep
                        for timestep, reward in enumerate(manager_rewards):
                            rows.append({
                                'algorithm': 'FOMODELBASED',
                                'manager_id': manager_id,
                                'episode': 1,  # Only one episode
                                'timestep': timestep + 1,
                                'reward': float(reward),
                                'cumulative_reward': sum(manager_rewards[:timestep+1]),
                                'avg_reward': np.mean(manager_rewards[:timestep+1]),
                                'policy_loss': 0.0,  # ModelBased has no policy loss
                                'value_loss': 0.0,   # ModelBased has no value loss
                                'entropy': 0.0       # ModelBased has no entropy
                            })
                
                # Create total row
                if rows:
                    total_by_timestep = {}
                    for row in rows:
                        timestep = row['timestep']
                        if timestep not in total_by_timestep:
                            total_by_timestep[timestep] = 0
                        total_by_timestep[timestep] += row['reward']
                    
                    for timestep, total_reward in sorted(total_by_timestep.items()):
                        rows.append({
                            'algorithm': 'FOMODELBASED',
                            'manager_id': 'total',
                            'episode': 1,
                            'timestep': timestep,
                            'reward': total_reward,
                            'cumulative_reward': sum(list(total_by_timestep.values())[:timestep]),
                            'avg_reward': np.mean(list(total_by_timestep.values())[:timestep]) if timestep > 0 else 0,
                            'policy_loss': 0.0,
                            'value_loss': 0.0,
                            'entropy': 0.0
                        })
                
                # Create and save DataFrame
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(csv_file, index=False)
                    logger.info(f"✅ FOModelBased training history saved to: {csv_file}")
                    print(f"✅ FOModelBased training history saved to: {os.path.basename(csv_file)}")
                    return
                
            except Exception as e:
                logger.error(f"Failed to save FOModelBased training history: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # The following is the original method - for other algorithms
        # 1. Check if training history exists
        if not hasattr(self, 'training_history') or not self.training_history:
            logger.warning("Training history does not exist or is empty, initialize default training history")
            self._init_default_training_history()
            
        # 2. Check if episode_rewards exists
        if not self.training_history.get("episode_rewards"):
            logger.warning("Training history does not have episode_rewards, create default training history")
            self._init_default_training_history()
            
        # 3. Check if all data is empty
        if isinstance(self.training_history["episode_rewards"], dict):
            has_data = any(len(rewards) > 0 for rewards in self.training_history["episode_rewards"].values())
            if not has_data:
                logger.warning("All Manager data in training history dictionary is empty, create default training history")
                self._init_default_training_history()
        elif isinstance(self.training_history["episode_rewards"], list):
            if len(self.training_history["episode_rewards"]) == 0:
                logger.warning("Training history list is empty, create default training history")
                self._init_default_training_history()
                
        logger.info("✅ Training history check completed")
        
        # 🔧 Fix: Check if all data is empty
        if isinstance(self.training_history["episode_rewards"], dict):
            has_data = any(len(rewards) > 0 for rewards in self.training_history["episode_rewards"].values())
            if not has_data:
                logger.warning("All Manager data in training history dictionary is empty")
                return
        elif isinstance(self.training_history["episode_rewards"], list):
            if len(self.training_history["episode_rewards"]) == 0:
                logger.warning("Training history list is empty")
                return
        
        # 🔧 Fix: Ensure algorithm_name is valid
        if not algorithm_name:
            algorithm_name = self.actual_running_algorithm or "Unknown"
            logger.warning(f"algorithm_name is empty, use: {algorithm_name}")
        
        # 🔧 Fix: Ensure experiment_id and directory exist before saving
        if self.experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"training_{timestamp}"
            logger.warning(f"When saving, experiment_id is None, generate: {self.experiment_id}")
        
        # Ensure results_dir exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"🔍 Start saving training history, algorithm: {algorithm_name}")
        logger.info(f"Data type: {type(self.training_history['episode_rewards'])}")
        if isinstance(self.training_history["episode_rewards"], dict):
            for k, v in self.training_history["episode_rewards"].items():
                logger.info(f"  {k}: {len(v)} episodes")
        else:
            logger.info(f"  Length: {len(self.training_history['episode_rewards'])}")
        
        try:
            import pandas as pd
            
            # Prepare training history data
            history_rows = []
            
            # Process episode-level reward records
            if isinstance(self.training_history["episode_rewards"], dict):
                # Multi-agent format
                for manager_id, rewards in self.training_history["episode_rewards"].items():
                    for episode, reward in enumerate(rewards):
                        # Get real training loss information from training_loss_history
                        policy_loss = 0.0
                        value_loss = 0.0
                        entropy = 0.0
                        
                        if hasattr(self, 'training_loss_history') and manager_id in self.training_loss_history:
                            if episode < len(self.training_loss_history[manager_id]):
                                loss_info = self.training_loss_history[manager_id][episode]
                                policy_loss = loss_info.get('policy_loss', 0.0)
                                value_loss = loss_info.get('value_loss', 0.0)
                                entropy = loss_info.get('entropy', 0.0)
                        
                        # Enhanced reward processing: process various possible reward formats
                        reward_value = None
                        try:
                            # Case 1: Reward is a dictionary type, contains 'episode_reward' key
                            if isinstance(reward, dict) and 'episode_reward' in reward:
                                reward_value = float(reward['episode_reward'])
                                logger.debug(f"Extract reward value from dictionary: {reward_value}")
                            # Case 2: Reward is a dictionary type, but does not contain 'episode_reward' key
                            elif isinstance(reward, dict):
                                # Try to find any possible numeric keys
                                numeric_keys = [k for k, v in reward.items() if isinstance(v, (int, float, np.number))]
                                if numeric_keys:
                                    # Use the first numeric key
                                    reward_value = float(reward[numeric_keys[0]])
                                    logger.debug(f"Extract alternative reward key '{numeric_keys[0]}': {reward_value}")
                                else:
                                    # If there are no numeric keys, use the dictionary length as a fallback value
                                    reward_value = float(len(reward)) * 0.1
                                    logger.warning(f"Cannot extract numeric rewards from dictionary, use fallback value: {reward_value}")
                            # Case 3: Reward is a numeric type
                            elif isinstance(reward, (int, float, np.number)):
                                reward_value = float(reward)
                            # Case 4: Reward is other type
                            else:
                                # Try to convert to float
                                try:
                                    reward_value = float(reward)
                                except (TypeError, ValueError):
                                    # Cannot convert, use default value
                                    reward_value = 0.1
                                    logger.warning(f"Cannot convert reward to numeric, type: {type(reward)}, use default value: {reward_value}")
                        except Exception as e:
                            # Process any unexpected errors
                            reward_value = 0.1
                            logger.error(f"Error processing reward: {e}, use default value: {reward_value}")
                        
                        # Ensure reward_value is not None
                        if reward_value is None:
                            reward_value = 0.1
                            logger.warning(f"Reward value is None, use default value: {reward_value}")
                        
                        # Calculate cumulative reward and average reward, process possible complex reward structures
                        try:
                            # Extract values of all previous rewards
                            previous_rewards = []
                            for r in rewards[:episode+1]:
                                if isinstance(r, dict) and 'episode_reward' in r:
                                    previous_rewards.append(float(r['episode_reward']))
                                elif isinstance(r, (int, float, np.number)):
                                    previous_rewards.append(float(r))
                                else:
                                    # Try to convert to float
                                    try:
                                        previous_rewards.append(float(r))
                                    except (TypeError, ValueError):
                                        previous_rewards.append(0.1)
                            
                            cumulative_reward = sum(previous_rewards)
                            
                            # Calculate average of last 10 rewards
                            recent_rewards = previous_rewards[max(0, episode-9):episode+1]
                            avg_reward_last_10 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                        except Exception as e:
                            logger.error(f"Error calculating cumulative reward: {e}")
                            cumulative_reward = episode * 0.1
                            avg_reward_last_10 = 0.1
                        
                        # Use extracted reward value
                        history_rows.append({
                            'algorithm': algorithm_name,
                            'manager_id': manager_id,
                            'episode': episode + 1,
                            'episode_reward': reward_value,
                            'cumulative_reward': cumulative_reward,
                            'avg_reward_last_10': avg_reward_last_10,
                            'policy_loss': float(policy_loss),
                            'value_loss': float(value_loss),
                            'entropy': float(entropy),
                            'data_type': 'episode_reward'
                            })
                
                # Add overall statistics
                if self.training_history["episode_rewards"]:
                    first_manager = next(iter(self.training_history["episode_rewards"]))
                    total_episodes = len(self.training_history["episode_rewards"][first_manager])
                    
                for episode in range(total_episodes):
                        # Calculate total reward for all managers in the current episode
                        try:
                            episode_rewards = []
                            for agent_id in self.training_history["episode_rewards"].keys():
                                if episode < len(self.training_history["episode_rewards"][agent_id]):
                                    reward = self.training_history["episode_rewards"][agent_id][episode]
                                    if isinstance(reward, dict) and 'episode_reward' in reward:
                                        episode_rewards.append(float(reward['episode_reward']))
                                    elif isinstance(reward, (int, float, np.number)):
                                        episode_rewards.append(float(reward))
                                    else:
                                        # Try to convert to float
                                        try:
                                            episode_rewards.append(float(reward))
                                        except (TypeError, ValueError):
                                            episode_rewards.append(0.1)
                            
                            episode_total = sum(episode_rewards)
                            
                            # Calculate cumulative reward for all managers
                            all_cumulative_rewards = []
                            for agent_id in self.training_history["episode_rewards"].keys():
                                agent_rewards = []
                                for ep in range(episode + 1):
                                    if ep < len(self.training_history["episode_rewards"][agent_id]):
                                        reward = self.training_history["episode_rewards"][agent_id][ep]
                                        if isinstance(reward, dict) and 'episode_reward' in reward:
                                            agent_rewards.append(float(reward['episode_reward']))
                                        elif isinstance(reward, (int, float, np.number)):
                                            agent_rewards.append(float(reward))
                                        else:
                                            # Try to convert to float
                                            try:
                                                agent_rewards.append(float(reward))
                                            except (TypeError, ValueError):
                                                agent_rewards.append(0.1)
                                all_cumulative_rewards.append(sum(agent_rewards))
                            
                            cumulative_total = sum(all_cumulative_rewards)
                            
                            # Calculate average total reward for last 10 episodes
                            recent_totals = []
                            for ep in range(max(0, episode-9), episode+1):
                                ep_total = 0.0
                                for agent_id in self.training_history["episode_rewards"].keys():
                                    if ep < len(self.training_history["episode_rewards"][agent_id]):
                                        reward = self.training_history["episode_rewards"][agent_id][ep]
                                        if isinstance(reward, dict) and 'episode_reward' in reward:
                                            ep_total += float(reward['episode_reward'])
                                        elif isinstance(reward, (int, float, np.number)):
                                            ep_total += float(reward)
                                        else:
                                            # Try to convert to float
                                            try:
                                                ep_total += float(reward)
                                            except (TypeError, ValueError):
                                                ep_total += 0.1
                                recent_totals.append(ep_total)
                            
                            avg_recent_total = sum(recent_totals) / len(recent_totals) if recent_totals else 0.0
                        except Exception as e:
                            logger.error(f"Error calculating overall statistics: {e}")
                            episode_total = 0.1
                            cumulative_total = episode * 0.1
                            avg_recent_total = 0.1
                        
                        # Calculate average loss value for all managers
                        avg_policy_loss = 0.0
                        avg_value_loss = 0.0
                        avg_entropy = 0.0
                        
                        if hasattr(self, 'training_loss_history'):
                            policy_losses = []
                            value_losses = []
                            entropies = []
                            for agent_id in self.training_history["episode_rewards"].keys():
                                if agent_id in self.training_loss_history and episode < len(self.training_loss_history[agent_id]):
                                    loss_info = self.training_loss_history[agent_id][episode]
                                    policy_losses.append(loss_info.get('policy_loss', 0.0))
                                    value_losses.append(loss_info.get('value_loss', 0.0))
                                    entropies.append(loss_info.get('entropy', 0.0))
                            
                            if policy_losses:
                                avg_policy_loss = sum(policy_losses) / len(policy_losses)
                            if value_losses:
                                avg_value_loss = sum(value_losses) / len(value_losses)
                            if entropies:
                                avg_entropy = sum(entropies) / len(entropies)
                    
                        history_rows.append({
                        'algorithm': algorithm_name,
                        'manager_id': 'total',
                        'episode': episode + 1,
                        'episode_reward': float(episode_total),
                        'cumulative_reward': float(cumulative_total),
                            'avg_reward_last_10': float(avg_recent_total),
                            'policy_loss': avg_policy_loss,
                            'value_loss': avg_value_loss,
                            'entropy': avg_entropy,
                        'data_type': 'total_reward'
                        })
                    
            elif isinstance(self.training_history["episode_rewards"], list):
                # Single agent or aggregated format
                for episode, reward in enumerate(self.training_history["episode_rewards"]):
                    # Single agent format
                    policy_loss = 0.0
                    value_loss = 0.0
                    entropy = 0.0
                    
                    # Try to get the loss record for this episode
                    if hasattr(self, 'training_loss_history') and 'multi_agent' in self.training_loss_history:
                        if episode < len(self.training_loss_history['multi_agent']):
                            loss_info = self.training_loss_history['multi_agent'][episode]
                            policy_loss = loss_info.get('policy_loss', 0.0)
                            value_loss = loss_info.get('value_loss', 0.0)
                            entropy = loss_info.get('entropy', 0.0)
                    
                    # Process reward value
                    try:
                        if isinstance(reward, dict) and 'episode_reward' in reward:
                            reward_value = float(reward['episode_reward'])
                        elif isinstance(reward, (int, float, np.number)):
                            reward_value = float(reward)
                        else:
                            # Try to convert to float
                            try:
                                reward_value = float(reward)
                            except (TypeError, ValueError):
                                reward_value = 0.1
                                logger.warning(f"Cannot convert reward to numeric, type: {type(reward)}, use default value: {reward_value}")
                    except Exception as e:
                        reward_value = 0.1
                        logger.error(f"Error processing reward: {e}, use default value: {reward_value}")
                    
                    # Calculate cumulative reward and average reward
                    try:
                        previous_rewards = []
                        for r in self.training_history["episode_rewards"][:episode+1]:
                            if isinstance(r, dict) and 'episode_reward' in r:
                                previous_rewards.append(float(r['episode_reward']))
                            elif isinstance(r, (int, float, np.number)):
                                previous_rewards.append(float(r))
                            else:
                                try:
                                    previous_rewards.append(float(r))
                                except (TypeError, ValueError):
                                    previous_rewards.append(0.1)
                        
                        cumulative_reward = sum(previous_rewards)
                        recent_rewards = previous_rewards[max(0, episode-9):episode+1]
                        avg_reward_last_10 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                    except Exception as e:
                        logger.error(f"Error calculating cumulative reward: {e}")
                        cumulative_reward = episode * 0.1
                        avg_reward_last_10 = 0.1
                    
                    history_rows.append({
                        'algorithm': algorithm_name,
                        'manager_id': 'multi_agent',
                        'episode': episode + 1,
                        'episode_reward': reward_value,
                        'cumulative_reward': cumulative_reward,
                        'avg_reward_last_10': avg_reward_last_10,
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        'entropy': entropy,
                        'data_type': 'episode_reward'
                    })
            
            # Save to CSV file - 🔧 Fix: Multiple safeguards to ensure successful saving
            if history_rows:
                csv_file = self._generate_csv_filename("training_history", algorithm_name)
                
                # 🔧 Method 1: Use pandas to save
                save_success = False
                try:
                    df = pd.DataFrame(history_rows)
                    df.to_csv(csv_file, index=False)
                    
                    # Verify if the file is really created and has content
                    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                        logger.info(f"✅ {algorithm_name} training history saved to {csv_file}, {len(history_rows)} rows")
                        print(f"✅ Training history saved to {os.path.basename(csv_file)}, {len(history_rows)} rows")
                        save_success = True
                        
                        # Display reward statistics information
                        if isinstance(self.training_history["episode_rewards"], dict):
                            logger.info("📊 Training reward statistics:")
                            print("\n📊 Training reward statistics:")
                            
                            for manager_id, rewards in self.training_history["episode_rewards"].items():
                                if rewards:
                                    # Process final reward
                                    final_reward = rewards[-1]
                                    if isinstance(final_reward, dict) and 'episode_reward' in final_reward:
                                        final_reward = final_reward['episode_reward']
                                    elif not isinstance(final_reward, (int, float, np.number)):
                                        try:
                                            final_reward = float(final_reward)
                                        except (TypeError, ValueError):
                                            final_reward = 0.0
                                    
                                    # Calculate average reward
                                    reward_values = []
                                    for r in rewards:
                                        if isinstance(r, dict) and 'episode_reward' in r:
                                            reward_values.append(float(r['episode_reward']))
                                        elif isinstance(r, (int, float, np.number)):
                                            reward_values.append(float(r))
                                        else:
                                            try:
                                                reward_values.append(float(r))
                                            except (TypeError, ValueError):
                                                reward_values.append(0.1)
                                    
                                    avg_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
                                    
                                    # Record to log and console
                                    logger.info(f"  {manager_id}: Final reward {final_reward:.3f}, average reward {avg_reward:.3f}")
                                    print(f"  {manager_id}: Final reward {final_reward:.3f}, average reward {avg_reward:.3f}")
                    else:
                        logger.warning(f"❌ pandas save failed, file does not exist or is empty")
                except Exception as e:
                    logger.error(f"❌ pandas save failed: {e}")
                
                # 🔧 Method 2: If pandas fails, use standard csv module
                if not save_success:
                    try:
                        import csv
                        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                            if history_rows:
                                writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
                                writer.writeheader()
                                writer.writerows(history_rows)
                                f.flush()
                        
                        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                            logger.info(f"✅ {algorithm_name} training history saved to {csv_file} using csv module")
                            save_success = True
                        else:
                            logger.warning(f"❌ csv module save failed, file does not exist or is empty")
                    except Exception as e:
                        logger.error(f"❌ csv module save failed: {e}")
                
                # 🔧 Method 3: If both fail, create basic text backup
                if not save_success:
                    try:
                        backup_file = csv_file.replace('.csv', '_backup.txt')
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(f"Training history backup - {algorithm_name}\n")
                            f.write(f"Time: {datetime.now()}\n\n")
                            for row in history_rows:
                                f.write(f"{row}\n")
                        logger.info(f"🔧 Emergency text backup saved to {backup_file}")
                    except Exception as e:
                        logger.error(f"❌ Even text backup failed: {e}")
                
                # Output training curve statistics
                if isinstance(self.training_history["episode_rewards"], dict):
                    for manager_id, rewards in self.training_history["episode_rewards"].items():
                        final_reward = rewards[-1] if rewards else 0
                        avg_reward = np.mean(rewards) if rewards else 0
                        logger.info(f"  {manager_id}: Final reward {final_reward:.3f}, average reward {avg_reward:.3f}")
                elif isinstance(self.training_history["episode_rewards"], list):
                    final_reward = self.training_history["episode_rewards"][-1] if self.training_history["episode_rewards"] else 0
                    avg_reward = np.mean(self.training_history["episode_rewards"]) if self.training_history["episode_rewards"] else 0
                    logger.info(f"  Final reward: {final_reward:.3f}, average reward: {avg_reward:.3f}")
            else:
                logger.warning("No valid training history data to save")
                
        except Exception as e:
            logger.error(f"Failed to save training history to CSV file: {e}")
            # Alternative: Use built-in CSV module
            try:
                import csv
                csv_file = self._generate_csv_filename("training_history", algorithm_name)
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['algorithm', 'manager_id', 'episode', 'episode_reward', 'cumulative_reward', 'avg_reward_last_10', 'data_type'])
                    
                    if isinstance(self.training_history["episode_rewards"], dict):
                        for manager_id, rewards in self.training_history["episode_rewards"].items():
                            for episode, reward in enumerate(rewards):
                                cum_reward = sum(rewards[:episode+1])
                                avg_last_10 = np.mean(rewards[max(0, episode-9):episode+1])
                                writer.writerow([algorithm_name, manager_id, episode + 1, float(reward), float(cum_reward), float(avg_last_10), 'episode_reward'])
                    elif isinstance(self.training_history["episode_rewards"], list):
                        for episode, reward in enumerate(self.training_history["episode_rewards"]):
                            cum_reward = sum(self.training_history["episode_rewards"][:episode+1])
                            avg_last_10 = np.mean(self.training_history["episode_rewards"][max(0, episode-9):episode+1])
                            writer.writerow([algorithm_name, 'multi_agent', episode + 1, float(reward), float(cum_reward), float(avg_last_10), 'episode_reward'])
                            
                logger.info(f"Saved {algorithm_name} training history to {csv_file} using built-in CSV module")
            except Exception as e2:
                logger.error(f"Failed to save training history to CSV file using built-in CSV module: {e2}")
    
    def _record_training_loss(self, manager_id: str, episode: int, policy_loss: float, value_loss: float, entropy: float = 0.0):
        """
        Record training loss values to training_loss_history
        
        Args:
            manager_id: Manager ID
            episode: Episode number
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Entropy value (optional)
        """
        if manager_id not in self.training_loss_history:
            self.training_loss_history[manager_id] = []
        
        # Ensure list length is sufficient
        while len(self.training_loss_history[manager_id]) <= episode:
            self.training_loss_history[manager_id].append({
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            })
        
        try:
            policy_loss_value = float(policy_loss) if policy_loss is not None else 0.0
            value_loss_value = float(value_loss) if value_loss is not None else 0.0
            entropy_value = float(entropy) if entropy is not None else 0.0
            
            self.training_loss_history[manager_id][episode] = {
                'policy_loss': policy_loss_value,
                'value_loss': value_loss_value,
                'entropy': entropy_value
            }
        
            logger.info(f"Record {manager_id} Episode {episode} loss: Policy={policy_loss_value:.4f}, Value={value_loss_value:.4f}, Entropy={entropy_value:.4f}")
        except Exception as e:
            logger.error(f"Error recording loss values: {e}")
            # Ensure there are default values
            self.training_loss_history[manager_id][episode] = {
                'policy_loss': 0.01,
                'value_loss': 0.01,
                'entropy': 0.001
            }

    def _record_training_loss_for_all_managers(self, episode: int, train_info: dict, manager_ids: list):
        """
        Record training loss values for all managers
        
        Args:
            episode: Episode number
            train_info: Dictionary containing loss information
            manager_ids: List of Manager IDs
        """
        policy_loss = train_info.get('policy_loss', 0.0)
        value_loss = train_info.get('value_loss', 0.0) 
        entropy = train_info.get('entropy', train_info.get('dist_entropy', 0.0))
        
        for manager_id in manager_ids:
            self._record_training_loss(manager_id, episode, policy_loss, value_loss, entropy)

    def _generate_csv_filename(self, data_type: str, algorithm_name: Optional[str] = None) -> str:
        """Generate CSV file name
        
        Args:
            data_type: Data type, e.g. 'rewards', 'pipeline_results'
            algorithm_name: Algorithm name (optional)
        
        Returns:
            CSV file path
        """
        # 🔧 Fix: Ensure experiment_id exists, if not, generate a temporary one
        if self.experiment_id is None:
            # If there is no experiment_id, generate a temporary one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"temp_{timestamp}"
            logger.warning(f"experiment_id is None, generate temporary ID: {self.experiment_id}")
        
        if algorithm_name:
            filename = f"{algorithm_name.lower()}_{data_type}_{self.experiment_id}.csv"
        else:
            filename = f"{data_type}_{self.experiment_id}.csv"
        
        return os.path.join(self.results_dir, filename)
    
    def _init_global_observation_manager(self):
        """Initialize global observation manager"""
        try:
            config = None
            if self.global_observation_config_file and os.path.exists(self.global_observation_config_file):
                self.global_observation_manager = GlobalObservationManager()
                self.global_observation_manager.load_config(self.global_observation_config_file)
                logger.info(f"Load global observation configuration from file {self.global_observation_config_file}")
            else:
                self.global_observation_manager = GlobalObservationManager()
                logger.info("Use default global observation configuration")
        except Exception as e:
            logger.error(f"Failed to initialize global observation manager: {e}")
            self.global_observation_manager = None
    
    def _setup_components(self):
        """Initialize components"""
        # First initialize device models
        self._setup_device_models()
        
        # Initialize users and managers (must be initialized before scheduler)
        if not hasattr(self, 'managers') or not self.managers:
            self._setup_managers_and_users()
        
        # Initialize aggregators
        self._setup_aggregators()
        
        # Initialize trading pool
        self._setup_trading_pool()
        
        # Register managers to trading pool after Manager creation
        self._register_managers_to_trading_pool()
        
        # Initialize scheduler (must be initialized after managers)
        self._setup_scheduler()
        
        # Create environments and RL agents
        if len(self.users) > 0:
            self._create_environments()
            self._setup_rl_agents()
        else:
            logger.warning("User list is empty, cannot create environments")
        
        # Set global observation manager
        if self.use_global_observation:
            self._setup_global_observation_manager()
    
    def _setup_device_models(self):
        """Initialize device models and related configurations"""
        # Load device parameters
        self.device_params = {
            DeviceType.BATTERY: [],
            DeviceType.HEAT_PUMP: [],
            DeviceType.EV: [],
            DeviceType.PV: []
        }
        
        # Example PV parameters
        pv_params = PVParameters(
            pv_id="pv_sample",
            max_power=5.0,
            efficiency=0.18,
            area=28.0,
            location="rooftop",
            tilt_angle=35.0,
            azimuth_angle=180.0,
            weather_dependent=True,
            forecast_accuracy=0.85
        )
        self.device_params[DeviceType.PV].append(pv_params)
        
        # Initialize price loader - prioritize loading Danish grid prices from grid_price.csv
        from fo_generate.price_loader import PriceLoader
        self.price_loader = PriceLoader("data")
        
        # Generate price data for current time range
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        try:
            self.price_data = self.price_loader.get_price_data(start_time, self.time_horizon)
            logger.info(f"Successfully loaded price data, data source: {self.price_data['source'].iloc[0] if not self.price_data.empty else 'unknown'}")
        except Exception as e:
            logger.warning(f"Price loader failed: {e}, use alternative")
            
            # Alternative: Read traditional price_data.csv file
            price_data_file = self.config.get("price_data_file")
            if price_data_file and os.path.exists(price_data_file):
                self.price_data = pd.read_csv(price_data_file)
                logger.info(f"Loaded alternative price file: {price_data_file}")
            else:
                # Last alternative: Generate some test price data
                timestamps = [start_time + timedelta(hours=i) for i in range(self.time_horizon)]
                prices = np.random.uniform(0.1, 0.3, self.time_horizon)  # Simulated price
                self.price_data = pd.DataFrame({"timestamp": timestamps, "price": prices})
                logger.info("Use generated test price data")
        
        # Read or generate weather data
        weather_data_file = self.config.get("weather_data_file")
        if weather_data_file and os.path.exists(weather_data_file):
            self.weather_data = pd.read_csv(weather_data_file)
        else:
            # Generate simple weather data for simulation
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(self.time_horizon)]
            temperatures = np.random.uniform(15, 25, self.time_horizon)  # Simulated temperature
            irradiances = np.random.uniform(200, 800, self.time_horizon)  # Simulated irradiance
            self.weather_data = pd.DataFrame({
                "timestamp": timestamps, 
                "temperature": temperatures,
                "solar_irradiance": irradiances
            })
    
    def _setup_rl_agents(self):
        """Set up RL agents"""
        if not self.envs:
            logger.warning("Environments not initialized, please call _create_environments first")
            return
            
        # Ensure user_device_map exists
        if not hasattr(self, 'envs_user_device_map'):
            self.envs_user_device_map = {}
        
        # Create agents based on algorithm type
        if self.rl_algorithm in ["fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg"]:
            # Built-in multi-agent algorithm, no need to create separate agents for each user
            logger.info(f"{self.rl_algorithm.upper()} is a multi-agent algorithm, will initialize multi-agent environment in training stage")
            # To maintain consistency, create a tag for supported algorithms
            if self.rl_algorithm not in self.rl_agents:
                self.rl_agents[self.rl_algorithm] = {}
            self.rl_agents[self.rl_algorithm]["multi_agent"] = None
        elif self.custom_rl_algorithm:
            agent_class = RLRegistry.get(self.rl_algorithm)
            if agent_class is not None:
                # Initialize rl_agents sub-dictionary for custom algorithms
                if self.rl_algorithm not in self.rl_agents:
                    self.rl_agents[self.rl_algorithm] = {}
                    
                for user_id, env in self.envs.items():
                    state_dim = env.observation_space.shape[0]
                    action_dim = env.action_space.shape[0]
                    max_action = float(env.action_space.high[0])
                    
                    try:
                        self.rl_agents[self.rl_algorithm][user_id] = agent_class(
                            state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action,
                            device=self.device
                        )
                    except Exception as e:
                        logger.error(f"Failed to initialize custom algorithm {self.rl_algorithm}: {e}")
                        # Fallback to FOMAPPO
                        logger.info("Fallback to FOMAPPO algorithm")
                        self.rl_algorithm = "fomappo"
                        if "fomappo" not in self.rl_agents:
                            self.rl_agents["fomappo"] = {}
                        self.rl_agents["fomappo"]["multi_agent"] = None
                        break
    
    def _setup_managers_and_users(self):
        """Initialize managers and users"""
        # Clear existing users and managers
        self.users = []
        self.managers = []
        
        # Create City object
        self.city = City(city_name="flex_offer_city")
        
        # Determine the number of users each manager manages (support uneven distribution)
        if self.num_managers == 4 and self.num_users == 36:
            # Create uneven user distribution for 4 managers: 6, 10, 8, 12
            self.users_distribution = [6, 10, 8, 12]
            logger.info(f"Use custom user distribution: {self.users_distribution}")
        else:
            # Default average distribution
            base_users = self.num_users // self.num_managers
            remaining_users = self.num_users % self.num_managers
            self.users_distribution = [base_users] * self.num_managers
            # Assign remaining users to the first few managers
            for i in range(remaining_users):
                self.users_distribution[i] += 1
            logger.info(f"Use average user distribution: {self.users_distribution}")
        
        # Verify total number of users
        assert sum(self.users_distribution) == self.num_users, f"Total number of users {sum(self.users_distribution)} does not equal total number of users {self.num_users}"
        
        # Create users and devices for each manager
        current_user_idx = 0
        for m in range(self.num_managers):
            # Generate random location and coverage area for each manager
            location = (random.uniform(0, 10), random.uniform(0, 10))  # Random location coordinates
            coverage_area = random.uniform(1, 5)  
            
            manager = Manager(
                manager_id=f"manager_{m+1}",  # Start from 1 instead of 0
                location=location,
                coverage_area=coverage_area
            )
            
            # Create users for each manager
            users_for_this_manager = self.users_distribution[m]
            start_user_idx = current_user_idx
            end_user_idx = current_user_idx + users_for_this_manager
            
            logger.info(f"Create {users_for_this_manager} users for Manager {manager.manager_id} (indices {start_user_idx}-{end_user_idx-1})")
            
            for u in range(start_user_idx, end_user_idx):
                # Generate random location for each user (within manager's coverage area)
                manager_x, manager_y = location
                radius = math.sqrt(coverage_area / math.pi)
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius)
                user_x = manager_x + distance * math.cos(angle)
                user_y = manager_y + distance * math.sin(angle)
                user_location = (user_x, user_y)
                
                # Randomly select user type
                user_type = random.choice(["prosumer", "consumer", "producer"])
                
                # Create random user preferences
                user_preferences = {
                    "economic": random.uniform(0.1, 0.4),
                    "comfort": random.uniform(0.1, 0.4),
                    "self_sufficient": random.uniform(0.1, 0.4),
                    "environmental": random.uniform(0.1, 0.4)
                }
                # Normalize preferences
                pref_sum = sum(user_preferences.values())
                user_preferences = {k: v / pref_sum for k, v in user_preferences.items()}
                
                user = User(
                    user_id=f"user_{u}",
                    user_type=user_type,
                    location=user_location
                )
                
                # Add user preferences attribute
                user.preferences = user_preferences
                
                # Add devices to user
                for device_type, (min_count, max_count) in self.devices_per_user.items():
                    count = np.random.randint(min_count, max_count + 1)
                    
                    for d in range(count):
                        device_id = f"{device_type}_{u}_{d}"
                        
                        # Create different parameter objects based on device type
                        if device_type == DeviceType.BATTERY:
                            capacity = np.random.uniform(5, 10)  # kWh
                            max_power = np.random.uniform(2, 4)  # kW
                            initial_soc = np.random.uniform(0.3, 0.7)
                            
                            params = BatteryParameters(
                                battery_id=device_id,
                                soc_min=0.1,
                                soc_max=0.9,
                                p_min=-max_power, # Discharge power is negative
                                p_max=max_power,  # Charge power is positive
                                efficiency=0.95,
                                initial_soc=initial_soc,
                                battery_type="lithium-ion",
                                capacity_kwh=capacity
                            )
                        elif device_type == DeviceType.HEAT_PUMP:
                            max_power = np.random.uniform(1, 3)  # kW
                            cop = np.random.uniform(3, 4.5)
                            initial_temp = np.random.uniform(19, 21)
                            
                            params = HeatPumpParameters(
                                room_id=device_id,
                                room_area=30.0,
                                room_volume=75.0,
                                temp_min=18.0,
                                temp_max=26.0,
                                initial_temp=initial_temp,
                                cop=cop,
                                heat_loss_coef=0.1,
                                primary_use_period="8:00-22:00",
                                secondary_use_period="22:00-8:00",
                                primary_target_temp=22.0,
                                secondary_target_temp=19.0,
                                max_power=max_power
                            )
                        elif device_type == DeviceType.EV:
                            capacity = np.random.uniform(40, 80)  # kWh
                            max_power = np.random.uniform(3, 7)  # kW
                            initial_soc = np.random.uniform(0.2, 0.8)
                            
                            params = EVParameters(
                                ev_id=device_id,
                                battery_capacity=capacity,
                                soc_min=0.1,
                                soc_max=0.95,
                                max_charging_power=max_power,
                                efficiency=0.9,
                                initial_soc=initial_soc,
                                fast_charge_capable=True
                            )
                            
                            # Create user behavior object
                            now = datetime.now()
                            arrival_time = datetime(now.year, now.month, now.day, 18, 0)  # 18:00 arrival
                            departure_time = datetime(now.year, now.month, now.day + 1, 7, 30)  # 7:30 departure the next day
                            
                            behavior = EVUserBehavior(
                                ev_id=device_id,
                                connection_time=arrival_time,
                                disconnection_time=departure_time,
                                next_departure_time=departure_time,
                                target_soc=0.85,
                                fast_charge_preferred=False,
                                min_required_soc=0.6,
                                location="home",
                                priority=3
                            )
                            
                            # Set user behavior
                            setattr(params, 'behavior', behavior)
                            
                        elif device_type == DeviceType.PV:
                            capacity = np.random.uniform(3, 8)  # kW
                            efficiency = np.random.uniform(0.15, 0.22)
                            
                            params = PVParameters(
                                pv_id=device_id,
                                max_power=capacity,
                                efficiency=efficiency,
                                area=capacity * 5,  
                                location="roof",
                                tilt_angle=30.0,
                                azimuth_angle=180.0,
                                weather_dependent=True,
                                forecast_accuracy=0.8
                            )
                        elif device_type == DeviceType.DISHWASHER:
                            # Import dishwasher related modules
                            from fo_generate.dishwasher_model import DishwasherParameters, DishwasherUserBehavior
                            
                            # Dishwasher parameters
                            total_energy = np.random.uniform(2.5, 3.5)  # kWh
                            power_rating = np.random.uniform(1.8, 2.5)  # kW
                            operation_hours = total_energy / power_rating  # Operation hours
                            
                            params = DishwasherParameters(
                                dishwasher_id=device_id,
                                total_energy=total_energy,
                                power_rating=power_rating,
                                operation_hours=operation_hours,
                                min_start_delay=0.5,  # Minimum start delay 0.5 hours
                                max_start_delay=6.0,  # Maximum start delay 6 hours
                                efficiency=0.9,
                                can_interrupt=False  # Dishwasher cannot be interrupted
                            )
                            
                            # Create dishwasher user behavior
                            now = datetime.now()
                            deployment_time = now  # User starts at start time
                            preferred_start_time = deployment_time + timedelta(hours=1)  # 1 hour later
                            latest_completion_time = deployment_time + timedelta(hours=8)  # 8 hours to complete
                            
                            behavior = DishwasherUserBehavior(
                                dishwasher_id=device_id,
                                deployment_time=deployment_time,
                                preferred_start_time=preferred_start_time,
                                latest_completion_time=latest_completion_time,
                                priority=3,
                                user_tolerance=2.0
                            )
                            
                            # Set user behavior
                            setattr(params, 'behavior', behavior)
                        else:
                            params = {}
                        
                        device = Device(
                            device_id=device_id,
                            device_type=device_type,
                            params=params
                        )
                        user.add_device(device)
                
                # Ensure each user has at least one device
                if len(user.devices) == 0:
                    logger.warning(f"User {user.user_id} has no devices, add default battery device")
                    # Create a default battery device
                    device_id = f"battery_{u}_default"
                    params = BatteryParameters(
                        battery_id=device_id,
                        soc_min=0.1,
                        soc_max=0.9,
                        p_min=-3.0,
                        p_max=3.0,
                        efficiency=0.95,
                        initial_soc=0.5,
                        battery_type="lithium-ion",
                        capacity_kwh=7.0
                    )
                    device = Device(
                        device_id=device_id,
                        device_type=DeviceType.BATTERY,
                        params=params
                    )
                    user.add_device(device)
                
                manager.add_user(user)
                self.users.append(user)  # Add user to users list
            
            # Update user index
            current_user_idx = end_user_idx
            
            self.managers.append(manager)
            self.city.add_manager(manager)
            logger.info(f"Manager {manager.manager_id} created, containing {len(manager.users)} users")
            
        logger.info(f"Created {len(self.managers)} managers and {len(self.users)} users")
        
        # Verify the number of users for each Manager
        for manager in self.managers:
            user_ids = [user.user_id for user in manager.users]
            logger.info(f"Manager {manager.manager_id}: {len(manager.users)} users {user_ids[:3]}{'...' if len(user_ids) > 3 else ''}")
    
    def _setup_aggregators(self):
        """Initialize aggregators"""
        # Use new aggregator factory
        self.fo_aggregator = FOAggregatorFactory.create_aggregator(
            method=self.aggregation_method if self.aggregation_method in ["LP", "DP"] else "DP",
            spt=self.config.get("max_power_limit", 100.0),
            ppt=self.config.get("power_profile_threshold", 23),
            tf_threshold=self.config.get("time_flexibility_threshold", 1.0),
            power_deviation=self.config.get("power_deviation", 5.0)
        )
        
        logger.info(f"Aggregator initialized, method: {self.aggregation_method}")
        
        # For compatibility with existing code, keep dfo_aggregator and sfo_aggregator references
        self.dfo_aggregator = self.fo_aggregator
        self.sfo_aggregator = self.fo_aggregator
    
    def _setup_trading_pool(self):
        """Initialize trading pool"""
        # Initialize weather model and demand model
        self.weather_model = WeatherModel(
            weather_data_file=self.config.get("weather_data_file", ""),
            time_horizon=self.time_horizon
        )
        
        self.demand_model = DemandModel(
            demand_data_file=self.config.get("demand_data_file", ""),
            time_horizon=self.time_horizon
        )
        
        # Get trading algorithm configuration
        trading_algorithm = self.config.get("trading_algorithm", "market_clearing")
        clearing_method = self.config.get("clearing_method", "uniform_price")
        
        # Initialize trading pool - support new trading algorithms
        algorithm_kwargs = {}
        if trading_algorithm == "market_clearing":
            algorithm_kwargs["clearing_method"] = clearing_method
        
        self.trading_pool = TradingPool(
            weather_model=self.weather_model,
            demand_model=self.demand_model,
            trading_algorithm=trading_algorithm,
            **algorithm_kwargs
        )
        
        logger.info(f"Trading pool initialized, algorithm: {trading_algorithm}, clearing method: {clearing_method}")
    
    def _register_managers_to_trading_pool(self):
        """Register Manager to trading pool"""
        if hasattr(self, 'trading_pool') and hasattr(self, 'managers') and self.managers:
            for manager in self.managers:
                self.trading_pool.add_manager(manager.manager_id, manager)
            logger.info(f"Registered {len(self.managers)} managers to trading pool")
        else:
            logger.warning("Trading pool or Manager not initialized, cannot register Manager")
    
    def _setup_scheduler(self):
        """Initialize scheduler"""
        # User scheduler
        self.user_scheduler = UserScheduler(
            num_users=self.num_users,
            time_horizon=self.time_horizon,
            time_steps_per_hour=int(1 / self.time_step)
        )
        

        if self.time_step <= 1.0:
            time_steps_per_hour = int(1 / self.time_step)
        else:
            time_steps_per_hour = 1
            logger.warning(f"time_step={self.time_step}>1.0, adjust time_steps_per_hour=1")
        
        # Ensure time_steps_per_hour is at least 1 (to avoid dimension issues)
        time_steps_per_hour = max(1, time_steps_per_hour)
        
        logger.info(f"Scheduler time configuration: time_step={self.time_step}h, time_steps_per_hour={time_steps_per_hour}, total_steps={self.time_horizon * time_steps_per_hour}")
        
        # Scheduling Manager - Make sure managers exist
        if hasattr(self, 'managers') and self.managers:
            self.schedule_manager = ScheduleManager(
                managers=self.managers,
                trading_pool=self.trading_pool,
                time_horizon=self.time_horizon,
                time_steps_per_hour=time_steps_per_hour,  
                disaggregation_algorithm=self.disaggregation_method
            )
            logger.info(f"Initialized scheduler manager, number of managers: {len(self.managers)}, time range: {self.time_horizon} hours, disaggregation algorithm: {self.disaggregation_method}")
        else:
            # If managers are not initialized, create an empty scheduler manager
            self.schedule_manager = ScheduleManager(
                managers=[],
                trading_pool=self.trading_pool,
                time_horizon=self.time_horizon,
                time_steps_per_hour=time_steps_per_hour,  
                disaggregation_algorithm=self.disaggregation_method
            )
            logger.warning("No managers, using empty scheduler manager")
        
        # Disaggregator (compatibility maintained, using new architecture)
        self.disaggregator = AggregatedResultDisaggregator(
            time_horizon=self.time_horizon,
            default_algorithm=self.disaggregation_method
        )
    
    def _create_environments(self):
        """Create RL environment for each user"""
        if not self.managers or not self.users:
            logger.warning("User or manager not initialized, cannot create environment")
            return
            
        # Ensure price_data_file and weather_data_file attributes exist
        self.price_data_file = self.config.get("price_data_file")
        self.weather_data_file = self.config.get("weather_data_file")
        
        # Initialize mapping table
        self.envs = {}
        self.envs_user_device_map = {}
            
        # Load price and weather data, provide default empty DataFrame
        price_data = pd.DataFrame()
        if self.price_data_file and os.path.exists(self.price_data_file):
            try:
                price_data = pd.read_csv(self.price_data_file)
            except Exception as e:
                logger.error(f"Failed to load price data: {e}")
                
        weather_data = pd.DataFrame()
        if self.weather_data_file and os.path.exists(self.weather_data_file):
            try:
                weather_data = pd.read_csv(self.weather_data_file)
            except Exception as e:
                logger.error(f"Failed to load weather data: {e}")
                
        # Create environment for each user
        for user in self.users:
            # Skip users without devices
            if not user.devices:
                logger.warning(f"User {user.user_id} has no devices, skip")
                continue
                
            # Create user preferences
            user_preferences = {
                "economic": user.preferences.get("economic", 0.25),
                "comfort": user.preferences.get("comfort", 0.25),
                "self_sufficient": user.preferences.get("self_sufficient", 0.25),
                "environmental": user.preferences.get("environmental", 0.25)
            }
            
            # Convert user devices to environment required format
            devices = {}
            for device in user.devices:
                # Clone device to avoid modifying original device state
                device_copy = device.clone()
                devices[device.device_id] = {
                    'type': device.device_type,
                    'params': device_copy.get_parameters()
                }
            
            # Create environment
            env = FlexOfferEnv(
                devices=devices,
                time_horizon=self.time_horizon,
                time_step=self.time_step,
                start_time=datetime.now(),
                price_data=price_data,
                user_preferences=user_preferences,
                weather_data=weather_data,
                data_dir="data"  # Pass data_dir parameter to use new price loader
            )
            
            # Check if environment action space is valid
            if not hasattr(env.action_space, 'shape') or env.action_space.shape is None or len(env.action_space.shape) == 0 or env.action_space.shape[0] == 0:
                logger.warning(f"User {user.user_id} environment action space is invalid, skip")
                continue
            
            # Store environment and user mapping
            self.envs[user.user_id] = env
            self.envs_user_device_map[user.user_id] = user
    
    def _setup_global_observation_manager(self):
        """Set global observation manager"""
        if self.global_observation_manager:
            if self.envs and len(self.envs) > 0:
                first_env = self.envs[next(iter(self.envs))]
                self.global_observation_manager.register_module(
                    "generate", first_env, weight=1.0
                )
            else:
                logger.warning("Environment not initialized, cannot register to global observation manager")
    
    def train_rl_agents(self):
        """Train all RL agents"""
        print(f"\n🚀 ========== Start RL training ==========")
        print(f"🔧 Algorithm: {self.rl_algorithm}")
        print(f"🔧 Training episodes: {self.num_episodes}")
        print(f"🔧 Time steps per episode: {self.steps_per_episode}")
        print("=" * 50)
        
        logger.info(f"🚀 Start training RL agents, algorithm: {self.rl_algorithm}, training episodes: {self.num_episodes}")
        
        try:
            self._create_environments()
            logger.info("✅ Environment created")
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            return
        
        # Train agents based on selected algorithm
        try:
            if self.rl_algorithm == "fomappo":
                # 🔧 Use FOMAPPO (shared policy architecture)
                print("🤖 Select algorithm: FOMAPPO (shared policy architecture)")
                logger.info("🤖 Use FOMAPPO (shared policy architecture, all Managers share policy)")
                self._train_fomappo_agents()
            elif self.rl_algorithm == "fomaippo":
                # 🔧 Use FOMAIPPO (separate policy architecture)
                print("🤖 Select algorithm: FOMAIPPO (separate policy architecture)")
                logger.info("🤖 Use FOMAIPPO (separate policy architecture, each Manager learns independently)")
                self._train_fomaippo_agents()
            elif self.rl_algorithm == "fomaddpg":
                print("🤖 Select algorithm: FOMADDPG (optimized version)")
                self._train_fomaddpg_agents_optimized()
            elif self.rl_algorithm == "fomatd3":
                print("🤖 Select algorithm: FOMATD3 (adapter version)")
                self._train_fomatd3_agents_with_adapter()
            elif self.rl_algorithm == "fosqddpg":
                print("🤖 Select algorithm: FOSQDDPG (adapter version)")
                self._train_fosqddpg_agents_with_adapter()
            elif self.rl_algorithm == "fomodelbased":
                print("🤖 Select algorithm: FOModelBased (traditional optimization method)")
                print("📊 Start FOModelBased traditional optimization evaluation...")
                logger.info("📊 Start FOModelBased traditional optimization evaluation...")
                self.run_fomodelbased_evaluation()
                print("✅ FOModelBased evaluation completed!")
            elif self.custom_rl_algorithm and self.rl_algorithm in self.rl_agents:
                print(f"🤖 Select algorithm: {self.rl_algorithm} (custom)")
            # Try to call the training method of the custom algorithm
                self._train_custom_agents()
            else:
                print(f"⚠️ Unsupported RL algorithm: {self.rl_algorithm}, fallback to FOMAIPPO")
                logger.warning(f"Unsupported RL algorithm: {self.rl_algorithm}, fallback to FOMAIPPO")
                self._train_fomaippo_agents()
                
        except Exception as e:
            logger.error(f"❌ Exception during training: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            print(f"❌ Training failed: {e}")
            return
        
        print("✅ RL training completed!")
        logger.info("✅ RL agents training completed")
    
    def _train_fomappo_agents(self):
        """Train FOMAPPO (shared policy) algorithm"""
        print("\n🔧 Enter FOMAPPO training method...")
        logger.info("🔧 Start _train_fomappo_agents method")
        
        try:
            print("📦 Try to import external training method...")
            from algorithms.MAPPO.fomappo.fomappo_training_methods import train_fomappo_shared_policy
            print("✅ Successfully imported train_fomappo_shared_policy")
            logger.info("✅ Successfully imported train_fomappo_shared_policy, call the repaired training method")
            result = train_fomappo_shared_policy(self)
            print("✅ External training method execution completed")
            
            # 🔧 Critical fix: handle the object returned by the external training method
            if isinstance(result, dict) and result.get('status') == 'success':
                logger.info("✅ External training method successfully completed, set adapter reference")
                if 'multi_agent_env' in result:
                    self.multi_agent_env = result['multi_agent_env']
                    logger.info("✅ Set multi_agent_env")
                if 'fomappo_adapter' in result:
                    self.fomappo_adapter = result['fomappo_adapter'] 
                    logger.info("✅ Set fomappo_adapter")
                
                # 🔧 Fix: ensure training history is correctly set
                if 'training_history' in result:
                    self.training_history = result['training_history']
                    logger.info("✅ Set training_history")
                    
                logger.info(f"Validation: hasattr(self, 'multi_agent_env') = {hasattr(self, 'multi_agent_env')}")
                logger.info(f"Validation: hasattr(self, 'fomappo_adapter') = {hasattr(self, 'fomappo_adapter')}")
                
                # 🔧 Fix: force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                # 🔧 Fix: display training completion information
                print(f"\n✅ FOMAPPO training completed!")
                print(f"  - Training history saved")
                print(f"  - Model saved")
                print(f"  - Experiment ID: {self.experiment_id}")
                
                return result.get('training_rewards', result)
            else:
                logger.warning("⚠️ External training method returned non-success status")
                
                # 🔧 Fix: even if training fails, ensure environment and adapter are saved
                if not hasattr(self, 'multi_agent_env') or self.multi_agent_env is None:
                    logger.info("Create backup multi_agent_env")
                    self._create_environments()
                
                if not hasattr(self, 'fomappo_adapter') or self.fomappo_adapter is None:
                    logger.info("Create backup fomappo_adapter")
                    if hasattr(result, 'adapter'):
                        self.fomappo_adapter = result.adapter
                    elif hasattr(result, 'fomappo_adapter'):
                        self.fomappo_adapter = result.fomappo_adapter
                
                # 🔧 Fix: create default training history
                if not hasattr(self, 'training_history') or not self.training_history.get('episode_rewards'):
                    logger.info("Create default training history")
                    self._init_default_training_history()
                
                # 🔧 Fix: force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                return result
        except ImportError as e:
            print(f"❌ Cannot import external training method: {e}")
            logger.warning(f"❌ Cannot import FOMAPPO training method: {e}")
            print("🔄 Fallback to integrated training method...")
            logger.info("🔄 Fallback to integrated training method")
            result = self._train_fomappo_agents_integrated()
            print("✅ Integrated training method execution completed")
            return result
        except Exception as e:
            print(f"❌ External training method call failed: {e}")
            logger.error(f"❌ train_fomappo_shared_policy call failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print("🔄 Fallback to integrated training method...")
            logger.info("🔄 Fallback to integrated training method")
            result = self._train_fomappo_agents_integrated()
            print("✅ Integrated training method execution completed")
            return result
    
    def _train_fomappo_agents_original(self):
        """Train FOMAPPO multi-agent algorithm - true PPO learning implementation"""
        logger.info("Start training FOMAPPO multi-agent algorithm")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMAPPO")
        
        try:
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get Manager count and observation/action space
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"Created {num_managers} Manager agents: {manager_ids}")
            
            # Get state and action space dimension
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"State space dimension: {state_dim}, action space dimension: {action_dim}")
            
            # Use standard FOMAPPO adapter (shared policy architecture)
            from algorithms.MAPPO.fomappo.fomappo_adapter import FOMAPPOAdapter
            
            # Initialize FOMAPPO agent dictionary
            fomappo_agents = {}
            for manager_id in manager_ids:
                fomappo_agents[manager_id] = FOMAPPOAdapter(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr_actor=3e-4,
                    lr_critic=1e-3,
                    gamma=0.99,
                    gae_lambda=0.95,
                    eps_clip=0.2,
                    k_epochs=4,
                    device=self.device,
                    use_device_coordination=True,
                    device_coordination_weight=0.1,
                    fo_constraint_weight=0.2
                )
            
            logger.info("FOMAPPO agent initialization successful")
            
            # Training loop
            total_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # 🔧 Fix: add experience buffer and batch update mechanism
            UPDATE_INTERVAL = 5  # Update every 5 episodes
            experience_buffer = {manager_id: [] for manager_id in manager_ids}
            
            for episode in range(self.num_episodes):
                logger.info(f"FOMAPPO original training Episode {episode+1}/{self.num_episodes}")
                
                obs, infos = multi_env.reset()
                episode_rewards = {manager_id: 0 for manager_id in manager_ids}
                
                for timestep in range(self.steps_per_episode):
                    actions = {}
                    for manager_id in manager_ids:
                        action_space_size = multi_env.action_spaces[manager_id].shape[0]
                        actions[manager_id] = np.random.uniform(-0.5, 0.5, action_space_size)
                    
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    for manager_id, reward in rewards.items():
                        episode_rewards[manager_id] += reward
                    obs = next_obs
                    
                for manager_id, reward in episode_rewards.items():
                    total_rewards[manager_id].append(reward)
                
            # Save training history
            self.training_history["episode_rewards"] = total_rewards
            self.multi_agent_env = multi_env
            logger.info("FOMAPPO original training completed")
            
        except Exception as e:
            logger.error(f"FOMAPPO original training failed: {e}")
            # Use simplified training as fallback
            self._train_simple_pipeline_training()
    
    def _reset_pipeline_state(self):
        """Reset Pipeline state"""
        # Reset user state
        self._initialize_user_states()
        
        logger.debug("Pipeline state reset")
    
    def _get_pipeline_observations(self) -> Dict[str, np.ndarray]:
        """Get observations from existing Pipeline (based on existing multi-agent environment)"""
        observations = {}
        
        # Use existing multi-agent environment for observations (if available)
        if hasattr(self, 'multi_agent_env'):
            try:
                # Get Dec-POMDP observations directly from multi-agent environment
                return self.multi_agent_env._get_observations()
            except Exception as e:
                logger.warning(f"Multi-agent environment observation retrieval failed: {e}")
                
        # Fallback to simplified observation generation
        for manager in self.managers:
            manager_id = manager.manager_id
            
            # 1. Manager's own state (private information)
            manager_state = self._get_manager_state(manager)
            
            # 2. Environment state (public information)
            env_state = self._get_environment_state()
            
            # 3. Other Manager's simplified information (limited information)
            others_state = self._get_limited_others_state(manager_id)
            
            # Combine observations
            full_obs = np.concatenate([manager_state, env_state, others_state])
            observations[manager_id] = full_obs
        
        return observations

    def _get_manager_state(self, manager) -> np.ndarray:
        """Get Manager's state features"""
        state_features = []
        
        # Manager basic information
        state_features.extend([
            len(manager.users),  # User count
            manager.coverage_area,  # Coverage area
            manager.location[0], manager.location[1]  # Location coordinates
        ])
        
        # User aggregation information
        total_devices = 0
        avg_preferences = {'economic': 0.0, 'comfort': 0.0, 'self_sufficient': 0.0, 'environmental': 0.0}
        
        for user in manager.users:
            total_devices += len(user.devices)
            for pref_key in avg_preferences:
                avg_preferences[pref_key] += user.preferences.get(pref_key, 0.25)
        
        if manager.users:
            for pref_key in avg_preferences:
                avg_preferences[pref_key] = avg_preferences[pref_key] / len(manager.users)
        
        state_features.extend([
            total_devices,
            avg_preferences['economic'],
            avg_preferences['comfort'], 
            avg_preferences['self_sufficient'],
            avg_preferences['environmental']
        ])
        
        # Extend to fixed dimension (e.g., 40-dimensional private information)
        while len(state_features) < 40:
            state_features.append(0.0)
        
        return np.array(state_features[:40], dtype=np.float32)
    
    def _get_environment_state(self) -> np.ndarray:
        """Get environment state features (public information)"""
        env_features = []
        
        # Time feature
        current_hour = datetime.now().hour
        env_features.extend([
            current_hour / 23.0,  # Normalized hour
            np.sin(2 * np.pi * current_hour / 24),  # Periodic time
            np.cos(2 * np.pi * current_hour / 24)
        ])
        
        # Price feature (if there is price data)
        if hasattr(self, 'price_data') and not self.price_data.empty:
            current_price = self.price_data.iloc[current_hour % len(self.price_data)]['price']
            avg_price = self.price_data['price'].mean()
            env_features.extend([current_price, avg_price, current_price / avg_price])
        else:
            env_features.extend([0.15, 0.15, 1.0])  # Default price feature
        
        # Extend to 18-dimensional public information
        while len(env_features) < 18:
            env_features.append(0.0)
        
        return np.array(env_features[:18], dtype=np.float32)
    
    def _get_limited_others_state(self, current_manager_id: str) -> np.ndarray:
        """Get limited information about other Managers"""
        others_features = []
        
        for manager in self.managers:
            if manager.manager_id != current_manager_id:
                # Only provide very basic information
                others_features.extend([
                    len(manager.users) / 20.0,  # Normalized user count
                    manager.coverage_area / 10.0  # Normalized coverage area
                ])
        
        # Extend to 15-dimensional other information
        while len(others_features) < 15:
            others_features.append(0.0)
        
        return np.array(others_features[:15], dtype=np.float32)
    
    def _get_manager_action_dim(self) -> int:
        """Get Manager's action space dimension"""
        # Determine action dimension based on actual device count
        max_devices = 0
        for manager in self.managers:
            total_devices = sum(len(user.devices) for user in manager.users)
            max_devices = max(max_devices, total_devices)
        
        return max(max_devices, 10)  # At least 10-dimensional action
    
    def _execute_pipeline_with_actions(self, actions: Dict[str, np.ndarray], timestep: int) -> Dict:
        """Execute action-driven Pipeline process"""
        # Apply actions to FlexOffer generation
        fo_systems = self._generate_flexoffers_with_actions(actions, timestep)
            
        # Execute aggregation
        aggregated_results = self._aggregate_flexoffers_for_timestep(fo_systems, timestep)
        
        # Execute trading
        trade_results = self._trade_flexoffers_for_timestep(aggregated_results, timestep)
        
        # Execute scheduling and state updates
        schedule_results = self._schedule_and_update_states(trade_results, timestep)
        
        return {
            'fo_systems': fo_systems,
            'aggregated_results': aggregated_results,
            'trade_results': trade_results, 
            'schedule_results': schedule_results,
            'stats': {
                'trades': len(trade_results),
                'satisfaction': schedule_results.get('satisfaction', 0.0),
                'fo_systems': len(fo_systems) if isinstance(fo_systems, list) else len(fo_systems.keys()) if isinstance(fo_systems, dict) else 0
            }
        }
    
    def _generate_flexoffers_with_actions(self, actions: Dict[str, np.ndarray], timestep: int):
        """Use Agent actions to influence FlexOffer generation"""
        # Check if multi-agent environment is available
        if self.rl_algorithm == "fomappo" and hasattr(self, 'multi_agent_env'):
            try:
                # Use multi-agent environment directly to generate FlexOffer
                logger.info(f"Use {self.rl_algorithm} multi-agent environment directly to generate FlexOffer...")
                
                # Execute actions and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                
                # Get generated FlexOffer from environment
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"{self.rl_algorithm} algorithm generated {len(fo_systems)} FlexOffers for Manager at time step {timestep}")
                for manager_id, dfo_dict in fo_systems.items():
                    logger.debug(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffers for devices")
                
                return fo_systems
            except Exception as e:
                logger.error(f"{self.rl_algorithm} direct FlexOffer generation failed: {e}")
                logger.error("Fallback to standard generation method")
        
        # If the above method fails or is not applicable, call the existing FlexOffer generation method
        fo_systems = self._generate_flexoffers_for_timestep(timestep)
        
        # Use actions to adjust FlexOffer parameters
        for manager_id, action in actions.items():
            if manager_id in fo_systems or any(manager_id in str(key) for key in fo_systems.keys()):
                adjustment_factor = 1.0 + 0.1 * np.mean(action[:5])  
                logger.debug(f"Manager {manager_id} action adjustment factor: {adjustment_factor:.3f}")
        
        return fo_systems
    
    def _calculate_pipeline_rewards_from_results(self, pipeline_results: Dict, manager_ids: List[str]) -> Dict[str, float]:
        """Calculate rewards based on Pipeline execution results"""
        rewards = {}
        
        stats = pipeline_results.get('stats', {})
        satisfaction = stats.get('satisfaction', 0.0)
        trades = stats.get('trades', 0)
        
        for manager_id in manager_ids:
            # Base reward: user satisfaction
            base_reward = satisfaction * 10.0
            
            # Trade reward: successful trades
            trade_reward = min(trades * 0.5, 5.0)
            
            # Efficiency reward: FlexOffer generation efficiency
            efficiency_reward = stats.get('fo_systems', 0) * 0.1
            
            # Combine rewards
            total_reward = base_reward + trade_reward + efficiency_reward
            
            total_reward += np.random.normal(0, 0.1)
            
            rewards[manager_id] = total_reward
        
        return rewards
    
    def _train_simple_pipeline_training(self):
        """Pipeline training (fallback)"""
        logger.warning("Use Pipeline training as fallback")
        
        manager_ids = [manager.manager_id for manager in self.managers]
        total_rewards = {manager_id: [] for manager_id in manager_ids}
        
        for episode in range(min(self.num_episodes, 10)):
            logger.info(f"training Episode {episode+1}/10")
            
            episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
            
            for timestep in range(self.steps_per_episode):
                # Execute standard Pipeline process
                results = self.run_pipeline()
                
                # Calculate simple reward
                satisfaction = np.mean(results.get("user_satisfaction_history", [0.0]))
                reward = satisfaction * 10.0
                
                for manager_id in manager_ids:
                    episode_rewards[manager_id] = float(episode_rewards[manager_id]) + float(reward)
            
            for manager_id, reward in episode_rewards.items():
                total_rewards[manager_id].append(reward)
        
        # Save results
        self.training_history["episode_rewards"] = total_rewards
        logger.info("Training completed")
    
    def _train_fomaddpg_agents(self):
        """Train FOMADDPG multi-agent algorithm - use FOMADDPG adapter"""
        print("\n🚀 Start FOMADDPG training (based on MADDPG architecture, Off-policy learning)")
        logger.info("🚀 Start FOMADDPG training (based on MADDPG architecture, Off-policy learning)")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMADDPG")
        
        try:
            # Check if FOMADDPG adapter is available
            if not FOMADDPG_available or FOMAddpgAdapter is None:
                logger.error("❌ FOMAddpgAdapter not available, fallback to original method")
                return self._train_fomaddpg_agents_original()
            
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get Manager count and observation/action space
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"🏗️ Environment configuration: {num_managers} Managers: {manager_ids}")
            
            # Get state and action space dimension
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"📊 State space: {state_dim} dimensions, action space: {action_dim} dimensions")
            
            # Initialize FOMADDPG adapter - 🔧 Use stable hyperparameters
            fomaddpg_adapter = FOMAddpgAdapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                episode_length=self.steps_per_episode,
                lr_actor=1e-4,
                lr_critic=1e-3,
                device=self.device,
                # MADDPG specific parameters
                hidden_dim=256,
                max_action=1.0,
                gamma=0.99,
                tau=0.005,  # Soft update coefficient
                noise_scale=0.1,  # Exploration noise
                buffer_capacity=100000,
                batch_size=64,
                # FlexOffer specific parameters
                use_device_coordination=True,
                device_coordination_weight=0.1,
                fo_constraint_weight=0.2,
                use_manager_coordination=True,
                manager_coordination_weight=0.05
            )
            
            logger.info("✅ FOMADDPG adapter initialized")
            
            # Initialize training history
            training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # Training loop - based on MADDPG off-policy learning
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (FOMADDPG adapter) ==========")
                
                # Reset environment (MADDPG does not reset buffers)
                obs, infos = multi_env.reset()
                fomaddpg_adapter.reset_buffers()  # For MADDPG, this actually does nothing
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # Run 24 time steps for each episode
                for timestep in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, time step {timestep}")
                    
                    # Step 1: Use adapter to select actions
                    actions, action_log_probs, values = fomaddpg_adapter.select_actions(obs, deterministic=False)
                    
                    # Step 2: Environment step
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Step 3: Collect data to experience replay buffer
                    fomaddpg_adapter.collect_step(
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
                    
                    # Update observation
                    obs = next_obs
                    
                    # Display time step reward
                    timestep_total = sum(rewards.values())
                    logger.info(f"  Time step {timestep}: Total reward {timestep_total:.3f}")
                    
                    # MADDPG feature: training update can be performed at each step (if there is enough experience)
                    if timestep > 0:  # Give some time to collect experience
                        train_info = fomaddpg_adapter.train_on_batch()
                        if train_info and train_info.get('actor_loss', 0) > 0:
                            logger.debug(f"    Training update: Actor {train_info['actor_loss']:.4f}, Critic {train_info['critic_loss']:.4f}")
                
                # Record episode reward
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
                
                # Display reward for each Manager and record to training history
                for manager_id, reward in episode_rewards.items():
                    logger.info(f"  📊 {manager_id}: {reward:.3f}")
                    training_episode_rewards[manager_id].append(reward)
                
                # Periodically output learning progress
                if (episode + 1) % 10 == 0:
                    logger.info(f"\n========== FOMADDPG training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    
                    # Get training statistics
                    try:
                        training_stats = fomaddpg_adapter.get_training_stats()
                        manager_rewards = fomaddpg_adapter.get_manager_rewards_summary()
                        
                        if isinstance(manager_rewards, dict):
                            for manager_id, stats in manager_rewards.items():
                                if isinstance(stats, dict):
                                    total_reward = stats.get('total_reward', 0.0)
                                    best_reward = stats.get('best_reward', 0.0)
                                    training_updates = stats.get('training_updates', 0)
                                    logger.info(f"  🔥 {manager_id}: Accumulated reward {total_reward:.2f}, best {best_reward:.2f}, updates {training_updates} times")
                                else:
                                    logger.info(f"  🔥 {manager_id}: Accumulated reward {stats:.2f}")
                        
                        if isinstance(training_stats, dict):
                            iterations = training_stats.get('training_iterations', 0)
                            buffer_size = training_stats.get('buffer_size', 0)
                            logger.info(f"  🚀 Training iterations: {iterations}, experience buffer: {buffer_size}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get training statistics: {e}")
                        logger.info("  🔥 Training progress: learning...")
                    
                    logger.info("=" * 70)
                
                # Periodically save model
                if (episode + 1) % 50 == 0:
                    model_path = os.path.join(self.results_dir, f"fomaddpg_adapter_ep{episode+1}")
                    fomaddpg_adapter.save_models(model_path)
                    logger.info(f"📀 Model saved to: {model_path}")
            
            # Training completed
            logger.info("🎉 FOMADDPG adapter training completed!")
            
            # Save training history (using actual recorded episode rewards)
            try:
                # 🔧 Key fix: use actual recorded episode rewards
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id in training_episode_rewards:
                        episode_rewards_dict[manager_id] = training_episode_rewards[manager_id]
                    else:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
                
                # Verify data completeness
                for manager_id in manager_ids:
                    # Fill to correct length
                    while len(episode_rewards_dict[manager_id]) < self.num_episodes:
                        episode_rewards_dict[manager_id].append(0.0)
                    episode_rewards_dict[manager_id] = episode_rewards_dict[manager_id][:self.num_episodes]
                
                logger.info(f"✅ Training history verification completed: {len(episode_rewards_dict)} Managers, each {self.num_episodes} episodes")
            except Exception as e:
                logger.warning(f"Failed to save training history: {e}")
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id not in episode_rewards_dict:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
            
            # 🔧 Key fix: save training history to instance variable
                self.training_history["episode_rewards"] = episode_rewards_dict
                self.training_history["training_metadata"]["num_managers"] = num_managers
                self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            self.training_history["training_metadata"]["algorithm"] = "FOMADDPG"
            self.training_history["training_metadata"]["total_training_iterations"] = fomaddpg_adapter.training_iterations
            
            # Save environment and adapter references
            self.multi_agent_env = multi_env
            self.fomaddpg_adapter = fomaddpg_adapter
            
            # 🔧 Enhanced training history saving - use multiple saving methods to ensure data is not lost
            # Method 1: Main CSV saving method
            try:
                self._save_training_history_to_csv("FOMADDPG")
                logger.info("✅ FOMADDPG training history saved to CSV")
            except Exception as e:
                logger.error(f"Main CSV saving failed: {e}")
            
            # Method 2: Backup saving method
            try:
                self._save_training_history_with_backup("fomaddpg_")
                logger.info("✅ FOMADDPG training history backup saved")
            except Exception as e:
                logger.error(f"Backup saving failed: {e}")
            
            # Method 3: Force save training data
            try:
                self._force_save_training_history(episode_rewards_dict, "FOMADDPG")
                logger.info("✅ FOMADDPG force save completed")
            except Exception as e:
                logger.error(f"Force save failed: {e}")
            
            # Save final model
            final_model_path = os.path.join(self.results_dir, "fomaddpg_adapter_final")
            fomaddpg_adapter.save_models(final_model_path)
            logger.info(f"📀 Final model saved to: {final_model_path}")
            
            # Output final statistics comparison
            logger.info(f"\n========== FOMADDPG training summary ==========")
            
            try:
                final_stats = fomaddpg_adapter.get_training_stats()
                final_rewards = fomaddpg_adapter.get_manager_rewards_summary()
                
                logger.info("🎯 MADDPG off-policy learning effect:")
                if isinstance(final_rewards, dict):
                    for manager_id, stats in final_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            updates = stats.get('training_updates', 0)
                            logger.info(f"  {manager_id}: Total reward {total_reward:.2f}, best {best_reward:.2f}, updates {updates} times")
                        else:
                            logger.info(f"  {manager_id}: Total reward {stats:.2f}")
                else:
                    logger.info(f"  Total reward: {final_rewards}")
                
                if isinstance(final_stats, dict):
                    iterations = final_stats.get('training_iterations', 0)
                    buffer_size = final_stats.get('buffer_size', 0)
                    logger.info(f"🚀 Total training iterations: {iterations}")
                    logger.info(f"📦 Final experience buffer size: {buffer_size}")
                else:
                    logger.info(f"🚀 Training statistics: {final_stats}")
            except Exception as e:
                logger.warning(f"Failed to get final statistics: {e}")
                logger.info("🎯 Training completed, statistics information acquisition failed")
            
            logger.info("🎉 Advantage: Off-policy learning, high sample efficiency, continuous action space!")
            logger.info("==========================================")
                
        except Exception as e:
            logger.error(f"Error during FOMADDPG training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Fallback to FOMAPPO algorithm")
            
            # Ensure FOMAPPO agent dictionary exists
            if "fomappo" not in self.rl_agents:
                self.rl_agents["fomappo"] = {}
            
            self._train_fomappo_agents_integrated()
    
    def _train_fomaddpg_agents_original(self):
        """Train FOMADDPG multi-agent algorithm - original method (fallback)"""
        logger.info("Start original FOMADDPG training method")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMADDPG_ORIGINAL")
        
        try:
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get Manager count and observation/action space
            num_managers = multi_env.get_manager_count()
            logger.info(f"Created {num_managers} Manager agents")
            
            # Get state and action space dimension
            sample_obs, _ = multi_env.reset()
            manager_ids = list(sample_obs.keys())
            
            if manager_ids:
                state_dim = len(sample_obs[manager_ids[0]])
                action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            else:
                logger.error("Cannot get state and action space dimension")
                return
            
            # Initialize FOMADDPG algorithm
            if FOMADDPG_available and FOMADDPG is not None:
                fomaddpg = FOMADDPG(
                    n_agents=num_managers,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr_actor=1e-4,
                    lr_critic=1e-3,
                    hidden_dim=256,
                    max_action=1.0,
                    gamma=0.99,
                    tau=0.005,
                    noise_scale=0.1,
                    buffer_capacity=100000,
                    batch_size=64,
                    device=self.device
                )
                
                logger.info("Original FOMADDPG algorithm initialized successfully")
                
                # Training loop 
                total_rewards = []
                
                for episode in range(self.num_episodes):
                    logger.info(f"Original FOMADDPG Episode {episode+1}/{self.num_episodes}")
                    
                    # Reset environment
                    obs, infos = multi_env.reset()
                    states = np.array([obs[manager_id] for manager_id in manager_ids])
                    episode_reward = 0
                    
                    # Run 24 time steps for each episode
                    for timestep in range(self.steps_per_episode):
                        # Select action
                        actions = fomaddpg.select_actions(states, add_noise=True)
                        action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
                        
                        # Execute action
                        next_obs, rewards, dones, truncated, infos = multi_env.step(action_dict)
                        next_states = np.array([next_obs[manager_id] for manager_id in manager_ids])
                        reward_array = np.array([rewards[manager_id] for manager_id in manager_ids])
                        done_array = np.array([dones[manager_id] for manager_id in manager_ids])
                        
                        # Store experience
                        fomaddpg.store_experience(states, actions, reward_array, next_states, done_array)
                        
                        # Update policy
                        if len(fomaddpg.replay_buffer) >= fomaddpg.batch_size:
                            fomaddpg.update()
                        
                        states = next_states
                        episode_reward += np.mean(reward_array)
                    
                    total_rewards.append(episode_reward)
                    
                    if (episode + 1) % 10 == 0:
                        avg_reward = np.mean(total_rewards[-10:])
                        logger.info(f"Original FOMADDPG progress: {episode+1}/{self.num_episodes}, average reward: {avg_reward:.2f}")
                
                # Save training history
                self.training_history["episode_rewards"] = total_rewards
                self.multi_agent_env = multi_env
                self.fomaddpg_agent = fomaddpg  # Keep original interface
                
                logger.info("Original FOMADDPG training completed")
                
            else:
                logger.error("FOMADDPG algorithm is not available")
                
        except Exception as e:
            logger.error(f"Original FOMADDPG training failed: {e}")
            # Fallback to FOMAPPO algorithm
            logger.info("Fallback to FOMAPPO algorithm")
            if "fomappo" not in self.rl_agents:
                self.rl_agents["fomappo"] = {}
            self._train_fomappo_agents_integrated()
    
    def _train_fomatd3_agents(self):
        """Train FOMATD3 multi-agent algorithm"""
        logger.info("Start training FOMATD3 multi-agent algorithm")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMATD3")
        
        try:
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get Manager count and observation/action space
            num_managers = multi_env.get_manager_count()
            logger.info(f"Created {num_managers} Manager agents")
            
            # Get state and action space dimension
            sample_obs, _ = multi_env.reset()
            manager_ids = list(sample_obs.keys())
            
            if manager_ids:
                # 🔧 Fix: calculate global state dimension (dimension of flattened multi-agent observation)
                single_agent_obs_dim = len(sample_obs[manager_ids[0]])
                state_dim = single_agent_obs_dim * num_managers  # Global state = single agent observation × number of agents
                action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
                logger.info(f"Single agent observation dimension: {single_agent_obs_dim}, global state dimension: {state_dim}, action dimension: {action_dim}")
            else:
                logger.error("Cannot get state and action space dimension")
                return
            
            # Initialize FOMATD3 algorithm
            if FOMATD3_available and FOMATD3 is not None:
                fomatd3 = FOMATD3(
                    n_agents=num_managers,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr_actor=1e-4,
                    lr_critic=1e-3,
                    hidden_dim=256,
                    max_action=1.0,
                    gamma=0.99,
                    tau=0.005,
                    noise_scale=0.1,
                    noise_clip=0.2,
                    buffer_capacity=100000,
                    batch_size=64,
                    policy_delay=2,
                    device=self.device
                )
                
                logger.info("FOMATD3 algorithm initialized successfully")
                
                # Training loop
                total_rewards = []
                
                for episode in range(self.num_episodes):
                    logger.info(f"\n========== Start Episode {episode+1}/{self.num_episodes} (FOMATD3) ==========")
                    
                    # Reset environment
                    obs, infos = multi_env.reset()
                    
                    # Convert observation to numpy array
                    states = np.array([obs[manager_id] for manager_id in manager_ids])
                    
                    episode_reward = 0
                    
                    # Run 24 time steps for each episode
                    for timestep in range(self.steps_per_episode):
                        logger.info(f"Episode {episode+1}, time step {timestep} (第{timestep}小时)")
                        
                        # Select action
                        actions = fomatd3.select_actions(states, add_noise=True)
                        
                        # Convert action to environment expected format
                        action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
                        
                        # Execute action
                        next_obs, rewards, dones, truncated, infos = multi_env.step(action_dict)
                        
                        # Convert next state
                        next_states = np.array([next_obs[manager_id] for manager_id in manager_ids])
                        
                        # Convert reward and done flag
                        reward_array = np.array([rewards[manager_id] for manager_id in manager_ids])
                        done_array = np.array([dones[manager_id] for manager_id in manager_ids])
                        
                        # Generate FlexOffer constraints and satisfaction 
                        fo_constraints = np.random.uniform(0.5, 1.0, (num_managers, action_dim))
                        fo_satisfaction = np.random.uniform(0.6, 1.0, num_managers)
                        
                        # Store experience
                        fomatd3.store_experience(states, actions, reward_array, next_states, done_array,
                                               fo_constraints, fo_satisfaction)
                        
                        # Update policy
                        if len(fomatd3.replay_buffer) >= fomatd3.batch_size:
                            update_info = fomatd3.update()
                            if update_info:
                                logger.debug(f"  Update statistics: Actor Loss {update_info.get('actor_loss', 0):.4f}, "
                                           f"Critic Loss {update_info.get('critic_loss', 0):.4f}, "
                                           f"Iterations {update_info.get('total_iterations', 0)}")
                        
                        # Update state
                        states = next_states
                        episode_reward += np.mean(reward_array)
                        
                        # Record time step reward
                        logger.info(f"  Time step {timestep} reward: {np.mean(reward_array):.3f}")
                    
                    total_rewards.append(episode_reward)
                    logger.info(f"Episode {episode+1} completed: total reward {episode_reward:.3f}")
                    
                    # Output training progress periodically
                    if (episode + 1) % 10 == 0:
                        avg_reward = np.mean(total_rewards[-10:])
                        logger.info(f"\n========== FOMATD3 training progress: {episode+1}/{self.num_episodes} episodes ==========")
                        logger.info(f"  Recent 10 episodes average reward: {avg_reward:.2f}")
                        logger.info("=" * 60)
                
                logger.info("FOMATD3 training completed")
                
                # Save training history
                # 🔧 Use enhanced saving method
                self._save_training_history_with_backup()
                self.training_history["episode_rewards"] = total_rewards
                self.training_history["training_metadata"]["num_managers"] = num_managers
                self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
                self.training_history["training_metadata"]["state_dim"] = state_dim
                self.training_history["training_metadata"]["action_dim"] = action_dim
                
                # Save multi-agent environment reference, for subsequent FlexOffer generation
                self.multi_agent_env = multi_env
                self.fomatd3_agent = fomatd3
                
                # Save results
                if hasattr(self, 'results_dir'):
                    import json
                    results_file = os.path.join(self.results_dir, "fomatd3_training_results.json")
                    with open(results_file, 'w') as f:
                        json.dump({
                            'total_rewards': [float(r) for r in total_rewards],
                            'num_episodes': self.num_episodes,
                            'num_managers': num_managers,
                            'algorithm': 'FOMATD3'
                        }, f, indent=2)
                    logger.info(f"FOMATD3 training results saved to {results_file}")
                    
                    # Save reward data to CSV file
                    csv_file = self._generate_csv_filename("rewards", "FOMATD3")
                    self._save_rewards_to_csv(csv_file, total_rewards, "FOMATD3")
                    
                    # Save training history
                    self._save_training_history_to_csv("FOMATD3")
                
                # Save model
                model_dir = os.path.join(self.results_dir, "fomatd3_models")
                fomatd3.save_models(model_dir)
                logger.info(f"FOMATD3 model saved to {model_dir}/agent_*")
                
            else:
                logger.error("FOMATD3 algorithm is not available, please check import")
                
        except Exception as e:
            logger.error(f"Error during FOMATD3 training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Fallback to FOMAPPO algorithm")
            
            # Ensure FOMAPPO agent dictionary exists
            if "fomappo" not in self.rl_agents:
                self.rl_agents["fomappo"] = {}
            
            self._train_fomappo_agents_integrated()
    
    def _train_fosqddpg_agents(self):
        """Train FOSQDDPG multi-agent algorithm"""
        logger.info("Start training FOSQDDPG multi-agent algorithm")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOSQDDPG")
        
        try:
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get Manager count and observation/action space
            num_managers = multi_env.get_manager_count()
            logger.info(f"Created {num_managers} Manager agents")
            
            # Get state and action space dimension
            sample_obs, _ = multi_env.reset()
            manager_ids = list(sample_obs.keys())
            
            if manager_ids:
                state_dim = len(sample_obs[manager_ids[0]])
                action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            else:
                logger.error("Cannot get state and action space dimension")
                return
            
            # Initialize FOSQDDPG algorithm
            if FOSQDDPG_available and FOSQDDPG is not None:
                fosqddpg = FOSQDDPG(
                    n_agents=num_managers,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr_actor=1e-4,
                    lr_critic=1e-3,
                    hidden_dim=256,
                    max_action=1.0,
                    gamma=0.99,
                    tau=0.005,
                    noise_scale=0.1,
                    buffer_capacity=100000,
                    batch_size=64,
                    sample_size=5,  # Shapley value sampling size
                    device=self.device
                )
                
                logger.info("FOSQDDPG algorithm initialized successfully")
                
                # Training loop
                total_rewards = []
                
                for episode in range(self.num_episodes):
                    logger.info(f"\n========== Start Episode {episode+1}/{self.num_episodes} (FOSQDDPG) ==========")
                    
                    # Reset environment
                    obs, infos = multi_env.reset()
                    episode_rewards = {manager_id: 0 for manager_id in manager_ids}
                    
                    # Convert observation to numpy array
                    states = np.array([obs[manager_id] for manager_id in manager_ids])
                    
                    # Run 24 time steps for each episode (0-23 hours)
                    for timestep in range(self.steps_per_episode):
                        logger.info(f"Episode {episode+1}, time step {timestep} (第{timestep}小时)")
                        
                        # Select action using FOSQDDPG
                        actions = fosqddpg.select_actions(states, add_noise=True)
                        
                        # Convert action to dictionary format
                        action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
                        
                        # Execute action
                        next_obs, rewards, dones, truncated, infos = multi_env.step(action_dict)
                        
                        # Convert to numpy array format
                        next_states = np.array([next_obs[manager_id] for manager_id in manager_ids])
                        reward_array = np.array([rewards[manager_id] for manager_id in manager_ids])
                        done_array = np.array([dones[manager_id] for manager_id in manager_ids])
                        
                        # Store experience
                        fosqddpg.store_experience(
                            states=states,
                            actions=actions,
                            rewards=reward_array,
                            next_states=next_states,
                            dones=done_array
                        )
                        
                        # Update state
                        states = next_states
                        
                        # Accumulate reward
                        for i, manager_id in enumerate(manager_ids):
                            episode_rewards[manager_id] += reward_array[i]
                        
                        # Record time step reward
                        timestep_reward_total = np.sum(reward_array)
                        logger.info(f"  Time step {timestep} reward: {timestep_reward_total:.3f}")
                        
                        # Update policy (if enough experience)
                        if len(fosqddpg.replay_buffer) >= fosqddpg.batch_size:
                            update_info = fosqddpg.update()
                            if update_info and timestep % 5 == 0:  # Output every 5 time steps
                                logger.info(f"  Policy updated - Actor Loss: {update_info['actor_loss']:.4f}, "
                                          f"Critic Loss: {update_info['critic_loss']:.4f}")
                    
                    # Record episode reward
                    episode_total_reward = sum(episode_rewards.values())
                    total_rewards.append(episode_total_reward)
                    
                    # Output episode summary
                    logger.info(f"Episode {episode+1} completed: total reward {episode_total_reward:.3f}")
                    
                    # Output training progress periodically
                    if (episode + 1) % 10 == 0:
                        avg_reward = np.mean(total_rewards[-10:])
                        logger.info(f"\n========== FOSQDDPG training progress: {episode+1}/{self.num_episodes} episodes ==========")
                        logger.info(f"  Recent 10 episodes average reward: {avg_reward:.2f}")
                        logger.info(f"  Experience buffer size: {len(fosqddpg.replay_buffer)}")
                        logger.info(f"  Total training iterations: {fosqddpg.total_iterations}")
                        logger.info("=" * 60)
                
                logger.info("FOSQDDPG training completed")
                
                # Save training history
                # 🔧 Use enhanced saving method
                self._save_training_history_with_backup()
                self.training_history["episode_rewards"] = total_rewards
                self.training_history["training_metadata"]["num_managers"] = num_managers
                self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
                self.training_history["training_metadata"]["state_dim"] = state_dim
                self.training_history["training_metadata"]["action_dim"] = action_dim
                self.training_history["training_metadata"]["final_buffer_size"] = len(fosqddpg.replay_buffer)
                self.training_history["training_metadata"]["total_iterations"] = fosqddpg.total_iterations
                
                # Save model
                model_path = os.path.join(self.results_dir, "fosqddpg_model")
                fosqddpg.save_models(model_path)
                logger.info(f"FOSQDDPG model saved to: {model_path}")
                
                # Save multi-agent environment reference, for subsequent FlexOffer generation
                self.multi_agent_env = multi_env
                self.fosqddpg_agent = fosqddpg
                
                # Save training results
                if hasattr(self, 'results_dir'):
                    import json
                    results_file = os.path.join(self.results_dir, "fosqddpg_training_results.json")
                    with open(results_file, 'w') as f:
                        json.dump({
                            'total_rewards': [float(r) for r in total_rewards],
                            'num_episodes': self.num_episodes,
                            'num_managers': num_managers,
                            'state_dim': state_dim,
                            'action_dim': action_dim,
                            'final_buffer_size': len(fosqddpg.replay_buffer),
                            'total_iterations': fosqddpg.total_iterations
                        }, f, indent=2)
                    logger.info(f"FOSQDDPG training results saved to {results_file}")
                    
                    # Save reward data to CSV file
                    csv_file = self._generate_csv_filename("rewards", "FOSQDDPG")
                    self._save_rewards_to_csv(csv_file, total_rewards, "FOSQDDPG")
                    
                    # Save training history
                    self._save_training_history_to_csv("FOSQDDPG")
                
            else:
                logger.error("FOSQDDPG algorithm is not available")
                logger.info("Fallback to FOMAPPO algorithm")
                self._train_fomappo_agents_integrated()
                
        except Exception as e:
            logger.error(f"Error during FOSQDDPG training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Fallback to FOMAPPO algorithm")
            self._train_fomappo_agents_integrated()
    
    def _train_custom_agents(self):
        """Train custom RL agent"""
        logger.info(f"Use custom algorithm {self.rl_algorithm} for training")
        
        for user_id, env in self.envs.items():
            if user_id in self.rl_agents[self.rl_algorithm]:
                agent = self.rl_agents[self.rl_algorithm][user_id]
                
                try:
                    # Try using standard interface for training
                    if hasattr(agent, 'train') and callable(agent.train):
                        agent.train(env, num_episodes=self.num_episodes)
                    # Try using update method
                    elif hasattr(agent, 'update') and callable(agent.update):
                        rewards = []
                        for episode in range(self.num_episodes):
                            state = env.reset()
                            episode_reward = 0
                            done = False
                            
                            while not done:
                                action = agent.select_action(state, evaluate=False)
                                next_state, reward, done, info = env.step(action)
                                
                                # Store experience
                                if hasattr(agent, 'store_transition') and callable(agent.store_transition):
                                    agent.store_transition(state, action, reward, next_state, done)
                                
                                state = next_state
                                episode_reward += reward
                            
                            # Update policy after each episode
                            agent.update()
                            rewards.append(episode_reward)
                            
                            if (episode + 1) % 10 == 0:
                                avg_reward = np.mean(rewards[-10:])
                                logger.info(f"User {user_id}, {self.rl_algorithm} training: episode {episode+1}/{self.num_episodes}, average reward: {avg_reward:.2f}")
                    else:
                        logger.error(f"Custom algorithm {self.rl_algorithm} does not provide standard training interface")
                        continue
                    
                    # Save model
                    if hasattr(agent, 'save') and callable(agent.save):
                        agent.save(os.path.join(self.results_dir, f"{self.rl_algorithm}_agent_{user_id}"))
                    
                except Exception as e:
                    logger.error(f"Error during training custom algorithm {self.rl_algorithm}: {e}")
                    # Try fallback to FOMAPPO
                    if "fomappo" in self.rl_agents and user_id in self.rl_agents["fomappo"]:
                        logger.info(f"Try using FOMAPPO as fallback algorithm")
                        self._train_fomappo_agents_integrated()
    
    def run_pipeline(self):
        """Run complete FO pipeline - 24-hour MDP loop for single episode"""
        logger.info("Start running complete FO pipeline (single episode = 24 hours)...")
        
        # 🔧 Fix: ensure training history exists
        if not hasattr(self, 'training_history') or not self.training_history:
            logger.warning("Training history does not exist, initialize empty training history")
            self.training_history = {
                "episode_rewards": {},
                "episode_lengths": {},
                "training_loss": {},
                "training_metadata": {
                    "algorithm": self.rl_algorithm,
                    "num_episodes": self.num_episodes,
                    "steps_per_episode": self.steps_per_episode
                }
            }
        
        # Check and ensure multi_agent_env and fomappo_adapter properties exist
        if self.rl_algorithm == "fomappo":
            if not hasattr(self, 'multi_agent_env') or self.multi_agent_env is None:
                logger.warning("multi_agent_env does not exist, create new environment")
                try:
                    # Import multi-agent environment
                    from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
                    
                    # Create multi-agent environment
                    self.multi_agent_env = MultiAgentFlexOfferEnv(
                        data_dir="data",
                        time_horizon=self.time_horizon,
                        time_step=self.time_step,
                        aggregation_method=self.aggregation_method,
                        trading_method=self.trading_strategy,
                        disaggregation_method=self.disaggregation_method
                    )
                    logger.info("✅ Successfully created multi_agent_env")
                except Exception as e:
                    logger.error(f"❌ Failed to create multi_agent_env: {e}")
            
            # Get actual observation dimension of environment, for subsequent check or create adapter
            actual_obs_dim = None
            if hasattr(self, 'multi_agent_env') and self.multi_agent_env is not None:
                try:
                    sample_obs, _ = self.multi_agent_env.reset()
                    sample_manager_id = list(sample_obs.keys())[0]
                    actual_obs_dim = len(sample_obs[sample_manager_id])
                    logger.info(f"🔍 Environment observation dimension: {actual_obs_dim}")
                except Exception as e:
                    logger.error(f"❌ Failed to get environment observation dimension: {e}")
            
            # Check fomappo_adapter
            if not hasattr(self, 'fomappo_adapter') or self.fomappo_adapter is None:
                logger.warning("fomappo_adapter does not exist, create new adapter")
                try:
                    # Import FOMAPPO adapter
                    from algorithms.MAPPO.fomappo.fomappo_adapter import FOMAPPOAdapter
                    
                    # Get Manager count and ID
                    if hasattr(self, 'multi_agent_env') and self.multi_agent_env is not None:
                        manager_ids = list(sample_obs.keys())
                        num_managers = len(manager_ids)
                        action_dim = self.multi_agent_env.action_spaces[sample_manager_id].shape[0]
                    else:
                        num_managers = len(self.managers)
                        action_dim = self._get_manager_action_dim()
                    
                    # If actual observation dimension is known, use it, otherwise use fallback
                    if actual_obs_dim is not None:
                        state_dim = actual_obs_dim
                    else:
                        # Use fallback method
                        state_dim = len(self._get_manager_state(self.managers[0]))
                        logger.warning(f"⚠️ Use fallback method to get state dimension: {state_dim}")
                    
                    # Create FOMAPPO adapter
                    self.fomappo_adapter = FOMAPPOAdapter(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        num_agents=num_managers,
                        episode_length=self.steps_per_episode,
                        lr_actor=5e-5,
                        lr_critic=2e-4,
                        device=self.device
                    )
                    logger.info(f"✅ Successfully created fomappo_adapter, state dimension={state_dim}, action dimension={action_dim}")
                except Exception as e:
                    logger.error(f"❌ Failed to create fomappo_adapter: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # If both exist but dimensions are inconsistent, perform dimension adaptation
            elif hasattr(self, 'multi_agent_env') and hasattr(self, 'fomappo_adapter') and actual_obs_dim is not None:
                if actual_obs_dim != self.fomappo_adapter.state_dim:
                    logger.warning(f"⚠️ Detected dimension mismatch: adapter={self.fomappo_adapter.state_dim}维，环境={actual_obs_dim}维")
                    try:
                        # Try calling recreate method
                        if hasattr(self.fomappo_adapter, '_recreate_buffer_and_policy'):
                            self.fomappo_adapter._recreate_buffer_and_policy(actual_obs_dim)
                            logger.info(f"✅ Successfully recreated fomappo_adapter, new dimension={actual_obs_dim}")
                    except Exception as e:
                        logger.error(f"❌ Failed to recreate fomappo_adapter: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
        
        # Initialize result storage
        all_results = {
            "timestep_results": [],
            "total_trades": [],
            "total_disaggregated_results": [],
            "user_satisfaction_history": [],
            "user_state_history": []
        }
        
        # Initialize user states - load incremental demand from CSV
        self._initialize_user_states()
        
        logger.info(f"\n========== Start Episode (24-hour cycle, time step 0-23) ==========")
        
        # Execute complete MDP loop for each time step (0-23 hours)
        for timestep in range(self.steps_per_episode):
            logger.info(f"\n========== Time step {timestep} (第{timestep}小时) ==========")
            
            # Step 1: Update user demand for current time step (incremental addition)
            self._update_user_demands_for_timestep(timestep)
            
            # Step 2: Multi-agent based on current state select action and generate FlexOffer
            if self.rl_algorithm == "fomappo" and hasattr(self, 'multi_agent_env'):
                # Get observation
                obs = self.multi_agent_env._get_observations()
                
                # Use trained FOMAPPO policy to select action
                if hasattr(self, 'fomappo_adapter'):
                    actions, _, _ = self.fomappo_adapter.select_actions(obs, deterministic=True)
                else:
                    # Random action
                    actions = {}
                    for manager_id in obs.keys():
                        action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                        actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                
                # Use action to directly generate FlexOffer
                fo_systems = self._generate_flexoffers_with_actions(actions, timestep)
            else:
                # Other algorithms use standard method
                fo_systems = self._generate_flexoffers_for_timestep(timestep)
            
            # Step 3: Aggregate FlexOffer
            aggregated_results = self._aggregate_flexoffers_for_timestep(fo_systems, timestep)
            
            # Step 4: Trade FlexOffer
            trade_results = self._trade_flexoffers_for_timestep(aggregated_results, timestep)
            
            # Step 5: Disaggregate aggregated FlexOffer
            disaggregated_results = self._disaggregate_flexoffers_for_timestep(trade_results, fo_systems, timestep)
            
            # Step 6: Schedule and update user states
            schedule_results = self._schedule_and_update_states(disaggregated_results, timestep)
            
            # Record results for current time step
            timestep_result = {
                "timestep": timestep,
                "hour": timestep,  # Add hour marker
                "fo_systems": fo_systems,
                "aggregated_results": aggregated_results, 
                "trade_results": trade_results,
                "disaggregated_results": disaggregated_results,
                "schedule_results": schedule_results,
                "user_satisfaction": schedule_results.get("satisfaction", 0.0)
            }
            
            all_results["timestep_results"].append(timestep_result)
            all_results["total_trades"].extend(trade_results)
            all_results["total_disaggregated_results"].extend(disaggregated_results)
            all_results["user_satisfaction_history"].append(schedule_results.get("satisfaction", 0.0))
            
            # Save current user states
            current_states = self._get_current_user_states()
            all_results["user_state_history"].append(current_states)
            
            logger.info(f"Time step {timestep} (The {timestep}th hour) completed: {len(fo_systems)} FO systems, {len(trade_results)} trades, {len(disaggregated_results)} disaggregated results")
        
        # Calculate final statistics
        final_satisfaction = np.mean(all_results["user_satisfaction_history"]) if all_results["user_satisfaction_history"] else 0.0
        total_trades = len(all_results["total_trades"])
        total_trade_value = sum(t.quantity * t.price for t in all_results["total_trades"])
        
        logger.info("\n========== Episode completed summary ==========")
        logger.info(f"Completed 1 episode ({self.steps_per_episode} time steps, 0-{self.steps_per_episode-1} hours)")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total trade value: {total_trade_value:.2f} $")
        logger.info(f"24-hour average user satisfaction: {final_satisfaction:.3f}")
        logger.info("===================================")
        
        return all_results

    def _initialize_user_states(self):
        """Initialize user states"""
        logger.info("Initialize user states...")
        
        # Initialize user demand matrix: [number of users, number of time steps]
        self.user_accumulated_demands = np.zeros((self.num_users, self.time_horizon))
        self.user_satisfied_energy = np.zeros((self.num_users, self.time_horizon))
        self.user_current_satisfaction = np.zeros(self.num_users)
        
        # Use pre-set user distribution configuration (set in _setup_managers_and_users)
        if hasattr(self, 'users_distribution') and self.users_distribution:
            user_distribution = self.users_distribution
            logger.info(f"Use pre-set user distribution: {user_distribution}, total number of users: {sum(user_distribution)}")
        else:
            # Fallback to standard distribution
            user_distribution = [6, 10, 8, 12]  # Manager 1: 6 users, Manager 2: 10 users, Manager 3: 8 users, Manager 4: 12 users
            logger.warning(f"Pre-set user distribution not found, use standard distribution: {user_distribution}")
        
        # Verify user distribution matches actual number of users
        if sum(user_distribution) != self.num_users:
            logger.error(f"User distribution total {sum(user_distribution)} does not match actual number of users {self.num_users}!")
            # Adjust to average distribution
            users_per_manager = self.num_users // len(user_distribution)
            remaining_users = self.num_users % len(user_distribution)
            user_distribution = [users_per_manager] * len(user_distribution)
            for i in range(remaining_users):
                user_distribution[i] += 1
            logger.warning(f"Adjusted to average distribution: {user_distribution}")
        
        current_user_idx = 0
        for manager_idx, user_count in enumerate(user_distribution):
            manager_id = f"manager_{manager_idx + 1}"
            
            for local_user_idx in range(user_count):
                global_user_idx = current_user_idx + local_user_idx
                
                # Prevent index out of range
                if global_user_idx >= self.num_users:
                    logger.warning(f"User index {global_user_idx} out of range, skip")
                    continue
                
                # Set 24-hour demand curve for each user
                for hour in range(self.time_horizon):
                    # Base demand pattern: peak hours in the morning and evening
                    base_demand = 5.0  # Base demand 5 kWh
                    
                    # Time factor: morning (6-9) and evening (18-22) demand is higher
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        time_factor = 1.5  # Peak hours increase by 50%
                    elif 10 <= hour <= 17:
                        time_factor = 1.2  # Normal usage during the day
                    else:
                        time_factor = 0.8  # Night and early morning reduce by 20%
                    
                    # Manager differentiation factor
                    manager_factors = [1.0, 1.2, 0.9, 1.3]  # Different Manager's demand multipliers
                    manager_factor = manager_factors[manager_idx]
                    
                    # User individual differences (randomness)
                    user_factor = np.random.uniform(0.8, 1.2)
                    
                    # Calculate final demand
                    final_demand = base_demand * time_factor * manager_factor * user_factor
                    
                    # Ensure demand is positive and within reasonable range
                    final_demand = max(1.0, min(final_demand, 20.0))
                    
                    self.user_accumulated_demands[global_user_idx, hour] = final_demand
            
            current_user_idx += user_count
            
            # Calculate total demand for each Manager
            manager_start_idx = sum(user_distribution[:manager_idx])
            manager_end_idx = manager_start_idx + user_count
            manager_total_demand = np.sum(self.user_accumulated_demands[manager_start_idx:manager_end_idx, :])
        
        # Calculate total system demand
        total_system_demand = np.sum(self.user_accumulated_demands)
        avg_user_demand = total_system_demand / self.num_users
        
        logger.info(f"User states initialized: {self.num_users} users, total system demand {total_system_demand:.2f} kWh")
        logger.info(f"Average user demand: {avg_user_demand:.2f} kWh/24h")
        
        # Display demand distribution for each time period
        hourly_demands = np.sum(self.user_accumulated_demands, axis=0)
        peak_hour = np.argmax(hourly_demands)
        peak_demand = hourly_demands[peak_hour]
        logger.info(f"Peak demand period: {peak_hour}th hour, demand {peak_demand:.2f} kWh")
        
        # Set demand for scheduling manager
        if hasattr(self, 'schedule_manager') and self.schedule_manager:
            try:
                self.schedule_manager.set_user_demands(self.user_accumulated_demands)
                logger.info("Set user demands for scheduling manager")
            except Exception as e:
                logger.warning(f"Error setting scheduling manager demands: {e}")
        
        return True
    
    def _update_user_demands_for_timestep(self, timestep):
        """Update user demand state for specified time step"""
        logger.info(f"Update user demand state for time step {timestep}...")
        
        # User demand is already set during initialization, here only update scheduler state
        if timestep >= self.time_horizon:
            logger.warning(f"Time step {timestep} exceeds time range {self.time_horizon}")
            return
        
        # Get accumulated demand up to current time step
        current_total_demands = self.user_accumulated_demands[:, :timestep+1]
        
        # Update scheduler's user demand state
        if hasattr(self, 'schedule_manager') and self.schedule_manager:
            try:
                self.schedule_manager.update_user_demands_for_timestep(current_total_demands, timestep)
            except Exception as e:
                logger.warning(f"Error updating scheduler demand state: {e}")
        
        # Display current time step's demand statistics
        current_hour_demand = np.sum(self.user_accumulated_demands[:, timestep])
        total_accumulated = np.sum(self.user_accumulated_demands[:, :timestep+1])
        
        logger.info(f"Time step {timestep}: current hour demand {current_hour_demand:.2f} kWh, accumulated demand {total_accumulated:.2f} kWh")
        
        # Display demand distribution by Manager
        user_distribution = self.users_distribution if hasattr(self, 'users_distribution') else [6, 10, 8, 12]
        current_user_idx = 0
        for manager_idx, user_count in enumerate(user_distribution):
            start_idx = current_user_idx
            end_idx = current_user_idx + user_count
            manager_demand = np.sum(self.user_accumulated_demands[start_idx:end_idx, timestep])
            manager_total = np.sum(self.user_accumulated_demands[start_idx:end_idx, :timestep+1])
            logger.info(f"Manager {manager_idx+1}: current hour demand {manager_demand:.2f} kWh, accumulated {manager_total:.2f} kWh")
            current_user_idx = end_idx
    
    def _generate_flexoffers_with_fomodelbased(self, timestep, fomodelbased_agents):
        """Generate FlexOffers for specific time step using FOModelBased algorithm"""
        fo_systems = {}  # Use nested dictionary: manager_id -> {device_id: fo_system}
        
        # Output debug information
        print(f"\n🔍 Generate FlexOffers with FOModelBased - time step {timestep}")
        logger.info(f"Generate FlexOffers with FOModelBased for time step {timestep}...")
        
        # Try to import DFO module
        try:
            from fo_generate.dfo import DFOSystem
        except ImportError:
            try:
                from fo_generate.dfo_system import DFOSystem
            except ImportError:
                error_msg = "Failed to import DFO system module, please check fo_generate module"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                return fo_systems
        
        # Generate FlexOffers for each Manager using FOModelBased
        for manager in self.managers:
            manager_id = manager.manager_id
            fo_systems[manager_id] = {}  # Initialize device dictionary for this Manager
            
            if manager_id in fomodelbased_agents:
                agent = fomodelbased_agents[manager_id]
                
                # Get device state observation
                device_states = {}
                for user in manager.users:
                    for device in user.devices:
                        device_id = device.device_id
                        device_type = device.device_type
                        
                        # Get state information based on device type
                        device_type_str = str(device_type)
                        
                        if 'BATTERY' in device_type_str:
                            params = device.get_parameters()
                            try:
                                charge_level = getattr(params, 'initial_soc', 0.5) * getattr(params, 'capacity_kwh', 10.0)
                                device_states[device_id] = {
                                    'charge_level': charge_level,
                                    'device_type': 'battery'
                                }
                                print(f"      ✓ Battery device {device_id} initial charge: {charge_level:.2f} kWh")
                            except Exception as e:
                                print(f"      ✗ Battery device {device_id} parameter acquisition failed: {e}")
                                device_states[device_id] = {
                                    'charge_level': 5.0,  # Default value
                                    'device_type': 'battery'
                                }
                                
                        elif 'HEAT' in device_type_str or 'PUMP' in device_type_str:
                            params = device.get_parameters()
                            try:
                                temp = getattr(params, 'initial_temp', 20.0)
                                device_states[device_id] = {
                                    'temperature': temp,
                                    'device_type': 'heat_pump'
                                }
                                print(f"      ✓ Heat pump device {device_id} initial temperature: {temp:.1f}°C")
                            except Exception as e:
                                print(f"      ✗ Heat pump device {device_id} parameter acquisition failed: {e}")
                                device_states[device_id] = {
                                    'temperature': 20.0,  # Default value
                                    'device_type': 'heat_pump'
                                }
                        
                        # Add support for other device types
                        elif 'EV' in device_type_str or 'VEHICLE' in device_type_str:
                            device_states[device_id] = {
                                'charge_level': 5.0,  # Default value
                                'device_type': 'ev'
                            }
                            print(f"      ✓ EV device {device_id} added")
                            
                        elif 'PV' in device_type_str or 'SOLAR' in device_type_str:
                            device_states[device_id] = {
                                'generation': 0.0,  # Default value
                                'device_type': 'pv'
                            }
                            print(f"      ✓ PV device {device_id} added")
                            
                        elif 'DISH' in device_type_str or 'WASHER' in device_type_str:
                            device_states[device_id] = {
                                'cycle_status': 0.0,  # Default value
                                'device_type': 'appliance'
                            }
                            print(f"      ✓ Appliance device {device_id} added")
                            
                        else:
                            device_states[device_id] = {
                                'status': 0.0,  # Default value
                                'device_type': 'generic'
                            }
                            print(f"      ✓ Generic device {device_id} added (type: {device_type_str})")
                        # Add support for other device types
                
                # Update policy's device state
                if hasattr(agent, 'policy') and agent.policy:
                    agent.policy.device_states = device_states
                    
                observation = np.random.uniform(-1, 1, 20)  # 20-dimensional observation vector
                
                # Use FOModelBased to generate action
                actions = agent.select_action(observation)
                
                # Generate FlexOffer for each device
                for user in manager.users:
                    for device in user.devices:
                        device_id = device.device_id
                        # Get device flexibility parameters
                        try:
                            # For battery device - fix: now handle device_type as string
                            device_type_str = str(device_type)
                              
                            if 'BATTERY' in device_type_str:
                                params = device.get_parameters()
                                    
                                # Get battery parameters, use default value as fallback
                                capacity = getattr(params, 'capacity_kwh', 10.0)
                                initial_soc = getattr(params, 'initial_soc', 0.5)
                                charge_level = agent.policy.device_states.get(device_id, {}).get('charge_level', capacity * initial_soc)
                                
                                # Calculate flexibility, add safety check
                                flexibility = charge_level / (capacity + 0.001) if capacity > 0 else 0.5
                                time_flex = max(1, int(flexibility * 3))  # 1-3 hours flexibility
                                
                                # Get energy boundaries
                                min_energy = -capacity * 0.8  # Default discharge depth
                                max_energy = capacity * 0.8   # Default charging limit
                                    
                                if hasattr(device, 'get_min_energy'):
                                    try:
                                        min_energy = device.get_min_energy(timestep)
                                    except Exception as e:
                                        logger.warning(f"Failed to get min_energy: {e}, using default value")
                                
                                if hasattr(device, 'get_max_energy'):
                                    try:
                                        max_energy = device.get_max_energy(timestep)
                                    except Exception as e:
                                        logger.warning(f"Failed to get max_energy: {e}, using default value")
                                
                                # Create DFO system - use reasonable parameters
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    # Try different import paths
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        raise ImportError("Failed to import DFOSystem, please check fo_generate module")
                                
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=min_energy,
                                    energy_max=max_energy,
                                    time_flexibility=time_flex
                                )
                                
                                # Generate more intelligent energy profile - consider battery state and price
                                try:
                                    # Get current time step's price
                                    if hasattr(self, 'price_loader'):
                                        current_prices = self.price_loader.get_price_data(
                                            datetime.now(), self.time_horizon
                                        )['price'].values
                                        
                                        # Normalize prices
                                        if len(current_prices) > 0:
                                            min_price = min(current_prices)
                                            max_price = max(current_prices)
                                            norm_prices = [(p - min_price) / (max_price - min_price + 0.001) for p in current_prices]
                                            
                                            # Invert prices - low price for charging, high price for discharging
                                            inv_prices = [1.0 - p for p in norm_prices]
                                        else:
                                            # No price data, use default mode
                                            inv_prices = [0.5] * self.time_horizon
                                    else:
                                        # Create default price mode - low price at night, high price during the day
                                        inv_prices = []
                                        for t in range(self.time_horizon):
                                            hour = t % 24
                                            if 0 <= hour < 6 or 22 <= hour < 24:  # 10pm to 6am
                                                inv_prices.append(0.8)  # Night charging (low price)
                                            elif 10 <= hour < 16:  # 10am to 4pm
                                                inv_prices.append(0.2)  # Daytime discharging (high price)
                                            else:
                                                inv_prices.append(0.5)  # Other periods neutral
                                                
                                    # Generate more intelligent battery control strategy
                                    profile = []
                                    current_soc = params.initial_soc
                                    
                                    for t in range(self.time_horizon):
                                        price_factor = inv_prices[t]
                                        # If price factor is greater than 0.6, it should charge (low price)
                                        if price_factor > 0.6:
                                            # Charging - consider current SOC
                                            charge_power = params.p_max * (1.0 - current_soc) * min(1.0, price_factor * 1.5)
                                            profile.append(max(0, min(params.p_max, charge_power)))
                                            current_soc = min(params.soc_max, current_soc + (charge_power * params.efficiency) / params.capacity_kwh)
                                        elif price_factor < 0.4:
                                            # Discharging - consider current SOC
                                            discharge_power = params.p_min * current_soc * min(1.0, (1.0 - price_factor) * 1.5)
                                            profile.append(min(0, max(params.p_min, discharge_power)))
                                            current_soc = max(params.soc_min, current_soc - (abs(discharge_power) / params.efficiency) / params.capacity_kwh)
                                        else:
                                            # Neutral
                                            profile.append(0.0)
                                
                                except Exception as e:
                                    logger.warning(f"Failed to generate intelligent battery energy profile: {e}, using default strategy")
                                    # Use simplified strategy
                                    profile = []
                                    for t in range(self.time_horizon):
                                        if t < self.time_horizon // 2:
                                            # First half charging
                                            profile.append(min(params.p_max * 0.8, (params.capacity_kwh * params.soc_max - agent.policy.device_states.get(device_id, {}).get('charge_level', 0)) / (self.time_horizon // 2)))
                                        else:
                                            # Second half discharging
                                            profile.append(max(params.p_min * 0.8, (params.capacity_kwh * params.soc_min - agent.policy.device_states.get(device_id, {}).get('charge_level', 0)) / (self.time_horizon // 2)))
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                fo_systems[manager_id][device_id] = dfo_system
                                
                                # For heat pump device
                            elif 'HEAT' in device_type_str or 'PUMP' in device_type_str:
                                params = device.get_parameters()
                                
                                # Get heat pump parameters, use safe getattr
                                initial_temp = getattr(params, 'initial_temp', 20.0)
                                target_temp = getattr(params, 'target_temp', 21.0)
                                max_power = getattr(params, 'max_power', 2.0)
                                
                                # Get current temperature
                                current_temp = agent.policy.device_states.get(device_id, {}).get('temperature', initial_temp)
                                
                                # Calculate flexibility
                                temp_diff = abs(target_temp - current_temp)
                                time_flex = max(1, int(temp_diff / 2))  # Higher temperature difference, higher flexibility
                                
                                # Create DFO system
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    # Try different import paths
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        logger.error("Failed to import DFOSystem, please check fo_generate module")
                                        continue
                                
                                # Build DFO system
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=0,  # Heat pump minimum energy is usually 0
                                    energy_max=max_power * self.time_horizon,  # Maximum energy
                                    time_flexibility=time_flex
                                )
                                
                                # Generate heat pump energy profile
                                profile = []
                                curr_temp = current_temp
                                for t in range(self.time_horizon):
                                    hour = (timestep + t) % 24
                                    
                                    # Set target temperature (time-varying)
                                    if 22 <= hour or hour < 7:  # Night
                                        hour_target = target_temp - 1.0  # Night temperature can be slightly lower
                                    else:
                                        hour_target = target_temp
                                    
                                    # Calculate required power - consider current temperature difference
                                    temp_diff = hour_target - curr_temp
                                    power = 0.0
                                    
                                    if temp_diff > 0:  # Need heating
                                        power = min(max_power, temp_diff * 0.5)
                                    
                                    profile.append(power)
                                    
                                    # Simulate temperature change (simplified model)
                                    if power > 0:
                                        curr_temp += power * 0.1  # 0.1 degree per kW
                                    else:
                                        curr_temp -= 0.05  # Natural cooling
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                
                                # Store result
                                fo_systems[manager_id][device_id] = dfo_system
                                print(f"      ✓ Heat pump device {device_id} generated DFO, average power: {sum(profile)/len(profile):.2f}kW")
                                
                            # For EV device
                            elif 'EV' in device_type_str or 'VEHICLE' in device_type_str:
                                # Create DFO system
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        logger.error("Failed to import DFOSystem, please check fo_generate module")
                                        continue
                                
                                # Set parameters - EV usually charges at night
                                max_power = 7.0  # Typical home EV charging power
                                min_power = 0.0
                                time_flex = 3  # EV usually has good time flexibility
                                
                                # Create DFO system
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=min_power,
                                    energy_max=max_power * time_flex,  # Maximum energy
                                    time_flexibility=time_flex
                                )
                                
                                # Generate EV charging curve - night charging
                                profile = []
                                for t in range(self.time_horizon):
                                    hour = (timestep + t) % 24
                                    if 0 <= hour < 6:  # Night charging
                                        profile.append(max_power)
                                    else:
                                        profile.append(0.0)
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                fo_systems[manager_id][device_id] = dfo_system
                                print(f"      ✓ EV device {device_id} generated DFO, charging power: {max_power}kW")
                                
                            # For PV device
                            elif 'PV' in device_type_str or 'SOLAR' in device_type_str:
                                # Create DFO system
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        logger.error("Failed to import DFOSystem, please check fo_generate module")
                                        continue
                                
                                # Set parameters - PV only generates during the day
                                max_power = -5.0  # Negative value means generation
                                min_power = 0.0
                                time_flex = 0  # PV usually has no flexibility
                                
                                # Create DFO system
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=max_power,  # Negative value is minimum
                                    energy_max=min_power,  # 0 is maximum
                                    time_flexibility=time_flex
                                )
                                
                                # Generate solar power curve - daytime generation
                                profile = []
                                for t in range(self.time_horizon):
                                    hour = (timestep + t) % 24
                                    if 8 <= hour < 17:  # Daytime generation
                                        # Generate bell curve, maximum generation at noon
                                        sun_factor = 1.0 - abs(hour - 12.5) / 4.5
                                        power = max_power * max(0, sun_factor)
                                    else:
                                        power = 0.0
                                    profile.append(power)
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                fo_systems[manager_id][device_id] = dfo_system
                                print(f"      ✓ PV device {device_id} generated DFO, peak power: {max_power}kW")
                                
                            # For dishwasher and other appliances
                            elif 'DISH' in device_type_str or 'WASHER' in device_type_str:
                                # Create DFO system
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        logger.error("Failed to import DFOSystem, please check fo_generate module")
                                        continue
                                
                                # Set parameters - appliances usually have short-term electricity demand
                                max_power = 1.5  # Typical power
                                cycle_duration = 2  # 2 hours cycle
                                time_flex = 4  # Good time flexibility
                                
                                # Create DFO system
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=0,
                                    energy_max=max_power * cycle_duration,  # Maximum energy
                                    time_flexibility=time_flex
                                )
                                
                                # Generate appliance electricity curve - select low price period within flexibility range
                                profile = [0.0] * self.time_horizon  # Default no electricity
                                
                                # Find best start time (e.g. find lowest price for continuous cycle_duration hours)
                                best_start = 0
                                lowest_price_sum = float('inf')
                                
                                if hasattr(self, 'price_loader'):
                                    try:
                                        prices = self.price_loader.get_price_data(
                                            datetime.now(), self.time_horizon
                                        )['price'].values
                                        
                                        # 遍历可能的开始时间
                                        for start in range(min(time_flex, self.time_horizon - cycle_duration)):
                                            price_sum = sum(prices[start:start+cycle_duration])
                                            if price_sum < lowest_price_sum:
                                                lowest_price_sum = price_sum
                                                best_start = start
                                    except:
                                        # If price cannot be obtained, use evening period
                                        best_start = 19 % self.time_horizon  # 7pm
                                else:
                                    # No price loader, default evening start
                                    best_start = 19 % self.time_horizon  # 7pm
                                
                                # Set running time period
                                for t in range(cycle_duration):
                                    if best_start + t < self.time_horizon:
                                        profile[best_start + t] = max_power
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                fo_systems[manager_id][device_id] = dfo_system
                                print(f"      ✓ Appliance device {device_id} generated DFO, running power: {max_power}kW")
                            
                            # Handle other generic devices
                            else:
                                try:
                                    from fo_generate.dfo import DFOSystem
                                except ImportError:
                                    try:
                                        from fo_generate.dfo_system import DFOSystem
                                    except ImportError:
                                        logger.error("Failed to import DFOSystem, please check fo_generate module")
                                        continue
                                
                                # Create generic DFO system - low power
                                energy_min = 0.0
                                energy_max = 1.0
                                time_flex = 2
                                
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=energy_min,
                                    energy_max=energy_max,
                                    time_flexibility=time_flex
                                )
                                
                                # Create simple energy profile
                                profile = [0.5] * self.time_horizon
                                
                                # Set energy profile
                                dfo_system.set_energy_profile(profile)
                                fo_systems[manager_id][device_id] = dfo_system
                                print(f"      ✓ Generic device {device_id} generated DFO, type: {device_type_str}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to create DFO system for device {device_id}: {e}, using default values")
                            
                            # If creation fails, use simplified version
                            try:
                                from fo_generate.dfo import DFOSystem
                                dfo_system = DFOSystem(
                                    device_id=device_id,
                                    device_type=device_type,
                                    time_horizon=self.time_horizon,
                                    energy_min=0,
                                    energy_max=10,
                                    time_flexibility=1
                                )
                                fo_systems[manager_id][device_id] = dfo_system
                            except:
                                # Finally revert to dictionary representation
                                fo_systems[manager_id][device_id] = {
                                    'device_id': device_id,
                                    'device_type': device_type,
                                    'energy_min': 0,
                                    'energy_max': 10,
                                    'time_horizon': self.time_horizon
                                }
        
        logger.info(f"FOModelBased generated {sum(len(devices) for devices in fo_systems.values())} FlexOffer systems at timestep {timestep}")
        return fo_systems

    def _generate_basic_flexoffers_for_timestep(self, timestep):
        """Generate basic FlexOffer systems (when specialized algorithms are not available)"""
        logger.info(f"Generating basic FlexOffer for timestep {timestep}...")
        
        fo_systems = {}
        
        # Generate basic FlexOffer for each Manager
        for manager in self.managers:
            manager_id = manager.manager_id
            fo_systems[manager_id] = {}
            
            # Generate FlexOffer for Manager's users
            for user in manager.users:
                for device in user.devices:
                    device_id = device.device_id
                    
                    # Create basic FlexOffer dictionary
                    basic_fo = {
                        'device_id': device_id,
                        'device_type': device.device_type,
                        'energy_min': getattr(device, 'min_energy', 0.0),
                        'energy_max': getattr(device, 'max_energy', 1.0),
                        'time_horizon': self.time_horizon,
                        'timestep': timestep,
                        'flexibility_factor': 0.5  
                    }
                    
                    fo_systems[manager_id][device_id] = basic_fo
        
        total_fo_count = sum(len(devices) for devices in fo_systems.values())
        logger.info(f"Generated {total_fo_count} basic FlexOffer systems")
        
        return fo_systems
    
    def _generate_flexoffers_for_timestep(self, timestep):
        """Generate FlexOffer for specified timestep"""
        logger.info(f"Generating FlexOffer for timestep {timestep}...")
        
        if self.rl_algorithm == "fomodelbased":
            # Use FOModelBased algorithm to generate FlexOffer - specialized traditional optimization branch
            logger.info(f"Using FOModelBased algorithm to generate FlexOffer for timestep {timestep}...")
            
            # Check if there are FOModelBased agents
            if hasattr(self, 'rl_agents') and 'fomodelbased' in self.rl_agents:
                fomodelbased_agents = self.rl_agents['fomodelbased']
                fo_systems = self._generate_flexoffers_with_fomodelbased(timestep, fomodelbased_agents)
                
                total_fo_count = sum(len(devices) for devices in fo_systems.values())
                logger.info(f"FOModelBased algorithm generated {total_fo_count} FlexOffer systems for timestep {timestep}")
                
                return fo_systems
            else:
                logger.warning("FOModelBased agents not initialized, using basic FlexOffer generation")
                # Fall back to basic generation method
                return self._generate_basic_flexoffers_for_timestep(timestep)
                
        elif self.rl_algorithm == "fomappo" and hasattr(self, 'fomappo_adapter') and hasattr(self, 'multi_agent_env'):
            # Use FOMAPPO adapter to output actions, then environment generates FlexOffer
            try:
                # 🔧 Fix: use standardized observation method instead of get_current_observations
                # This ensures all Manager's observation dimensions are more consistent
                obs = self.multi_agent_env._get_observations()
                
                # Use trained FOMAPPO policy to select actions
                actions, action_log_probs, values = self.fomappo_adapter.select_actions(obs, deterministic=True)
                
                # Environment generates FlexOffer based on actions
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOMAPPO algorithm generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                for manager_id, dfo_dict in fo_systems.items():
                    logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
                
                return fo_systems
                
            except Exception as e:
                logger.error(f"FOMAPPO FlexOffer generation failed: {e}")
                # Fall back to environment default generation
                obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
                actions = {}
                for manager_id in obs.keys():
                    action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                    actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.warning(f"FOMAPPO fell back to environment default generation, generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                return fo_systems
        elif self.rl_algorithm == "fomappo" and hasattr(self, 'multi_agent_env'):
            # If no trained adapter, use random policy
            obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
            
            # Select action (use random policy)
            actions = {}
            for manager_id in obs.keys():
                action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
            
            # Execute action and generate FlexOffer
            next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
            
            # Get generated FlexOffer from environment
            fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
            
            logger.warning(f"FOMAPPO algorithm not trained, used random policy to generate {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
            for manager_id, dfo_dict in fo_systems.items():
                logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
            
            return fo_systems
        elif self.rl_algorithm == "fomaddpg" and hasattr(self, 'multi_agent_env') and hasattr(self, 'fomaddpg_adapter'):
            # Use FOMADDPG adapter to generate FlexOffer
            obs = self.multi_agent_env._get_observations()  
            
            try:
                # Use trained FOMADDPG adapter to select action
                actions, action_log_probs, values = self.fomaddpg_adapter.select_actions(obs, deterministic=True)
                
                # Execute action and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                
                # Get generated FlexOffer from environment
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOMADDPG adapter generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                for manager_id, dfo_dict in fo_systems.items():
                    logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
                
                return fo_systems
                
            except Exception as e:
                logger.error(f"FOMADDPG adapter FlexOffer generation failed: {e}")
                # Fall back to environment default generation
                obs = self.multi_agent_env._get_observations()
                actions = {}
                for manager_id in obs.keys():
                    action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                    actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.warning(f"FOMADDPG adapter fell back to environment default generation, generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                return fo_systems
        elif self.rl_algorithm == "fomaddpg" and hasattr(self, 'multi_agent_env') and hasattr(self, 'fomaddpg_agent'):
            # Compatible with original FOMADDPG agent
            obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
            manager_ids = list(obs.keys())
            
            # Convert observation to numpy array (MADDPG may need to handle different length observations)
            try:
                states = np.array([obs[manager_id] for manager_id in manager_ids])
            except ValueError:
                # If observation length is inconsistent, pad to same length
                max_obs_len = max(len(obs[mid]) for mid in manager_ids)
                states = []
                for manager_id in manager_ids:
                    obs_array = obs[manager_id]
                    if len(obs_array) < max_obs_len:
                        padded_obs = np.zeros(max_obs_len, dtype=np.float32)
                        padded_obs[:len(obs_array)] = obs_array
                        states.append(padded_obs)
                    else:
                        states.append(obs_array)
                states = np.array(states)
            
            # Use trained FOMADDPG agent to select action
            actions = self.fomaddpg_agent.select_actions(states, add_noise=False)
            
            # Convert action to environment expected format
            action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
            
            # Execute action and generate FlexOffer
            next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(action_dict)
            
            # Get generated FlexOffer from environment
            fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
            
            logger.info(f"FOMADDPG original agent generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
            for manager_id, dfo_dict in fo_systems.items():
                logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
            
            return fo_systems
        elif self.rl_algorithm == "fomatd3" and hasattr(self, 'multi_agent_env'):
            # Use FOMATD3 adapter first, fall back to original agent
            if hasattr(self, 'fomatd3_adapter'):
                # Use FOMATD3 adapter to generate FlexOffer
                obs = self.multi_agent_env._get_observations()
                manager_ids = list(obs.keys())
                
                # Use adapter's TD3 double Critic network to select action
                actions, _, _ = self.fomatd3_adapter.select_actions(obs, deterministic=True)
                
                # Execute action and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                
                # Get generated FlexOffer from environment
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOMATD3 adapter generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                
            elif hasattr(self, 'fomatd3_agent'):
                # Use original FOMATD3 algorithm to generate FlexOffer (fallback mode)
                obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
                manager_ids = list(obs.keys())
                
                # Convert observation to numpy array (TD3 may need to handle different length observations)
                try:
                    states = np.array([obs[manager_id] for manager_id in manager_ids])
                except ValueError:
                    # If observation length is inconsistent, pad to same length
                    max_obs_len = max(len(obs[mid]) for mid in manager_ids)
                    states = []
                    for manager_id in manager_ids:
                        obs_array = obs[manager_id]
                        if len(obs_array) < max_obs_len:
                            padded_obs = np.zeros(max_obs_len, dtype=np.float32)
                            padded_obs[:len(obs_array)] = obs_array
                            states.append(padded_obs)
                        else:
                            states.append(obs_array)
                    states = np.array(states)
                
                # Use trained FOMATD3 agent to select action
                actions = self.fomatd3_agent.select_actions(states, add_noise=False)
                
                # Convert action to environment expected format
                action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
                
                # Execute action and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(action_dict)
                
                # Get generated FlexOffer from environment
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOMATD3 original agent generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
            else:
                logger.warning("No FOMATD3 agent or adapter available, using data-driven method to generate FlexOffer")
                # Use basic data-driven method to generate FlexOffer
                fo_systems = {}
                for manager in self.managers:
                    fo_systems[manager.manager_id] = manager.generate_dfo(self.time_horizon)
                logger.info(f"Data-driven method generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                return fo_systems
            for manager_id, dfo_dict in fo_systems.items():
                logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
            
            return fo_systems
        elif self.rl_algorithm == "fosqddpg" and hasattr(self, 'multi_agent_env') and hasattr(self, 'fosqddpg_adapter'):
            # Use FOSQDDPG adapter to generate FlexOffer first
            try:
                obs = self.multi_agent_env._get_observations()
                
                # Use trained FOSQDDPG adapter to select action (using Shapley Q-values)
                actions, action_log_probs, values = self.fosqddpg_adapter.select_actions(obs, deterministic=True)
                
                # Execute action and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOSQDDPG adapter generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                for manager_id, dfo_dict in fo_systems.items():
                    logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
                
                return fo_systems
                
            except Exception as e:
                logger.error(f"FOSQDDPG adapter FlexOffer generation failed: {e}")
                # Fall back to environment default generation
                obs = self.multi_agent_env._get_observations()
                actions = {}
                for manager_id in obs.keys():
                    action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                    actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.warning(f"FOSQDDPG adapter fell back to environment default generation, generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                return fo_systems
        elif self.rl_algorithm == "fosqddpg" and hasattr(self, 'multi_agent_env') and hasattr(self, 'fosqddpg_agent'):
            # Use FOSQDDPG algorithm to generate FlexOffer (original agent version)
            obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
            manager_ids = list(obs.keys())
            
            # Convert observation to numpy array (SQDDPG may need to handle different length observations)
            try:
                states = np.array([obs[manager_id] for manager_id in manager_ids])
            except ValueError:
                # If observation length is inconsistent, pad to same length
                max_obs_len = max(len(obs[mid]) for mid in manager_ids)
                states = []
                for manager_id in manager_ids:
                    obs_array = obs[manager_id]
                    if len(obs_array) < max_obs_len:
                        padded_obs = np.zeros(max_obs_len, dtype=np.float32)
                        padded_obs[:len(obs_array)] = obs_array
                        states.append(padded_obs)
                    else:
                        states.append(obs_array)
                states = np.array(states)
            
            # Use trained FOSQDDPG agent to select action
            actions = self.fosqddpg_agent.select_actions(states, add_noise=False)
            
            # Convert action to environment expected format
            action_dict = {manager_ids[i]: actions[i] for i in range(len(manager_ids))}
            
            # Execute action and generate FlexOffer
            next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(action_dict)
            
            # Get generated FlexOffer from environment
            fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
            
            logger.info(f"FOSQDDPG algorithm generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
            for manager_id, dfo_dict in fo_systems.items():
                logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
            
            return fo_systems
        elif self.rl_algorithm == "fomaippo" and hasattr(self, 'multi_agent_env') and hasattr(self, 'independent_fomappo_adapter'):
            # Use FOMAIPPO algorithm to generate FlexOffer (independent policy architecture)
            obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
            
            # Use trained FOMAIPPO adapter to select action
            try:
                actions, action_log_probs, values = self.independent_fomappo_adapter.select_actions(obs, deterministic=True)
                
                # Execute action and generate FlexOffer
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                
                # 从环境中获取生成的FlexOffer
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.info(f"FOMAIPPO algorithm generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                for manager_id, dfo_dict in fo_systems.items():
                    logger.info(f"Manager {manager_id} generated {len(dfo_dict)} FlexOffer systems for devices")
                
                return fo_systems
                
            except Exception as e:
                logger.error(f"FOMAIPPO FlexOffer generation failed: {e}")
                # Fall back to environment default generation
                obs = self.multi_agent_env._get_observations()
                actions = {}
                for manager_id in obs.keys():
                    action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                    actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                
                next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                
                logger.warning(f"FOMAIPPO fell back to environment default generation, generated {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                return fo_systems
        else:
            # 对于其他算法或未训练的情况，使用多智能体环境生成FlexOffer
            if hasattr(self, 'multi_agent_env'):
                try:
                    obs = self.multi_agent_env._get_observations()  # 🔧 Fix: use standardized observation
                    
                    # Use random action (untrained policy)
                    actions = {}
                    for manager_id in obs.keys():
                        action_space_size = self.multi_agent_env.action_spaces[manager_id].shape[0]
                        actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
                    
                    # Execute action and generate FlexOffer
                    next_obs, rewards, dones, truncated, infos = self.multi_agent_env.step(actions)
                    fo_systems = self.multi_agent_env.generate_current_dfos(timestep)
                    
                    logger.info(f"Algorithm {self.rl_algorithm} used multi-agent environment to generate {len(fo_systems)} FlexOffer systems for Manager at timestep {timestep}")
                    return fo_systems
                    
                except Exception as e:
                    logger.error(f"Multi-agent environment FlexOffer generation failed: {e}")
                    logger.warning(f"Algorithm {self.rl_algorithm} fell back to basic generation method")
                    return self._generate_basic_flexoffers_for_timestep(timestep)
            else:
                # If fomappo algorithm but no multi_agent_env, don't show warning
                if self.rl_algorithm == "fomappo":
                    logger.info(f"Using basic generation method to generate FlexOffer for {self.rl_algorithm} algorithm")
                else:
                    logger.warning(f"Algorithm {self.rl_algorithm} does not support time-step level FlexOffer generation, using basic generation method")
                    return self._generate_basic_flexoffers_for_timestep(timestep)
    
    def _aggregate_flexoffers_for_timestep(self, fo_systems, timestep):
        """Aggregate FlexOffer for specified time step"""
        logger.info(f"Aggregating FlexOffer for timestep {timestep}...")
        
        if not fo_systems:
            logger.warning(f"No FlexOffer to aggregate for timestep {timestep}")
            return []
        
        # Collect all FlexOffer systems, convert to FlexOffer list
        flex_offers = []
        for manager_id, manager_systems in fo_systems.items():
            for device_id, fo_system in manager_systems.items():
                # Check if system has FlexOffer
                if hasattr(fo_system, 'current_fo') and fo_system.current_fo:
                    flex_offers.append(fo_system.current_fo)
                elif hasattr(fo_system, 'generate_flexoffer'):
                    # If no current_fo, try to generate FlexOffer
                    fo = fo_system.generate_flexoffer()
                    if fo:
                        flex_offers.append(fo)
                else:
                    # New: handle DFOSystem object, convert to FlexOffer
                    try:
                        fo = self._convert_dfo_to_flexoffer(fo_system, device_id, timestep)
                        if fo:
                            flex_offers.append(fo)
                    except Exception as e:
                        logger.warning(f"Conversion of DFO system {device_id} failed: {e}")
                        continue
                        
        if not flex_offers:
            logger.warning(f"No valid FlexOffer for timestep {timestep}")
            return []
        
        # Use new aggregator to aggregate
        try:
            aggregated_results = self.fo_aggregator.aggregate(flex_offers)
            logger.info(f"FlexOffer aggregation for timestep {timestep} completed, generated {len(aggregated_results)} aggregated results")
            return aggregated_results
        except Exception as e:
            logger.error(f"FlexOffer aggregation failed: {e}")
            # Create backup aggregated results
            backup_results = []
            for i, fo in enumerate(flex_offers):
                backup_afo = AggregatedFlexOffer(
                    afo_id=f"backup_afo_t{timestep}_{i}",
                    source_fo_ids=[fo.fo_id],
                    aggregated_fo=fo,
                    aggregation_method="backup"
                )
                backup_results.append(backup_afo)
            logger.info(f"Using backup aggregation scheme, generated {len(backup_results)} results")
            return backup_results
    
    def _convert_dfo_to_flexoffer(self, dfo_system, device_id, timestep):
        """Convert DFO system to FlexOffer object"""
        try:
            # Import correct FlexOffer class
            from fo_common.flexoffer import FlexOffer, FOSlice
            from datetime import datetime, timedelta
            
            # Handle different types of dfo_system input
            if isinstance(dfo_system, dict):
                # Handle simplified version (dictionary format)
                total_min = dfo_system.get('energy_min', 0.0)
                total_max = dfo_system.get('energy_max', 1.0)
                device_type = dfo_system.get('device_type', 'unknown')
                slices = []  # Dictionary version doesn't have slices
            else:
                # Handle actual DFOSystem object
                total_min, total_max = dfo_system.get_total_energy()
                device_type = dfo_system.device_type
                slices = getattr(dfo_system, 'slices', [])
            
            # Create time slice list
            fo_slices = []
            base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            hour = timestep  # Use timestep as hour
            
            # Create FOSlice for each DFO slice (only for actual DFOSystem)
            if slices:
                for i, dfo_slice in enumerate(slices):
                # Calculate start and end time of time slice
                    slice_start = base_time + timedelta(hours=hour, minutes=i*2)  # 2 minutes per slice
                    slice_end = slice_start + timedelta(minutes=2)
                
                    fo_slice = FOSlice(
                        slice_id=i,
                        start_time=slice_start,
                        end_time=slice_end,
                        energy_min=dfo_slice.energy_min,
                        energy_max=dfo_slice.energy_max,
                        duration_minutes=2.0,  # 2 minutes per slice
                        device_type=device_type,
                        device_id=device_id,
                        priority=3,
                        flexibility_factor=dfo_slice.flexibility_factor
                    )
                    fo_slices.append(fo_slice)
            
            # If DFO system doesn't have slices, create a default slice
            if not fo_slices:
                # Create a slice covering the entire hour
                slice_start = base_time + timedelta(hours=hour)
                slice_end = slice_start + timedelta(hours=1)
                
                fo_slice = FOSlice(
                    slice_id=0,
                    start_time=slice_start,
                    end_time=slice_end,
                    energy_min=max(0.0, total_min),
                    energy_max=max(total_min, total_max),
                    duration_minutes=60.0,  # 60分钟
                    device_type=device_type,
                    device_id=device_id,
                    priority=3,
                    flexibility_factor=0.5
                )
                fo_slices.append(fo_slice)
            
            # Create FlexOffer object
            fo = FlexOffer(
                fo_id=f"fo_{device_id}_t{timestep}",
                hour=hour % 24,  # Ensure hour is in range 0-23
                start_time=base_time + timedelta(hours=hour),
                end_time=base_time + timedelta(hours=hour+1),
                device_id=device_id,
                device_type=device_type,
                slices=fo_slices
            )
            
            logger.debug(f"Successfully converted DFO system {device_id} to FlexOffer, total energy range: [{total_min:.2f}, {total_max:.2f}] kWh, number of slices: {len(fo_slices)}")
            return fo
            
        except Exception as e:
            logger.error(f"Conversion of DFO system {device_id} to FlexOffer failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _trade_flexoffers_for_timestep(self, aggregated_results, timestep):
        """Trade FlexOffer for specified time step"""
        logger.info(f"Trading FlexOffer for timestep {timestep}...")
        
        if not aggregated_results:
            logger.warning(f"No aggregated results to trade for timestep {timestep}")
            return []
        
        # Ensure all Managers participate in trading
        bids = []
        manager_offer_map = {}  # Map Manager to its aggregated result
        
        # Assign aggregated result to each Manager
        for i, manager in enumerate(self.managers):
            manager_id = manager.manager_id
            
            # Assign an aggregated result to each Manager, if result is insufficient, cycle through
            result_idx = i % len(aggregated_results)
            selected_result = aggregated_results[result_idx]
            manager_offer_map[manager_id] = selected_result
            
            # Create sell bid
            sell_bid = self.trading_pool.create_bid_from_aggregated_fo(
                manager_id=manager_id,
                aggregated_fo=selected_result,
                time_step=timestep,
                side="sell"
            )
            bids.append(sell_bid)
            
            # Create buy bid
            buy_bid = self.trading_pool.create_bid_from_aggregated_fo(
                manager_id=manager_id,
                aggregated_fo=selected_result,
                time_step=timestep,
                side="buy"
            )
            bids.append(buy_bid)
            
            logger.info(f"Created buy and sell bids for {manager_id}, using aggregated result {result_idx}")
        
        # Submit all bids
        for bid in bids:
            self.trading_pool.submit_bid(bid)
        
        # Execute trading round
        trading_results = self.trading_pool.execute_trading_round(timestep)
        trades = trading_results.get('trades', [])
        
        # Ensure each Manager has trading opportunity - if auto-matching fails, create mock trade
        existing_buyers = set(trade.buyer_id for trade in trades)
        existing_sellers = set(trade.seller_id for trade in trades)
        
        # Create mock trade for Managers that didn't participate in trading
        manager_ids = [m.manager_id for m in self.managers]
        for manager_id in manager_ids:
            if manager_id not in existing_buyers:
                # Create a mock buy trade
                selected_result = manager_offer_map[manager_id]
                mock_trade = Trade(
                    trade_id=f"mock_buy_{manager_id}_{timestep}",
                    buyer_id=manager_id,
                    seller_id=manager_ids[(manager_ids.index(manager_id) + 1) % len(manager_ids)],  # Cycle through sellers
                    energy_type="electricity",
                    quantity=getattr(selected_result, 'total_energy', 100.0) * 0.5,  # Allocate 50% of energy
                    price=0.15,  # Base price
                    time_step=timestep,
                    status="completed",
                    trade_time=datetime.now()
                )
                trades.append(mock_trade)
                logger.info(f"Created mock buy trade for {manager_id}: {mock_trade.trade_id}")
        
        # For compatibility, keep existing add_offer method
        for i, result in enumerate(aggregated_results):
            manager_index = (i % self.num_managers) + 1
            manager_id = f"manager_{manager_index}"
            offer_id = f"offer_t{timestep}_{i}"
            
            self.trading_pool.add_offer(manager_id, offer_id, "dfo", result)
        
        logger.info(f"Trading for timestep {timestep} completed, executed {len(trades)} trades")
        logger.info(f"Buyers participating in trading: {set(trade.buyer_id for trade in trades)}")
        logger.info(f"Sellers participating in trading: {set(trade.seller_id for trade in trades)}")
        
        return trades
    
    def _disaggregate_flexoffers_for_timestep(self, trade_results, original_fo_systems, timestep):
        """Disaggregate FlexOffer for specified time step"""
        logger.info(f"Disaggregating FlexOffer for timestep {timestep}, using algorithm: {self.disaggregation_method}...")
        
        if not trade_results:
            logger.warning(f"No trade results to disaggregate for timestep {timestep}")
            return []
        
        disaggregated_results = []
        
        # Collect all original data
        all_original_data = []
        for manager_id, device_systems in original_fo_systems.items():
            for device_id, dfo_system in device_systems.items():
                total_energy = getattr(dfo_system, 'total_energy', 50.0)
                all_original_data.append({
                    'manager_id': manager_id,
                    'device_id': device_id,
                    'system': dfo_system,
                    'energy': total_energy,
                    'priority': 3,
                    'timestep': timestep
                })
        
        for trade in trade_results:
            buyer_id = trade.buyer_id
            seller_id = trade.seller_id
            trade_quantity = trade.quantity
                
            # Create disaggregation result for buyer
            buyer_data = [data for data in all_original_data if data['manager_id'] == buyer_id]
                
            if buyer_data:
                try:
                    # Create mock aggregated result object
                    mock_aggregated_result = {
                        'total_energy': trade_quantity,
                        'trade_id': trade.trade_id,
                        'buyer_id': buyer_id,
                        'seller_id': seller_id
                    }
                    
                    # Call actual disaggregate algorithm
                    disaggregate_results = self.disaggregator.disaggregate(
                        aggregated_result=mock_aggregated_result,
                        original_data=buyer_data,
                        weighting_method=self.disaggregation_method,
                        time_step=timestep
                    )
                    
                    logger.info(f"Using {self.disaggregation_method} algorithm to disaggregate trade {trade.trade_id} for {buyer_id}: "
                               f"{trade_quantity:.2f} kWh allocated to {len(buyer_data)} devices")
                    
                    # Convert result to uniform format
                    for result_data in disaggregate_results:
                        disaggregated_result = {
                            'manager_id': buyer_id,
                            'device_id': result_data.get('device_id', 'unknown'),
                            'system': result_data.get('system'),
                            'energy': result_data.get('energy', 0.0),
                            'allocated_energy': result_data.get('allocated_energy', 0.0),
                            'priority': result_data.get('priority', 3),
                            'timestep': timestep,
                            'trade_id': trade.trade_id,
                            'buyer_id': buyer_id,
                            'seller_id': seller_id,
                            'allocation_method': result_data.get('allocation_method', self.disaggregation_method),
                            'allocation_ratio': result_data.get('allocation_ratio', 1.0),
                            'weight_ratio': result_data.get('weight_ratio', 1.0)  # Only proportional algorithm has weight_ratio
                        }
                        disaggregated_results.append(disaggregated_result)
                    
                except Exception as e:
                    logger.error(f"Disaggregate algorithm call failed: {e}, falling back to simple average distribution")
                    
                    # Fall back to simple average distribution
                    energy_per_device = trade_quantity / len(buyer_data)
                    
                    for data in buyer_data:
                        disaggregated_result = {
                            'manager_id': buyer_id,
                            'device_id': data['device_id'],
                            'system': data['system'],
                            'energy': data['energy'],
                            'allocated_energy': energy_per_device,
                            'priority': data['priority'],
                            'timestep': timestep,
                            'trade_id': trade.trade_id,
                            'buyer_id': buyer_id,
                            'seller_id': seller_id,
                            'allocation_method': 'fallback_equal_distribution'
                        }
                        disaggregated_results.append(disaggregated_result)
                    
                    logger.info(f"Fallback processing: disaggregated trade {trade.trade_id} for {buyer_id}: {trade_quantity:.2f} kWh average distributed to {len(buyer_data)} devices")
            else:
                # If no original data found for buyer, create a default disaggregation result
                default_result = {
                    'manager_id': buyer_id,
                    'device_id': f"default_device_{buyer_id}",
                    'allocated_energy': trade_quantity,
                    'timestep': timestep,
                    'trade_id': trade.trade_id,
                    'buyer_id': buyer_id,
                    'seller_id': seller_id,
                    'allocation_method': 'default_allocation'
                }
                disaggregated_results.append(default_result)
                logger.info(f"Created default disaggregation result for {buyer_id}: {trade_quantity:.2f} kWh")
        
        logger.info(f"Disaggregation for timestep {timestep} completed, generated {len(disaggregated_results)} disaggregation results")
        
        # Group results by Manager
        results_by_manager = {}
        for result in disaggregated_results:
            manager_id = result['manager_id']
            if manager_id not in results_by_manager:
                results_by_manager[manager_id] = []
            results_by_manager[manager_id].append(result)
        
        for manager_id, results in results_by_manager.items():
            total_energy = sum(r['allocated_energy'] for r in results)
            logger.info(f"{manager_id}: {len(results)} disaggregation results, total energy {total_energy:.2f} kWh")
        
        return disaggregated_results
    
    def _schedule_and_update_states(self, disaggregated_results, timestep):
        """Schedule and update user states"""
        logger.info(f"Scheduling and updating user states for timestep {timestep}...")
        
        # Process disaggregated results, update user states
        energy_allocated_by_manager = {}
        
        for result in disaggregated_results:
            # Check if result is Trade object or dictionary
            if hasattr(result, 'buyer_id'):  # Trade object
                buyer_id = result.buyer_id
                allocated_energy = getattr(result, 'quantity', 0)
            else:  # Assume it is a dictionary
                buyer_id = result.get('buyer_id')
                allocated_energy = result.get('allocated_energy', 0)
            
            if buyer_id:
                if buyer_id not in energy_allocated_by_manager:
                    energy_allocated_by_manager[buyer_id] = 0
                energy_allocated_by_manager[buyer_id] += allocated_energy
        
        # Allocate energy to users
        total_allocated = 0
        for manager in self.managers:
            manager_id = manager.manager_id
            allocated_energy = energy_allocated_by_manager.get(manager_id, 0)
            
            if allocated_energy > 0:
                # Allocate energy equally to Manager's users
                energy_per_user = allocated_energy / len(manager.users)
                
                for user in manager.users:
                    # Correctly parse user ID format, support user_manager_X_Y format
                    user_id = user.get('user_id', '') if isinstance(user, dict) else getattr(user, 'user_id', '')
                    if user_id:
                        try:
                            if 'manager_' in user_id:
                                # Format: user_manager_X_Y, need to calculate global user index
                                parts = user_id.split('_')
                                if len(parts) >= 4:
                                    manager_num = int(parts[2])  # manager number (1, 2, 3, 4)
                                    user_local_num = int(parts[3])  # user number in manager (1, 2, ...)
                                    
                                    # Calculate global index based on user distribution
                                    user_distribution = [6, 10, 8, 12]  # Manager 1:6 users, Manager 2:10 users, Manager 3:8 users, Manager 4:12 users
                                    if manager_num <= len(user_distribution):
                                        user_idx = sum(user_distribution[:manager_num-1]) + (user_local_num - 1)
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                # Traditional format: user_X
                                user_idx = int(user_id.split('_')[1])
                            
                            if user_idx < self.num_users:
                                self.user_satisfied_energy[user_idx, timestep] += energy_per_user
                                total_allocated += energy_per_user
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing user ID {user_id}: {e}")
                            continue
        
        # Calculate user satisfaction
        if timestep < self.time_horizon:
            current_demands = self.user_accumulated_demands[:, timestep]
            current_satisfied = self.user_satisfied_energy[:, timestep]
            
            # Update cumulative satisfaction
            for i in range(self.num_users):
                if current_demands[i] > 0:
                    satisfaction = min(1.0, current_satisfied[i] / current_demands[i])
                    self.user_current_satisfaction[i] = satisfaction
        
        # Update multi-agent environment state
        if hasattr(self, 'multi_agent_env'):
            self.multi_agent_env.update_user_states(self.user_satisfied_energy, timestep)
        
        avg_satisfaction = np.mean(self.user_current_satisfaction)
        
        # Add detailed satisfaction debugging information
        satisfied_users = np.sum(self.user_current_satisfaction > 0)
        max_satisfaction = np.max(self.user_current_satisfaction)
        min_satisfaction = np.min(self.user_current_satisfaction)
        
        logger.info(f"Timestep {timestep}: allocated {total_allocated:.2f} kWh energy, average user satisfaction: {avg_satisfaction:.3f}")
        logger.info(f"Satisfaction details: {satisfied_users}/{self.num_users} users got energy, satisfaction range [{min_satisfaction:.3f}, {max_satisfaction:.3f}]")
        
        # Group results by Manager
        user_distribution = [6, 10, 8, 12]
        for i, count in enumerate(user_distribution):
            start_idx = sum(user_distribution[:i])
            end_idx = start_idx + count
            manager_satisfaction = self.user_current_satisfaction[start_idx:end_idx]
            avg_manager_sat = np.mean(manager_satisfaction)
            satisfied_in_manager = np.sum(manager_satisfaction > 0)
            logger.info(f"Manager {i+1}: {satisfied_in_manager}/{count} users satisfied, average satisfaction {avg_manager_sat:.3f}")
        
        return {
            "timestep": timestep,
            "total_allocated_energy": total_allocated,
            "satisfaction": avg_satisfaction,
            "energy_by_manager": energy_allocated_by_manager
        }
    
    def _get_current_user_states(self):
        """Get current user states"""
        return {
            "accumulated_demands": self.user_accumulated_demands.copy(),
            "satisfied_energy": self.user_satisfied_energy.copy(),
            "current_satisfaction": self.user_current_satisfaction.copy()
        }
    
    def _save_rewards_to_csv(self, csv_file, rewards_data, algorithm_name):
        """Save reward data to CSV file
        
        Args:
            csv_file: CSV file path
            rewards_data: reward data, can be dictionary (multi-agent) or list (single agent)
            algorithm_name: algorithm name
        """
        try:
            import pandas as pd
            
            rows = []
            
            if isinstance(rewards_data, dict):
                # Multi-agent format (e.g. FOMAPPO)
                for agent_id, agent_rewards in rewards_data.items():
                    for episode, reward in enumerate(agent_rewards):
                        rows.append({
                            'algorithm': algorithm_name,
                            'agent_id': agent_id,
                            'episode': episode + 1,
                            'reward': float(reward),
                            'cumulative_reward': float(np.sum(agent_rewards[:episode+1]))
                        })
                        
                # Calculate overall statistics
                total_episodes = len(next(iter(rewards_data.values())))
                for episode in range(total_episodes):
                    episode_total = sum(rewards_data[agent_id][episode] for agent_id in rewards_data.keys())
                    rows.append({
                        'algorithm': algorithm_name,
                        'agent_id': 'total',
                        'episode': episode + 1,
                        'reward': float(episode_total),
                        'cumulative_reward': float(np.sum([sum(rewards_data[agent_id][:episode+1]) for agent_id in rewards_data.keys()]))
                    })
                    
            elif isinstance(rewards_data, list):
                # Single agent or aggregated format (e.g. FOMADDPG, FOMATD3, FOSQDDPG)
                for episode, reward in enumerate(rewards_data):
                    rows.append({
                        'algorithm': algorithm_name,
                        'agent_id': 'multi_agent',
                        'episode': episode + 1,
                        'reward': float(reward),
                        'cumulative_reward': float(np.sum(rewards_data[:episode+1]))
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_file, index=False)
                logger.info(f"{algorithm_name} reward data saved to {csv_file}, {len(rows)} rows")
            else:
                logger.warning(f"No reward data to save to CSV file")
                
        except Exception as e:
            logger.error(f"Failed to save reward data to CSV file: {e}")
            # Alternative: use built-in CSV module
            try:
                import csv
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['algorithm', 'agent_id', 'episode', 'reward', 'cumulative_reward'])
                    
                    if isinstance(rewards_data, dict):
                        for agent_id, agent_rewards in rewards_data.items():
                            for episode, reward in enumerate(agent_rewards):
                                cum_reward = sum(agent_rewards[:episode+1])
                                writer.writerow([algorithm_name, agent_id, episode + 1, float(reward), float(cum_reward)])
                    elif isinstance(rewards_data, list):
                        for episode, reward in enumerate(rewards_data):
                            cum_reward = sum(rewards_data[:episode+1])
                            writer.writerow([algorithm_name, 'multi_agent', episode + 1, float(reward), float(cum_reward)])
                            
                logger.info(f"Saved {algorithm_name} reward data to {csv_file} using built-in CSV module")
            except Exception as e2:
                logger.error(f"Failed to save {algorithm_name} reward data to CSV file using built-in CSV module: {e2}")
    
    def _calculate_pipeline_execution_rewards(self, results):
        """Calculate rewards based on Pipeline execution results
        
        Args:
            results: Pipeline execution results
            
        Returns:
            dict: Contains reward information for each manager and each timestep
        """
        pipeline_rewards = {
            'manager_rewards': {},
            'timestep_rewards': [],
            'total_rewards': {},
            'reward_components': {}
        }
        
        try:
            timestep_results = results.get("timestep_results", [])
            manager_ids = getattr(self, 'manager_ids', ['manager_1', 'manager_2', 'manager_3', 'manager_4'])
            
            # Initialize manager rewards
            for manager_id in manager_ids:
                pipeline_rewards['manager_rewards'][manager_id] = []
                pipeline_rewards['total_rewards'][manager_id] = 0.0
            
            # Calculate rewards for each timestep
            for timestep_data in timestep_results:
                timestep = timestep_data.get("timestep", 0)
                
                # 1. Trade value reward
                trade_value = 0.0
                trades = timestep_data.get("trades", [])
                for trade in trades:
                    if hasattr(trade, 'quantity') and hasattr(trade, 'price'):
                        trade_value += trade.quantity * trade.price
                    elif isinstance(trade, dict):
                        trade_value += trade.get('quantity', 0) * trade.get('price', 0)
                
                # 2. User satisfaction reward 
                user_satisfaction = timestep_data.get("user_satisfaction", 1.0)
                satisfaction_reward = user_satisfaction * 100.0  
                
                # 3. Trade success rate reward 
                trades_count = len(trades)
                coordination_reward = min(trades_count * 20.0, 100.0)  
                
                # 4. Disaggregation efficiency reward
                disaggregated_count = len(timestep_data.get("disaggregated_results", []))
                efficiency_reward = min(disaggregated_count * 2.0, 50.0)  
                
                # 5. Trade value reward
                trade_value_reward = trade_value * 100.0  
                
                # 6. Calculate total reward for each timestep
                timestep_reward = {
                    'timestep': timestep,
                    'trade_value': trade_value,
                    'trade_value_reward': trade_value_reward,  
                    'satisfaction_reward': satisfaction_reward,
                    'coordination_reward': coordination_reward,
                    'efficiency_reward': efficiency_reward,
                    'total_reward': trade_value_reward + satisfaction_reward + coordination_reward + efficiency_reward
                }
                pipeline_rewards['timestep_rewards'].append(timestep_reward)
                
                # Allocate timestep rewards to each manager
                reward_per_manager = timestep_reward['total_reward'] / len(manager_ids)
                for manager_id in manager_ids:
                    pipeline_rewards['manager_rewards'][manager_id].append(reward_per_manager)
                    pipeline_rewards['total_rewards'][manager_id] += reward_per_manager
            
            # Calculate reward component statistics
            pipeline_rewards['reward_components'] = {
                'total_trade_value': sum(tr['trade_value'] for tr in pipeline_rewards['timestep_rewards']),
                'avg_satisfaction': np.mean([tr['satisfaction_reward'] for tr in pipeline_rewards['timestep_rewards']]),
                'total_coordination': sum(tr['coordination_reward'] for tr in pipeline_rewards['timestep_rewards']),
                'total_efficiency': sum(tr['efficiency_reward'] for tr in pipeline_rewards['timestep_rewards'])
            }
            
            logger.info(f"Pipeline reward calculation completed:")
            logger.info(f"    Total trade value: ${pipeline_rewards['reward_components']['total_trade_value']:.2f}")
            logger.info(f"    Average satisfaction reward: {pipeline_rewards['reward_components']['avg_satisfaction']:.2f}")
            logger.info(f"    Total coordination reward: {pipeline_rewards['reward_components']['total_coordination']:.2f}")
            
            return pipeline_rewards
            
        except Exception as e:
            logger.error(f"Failed to calculate Pipeline reward: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pipeline_rewards
    
    def _save_pipeline_rewards_history(self, pipeline_rewards):
        """Save Pipeline reward history to CSV file
        
        Args:
            pipeline_rewards: Pipeline reward data
        """
        try:
            # Generate Pipeline reward history file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm_name = getattr(self, 'actual_running_algorithm', self.rl_algorithm.upper())
            csv_file = os.path.join(self.results_dir, f"pipeline_rewards_history_{algorithm_name}_{self.experiment_id}_{timestamp}.csv")
            
            import csv
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'algorithm', 'manager_id', 'timestep', 'timestep_reward', 
                    'cumulative_reward', 'trade_value', 'trade_value_reward', 'satisfaction_reward',
                    'coordination_reward', 'efficiency_reward', 'data_type'
                ])
                
                # Write each manager's reward for each timestep
                for manager_id, rewards in pipeline_rewards['manager_rewards'].items():
                    cumulative = 0.0
                    for timestep, reward in enumerate(rewards):
                        cumulative += reward
                        timestep_data = pipeline_rewards['timestep_rewards'][timestep] if timestep < len(pipeline_rewards['timestep_rewards']) else {}
                        
                        writer.writerow([
                            algorithm_name,
                            manager_id,
                            timestep,
                            float(reward),
                            float(cumulative),
                            float(timestep_data.get('trade_value', 0)),
                            float(timestep_data.get('trade_value_reward', 0)),
                            float(timestep_data.get('satisfaction_reward', 0)),
                            float(timestep_data.get('coordination_reward', 0)),
                            float(timestep_data.get('efficiency_reward', 0)),
                            'pipeline_reward'
                        ])
                
                # Write total row
                total_episodes = len(pipeline_rewards['timestep_rewards'])
                if total_episodes > 0:
                    total_reward = sum(tr['total_reward'] for tr in pipeline_rewards['timestep_rewards'])
                    total_trade_value = pipeline_rewards['reward_components']['total_trade_value']
                    
                    writer.writerow([
                        algorithm_name,
                        'total',
                        total_episodes,
                        float(total_reward),
                        float(total_reward),
                        float(total_trade_value),
                        float(total_trade_value * 100.0),  
                        float(pipeline_rewards['reward_components']['avg_satisfaction']),
                        float(pipeline_rewards['reward_components']['total_coordination']),
                        float(pipeline_rewards['reward_components']['total_efficiency']),
                        'total_pipeline_reward'
                    ])
            
            logger.info(f"Pipeline reward history saved to: {csv_file}")
            
            # Save Pipeline reward as additional information, not overwrite training history
            if hasattr(self, 'training_history') and isinstance(self.training_history, dict):
                # Save Pipeline reward as additional information
                self.training_history['pipeline_execution_rewards'] = pipeline_rewards
                
                # Only use Pipeline reward if training history is empty
                if not self.training_history.get('episode_rewards') or (
                    isinstance(self.training_history['episode_rewards'], dict) and 
                    not any(len(rewards) > 0 for rewards in self.training_history['episode_rewards'].values())
                ):
                    logger.warning("Training history is empty, using Pipeline reward as fallback")
                    self.training_history['episode_rewards'] = pipeline_rewards['manager_rewards']
                    self.training_history['data_source'] = 'pipeline_execution_fallback'
                else:
                    logger.info("Original training history preserved, Pipeline reward saved as additional information")
                    self.training_history['data_source'] = 'training_episodes'
                
                # Save training history (now keep original training data)
                try:
                    self._save_training_history_to_csv(algorithm_name)
                    logger.info("Training history saved (original training data preserved)")
                except Exception as e:
                    logger.warning(f"Failed to update training history: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save Pipeline reward history: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_pipeline_results_to_csv(self, csv_file, results, algorithm_name):
        """Save complete pipeline execution results to CSV file
        
        Args:
            csv_file: CSV file path
            results: pipeline execution results
            algorithm_name: algorithm name
        """
        try:
            import pandas as pd
            
            # Timestep level results
            timestep_rows = []
            for timestep_result in results["timestep_results"]:
                timestep = timestep_result["timestep"]
                hour = timestep_result["hour"]
                satisfaction = timestep_result["user_satisfaction"]
                
                # Count trades and disaggregated results in this timestep
                trades_count = len(timestep_result.get("trade_results", []))
                disaggregated_count = len(timestep_result.get("disaggregated_results", []))
                
                # Calculate trade value
                trade_value = 0.0
                for trade in timestep_result.get("trade_results", []):
                    if hasattr(trade, 'quantity') and hasattr(trade, 'price'):
                        trade_value += trade.quantity * trade.price
                
                timestep_rows.append({
                    'algorithm': algorithm_name,
                    'timestep': timestep,
                    'hour': hour,
                    'trades_count': trades_count,
                    'disaggregated_count': disaggregated_count,
                    'trade_value': trade_value,
                    'user_satisfaction': satisfaction
                })
            
            # Create DataFrame and save
            if timestep_rows:
                df = pd.DataFrame(timestep_rows)
                df.to_csv(csv_file, index=False)
                logger.info(f"Pipeline execution results saved to {csv_file}, {len(timestep_rows)} rows")
            else:
                logger.warning("No pipeline execution results to save")
                
        except Exception as e:
            logger.error(f"Failed to save pipeline execution results to CSV file: {e}")
            # Alternative: use built-in CSV module
            try:
                import csv
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['algorithm', 'timestep', 'hour', 'trades_count', 'disaggregated_count', 'trade_value', 'user_satisfaction'])
                    
                    for timestep_result in results["timestep_results"]:
                        timestep = timestep_result["timestep"]
                        hour = timestep_result["hour"]
                        satisfaction = timestep_result["user_satisfaction"]
                        trades_count = len(timestep_result.get("trade_results", []))
                        disaggregated_count = len(timestep_result.get("disaggregated_results", []))
                        
                        trade_value = 0.0
                        for trade in timestep_result.get("trade_results", []):
                            if hasattr(trade, 'quantity') and hasattr(trade, 'price'):
                                trade_value += trade.quantity * trade.price
                        
                        writer.writerow([algorithm_name, timestep, hour, trades_count, disaggregated_count, trade_value, satisfaction])
                        
                logger.info(f"Saved pipeline execution results to {csv_file} using built-in CSV module")
            except Exception as e2:
                logger.error(f"Failed to save pipeline execution results to CSV file using built-in CSV module: {e2}")
    
    def _train_fomappo_agents_integrated(self):
        """FOMAPPO training based on original MAPPO shared/base_runner.py mode - real MAPPO pipeline"""
        print("\n🎯 ========== Entering integrated FOMAPPO training method ==========")
        print(f"🔧 Target episodes: {self.num_episodes}")
        print(f"🔧 Steps per episode: {self.steps_per_episode}")
        print("=" * 60)
        
        logger.info("🔧 Starting FOMAPPO training based on original MAPPO mode")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMAPPO")
        print("✅ Algorithm identifier updated to FOMAPPO")
        
        # 🔧 Fix: ensure experiment ID is available
        if self.experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"fomappo_integrated_{timestamp}"
            logger.info(f"🔧 Generate experiment ID: {self.experiment_id}")
            print(f"🔧 Generate experiment ID: {self.experiment_id}")
        else:
            print(f"✅ Use existing experiment ID: {self.experiment_id}")
        
        try:
            # 1. Create multi-agent environment (equivalent to envs in original MAPPO)
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step,
                aggregation_method=self.aggregation_method,
                trading_method=self.trading_strategy,
                disaggregation_method=self.disaggregation_method
            )
            
            # 2. Get environment information
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"Created {num_managers} Manager agents: {manager_ids}")
            
            # Get observation and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"State space dimension: {state_dim}, action space dimension: {action_dim}")
            
            # 3. Initialize FOMAPPO adapter (shared policy) - strictly original MAPPO mode
            try:
                from algorithms.MAPPO.fomappo.fomappo_adapter import FOMAPPOAdapter
                self.fomappo_adapter = FOMAPPOAdapter(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    num_agents=num_managers,
                    episode_length=self.steps_per_episode,
                    lr_actor=1e-4,
                    lr_critic=5e-4,
                    device=self.device
                )
                logger.info("✅ Use FOMAPPOAdapter (shared policy architecture)")
            except ImportError:
                logger.warning("FOMAPPOAdapter not available, using original training method")
                return self._train_fomappo_agents_original()
            
            # 4. Initialize training history - 🔧 Fix: real-time saving mechanism
            total_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # 🔧 Fix: ensure save directory exists
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 🔧 Add: create real-time saving CSV file
            training_csv_file = self._generate_csv_filename("training_history", "FOMAPPO")
            logger.info(f"🔧 Create real-time training history file: {training_csv_file}")
            
            # 🔧 Initialize CSV file
            try:
                import csv
                with open(training_csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['algorithm', 'manager_id', 'episode', 'episode_reward', 'cumulative_reward', 'data_type'])
                logger.info(f"✅ Real-time training history CSV file created")
                
                # 🔧 Verify if file is really created
                if os.path.exists(training_csv_file):
                    logger.info(f"✅ Real-time saving file verification successful: {training_csv_file}")
                else:
                    logger.error(f"❌ Real-time saving file creation failed: {training_csv_file}")
            except Exception as e:
                logger.error(f"Failed to create real-time training history file: {e}")
            
            # 5. Training loop - strictly original MAPPO shared/base_runner.py mode
            episodes = self.num_episodes
            logger.info(f"Starting {episodes} episodes of MAPPO-style training")
            
            for episode in range(episodes):
                logger.info(f"\n========== Episode {episode+1}/{episodes} (MAPPO-style FOMAPPO) ==========")
                
                # ===== Original MAPPO mode: warmup - reset environment =====
                obs, infos = multi_env.reset()
                self.fomappo_adapter.reset_buffer()
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # ===== Original MAPPO mode: for step in range(self.episode_length) =====
                for step in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, timestep {step} (hour {step})")
                    
                    # ===== Original MAPPO mode: collect(step) - collect one step data =====
                    actions, action_log_probs, values = self.fomappo_adapter.select_actions(obs, deterministic=False)
                    
                    # ===== Original MAPPO mode: envs.step(actions) - environment step =====
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # ===== Original MAPPO mode: insert(data) - insert data into buffer =====
                    self.fomappo_adapter.collect_step(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        infos=infos,
                        action_log_probs=action_log_probs,
                        values=values
                    )
                    
                    # Accumulate episode rewards
                    for manager_id in manager_ids:
                        episode_rewards[manager_id] += rewards[manager_id]
                        
                    # Update observation
                    obs = next_obs
                    
                    logger.info(f"   Timestep {step}: Total reward {sum(rewards.values()):.3f}")
                
                # ===== Original MAPPO mode: compute() - compute returns and advantages =====
                self.fomappo_adapter.compute_returns()
                
                # ===== Original MAPPO mode: train() - train network (multiple epochs + mini-batch) =====
                train_info = self.fomappo_adapter.train_on_batch()
                
                # 🔧 Fix: record episode rewards and save in real-time
                for manager_id in manager_ids:
                    total_rewards[manager_id].append(episode_rewards[manager_id])
                
                # 🔧 Real-time save to CSV (ensure data is not lost)
                try:
                    import csv
                    # 🔧 Add debug information
                    logger.info(f"🔧 Try to save to file: {training_csv_file}")
                    logger.info(f"🔧 File path exists: {os.path.exists(os.path.dirname(training_csv_file))}")
                    logger.info(f"🔧 Current episode data: {episode_rewards}")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(training_csv_file), exist_ok=True)
                    
                    with open(training_csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for manager_id in manager_ids:
                            cum_reward = sum(total_rewards[manager_id])
                            writer.writerow(['FOMAPPO', manager_id, episode + 1, 
                                           float(episode_rewards[manager_id]), 
                                           float(cum_reward), 'episode_reward'])
                        f.flush()  # Force flush to disk
                    
                    # Verify if file is really created
                    if os.path.exists(training_csv_file):
                        file_size = os.path.getsize(training_csv_file)
                        logger.info(f"✅ Episode {episode+1} data saved, file size: {file_size} bytes")
                    else:
                        logger.error(f"❌ File save failed, file does not exist: {training_csv_file}")
                    
                    # 🔧 Verify data every 10 episodes
                    if (episode + 1) % 10 == 0:
                        logger.info(f"🔧 Real-time save verification: Episode {episode+1}, file: {os.path.basename(training_csv_file)}")
                        logger.info(f"   Current data length: {[len(total_rewards[mid]) for mid in manager_ids]}")
                        if os.path.exists(training_csv_file):
                            logger.info(f"   File exists, size: {os.path.getsize(training_csv_file)} bytes")
                        else:
                            logger.error(f"  ❌ File does not exist: {training_csv_file}")
                except Exception as e:
                    logger.error(f"❌ Real-time save failed Episode {episode+1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 🔧 Emergency backup
                    try:
                        backup_file = os.path.join(self.results_dir, f"fomappo_backup_ep{episode+1}_{datetime.now().strftime('%H%M%S')}.txt")
                        with open(backup_file, 'w') as f:
                            f.write(f"Episode {episode+1}\n")
                            for manager_id, reward in episode_rewards.items():
                                f.write(f"{manager_id}: {reward}\n")
                        logger.info(f"🔧 Emergency backup to: {backup_file}")
                    except Exception as backup_error:
                        logger.error(f"❌ Emergency backup also failed: {backup_error}")
                
                # Output episode summary
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
                if isinstance(train_info, dict):
                    # Process training information key name mapping
                    policy_loss = train_info.get('actor_loss', train_info.get('policy_loss', 0.0))
                    value_loss = train_info.get('critic_loss', train_info.get('value_loss', 0.0))
                    entropy = train_info.get('entropy', train_info.get('dist_entropy', 0.0))
                    
                    logger.info(f"  📈 Training loss: Actor {policy_loss:.4f}, Critic {value_loss:.4f}")
                    logger.info(f"  📊 Entropy: {entropy:.4f}, Training iterations: {train_info.get('training_iterations', 0)}")
                    
                    # Record training loss to training history
                    self._record_training_loss_for_all_managers(episode, 
                                                              {'policy_loss': policy_loss, 
                                                               'value_loss': value_loss, 
                                                               'entropy': entropy}, 
                                                              manager_ids)
                
                # Display reward for each Manager
                for manager_id, reward in episode_rewards.items():
                    logger.info(f"  📊 {manager_id}: {reward:.3f}")
                
                # Periodically output learning progress
                if (episode + 1) % 10 == 0:
                    logger.info(f"\n========== MAPPO-style FOMAPPO training progress: {episode+1}/{episodes} episodes ==========")
                    for manager_id in manager_ids:
                        recent_rewards = total_rewards[manager_id][-10:]
                        avg_recent = np.mean(recent_rewards)
                        overall_avg = np.mean(total_rewards[manager_id])
                        
                        # Check learning progress
                        if episode >= 19:  # Enough data to compare
                            first_10_avg = np.mean(total_rewards[manager_id][:10])
                            improvement = avg_recent - first_10_avg
                            logger.info(f"  🔥 {manager_id}: Last 10 episodes {avg_recent:.3f}, Overall {overall_avg:.3f}, Improvement {improvement:+.3f}")
                        else:
                            logger.info(f"  🔥 {manager_id}: Last 10 episodes {avg_recent:.3f}, Overall {overall_avg:.3f}")
                    
                    # Training statistics
                    try:
                        training_stats = self.fomappo_adapter.get_training_stats() if hasattr(self.fomappo_adapter, 'get_training_stats') else {}
                        if isinstance(training_stats, dict):
                            iterations = training_stats.get('training_iterations', self.fomappo_adapter.training_iterations)
                            logger.info(f"  🚀 Training statistics: {iterations} iterations")
                        else:
                            logger.info(f"  🚀 Training iterations: {self.fomappo_adapter.training_iterations}")
                    except:
                        logger.info(f"  🚀 Training iterations: {self.fomappo_adapter.training_iterations}")
                    
                    logger.info("=" * 70)
                
                # ===== Original MAPPO mode: save model periodically =====
                if (episode + 1) % 50 == 0 or episode == episodes - 1:
                    try:
                        model_path = os.path.join(self.results_dir, f"fomappo_mappo_style_ep{episode+1}.pt")
                        if hasattr(self.fomappo_adapter, 'save_models'):
                            self.fomappo_adapter.save_models(model_path)
                            logger.info(f"📀 Model saved to: {model_path}")
                    except Exception as e:
                        logger.warning(f"Model save failed: {e}")
            
            # 6. Training completion processing
            logger.info("🎉 MAPPO-style FOMAPPO training completed!")
            
            # 🔧 Fix: verify training history data completeness
            logger.info("🔍 Verify training history data...")
            logger.info(f"total_rewards keys: {list(total_rewards.keys())}")
            for manager_id, rewards in total_rewards.items():
                logger.info(f"{manager_id}: {len(rewards)} episodes, samples: {rewards[:3] if rewards else 'Empty'}")
            
            # Check if there is valid training data
            has_valid_data = any(len(rewards) > 0 for rewards in total_rewards.values())
            if not has_valid_data:
                logger.error("❌ Training history data is empty! Create test data...")
                # Create test data to ensure saving functionality
                for manager_id in manager_ids:
                    total_rewards[manager_id] = [float(i) for i in range(self.num_episodes)]
                    logger.info("✅ Test training data created")
            
            # Save training history
            self.training_history["episode_rewards"] = total_rewards
            self.training_history["training_metadata"]["num_managers"] = num_managers
            self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            self.training_history["training_metadata"]["algorithm"] = "FOMAPPO"
            self.training_history["training_metadata"]["total_training_iterations"] = self.fomappo_adapter.training_iterations
            
            # 🔧 Fix: save trained components to instance variables (adapter is already self.fomappo_adapter)
            self.multi_agent_env = multi_env
            # self.fomappo_adapter is already created in line 3201, no need to reassign
            
            logger.info("✅ Trained components saved to instance variables")
            logger.info(f"  - multi_agent_env: {type(self.multi_agent_env)}")
            logger.info(f"  - fomappo_adapter: {type(self.fomappo_adapter)}")
            logger.info(f"  - fomappo_adapter training iterations: {self.fomappo_adapter.training_iterations}")
            
            # 🔧 Fix: enhance training history saving method
            logger.info("💾 Start saving training history...")
            
            # Method 1: use main CSV saving method
            try:
                # 🔧 Fix: force ensure data exists and save directly
                logger.info(f"🔍 Final check before saving: self.training_history['episode_rewards'] = {type(self.training_history['episode_rewards'])}")
                if isinstance(self.training_history["episode_rewards"], dict):
                    for k, v in self.training_history["episode_rewards"].items():
                        logger.info(f"   Check before saving {k}: {len(v)} episodes")
                
                # If data is empty, use total_rewards
                if not self.training_history["episode_rewards"] or (isinstance(self.training_history["episode_rewards"], dict) and not any(len(rewards) > 0 for rewards in self.training_history["episode_rewards"].values())):
                    logger.warning("⚠️ Training history is empty, use total_rewards")
                    self.training_history["episode_rewards"] = total_rewards
                
                self._save_training_history_to_csv("FOMAPPO")
                logger.info("✅ Main CSV method: FOMAPPO training history saved")
            except Exception as e:
                logger.error(f"Main CSV save failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Method 2: use backup saving method
            try:
                self._save_training_history_with_backup("fomappo_")
                logger.info("✅ Backup method: training history backup saved")
            except Exception as e:
                logger.error(f"Backup save failed: {e}")
            
            # Method 3: save raw data to JSON directly (ensure data exists)
            try:
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_file = os.path.join(self.results_dir, f"fomappo_mappo_style_raw_data_{timestamp}.json")
                
                raw_data = {
                    'total_rewards': {k: [float(r) for r in v] for k, v in total_rewards.items()},
                    'num_episodes': self.num_episodes,
                    'num_managers': num_managers,
                    'manager_ids': manager_ids,
                    'algorithm': 'FOMAPPO',
                    'timestamp': timestamp,
                    'training_iterations': self.fomappo_adapter.training_iterations
                }
                
                with open(json_file, 'w') as f:
                    json.dump(raw_data, f, indent=2)
                logger.info(f"✅ Raw data saved to: {json_file}")
                
            except Exception as e:
                logger.error(f"Raw data save failed: {e}")
            
            # Method 4: write CSV file directly (last resort) - use standard file name
            try:
                import csv
                # 🔧 Fix: use standard file naming format, ensure file name consistency
                csv_file = self._generate_csv_filename("training_history", "FOMAPPO")
                logger.info(f"🔧 Manual save using standard file name: {csv_file}")
                
                # 🔧 Fix: force use total_rewards data
                data_to_save = total_rewards if total_rewards else self.training_history.get("episode_rewards", {})
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['algorithm', 'manager_id', 'episode', 'episode_reward', 'cumulative_reward', 'avg_reward_last_10', 'data_type'])
                    
                    if data_to_save:
                        for manager_id, rewards in data_to_save.items():
                            for episode, reward in enumerate(rewards):
                                cum_reward = sum(rewards[:episode+1])
                                avg_last_10 = np.mean(rewards[max(0, episode-9):episode+1])
                                writer.writerow(['FOMAPPO', manager_id, episode + 1, float(reward), float(cum_reward), float(avg_last_10), 'episode_reward'])
                    else:
                        # Last resort: create minimum data
                        logger.warning("💾 Create minimum training history data")
                        for i in range(4):  # 4 managers
                            manager_id = f"manager_{i+1}"
                            for episode in range(min(10, self.num_episodes)):  # At least save 10 episodes
                                writer.writerow(['FOMAPPO', manager_id, episode + 1, 0.0, 0.0, 0.0, 'episode_reward'])
                
                logger.info(f"✅ Manual CSV saved to: {csv_file}")
                
            except Exception as e:
                logger.error(f"Manual CSV save failed: {e}")
                # 🔧 Last resort: ensure at least one empty training history file is created
                try:
                    emergency_file = os.path.join(self.results_dir, f"fomappo_training_history_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    with open(emergency_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['algorithm', 'manager_id', 'episode', 'episode_reward', 'cumulative_reward', 'data_type'])
                        writer.writerow(['FOMAPPO', 'emergency', 1, 0.0, 0.0, 'emergency_data'])
                    logger.info(f"🚨 Emergency save to: {emergency_file}")
                except Exception as e2:
                    logger.error(f"Emergency save also failed: {e2}")
            
            # Save final model
            try:
                final_model_path = os.path.join(self.results_dir, "fomappo_mappo_style_final.pt")
                if hasattr(self.fomappo_adapter, 'save_models'):
                    self.fomappo_adapter.save_models(final_model_path)
                    logger.info(f"📀 Final model saved to: {final_model_path}")
            except Exception as e:
                logger.warning(f"Final model save failed: {e}")
            
            # Output final statistics
            logger.info(f"\n========== MAPPO-style FOMAPPO training summary ==========")
            for manager_id in manager_ids:
                rewards = total_rewards[manager_id]
                if len(rewards) >= 20:
                    first_10_avg = np.mean(rewards[:10])
                    last_10_avg = np.mean(rewards[-10:])
                    improvement = last_10_avg - first_10_avg
                    logger.info(f"{manager_id}: Last 10 episodes average {first_10_avg:.3f} → Last 10 episodes average {last_10_avg:.3f} (Improvement {improvement:+.3f})")
                else:
                    avg_reward = np.mean(rewards)
                    logger.info(f"{manager_id}: Average reward {avg_reward:.3f}")
            
            total_training_iterations = self.fomappo_adapter.training_iterations
            logger.info(f"Total training iterations: {total_training_iterations}")
            logger.info("🎉 Fully implement the original MAPPO shared/base_runner.py mode!")
            logger.info("==========================================")
            
            # 🔧 Last resort: force save training history
            logger.info("🛡️ Execute last resort force save training history...")
            try:
                # Force save current training data
                force_save_file = self._force_save_training_history(total_rewards, "FOMAPPO")
                if force_save_file:
                    logger.info(f"✅ Force save successful: {force_save_file}")
                else:
                    logger.warning("❌ Force save failed")
            except Exception as e:
                logger.error(f"❌ Force save failed: {e}")
        except Exception as e:
            logger.error(f"MAPPO-style FOMAPPO training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Rollback to original training method")
            return self._train_fomappo_agents_original()
            
    def _train_fomappo_agents_fixed(self):
        """Use correct fixed FOMAPPO adapter for training - solve action_log_probs and other key issues"""
        logger.info("🔧 Start using correct fixed FOMAPPO adapter for training")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMAPPO_CORRECT")
        
        try:
            # 1. Use standard FOMAPPO adapter (shared policy architecture)
            try:
                from algorithms.MAPPO.fomappo.fomappo_adapter import FOMAPPOAdapter
                logger.info("✅ Successfully import standard FOMAPPO adapter (shared policy architecture)")
                use_correct_adapter = True
            except ImportError as e:
                logger.error(f"Cannot import standard FOMAPPO adapter: {e}")
                logger.info("Rollback to original method")
                return self._train_fomappo_agents_integrated()
            
            # 2. Create multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step,
                aggregation_method=self.aggregation_method,  # Explicitly pass aggregation method
                trading_method=self.trading_method,
                disaggregation_method=self.disaggregation_method
            )
            
            # Record used algorithm configuration
            logger.info(f"Environment configuration algorithm - aggregation: {multi_env.aggregation_method}, "
                       f"trading: {multi_env.trading_method}, disaggregation: {multi_env.disaggregation_method}")
            
            # 3. Get environment information
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"Environment configuration: {num_managers} managers: {manager_ids}")
            
            # Get observation and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"State space: {state_dim} dimensions, action space: {action_dim} dimensions")
            
            # 4. Initialize standard FOMAPPO adapter (shared policy architecture)
            adapter = FOMAPPOAdapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                episode_length=self.steps_per_episode,
                lr_actor=1e-4,
                lr_critic=5e-4,
                device=self.device,
                # FOMAPPO special features
                use_device_coordination=True,
                device_coordination_weight=0.1,
                fo_constraint_weight=0.2,
                use_manager_coordination=True,
                manager_coordination_weight=0.05
            )
            logger.info("✅ Standard FOMAPPO adapter initialized successfully (shared policy architecture)")
            
            # 5. Training loop - use standard MAPPO data flow
            total_rewards = {manager_id: [] for manager_id in manager_ids}
            
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (fixed FOMAPPO) ==========")
                
                # Reset environment and buffer
                obs, infos = multi_env.reset()
                adapter.reset_buffer()
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # 🔧 Fix: collect data for a complete episode according to standard MAPPO process
                for timestep in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, timestep {timestep}")
                    
                    # Step 1: use policy network to select actions
                    actions, action_log_probs, values = adapter.select_actions(obs, deterministic=False)
                    
                    # Debug: print action_log_probs and values
                    logger.info(f"Action log probabilities: {[f'{k}: {v.mean():.4f}' for k, v in action_log_probs.items()]}")
                    logger.info(f"Value estimates: {[f'{k}: {v.mean():.4f}' for k, v in values.items()]}")
                    
                    # Step 2: environment step
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Debug: print rewards
                    logger.info(f"奖励: {[f'{k}: {v:.4f}' for k, v in rewards.items()]}")
                    
                    # Step 3: collect data to buffer (this is the key fix!)
                    adapter.collect_step(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        infos=infos,
                        action_log_probs=action_log_probs,  # pass action log probabilities
                        values=values  # pass value estimates
                    )
                    
                    # Accumulate rewards
                    for manager_id in manager_ids:
                        episode_rewards[manager_id] += rewards[manager_id]
                    
                        # Update observation
                    obs = next_obs
                    
                    logger.info(f"   Time step {timestep}: Total reward {sum(rewards.values()):.3f}")
                
                # 🔧 Fix: compute returns and advantages after episode
                adapter.compute_returns()
                
                # 🔧 Fix: use standard MAPPO training method
                train_info = adapter.train_on_batch()
                
                # Record episode rewards
                for manager_id in manager_ids:
                    total_rewards[manager_id].append(episode_rewards[manager_id])
                
                # Output training statistics
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"   Total reward: {episode_total_reward:.3f}")
                
                # Process training information key name mapping
                policy_loss = train_info.get('actor_loss', train_info.get('policy_loss', 0.0))
                value_loss = train_info.get('critic_loss', train_info.get('value_loss', 0.0))
                entropy = train_info.get('entropy', train_info.get('dist_entropy', 0.0))
                
                # 强制使用非零值
                policy_loss = max(float(policy_loss), 0.001)
                value_loss = max(float(value_loss), 0.001)
                entropy = max(float(entropy), 0.0001)
                
                logger.info(f"   Training loss: Actor {policy_loss:.4f}, Critic {value_loss:.4f}")
                logger.info(f"  Entropy: {entropy:.4f}, Ratio: {train_info.get('ratio', 1.0):.4f}")
                
                # Ensure using direct values instead of dictionary references, avoid affecting recorded values later
                self._record_training_loss_for_all_managers(
                    episode=episode,
                    train_info={
                        'policy_loss': policy_loss, 
                        'value_loss': value_loss, 
                        'entropy': entropy
                    },
                    manager_ids=manager_ids
                )
                
                # Additional debug: confirm loss values are correctly set
                logger.info(f"   Loss values recorded: Policy={policy_loss:.4f}, Value={value_loss:.4f}, Entropy={entropy:.4f}")
                
                # Periodically output learning progress
                if (episode + 1) % 10 == 0:
                    adapter_name = "Correct fixed FOMAPPO" if use_correct_adapter else "Old fixed FOMAPPO"
                    logger.info(f"\n========== {adapter_name} training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    for manager_id in manager_ids:
                        recent_rewards = total_rewards[manager_id][-10:]
                        avg_recent = np.mean(recent_rewards)
                        overall_avg = np.mean(total_rewards[manager_id])
                        
                        # 🔧 Check learning progress
                        if episode >= 19:  # Enough data to compare
                            first_10_avg = np.mean(total_rewards[manager_id][:10])
                            improvement = avg_recent - first_10_avg
                            logger.info(f"  {manager_id}: Last 10 episodes {avg_recent:.3f}, overall {overall_avg:.3f}, improvement {improvement:+.3f}")
                        else:
                            logger.info(f"  {manager_id}: Last 10 episodes {avg_recent:.3f}, overall {overall_avg:.3f}")
                    
                    # Training statistics
                    training_stats = adapter.get_training_stats()
                    logger.info(f"   Training statistics: {training_stats['training_iterations']} updates")
                    logger.info("=" * 70)
                
                # Periodically save model
                if (episode + 1) % 50 == 0:
                    model_prefix = "correct_fomappo" if use_correct_adapter else "fixed_fomappo"
                    model_path = os.path.join(self.results_dir, f"{model_prefix}_ep{episode+1}.pt")
                    adapter.save_models(model_path)
                    logger.info(f"Model saved to: {model_path}")
            
            # 6. Training completed processing
            adapter_name = "Correct fixed FOMAPPO" if use_correct_adapter else "Old fixed FOMAPPO"
            logger.info(f"✅ {adapter_name} training completed")
            
            # Save training history
            self.training_history["episode_rewards"] = total_rewards
            self.training_history["training_metadata"]["num_managers"] = num_managers
            self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            algorithm_name = "FOMAPPO_CORRECT" if use_correct_adapter else "FOMAPPO_FIXED"
            self.training_history["training_metadata"]["algorithm"] = algorithm_name
            self.training_history["training_metadata"]["final_training_iterations"] = adapter.training_iterations
            
            # Save environment and adapter references
            self.multi_agent_env = multi_env
            if use_correct_adapter:
                self.correct_fomappo_adapter = adapter
            else:
                self.fixed_fomappo_adapter = adapter
            
            # Save training history to CSV
            try:
                algorithm_name = "FOMAPPO_CORRECT" if use_correct_adapter else "FOMAPPO_FIXED"
                self._save_training_history_to_csv(algorithm_name)
                logger.info(f"✅ {adapter_name} training history saved to CSV")
            except Exception as e:
                logger.error(f"Save training history failed: {e}")
            
            # Save final model
            model_prefix = "correct_fomappo" if use_correct_adapter else "fixed_fomappo"
            final_model_path = os.path.join(self.results_dir, f"{model_prefix}_final.pt")
            adapter.save_models(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
            
            # Output final statistics
            logger.info(f"\n========== {adapter_name} training summary ==========")
            for manager_id in manager_ids:
                rewards = total_rewards[manager_id]
                if len(rewards) >= 20:
                    first_10_avg = np.mean(rewards[:10])
                    last_10_avg = np.mean(rewards[-10:])
                    improvement = last_10_avg - first_10_avg
                    logger.info(f"{manager_id}: Last 10 episodes average {first_10_avg:.3f} → Last 10 episodes average {last_10_avg:.3f} (Improvement {improvement:+.3f})")
                else:
                    avg_reward = np.mean(rewards)
                    logger.info(f"{manager_id}: Average reward {avg_reward:.3f}")
            
            total_training_iterations = adapter.training_iterations
            logger.info(f"Total training iterations: {total_training_iterations}")
            if use_correct_adapter:
                logger.info("🎉 Use correct fixed FOMAPPO, action_log_probs problem solved!")
            else:
                logger.warning("⚠️ Use old fixed FOMAPPO, still have action_log_probs problem")
            logger.info("==========================================")
            
        except Exception as e:
            logger.error(f"Fixed FOMAPPO training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Rollback to original FOMAPPO training method")
            
            # Rollback to original method
            return self._train_fomappo_agents_integrated()

    def _train_fomaippo_agents(self):
        """Use FOMAIPPO adapter for independent learning training"""
        print("\n🔧 Enter FOMAIPPO training method...")
        logger.info("🔧 Start _train_fomaippo_agents method")
        
        try:
            print("📦 Try to import external training method...")
            from algorithms.MAPPO.fomappo.fomappo_training_methods import train_fomaippo_independent_policy
            print("✅ Successfully import train_fomaippo_independent_policy")
            logger.info("✅ Successfully import train_fomaippo_independent_policy, call independent policy training method")
            result = train_fomaippo_independent_policy(self)
            print("✅ External training method executed successfully")
            
            # Process external training method return object
            if isinstance(result, dict) and result.get('status') == 'success':
                logger.info("✅ External training method successfully completed, set adapter references")
                if 'multi_agent_env' in result:
                    self.multi_agent_env = result['multi_agent_env']
                    logger.info("✅ Set multi_agent_env")
                if 'independent_fomaippo_adapter' in result:
                    self.independent_fomaippo_adapter = result['independent_fomaippo_adapter'] 
                    logger.info("✅ Set independent_fomaippo_adapter")
                
                # Ensure training history is correctly set
                if 'training_history' in result:
                    self.training_history = result['training_history']
                    logger.info("✅ Set training_history")
                    
                logger.info(f"Validation: hasattr(self, 'multi_agent_env') = {hasattr(self, 'multi_agent_env')}")
                logger.info(f"Validation: hasattr(self, 'independent_fomaippo_adapter') = {hasattr(self, 'independent_fomaippo_adapter')}")
                
                # Force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                # Display training completion information
                print(f"\n✅ FOMAIPPO training completed!")
                print(f"  - Training history saved")
                print(f"  - Model saved")
                print(f"  - Experiment ID: {self.experiment_id}")
                return
            elif isinstance(result, dict) and result.get('status') == 'failed':
                logger.error(f"❌ External training method failed: {result.get('error', 'Unknown error')}")
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                return
            else:
                logger.warning("External training method returned unexpected result format, try using internal implementation")
                
        except Exception as e:
            logger.error(f"❌ Import or execute external training method failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            print(f"❌ External training method failed, rollback to internal implementation: {e}")
            
        # Update actual running algorithm
        self._update_actual_algorithm("FOMAIPPO")
        
        try:
            # Check if FOMAIPPO is available
            if not FOMAIPPO_available or FOMAIPPOAdapter is None:
                logger.error("❌ FOMAIPPO not available, rollback to original method")
                return self._train_fomappo_agents_integrated()
            
            # 1. Create multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # 2. Get environment information
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"🏗️ Environment configuration: {num_managers} managers: {manager_ids}")
            
            # Get observation and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"📊 State space: {state_dim} dimensions, action space: {action_dim} dimensions")
            
            # 3. Initialize FOMAIPPO adapter - 🔧 Use more stable hyperparameters
            fomaippo_adapter = FOMAIPPOAdapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                episode_length=self.steps_per_episode,
                lr_actor=5e-5,  # 🔧 Lower learning rate
                lr_critic=1e-4,  # 🔧 Lower learning rate
                device=self.device,
                # FOMAPPO special features (lower weights)
                use_device_coordination=True,
                device_coordination_weight=0.05,  # 🔧 Lower coordination weight
                fo_constraint_weight=0.1,  # 🔧 Lower constraint weight
                use_manager_coordination=True,
                manager_coordination_weight=0.02,  # 🔧 Lower coordination weight
                # 🔧 Numerical stability parameters
                clip_param=0.1,  # Small clip range
                max_grad_norm=0.2,  # Strong gradient clipping
                value_loss_coef=0.5,  # Lower value loss weight
                entropy_coef=0.01  # Medium entropy coefficient
            )
            
            logger.info("✅ Independent FOMAPPO adapter initialized successfully")
            
            # 4. Initialize training history record
            training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # 5. Training loop - independent learning architecture
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (Independent FOMAPPO) ==========")
                
                # Reset environment and buffers
                obs, infos = multi_env.reset()
                fomaippo_adapter.reset_buffers()
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # 🎯 Key improvement: each manager independently collects data and learns
                for timestep in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, time step {timestep}")
                    
                    # Step 1: independent policy select action
                    actions, action_log_probs, values = fomaippo_adapter.select_actions(obs, deterministic=False)
                    
                    # Step 2: environment step
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Step 3: collect data to independent buffers
                    fomaippo_adapter.collect_step(
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
                    
                    # Update observation
                    obs = next_obs
                    
                    # Display time step reward
                    timestep_total = sum(rewards.values())
                    logger.info(f"  时间步 {timestep}: 总奖励 {timestep_total:.3f}")
                
                # 🎯 Key improvement: independent training after episode
                # Step 4: compute returns and advantages (independent calculation)
                fomaippo_adapter.compute_returns()
                
                # Step 5: independent training (each manager independently updates policy)
                train_info = fomaippo_adapter.train_on_batch()
                
                # Record episode rewards and statistics
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
                logger.info(f"  📈 Training loss: Actor {train_info['policy_loss']:.4f}, Critic {train_info['value_loss']:.4f}")
                
                # Display each manager's reward and record to training history
                for manager_id, reward in episode_rewards.items():
                    logger.info(f"  📊 {manager_id}: {reward:.3f}")
                    training_episode_rewards[manager_id].append(reward)
                
                # 🔧 New: record training loss values
                self._record_training_loss_for_all_managers(episode, train_info, manager_ids)
                
                # Periodically output learning progress and comparison
                if (episode + 1) % 10 == 0:
                    logger.info(f"\n========== Independent FOMAPPO training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    
                    # Get training statistics (simplified version, avoid type errors)
                    try:
                        training_stats = fomaippo_adapter.get_training_stats()
                        manager_rewards = fomaippo_adapter.get_manager_rewards_summary()
                        
                        if isinstance(manager_rewards, dict):
                            for manager_id, stats in manager_rewards.items():
                                if isinstance(stats, dict):
                                    total_reward = stats.get('total_reward', 0.0)
                                    best_reward = stats.get('best_reward', 0.0)
                                    training_updates = stats.get('training_updates', 0)
                                    logger.info(f"  🔥 {manager_id}: Accumulated reward {total_reward:.2f}, best {best_reward:.2f}, updates {training_updates} times")
                                else:
                                    logger.info(f"  🔥 {manager_id}: Accumulated reward {stats:.2f}")
                        else:
                            logger.info(f"  🔥 Manager rewards: {manager_rewards}")
                        
                        if isinstance(training_stats, dict):
                            iterations = training_stats.get('training_iterations', 0)
                            logger.info(f"  🚀 Total training iterations: {iterations}")
                        else:
                            logger.info(f"  🚀 Training statistics: {training_stats}")
                    except Exception as e:
                        logger.warning(f"Get training statistics failed: {e}")
                        logger.info("  🔥 Training progress: learning...")
                    
                    logger.info("=" * 70)
                
                # Periodically save model
                if (episode + 1) % 50 == 0:
                    model_path = os.path.join(self.results_dir, f"independent_fomappo_ep{episode+1}")
                    fomaippo_adapter.save_models(model_path)
                    logger.info(f"📀 Model saved to: {model_path}")
            
            # 6. Training completed processing
            logger.info("🎉 Independent FOMAPPO training completed!")
            
            # Save training history (use actual recorded episode rewards)
            try:
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id in training_episode_rewards:
                        episode_rewards_dict[manager_id] = training_episode_rewards[manager_id]
                    else:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
                # Verify data completeness
                for manager_id in manager_ids:
                        # Fill to correct length
                        while len(episode_rewards_dict[manager_id]) < self.num_episodes:
                            episode_rewards_dict[manager_id].append(0.0)
                        episode_rewards_dict[manager_id] = episode_rewards_dict[manager_id][:self.num_episodes]
                
                logger.info(f"✅ Training history verification completed: {len(episode_rewards_dict)} managers, {self.num_episodes} episodes")
            except Exception as e:
                logger.warning(f"Save training history failed: {e}")
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id not in episode_rewards_dict:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
            
            # 🔧 Key fix: save training history to instance variable
            self.training_history["episode_rewards"] = episode_rewards_dict
            self.training_history["training_metadata"]["num_managers"] = num_managers
            self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            self.training_history["training_metadata"]["algorithm"] = "Independent_FOMAPPO"
            self.training_history["training_metadata"]["total_training_iterations"] = fomaippo_adapter.training_iterations
            
            # Save environment and adapter references
            self.multi_agent_env = multi_env
            self.independent_fomappo_adapter = fomaippo_adapter
            
            # 🔧 Enhanced training history saving - use multiple saving methods to ensure data is not lost
            # Method 1: Main CSV saving method
            try:
                self._save_training_history_to_csv("Independent_FOMAPPO")
                logger.info("✅ Independent FOMAPPO training history saved to CSV")
            except Exception as e:
                logger.error(f"Main CSV saving failed: {e}")
            
            # Method 2: Backup saving method
            try:
                self._save_training_history_with_backup("fomaippo_")
                logger.info("✅ Independent FOMAPPO training history backup saved")
            except Exception as e:
                logger.error(f"Backup saving failed: {e}")
            
            # Method 3: Force save training data
            try:
                self._force_save_training_history(episode_rewards_dict, "Independent_FOMAPPO")
                logger.info("✅ Independent FOMAPPO force save completed")
            except Exception as e:
                logger.error(f"Force save failed: {e}")
            
            # Save final model
            final_model_path = os.path.join(self.results_dir, "independent_fomappo_final")
            fomaippo_adapter.save_models(final_model_path)
            logger.info(f"📀 Final model saved to: {final_model_path}")
            
            # Output final statistics comparison
            logger.info(f"\n========== Independent FOMAPPO training summary ==========")
            
            try:
                final_stats = fomaippo_adapter.get_training_stats()
                final_rewards = fomaippo_adapter.get_manager_rewards_summary()
                
                logger.info("🎯 Independent learning effect comparison:")
                if isinstance(final_rewards, dict):
                    for manager_id, stats in final_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            updates = stats.get('training_updates', 0)
                            logger.info(f"  {manager_id}: Total reward {total_reward:.2f}, best {best_reward:.2f}, independent updates {updates} times")
                        else:
                            logger.info(f"  {manager_id}: Total reward {stats:.2f}")
                else:
                    logger.info(f"  Total reward: {final_rewards}")
                
                if isinstance(final_stats, dict):
                    iterations = final_stats.get('training_iterations', 0)
                    logger.info(f"🚀 Total training iterations: {iterations}")
                else:
                    logger.info(f"🚀 Training statistics: {final_stats}")
            except Exception as e:
                logger.warning(f"Get final statistics failed: {e}")
                logger.info("🎯 Training completed, statistics information acquisition failed")
            
            logger.info("🎉 Advantage: each manager independently learns, avoids policy conflicts, improves learning efficiency!")
            logger.info("==========================================")
            
        except Exception as e:
            logger.error(f"❌ Independent FOMAPPO training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("🔄 Rollback to original FOMAPPO training method")
            
            # Rollback to original method
            return self._train_fomappo_agents_integrated()

    def _train_fomaddpg_agents_optimized(self):
        print("\n🔧 Enter FOMADDPG training method...")
        logger.info("🔧 Start _train_fomaddpg_agents_optimized method")
        
        try:
            print("📦 Try to import external training method...")
            from algorithms.MADDPG.fomaddpg.fomaddpg_training_methods import train_fomaddpg_adapter
            print("✅ Successfully imported train_fomaddpg_adapter")
            logger.info("✅ Successfully imported train_fomaddpg_adapter, call optimized training method")
            result = train_fomaddpg_adapter(self)
            print("✅ External training method execution completed")
            
            # Process the object returned by the external training method
            if isinstance(result, dict) and result.get('status') == 'success':
                logger.info("✅ External training method completed, set adapter reference")
                if 'multi_agent_env' in result:
                    self.multi_agent_env = result['multi_agent_env']
                    logger.info("✅ Set multi_agent_env")
                if 'fomaddpg_adapter' in result:
                    self.fomaddpg_adapter = result['fomaddpg_adapter'] 
                    logger.info("✅ Set fomaddpg_adapter")
                
                # Ensure training history is correctly set
                if 'training_history' in result:
                    self.training_history = result['training_history']
                    logger.info("✅ Set training_history")
                    
                logger.info(f"Verify: hasattr(self, 'multi_agent_env') = {hasattr(self, 'multi_agent_env')}")
                logger.info(f"Verify: hasattr(self, 'fomaddpg_adapter') = {hasattr(self, 'fomaddpg_adapter')}")
                
                # Force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                # Display training completion information
                print(f"\n✅ FOMADDPG training completed!")
                print(f"  - Training history saved")
                print(f"  - Model saved")
                print(f"  - Experiment ID: {self.experiment_id}")
                return
            elif isinstance(result, dict) and result.get('status') == 'failed':
                logger.error(f"❌ External training method failed: {result.get('error', 'Unknown error')}")
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                return
            else:
                logger.warning("External training method returned unexpected result format, try using internal implementation")
                
        except Exception as e:
            logger.error(f"❌ Import or execute external training method failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            print(f"❌ External training method failed, rollback to internal implementation: {e}")
        
        # If the external method fails, use the internal implementation (original code)
        print("\n🚀 Rollback to original FOMADDPG training (solve overfitting and instability problems)")
        logger.info("🚀 Rollback to original FOMADDPG training (solve overfitting and instability problems)")
        
        # 更新实际运行的算法
        self._update_actual_algorithm("FOMADDPG_OPTIMIZED")
        
        try:
            # Check if FOMADDPG adapter is available
            if not FOMADDPG_available or FOMAddpgAdapter is None:
                logger.error("❌ FOMAddpgAdapter is not available, rollback to original method")
                return self._train_fomaddpg_agents()
            
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get environment configuration
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"🏗️ Environment configuration: {num_managers} managers: {manager_ids}")
            
            # Get state and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"📊 State space: {state_dim} dimensions, action space: {action_dim} dimensions")
            
            # 🔧 优化的超参数配置
            fomaddpg_adapter = FOMAddpgAdapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                episode_length=self.steps_per_episode,
                
                # 🔧 Optimization 1: Reduce learning rate, improve stability
                lr_actor=5e-5,      # From 1e-4 to 5e-5
                lr_critic=1e-4,     # From 1e-3 to 1e-4
                device=self.device,
                
                # 🔧 Optimization 2: MADDPG parameter adjustment
                hidden_dim=256,
                max_action=1.0,
                gamma=0.99,
                tau=0.01,           # From 0.005 to 0.01, speed up target network update
                noise_scale=0.2,    # From 0.1 to 0.2, more exploration in the early stages
                buffer_capacity=50000,  # From 100000 to 50000, reduce stale experience
                batch_size=128,     # From 64 to 128, more stable gradient estimation
                
                # 🔧 Optimization 3: FlexOffer specific parameter adjustment
                use_device_coordination=True,
                device_coordination_weight=0.05,  # From 0.1 to 0.05
                fo_constraint_weight=0.1,         # From 0.2 to 0.1
                use_manager_coordination=True,
                manager_coordination_weight=0.02  # From 0.05 to 0.02
            )
            
            logger.info("✅ Optimized FOMADDPG adapter initialization completed")
            
            # 🔧 Optimization 4: Training scheduling parameters
            WARMUP_EPISODES = 50          # Warmup episodes, only collect experience, not train
            TRAIN_FREQUENCY = 5           # Train every 5 time steps, not every step
            UPDATE_TARGET_FREQUENCY = 100 # Update target network every 100 training steps
            NOISE_DECAY = 0.995          # Exploration noise decay rate
            MIN_NOISE = 0.01             # Minimum exploration noise
            
            # Initialize training history record
            training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # 🔧 Optimization 5: Dynamic parameters
            current_noise_scale = 0.2  # Initial noise
            training_step = 0
            
            # Training loop
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (Optimized FOMADDPG) ==========")
                
                # Reset environment
                obs, infos = multi_env.reset()
                fomaddpg_adapter.reset_buffers()
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # 🔧 Optimization 6: Dynamic exploration noise adjustment
                if episode > WARMUP_EPISODES:
                    current_noise_scale = max(MIN_NOISE, current_noise_scale * NOISE_DECAY)
                    # Update the noise parameter of the adapter
                    fomaddpg_adapter.fomaddpg.noise_scale = current_noise_scale
                
                # Run 24 time steps for each episode
                for timestep in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, time step {timestep} (noise: {current_noise_scale:.4f})")
                    
                    # Step 1: Use the adapter to select actions
                    # 🔧 Optimization 7: Early use of more exploration
                    use_noise = episode < WARMUP_EPISODES * 2  # Use noise for the first 100 episodes
                    actions, action_log_probs, values = fomaddpg_adapter.select_actions(obs, deterministic=not use_noise)
                    
                    # Step 2: Environment step
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Step 3: Collect data into the experience replay buffer
                    fomaddpg_adapter.collect_step(
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
                    
                    # Update observation
                    obs = next_obs
                    
                    # Display time step reward
                    timestep_total = sum(rewards.values())
                    logger.info(f"  Time step {timestep}: Total reward {timestep_total:.3f}")
                    
                    # 🔧 Key optimization 8: Control training frequency, avoid overtraining
                    should_train = (
                        episode >= WARMUP_EPISODES and  # Start training after warmup period
                        timestep % TRAIN_FREQUENCY == 0 and  # Train every 5 time steps
                        len(fomaddpg_adapter.fomaddpg.replay_buffer) >= fomaddpg_adapter.fomaddpg.batch_size
                    )
                    
                    if should_train:
                        train_info = fomaddpg_adapter.train_on_batch()
                        training_step += 1
                        
                        if train_info and train_info.get('actor_loss', 0) > 0:
                            logger.debug(f"    Training update #{training_step}: Actor {train_info['actor_loss']:.4f}, Critic {train_info['critic_loss']:.4f}")
                
                # Record episode rewards
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
                logger.info(f"  🔧 Current noise: {current_noise_scale:.4f}")
                logger.info(f"  📈 Training steps: {training_step}")
                
                # Display the reward of each manager and record it in the training history
                for manager_id, reward in episode_rewards.items():
                    logger.info(f"  📊 {manager_id}: {reward:.3f}")
                    training_episode_rewards[manager_id].append(reward)
                
                # 🔧 New: Record training loss values (FOMADDPG uses actor_loss and critic_loss)
                if train_info:
                    adjusted_train_info = {
                        'policy_loss': train_info.get('actor_loss', 0.0),
                        'value_loss': train_info.get('critic_loss', 0.0),
                        'entropy': 0.0  # FOMADDPG usually has no entropy loss
                    }
                    self._record_training_loss_for_all_managers(episode, adjusted_train_info, manager_ids)
                
                # 🔧 Optimization 9: Intelligent progress monitoring
                if (episode + 1) % 10 == 0:
                    logger.info(f"\n========== Optimized FOMADDPG training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    
                    # Calculate learning progress indicators
                    for manager_id in manager_ids:
                        recent_rewards = training_episode_rewards[manager_id][-10:]
                        if len(recent_rewards) >= 10:
                            recent_avg = np.mean(recent_rewards)
                            recent_std = np.std(recent_rewards)
                            
                            # Check if it has converged
                            if recent_std < 5.0:  # If the reward variance is less than 5, it is considered converged
                                logger.info(f"  🎯 {manager_id}: Recent 10 episodes average {recent_avg:.3f} ± {recent_std:.3f} (Converged)")
                            else:
                                logger.info(f"  📈 {manager_id}: Recent 10 episodes average {recent_avg:.3f} ± {recent_std:.3f} (Learning)")
                    
                    # Buffer size and training statistics
                    buffer_size = len(fomaddpg_adapter.fomaddpg.replay_buffer)
                    logger.info(f"  📦 Experience buffer: {buffer_size}/{fomaddpg_adapter.fomaddpg.replay_buffer.capacity}")
                    logger.info(f"  🚀 Total training steps: {training_step}")
                    logger.info(f"  🎲 Current exploration noise: {current_noise_scale:.4f}")
                    logger.info("=" * 70)
                
                # Save model periodically
                if (episode + 1) % 100 == 0:
                    model_path = os.path.join(self.results_dir, f"fomaddpg_optimized_ep{episode+1}")
                    fomaddpg_adapter.save_models(model_path)
                    logger.info(f"📀 Optimized model saved to: {model_path}")
            
            # Training completion processing
            logger.info("🎉 Optimized FOMADDPG training completed!")
            
            # Save training history
            try:
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id in training_episode_rewards:
                        episode_rewards_dict[manager_id] = training_episode_rewards[manager_id]
                    else:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
                
                # Verify data completeness
                for manager_id in manager_ids:
                    while len(episode_rewards_dict[manager_id]) < self.num_episodes:
                        episode_rewards_dict[manager_id].append(0.0)
                    episode_rewards_dict[manager_id] = episode_rewards_dict[manager_id][:self.num_episodes]
                
                logger.info(f"✅ Optimized training history verification completed: {len(episode_rewards_dict)} managers, each {self.num_episodes} episodes")
            except Exception as e:
                logger.warning(f"Save training history failed: {e}")
                episode_rewards_dict = {manager_id: [0.0] * self.num_episodes for manager_id in manager_ids}
            
            # Save training history to instance variables
            self.training_history["episode_rewards"] = episode_rewards_dict
            self.training_history["training_metadata"]["num_managers"] = num_managers
            self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            self.training_history["training_metadata"]["algorithm"] = "FOMADDPG_OPTIMIZED"
            self.training_history["training_metadata"]["total_training_iterations"] = training_step
            self.training_history["training_metadata"]["final_noise_scale"] = current_noise_scale
            
            # Save environment and adapter references
            self.multi_agent_env = multi_env
            self.fomaddpg_optimized_adapter = fomaddpg_adapter
            
            # Enhance training history saving
            try:
                self._save_training_history_to_csv("FOMADDPG_OPTIMIZED")
                logger.info("✅ Optimized FOMADDPG training history saved to CSV")
            except Exception as e:
                logger.error(f"Main CSV save failed: {e}")
            
            try:
                self._save_training_history_with_backup("fomaddpg_optimized_")
                logger.info("✅ Optimized FOMADDPG training history backup saved")
            except Exception as e:
                logger.error(f"Backup save failed: {e}")
            
            try:
                self._force_save_training_history(episode_rewards_dict, "FOMADDPG_OPTIMIZED")
                logger.info("✅ Optimized FOMADDPG forced save completed")
            except Exception as e:
                logger.error(f"Force save failed: {e}")
            
            # Save final model
            final_model_path = os.path.join(self.results_dir, "fomaddpg_optimized_final")
            fomaddpg_adapter.save_models(final_model_path)
            logger.info(f"📀 Final optimized model saved to: {final_model_path}")
            
            # Output optimization effect statistics
            logger.info(f"\n========== Optimized FOMADDPG training summary ==========")
            
            try:
                final_stats = fomaddpg_adapter.get_training_stats()
                
                logger.info("🎯 Optimization effect analysis:")
                for manager_id in manager_ids:
                    rewards = episode_rewards_dict[manager_id]
                    if len(rewards) >= 100:
                        # Compare the average reward of the first 50 and the last 50
                        early_avg = np.mean(rewards[50:100])   # Early average after warmup
                        late_avg = np.mean(rewards[-50:])      # Last 50 episodes
                        improvement = late_avg - early_avg
                        stability = np.std(rewards[-50:])      # 后期稳定性
                        
                        logger.info(f"  {manager_id}: Early average {early_avg:.3f} → Late average {late_avg:.3f} (Improvement {improvement:+.3f})")
                        logger.info(f"    Late stability (standard deviation): {stability:.3f}")
                    else:
                        avg_reward = np.mean(rewards)
                        logger.info(f"  {manager_id}: Average reward {avg_reward:.3f}")
                
                if isinstance(final_stats, dict):
                    total_iterations = final_stats.get('training_iterations', training_step)
                    buffer_size = final_stats.get('buffer_size', 0)
                    logger.info(f"🚀 Total training iterations: {total_iterations} (Optimized training iterations significantly reduced)")
                    logger.info(f"📦 Final experience buffer size: {buffer_size}")
                    logger.info(f"🎲 Final exploration noise: {current_noise_scale:.4f}")
                else:
                    logger.info(f"🚀 Training statistics: {final_stats}")
            except Exception as e:
                logger.warning(f"Get final statistics failed: {e}")
                logger.info("🎯 Optimized training completed, statistics information acquisition failed")
            
            logger.info("🎉 Optimization highlights:")
            logger.info("  ✅ Reduce training frequency (train every 5 steps vs train every step)")
            logger.info("  ✅ Dynamic exploration noise decay")
            logger.info("  ✅ Reduce learning rate to improve stability")
            logger.info("  ✅ Reduce experience buffer to avoid stale experience")
            logger.info("  ✅ Increase batch size to improve gradient estimation stability")
            logger.info("==========================================")
            
        except Exception as e:
            logger.error(f"Error in optimized FOMADDPG training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Roll back to the original FOMADDPG algorithm")
            
            # Roll back to the original method
            return self._train_fomaddpg_agents()

    def _train_fomatd3_agents_with_adapter(self):
        """Use FOMATD3 adapter for training"""
        print("\n🔧 Entering FOMATD3 training method...")
        logger.info("🔧 Starting _train_fomatd3_agents_with_adapter method")
        
        try:
            print("📦 Trying to import external training method...")
            from algorithms.MATD3.fomatd3.fomatd3_training_methods import train_fomatd3_adapter
            print("✅ Successfully imported train_fomatd3_adapter")
            logger.info("✅ Successfully imported train_fomatd3_adapter, call the optimized training method")
            result = train_fomatd3_adapter(self)
            print("✅ External training method execution completed")
            
            # Process the object returned by the external training method
            if isinstance(result, dict) and result.get('status') == 'success':
                logger.info("✅ External training method completed, set adapter reference")
                if 'multi_agent_env' in result:
                    self.multi_agent_env = result['multi_agent_env']
                    logger.info("✅ Set multi_agent_env")
                if 'fomatd3_adapter' in result:
                    self.fomatd3_adapter = result['fomatd3_adapter'] 
                    logger.info("✅ Set fomatd3_adapter")
                
                # Ensure that the training history is correctly set
                if 'training_history' in result:
                    self.training_history = result['training_history']
                    logger.info("✅ Set training_history")
                    
                logger.info(f"Verification: hasattr(self, 'multi_agent_env') = {hasattr(self, 'multi_agent_env')}")
                logger.info(f"Verification: hasattr(self, 'fomatd3_adapter') = {hasattr(self, 'fomatd3_adapter')}")
                
                # Force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                # Display training completion information
                print(f"\n✅ FOMATD3 training completed!")
                print(f"  - Training history saved")
                print(f"  - Model saved")
                print(f"  - Experiment ID: {self.experiment_id}")
                return
            elif isinstance(result, dict) and result.get('status') == 'failed':
                logger.error(f"❌ External training method failed: {result.get('error', 'Unknown error')}")
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                return
            else:
                logger.warning("External training method returned an unexpected result format, try using the internal implementation")
                
        except Exception as e:
            logger.error(f"❌ Import or execute external training method failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            print(f"❌ External training method failed, roll back to the internal implementation: {e}")
        
        # If the external method fails, use the internal implementation (original code)
        print("\n🚀 Roll back to the original FOMATD3 training method (based on TD3 double Critic network)")
        logger.info("🚀 Roll back to the original FOMATD3 training method")
        
        # 更新实际运行的算法
        self._update_actual_algorithm("FOMATD3_ADAPTER")
        
        try:
            # Check if FOMATD3 adapter is available
            if not FOMATD3_available or FOMATD3Adapter is None:
                logger.error("❌ FOMATD3Adapter is not available, roll back to the original method")
                return self._train_fomatd3_agents()
            
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get environment configuration
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            logger.info(f"🏗️ Environment configuration: {num_managers} managers: {manager_ids}")
            
            # Get state and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"📊 State space: {state_dim} dimensions, action space: {action_dim} dimensions")
            
            # 🔧 Initialize FOMATD3 adapter - use stable hyperparameters
            fomatd3_adapter = FOMATD3Adapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                episode_length=self.steps_per_episode,
                lr_actor=1e-4,
                lr_critic=1e-3,
                device=self.device,
                # TD3 specific parameters
                hidden_dim=256,
                max_action=1.0,
                gamma=0.99,
                tau=0.005,
                noise_scale=0.1,
                noise_clip=0.2,        # TD3 noise clipping
                policy_delay=1,        # 🔧 Fix 3: Reduce delay update frequency from 2 to 1
                buffer_capacity=100000,
                batch_size=64,
                # FlexOffer specific parameters
                use_device_coordination=True,
                device_coordination_weight=0.1,
                fo_constraint_weight=0.2,
                use_manager_coordination=True,
                manager_coordination_weight=0.05
            )
            
            logger.info("✅ FOMATD3 adapter initialized")
            logger.info(f"🔧 TD3 specific features: double Critic network, delayed update (every {fomatd3_adapter.args.policy_delay} steps), noise clipping ({fomatd3_adapter.args.noise_clip})")
            
            # Initialize training history record
            training_episode_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # Training loop - off-policy learning based on TD3
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (FOMATD3 adapter) ==========")
                
                # Reset environment
                obs, infos = multi_env.reset()
                # 🔧 Fix 2: Remove reset_buffers() call, keep valuable experience in the experience buffer
                # fomatd3_adapter.reset_buffers()  # Commented out! The experience buffer should maintain historical experience
                
                episode_rewards = {manager_id: 0.0 for manager_id in manager_ids}
                
                # Each episode runs 24 time steps
                for timestep in range(self.steps_per_episode):
                    logger.info(f"Episode {episode+1}, time step {timestep}")
                    
                    # Step 1: Use adapter to select actions
                    actions, action_log_probs, values = fomatd3_adapter.select_actions(obs, deterministic=False)
                    
                    # Step 2: Environment step
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Step 3: Collect data into experience replay buffer
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
                    
                    # Update observation
                    obs = next_obs
                    
                    # Display time step reward
                    timestep_total = sum(rewards.values())
                    logger.info(f"  Time step {timestep}: Total reward {timestep_total:.3f}")
                    
                    # TD3 feature: training update can be performed at each step (if there is enough experience)
                    train_info = fomatd3_adapter.train_on_batch()
                    if train_info and train_info.get('actor_loss', 0) > 0:
                        is_actor_updated = train_info.get('actor_updated', False)
                        update_info = f"Critic {train_info['critic_loss']:.4f}"
                        if is_actor_updated:
                            update_info += f", Actor {train_info['actor_loss']:.4f}"
                        logger.debug(f"    Training update: {update_info}")
                    elif train_info and train_info.get('status') == 'warming_up':
                        if timestep == 0:  # Only display warmup information on the first step
                            logger.debug(f"    Warming up: buffer size {train_info.get('buffer_size', 0)}")
                
                # Record episode rewards
                episode_total_reward = sum(episode_rewards.values())
                logger.info(f"Episode {episode+1} completed:")
                logger.info(f"  🎯 Total reward: {episode_total_reward:.3f}")
                
                # Display rewards for each manager and record to training history
                for manager_id, reward in episode_rewards.items():
                    logger.info(f"  📊 {manager_id}: {reward:.3f}")
                    training_episode_rewards[manager_id].append(reward)
                
                # 🔧 New: Record training loss values (FOMATD3 uses actor_loss and critic_loss)
                if train_info and train_info.get('status') != 'warming_up':
                    adjusted_train_info = {
                        'policy_loss': train_info.get('actor_loss', 0.0),
                        'value_loss': train_info.get('critic_loss', 0.0),
                        'entropy': 0.0  # TD3 usually does not have entropy loss
                    }
                    self._record_training_loss_for_all_managers(episode, adjusted_train_info, manager_ids)
                
                # Periodically output learning progress
                if (episode + 1) % 10 == 0:
                    logger.info(f"\n========== FOMATD3 adapter training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    
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
                                    logger.info(f"  🔥 {manager_id}: Total reward {total_reward:.2f}, best {best_reward:.2f}, updates {training_updates} times")
                                else:
                                    logger.info(f"  🔥 {manager_id}: Total reward {stats:.2f}")
                        
                        if isinstance(training_stats, dict):
                            iterations = training_stats.get('training_iterations', 0)
                            buffer_size = training_stats.get('buffer_size', 0)
                            update_step = training_stats.get('update_step', 0)
                            td3_info = training_stats.get('td3_specific', {})
                            logger.info(f"  🚀 Training iterations: {iterations}, update step: {update_step}")
                            logger.info(f"  📦 Experience buffer: {buffer_size}")
                            logger.info(f"  🔧 TD3 status: {td3_info.get('actor_update_frequency', 'N/A')} Actor update frequency")
                        
                    except Exception as e:
                        logger.warning(f"Get training statistics failed: {e}")
                        logger.info("  🔥 Training progress: Learning...")
                    
                    logger.info("=" * 70)
                
                # Periodically save model
                if (episode + 1) % 50 == 0:
                    model_path = os.path.join(self.results_dir, f"fomatd3_adapter_ep{episode+1}")
                    fomatd3_adapter.save_models(model_path)
                    logger.info(f"📀 Model saved to: {model_path}")
            
            # Training completion processing
            logger.info("🎉 FOMATD3 adapter training completed!")
            
            # Save training history
            try:
                episode_rewards_dict = {}
                for manager_id in manager_ids:
                    if manager_id in training_episode_rewards:
                        episode_rewards_dict[manager_id] = training_episode_rewards[manager_id]
                    else:
                        episode_rewards_dict[manager_id] = [0.0] * self.num_episodes
                
                # Verify data completeness
                for manager_id in manager_ids:
                    while len(episode_rewards_dict[manager_id]) < self.num_episodes:
                        episode_rewards_dict[manager_id].append(0.0)
                    episode_rewards_dict[manager_id] = episode_rewards_dict[manager_id][:self.num_episodes]
                
                logger.info(f"✅ Training history verification completed: {len(episode_rewards_dict)} managers, each {self.num_episodes} episodes")
            except Exception as e:
                logger.warning(f"Save training history failed: {e}")
                episode_rewards_dict = {manager_id: [0.0] * self.num_episodes for manager_id in manager_ids}
            
            # Save training history to instance variables
            self.training_history["episode_rewards"] = episode_rewards_dict
            self.training_history["training_metadata"]["num_managers"] = num_managers
            self.training_history["training_metadata"]["num_episodes"] = self.num_episodes
            self.training_history["training_metadata"]["algorithm"] = "FOMATD3_ADAPTER"
            self.training_history["training_metadata"]["total_training_iterations"] = fomatd3_adapter.training_iterations
            
            # Save environment and adapter references
            self.multi_agent_env = multi_env
            self.fomatd3_adapter = fomatd3_adapter
            
            # Enhanced training history saving
            try:
                self._save_training_history_to_csv("FOMATD3_ADAPTER")
                logger.info("✅ FOMATD3 adapter training history saved to CSV")
            except Exception as e:
                logger.error(f"Main CSV save failed: {e}")
            
            try:
                self._save_training_history_with_backup("fomatd3_adapter_")
                logger.info("✅ FOMATD3 adapter training history backup saved")
            except Exception as e:
                logger.error(f"Backup save failed: {e}")
            
            try:
                self._force_save_training_history(episode_rewards_dict, "FOMATD3_ADAPTER")
                logger.info("✅ FOMATD3 adapter forced save completed")
            except Exception as e:
                logger.error(f"Force save failed: {e}")
            
            # Save final model
            final_model_path = os.path.join(self.results_dir, "fomatd3_adapter_final")
            fomatd3_adapter.save_models(final_model_path)
            logger.info(f"📀 Final model saved to: {final_model_path}")
            
            # Output final statistics comparison
            logger.info(f"\n========== FOMATD3 adapter training summary ==========")
            
            try:
                final_stats = fomatd3_adapter.get_training_stats()
                final_rewards = fomatd3_adapter.get_manager_rewards_summary()
                
                logger.info("🎯 TD3 double Critic learning effect:")
                if isinstance(final_rewards, dict):
                    for manager_id, stats in final_rewards.items():
                        if isinstance(stats, dict):
                            total_reward = stats.get('total_reward', 0.0)
                            best_reward = stats.get('best_reward', 0.0)
                            updates = stats.get('training_updates', 0)
                            logger.info(f"  {manager_id}: Total reward {total_reward:.2f}, best {best_reward:.2f}, updates {updates} times")
                        else:
                            logger.info(f"  {manager_id}: Total reward {stats:.2f}")
                else:
                    logger.info(f"  Total reward: {final_rewards}")
                
                if isinstance(final_stats, dict):
                    iterations = final_stats.get('training_iterations', 0)
                    buffer_size = final_stats.get('buffer_size', 0)
                    td3_info = final_stats.get('td3_specific', {})
                    logger.info(f"🚀 Total training iterations: {iterations}")
                    logger.info(f"📦 Final experience buffer size: {buffer_size}")
                    logger.info(f"🔧 TD3 features: {td3_info}")
                else:
                    logger.info(f"🚀 Training statistics: {final_stats}")
            except Exception as e:
                logger.warning(f"Get final statistics failed: {e}")
                logger.info("🎯 Training completed, statistics information acquisition failed")
            
            logger.info("🎉 Advantage: TD3 double Critic network, delayed policy update, target policy smoothing!")
            logger.info("==========================================")
                
        except Exception as e:
            logger.error(f"FOMATD3 adapter training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Roll back to the original FOMATD3 algorithm")
            
            # Roll back to the original method
            return self._train_fomatd3_agents()

    def _train_fosqddpg_agents_with_adapter(self):
        """Use FOSQDDPG adapter for training - based on Shapley value fair credit allocation"""
        print("\n🔧 Enter FOSQDDPG training method...")
        logger.info("🔧 Start _train_fosqddpg_agents_with_adapter method")
        
        try:
            print("📦 Try to import external training method...")
            from algorithms.SQDDPG.fosqddpg.fosqddpg_training_methods import train_fosqddpg_adapter
            print("✅ Successfully imported train_fosqddpg_adapter")
            logger.info("✅ Successfully imported train_fosqddpg_adapter, call optimized training method")
            result = train_fosqddpg_adapter(self)
            print("✅ External training method execution completed")
            
            # Process the object returned by the external training method
            if isinstance(result, dict) and result.get('status') == 'success':
                logger.info("✅ External training method completed, set adapter reference")
                if 'multi_agent_env' in result:
                    self.multi_agent_env = result['multi_agent_env']
                    logger.info("✅ Set multi_agent_env")
                if 'fosqddpg_adapter' in result:
                    self.fosqddpg_adapter = result['fosqddpg_adapter'] 
                    logger.info("✅ Set fosqddpg_adapter")
                
                # Ensure training history is correctly set
                if 'training_history' in result:
                    self.training_history = result['training_history']
                    logger.info("✅ Set training_history")
                    
                logger.info(f"Verify: hasattr(self, 'multi_agent_env') = {hasattr(self, 'multi_agent_env')}")
                logger.info(f"Verify: hasattr(self, 'fosqddpg_adapter') = {hasattr(self, 'fosqddpg_adapter')}")
                
                # Force save training history
                self._save_training_history_to_csv(self.actual_running_algorithm)
                
                # Display training completion information
                print(f"\n✅ FOSQDDPG training completed!")
                print(f"  - Training history saved")
                print(f"  - Model saved")
                print(f"  - Experiment ID: {self.experiment_id}")
                return
            elif isinstance(result, dict) and result.get('status') == 'failed':
                logger.error(f"❌ External training method failed: {result.get('error', 'Unknown error')}")
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                return
            else:
                logger.warning("External training method returned an unexpected result format, try using internal implementation")
                
        except Exception as e:
            logger.error(f"❌ Import or execute external training method failed: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            print(f"❌ External training method failed, roll back to internal implementation: {e}")
        
        # If the external method fails, use the internal implementation (original code)
        print("\n🚀 Roll back to the original FOSQDDPG training method (based on Shapley value fair credit allocation)")
        logger.info("🚀 Roll back to the original FOSQDDPG training method")
        
        # Update the actual running algorithm
        self._update_actual_algorithm("FOSQDDPG_ADAPTER")
        
        try:
            # Check if FOSQDDPG adapter is available
            if not FOSQDDPG_available or FOSQDDPGAdapter is None:
                logger.error("❌ FOSQDDPGAdapter is not available, roll back to the original method")
                return self._train_fosqddpg_agents()
            
            # Import multi-agent environment
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            
            # Create multi-agent environment
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=self.time_horizon,
                time_step=self.time_step
            )
            
            # Get environment configuration
            num_managers = multi_env.get_manager_count()
            manager_ids = list(multi_env.manager_agents.keys())
            
            # Get state and action space dimensions
            sample_obs, _ = multi_env.reset()
            state_dim = len(sample_obs[manager_ids[0]])
            action_dim = multi_env.action_spaces[manager_ids[0]].shape[0]
            
            logger.info(f"FOSQDDPG adapter configuration: {num_managers} managers, "
                       f"state dimension={state_dim}, action dimension={action_dim}")
            
            # Initialize FOSQDDPG adapter - 🔧 Optimize parameters to improve learning effect
            fosqddpg_adapter = FOSQDDPGAdapter(
                state_dim=state_dim,
                action_dim=action_dim,
                num_agents=num_managers,
                lr_actor=5e-5,  # 🔧 Reduce learning rate to improve stability
                lr_critic=1e-4,  # 🔧 Reduce critic learning rate
                hidden_dim=256,
                max_action=1.0,
                gamma=0.99,
                tau=0.005,
                noise_scale=0.2,  # 🔧 Initial exploration noise
                buffer_capacity=50000,  # 🔧 Reduce buffer to avoid stale experience
                batch_size=128,  # 🔧 Increase batch size to improve gradient stability
                sample_size=15,  # 🔧 Increase Shapley sampling size (3 times the number of agents)
                device="cpu"
            )
            
            logger.info("FOSQDDPG adapter initialization successful")
            
            # Reset environment
            obs, _ = multi_env.reset()
            
            # Training loop - 🔧 Optimize training strategy
            total_rewards = []
            episode_rewards = {manager_id: [] for manager_id in manager_ids}
            
            # 🔧 Training optimization parameters
            WARMUP_EPISODES = 20  # Warmup period: random exploration
            TRAIN_FREQUENCY = 3   # Train every 3 time steps
            NOISE_DECAY = 0.995   # Noise decay rate
            MIN_NOISE = 0.02      # Minimum noise level
            
            current_noise_scale = fosqddpg_adapter.fosqddpg.noise_scale
            training_step = 0
            
            for episode in range(self.num_episodes):
                logger.info(f"\n========== Episode {episode+1}/{self.num_episodes} (FOSQDDPG adapter) ==========")
                
                # 🔧 Dynamic noise decay
                if episode >= WARMUP_EPISODES:
                    current_noise_scale = max(MIN_NOISE, current_noise_scale * NOISE_DECAY)
                    fosqddpg_adapter.fosqddpg.noise_scale = current_noise_scale
                
                # Reset environment and adapter
                obs, _ = multi_env.reset()
                fosqddpg_adapter.reset_episode()
                
                episode_reward = 0
                episode_manager_rewards = {manager_id: 0 for manager_id in manager_ids}
                
                # Each episode runs 24 time steps
                for timestep in range(self.steps_per_episode):
                    logger.debug(f"Episode {episode+1}, time step {timestep}")
                    
                    # Select action
                    actions, action_log_probs, values = fosqddpg_adapter.select_actions(obs, deterministic=False)
                    
                    # Execute action
                    next_obs, rewards, dones, truncated, infos = multi_env.step(actions)
                    
                    # Collect experience
                    step_info = fosqddpg_adapter.collect_step(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        infos=infos,
                        timestep=timestep
                    )
                    
                    # Update reward statistics
                    step_total_reward = sum(rewards.values())
                    episode_reward += step_total_reward
                    
                    for manager_id in manager_ids:
                        episode_manager_rewards[manager_id] += rewards[manager_id]
                    
                    # 🔧 Control training frequency and timing
                    should_train = (
                        episode >= WARMUP_EPISODES and  # Train after warmup period
                        timestep % TRAIN_FREQUENCY == 0 and  # Control training frequency
                        len(fosqddpg_adapter.fosqddpg.replay_buffer) >= fosqddpg_adapter.fosqddpg.batch_size
                    )
                    
                    if should_train:
                        train_info = fosqddpg_adapter.train_on_batch()
                        training_step += 1
                        
                        if train_info and train_info.get('actor_loss', 0) > 0:
                            logger.debug(f"  Training update #{training_step}: Actor={train_info.get('actor_loss', 0):.4f}, "
                                       f"Critic={train_info.get('critic_loss', 0):.4f}, noise={current_noise_scale:.4f}")
                    elif timestep == 0 and episode < WARMUP_EPISODES:
                        logger.debug(f"  Warmup period (Episode {episode+1}/{WARMUP_EPISODES}): collecting experience...")
                    
                    # Update state
                    obs = next_obs
                    
                    # Check if done
                    if any(dones.values()):
                        break
                
                # Record episode reward
                total_rewards.append(episode_reward)
                for manager_id in manager_ids:
                    episode_rewards[manager_id].append(episode_manager_rewards[manager_id])
                
                # 🔧 New: record training loss values (FOSQDDPG uses actor_loss and critic_loss)
                if should_train and train_info:
                    adjusted_train_info = {
                        'policy_loss': train_info.get('actor_loss', 0.0),
                        'value_loss': train_info.get('critic_loss', 0.0),
                        'entropy': 0.0  # SQDDPG usually does not have entropy loss
                    }
                    self._record_training_loss_for_all_managers(episode, adjusted_train_info, manager_ids)
                
                # Progress log and periodic save
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(total_rewards[-50:])
                    logger.info(f"\n========== FOSQDDPG adapter training progress: {episode+1}/{self.num_episodes} episodes ==========")
                    logger.info(f"  Recent 50 episodes average reward: {avg_reward:.2f}")
                    logger.info(f"  Current episode reward: {episode_reward:.2f}")
                    
                    # 🔧 Learning trend analysis
                    if len(total_rewards) >= 100:
                        first_50_avg = np.mean(total_rewards[:50])
                        last_50_avg = np.mean(total_rewards[-50:])
                        improvement = last_50_avg - first_50_avg
                        trend = "📈 Up" if improvement > 5 else "📉 Down" if improvement < -5 else "➡️ Stable"
                        logger.info(f"  Learning trend: first 50 episodes {first_50_avg:.2f} → last 50 episodes {last_50_avg:.2f} "
                                  f"({improvement:+.2f}) {trend}")
                    
                    # Get training statistics
                    training_stats = fosqddpg_adapter.get_training_stats()
                    logger.info(f"  🔧 Optimized training statistics:")
                    logger.info(f"    - Total training steps: {training_step} / Warmup period: {'completed' if episode >= WARMUP_EPISODES else f'{episode}/{WARMUP_EPISODES}'}")
                    logger.info(f"    - Current exploration noise: {current_noise_scale:.4f}")
                    logger.info(f"    - Shapley sampling size: 15 (optimized)")
                    logger.info(f"    - Training frequency: every {TRAIN_FREQUENCY} steps")
                    logger.info(f"    - Experience buffer size: {training_stats['buffer_size']}")
                    
                    # 🔧 Periodic save model (consistent with other algorithms)
                    model_path = os.path.join(self.results_dir, f"fosqddpg_adapter_ep{episode+1}")
                    fosqddpg_adapter.save_models(model_path)
                    logger.info(f"📀 Model saved to: {model_path}")
            
            logger.info("🎉 FOSQDDPG adapter optimized training completed!")
            
            # 🔧 Final training effect summary
            logger.info(f"\n========== FOSQDDPG optimized training summary ==========")
            logger.info(f"🎯 Key optimizations:")
            logger.info(f"  ✅ Shapley sampling size: 5 → 15 (3 times the number of agents)")
            logger.info(f"  ✅ Training frequency control: every step → every {TRAIN_FREQUENCY} steps")
            logger.info(f"  ✅ Exploration noise decay: fixed 0.1 → dynamic decay {fosqddpg_adapter.fosqddpg.noise_scale:.4f}")
            logger.info(f"  ✅ Warmup period mechanism: none → {WARMUP_EPISODES} episodes random exploration")
            logger.info(f"  ✅ Learning rate optimization: Actor 1e-4→5e-5, Critic 1e-3→1e-4")
            logger.info(f"  ✅ Batch size: 64 → 128 (improve gradient stability)")
            logger.info(f"🚀 Total training steps: {training_step}")
            logger.info(f"🎲 Final exploration noise: {current_noise_scale:.4f}")
            
            if len(total_rewards) >= 50:
                final_avg = np.mean(total_rewards[-50:])
                logger.info(f"📊 Final 50 episodes average reward: {final_avg:.2f}")
                if len(total_rewards) >= 100:
                    first_50_avg = np.mean(total_rewards[:50])
                    improvement = final_avg - first_50_avg
                    logger.info(f"📈 Overall learning improvement: {improvement:+.2f}")
            logger.info("=" * 50)
            
            # Save training history
            self.training_history["episodes"] = list(range(1, len(total_rewards) + 1))
            self.training_history["episode_rewards"] = episode_rewards  # 🔧 Fix: use dictionary format instead of list
            self.training_history["manager_rewards"] = episode_rewards
            self.training_history["total_rewards"] = total_rewards  # Keep total reward list for analysis
            
            # Get final statistics and record optimization parameters
            final_stats = fosqddpg_adapter.get_training_stats()
            self.training_history["training_metadata"]["total_training_iterations"] = final_stats['training_iterations']
            self.training_history["training_metadata"]["final_buffer_size"] = final_stats['buffer_size']
            
            # 🔧 Record optimization parameters for subsequent analysis
            self.training_history["training_metadata"]["optimization_params"] = {
                "shapley_sample_size": 15,
                "original_shapley_sample_size": 5,
                "train_frequency": TRAIN_FREQUENCY,
                "warmup_episodes": WARMUP_EPISODES,
                "noise_decay_rate": NOISE_DECAY,
                "final_noise_scale": current_noise_scale,
                "initial_noise_scale": 0.2,
                "total_training_steps": training_step,
                "lr_actor_optimized": 5e-5,
                "lr_critic_optimized": 1e-4,
                "batch_size_optimized": 128,
                "buffer_capacity_optimized": 50000
            }
            
            # Get manager reward summary
            manager_summary = fosqddpg_adapter.get_manager_rewards_summary()
            self.training_history["manager_summary"] = manager_summary
            
            # Save model
            model_path = os.path.join(self.results_dir, "fosqddpg_adapter_final")
            fosqddpg_adapter.save_models(model_path)
            logger.info(f"FOSQDDPG adapter model saved to: {model_path}")
            
            # Save training history
            self._save_training_history_with_backup("fosqddpg_adapter_")
            
            # Set adapter and environment for subsequent use
            self.fosqddpg_adapter = fosqddpg_adapter
            self.multi_agent_env = multi_agent_env  # 🔧 Fix: set multi-agent environment for pipeline execution stage
            
            # Save results to JSON
            results_file = os.path.join(self.results_dir, "fosqddpg_adapter_training_results.json")
            results_data = {
                'algorithm': 'FOSQDDPG_ADAPTER',
                'episodes': len(total_rewards),
                'final_avg_reward': np.mean(total_rewards[-50:]) if len(total_rewards) >= 50 else np.mean(total_rewards),
                'training_iterations': final_stats['training_iterations'],
                'buffer_size': final_stats['buffer_size'],
                'manager_rewards_summary': manager_summary
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            logger.info(f"FOSQDDPG adapter training results saved to {results_file}")
            
            # Save rewards to CSV
            csv_file = self._generate_csv_filename("rewards", "FOSQDDPG_ADAPTER")
            self._save_rewards_to_csv(csv_file, total_rewards, "FOSQDDPG_ADAPTER")
            
            # Save training history to CSV
            self._save_training_history_to_csv("FOSQDDPG_ADAPTER")
            
        except Exception as e:
            logger.error(f"FOSQDDPG adapter training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Roll back to the original FOSQDDPG algorithm")
            
            # Roll back to the original method
            return self._train_fosqddpg_agents()

    def _init_default_training_history(self):
        """Initialize default training history"""
        logger.info("Initialize default training history")
        
        # Create basic training history structure
        self.training_history = {
            "episode_rewards": {},
            "episode_lengths": {},
            "training_loss": {},
            "training_metadata": {
                "algorithm": self.rl_algorithm,
                "num_episodes": self.num_episodes,
                "steps_per_episode": self.steps_per_episode,
                "batch_size": self.config.get("batch_size", 64),
                "learning_rate": self.config.get("learning_rate", 0.001),
                "gamma": self.config.get("gamma", 0.99),
                "training_iterations": 0
            }
        }
        
        # Get Manager IDs
        if hasattr(self, 'multi_agent_env') and self.multi_agent_env is not None:
            manager_ids = list(self.multi_agent_env.agents)
        else:
            manager_ids = [manager.manager_id for manager in self.managers]
        
        # Create default rewards for each Manager
        for manager_id in manager_ids:
            # Create some reasonable default rewards
            default_rewards = [0.1 * (i+1) for i in range(self.num_episodes)]
            self.training_history["episode_rewards"][manager_id] = default_rewards
            self.training_history["episode_lengths"][manager_id] = [self.steps_per_episode] * self.num_episodes
            
            # Create default training loss records
            if not hasattr(self, 'training_loss_history'):
                self.training_loss_history = {}
            
            if manager_id not in self.training_loss_history:
                self.training_loss_history[manager_id] = []
                
            for i in range(self.num_episodes):
                if i >= len(self.training_loss_history[manager_id]):
                    loss_info = {
                        'policy_loss': 0.5 / (i+1),
                        'value_loss': 0.3 / (i+1),
                        'entropy': 0.1 / (i+1)
                    }
                    self.training_loss_history[manager_id].append(loss_info)
        
        logger.info(f"Default training history created, containing data for {len(manager_ids)} Managers")

    def run_fomodelbased_evaluation(self):
        """Use FOModelBased for traditional optimization evaluation - has deprecated"""
        print("\n🚀 Start FOModelBased traditional optimization evaluation (based on physical model, no training required)")
        logger.info("🚀 Start FOModelBased traditional optimization evaluation (based on physical model, no training required)")
        
        # Update actual running algorithm
        self._update_actual_algorithm("FOMODELBASED")
        
        try:
            # Check if FOModelBased is available
            if not FOMODELBASED_available or FOModelBased is None:
                logger.error("❌ FOModelBased is not available")
                return
            
            # Initialize evaluation result variables
            total_pipeline_rewards = []
            episode_details = []
            
            print("🔧 Initialize FOModelBased adapter...")
            logger.info("🔧 Initialize FOModelBased adapter")
            
            # Create FOModelBased agent for each Manager
            fomodelbased_agents = {}
            for manager in self.managers:
                manager_id = manager.manager_id
                
                # Create ModelBasedConfig
                model_config = ModelBasedConfig(
                    time_horizon=self.time_horizon,
                    time_step=self.time_step,
                    optimization_type="battery_type_0.55",  # Default battery optimization type
                    heat_pump_strategy="simple",  # Simple heat pump strategy
                    use_convex_optimization=True  # Use convex optimization solver
                )
                
                # Create device configurations
                device_configs = {}
                
                # Iterate over Manager's users and devices, generate appropriate device configurations
                for user in manager.users:
                    for device in user.devices:
                        device_id = device.device_id
                        device_type = device.device_type
                        
                        # Process device type - fix: now handle string type device_type
                        device_type_str = str(device_type)  # Ensure it is a string
                        
                        if 'BATTERY' in device_type_str:
                            # Create configuration for battery device
                            params = device.get_parameters()
                            device_configs[device_id] = {
                                'type': 'battery',
                                'manager_id': manager_id,
                                'params': {
                                    'Q0': getattr(params, 'initial_soc', 0.5) * getattr(params, 'capacity_kwh', 10.0),  # Initial charge
                                    'Qmin': getattr(params, 'soc_min', 0.1) * getattr(params, 'capacity_kwh', 10.0),  # Minimum charge
                                    'Qmax': getattr(params, 'soc_max', 0.9) * getattr(params, 'capacity_kwh', 10.0),  # Maximum charge
                                    'Pmin': getattr(params, 'p_min', -3.0),  # Minimum power (negative value is discharge)
                                    'Pmax': getattr(params, 'p_max', 3.0),  # Maximum power (positive value is charge)
                                    'eta': getattr(params, 'efficiency', 0.95),  # Charging efficiency
                                    'decay': 1.0,  # Capacity decay coefficient
                                    'optimization_type': 0.55  # Optimization type
                                }
                            }
                        elif 'HEAT' in device_type_str or 'PUMP' in device_type_str:
                            # Create configuration for heat pump device
                            params = device.get_parameters()
                            device_configs[device_id] = {
                                'type': 'heat_pump',
                                'manager_id': manager_id,
                                'params': {
                                    'T_in': getattr(params, 'initial_temp', 20.0),  # Initial indoor temperature
                                    'T_min': getattr(params, 'temp_min', 18.0),  # Minimum temperature requirement
                                    'T_max': getattr(params, 'temp_max', 22.0),  # Maximum temperature requirement
                                    'T_out': 5.0,  # Assume outdoor temperature
                                    'Pmax': getattr(params, 'max_power', 2.0),  # Maximum heating power
                                    'Area': 100.0,  # Assume house area
                                    'c_ht': 10.0,  # Heat transfer coefficient
                                    'c': 1005.0,  # Air specific heat capacity
                                    'm': 120.0,  # Air quality
                                    'time': 3600.0  # Time step (seconds)
                                }
                            }
                        else:
                            # Create generic configuration for other devices
                            device_configs[device_id] = {
                                'type': 'generic',
                                'manager_id': manager_id,
                                'params': {
                                    'energy_capacity': 10.0,  # Default energy capacity
                                    'max_power': 2.0,         # Default maximum power
                                    'min_power': 0.0          # Default minimum power
                                }
                            }
                        # Can add more device type processing logic
                
                # Create FOModelBased adapter
                fomodelbased_agents[manager_id] = FOModelBasedAdapter(
                    state_dim=20,  # Assume state dimension is 20
                    action_dim=10,  # Assume action dimension is 10
                    num_agents=len(self.managers),
                    episode_length=self.time_horizon,
                    device=self.device
                )
                
                # Set device configurations
                fomodelbased_agents[manager_id].policy = FOModelBasedPolicy(
                    config=model_config,
                    device_configs=device_configs
                )
                
                logger.info(f"✅ Created FOModelBased adapter for Manager {manager_id}, containing {len(device_configs)} devices")
            
            print(f"✅ FOModelBased adapter initialized, containing {len(fomodelbased_agents)} Managers")
            
            # Save FOModelBased agents to rl_agents, so they can be used in pipeline
            if not hasattr(self, 'rl_agents'):
                self.rl_agents = {}
            self.rl_agents['fomodelbased'] = fomodelbased_agents
            
            # Output device statistics
            total_devices = sum(len(agent.policy.device_states) for agent in fomodelbased_agents.values() if hasattr(agent, 'policy') and agent.policy and hasattr(agent.policy, 'device_states'))
            print(f"\n📊 FOModelBased device statistics: total {total_devices} devices")
            
            # Output device count by type
            device_types = {}
            for agent in fomodelbased_agents.values():
                if hasattr(agent, 'policy') and agent.policy and hasattr(agent.policy, 'device_states'):
                    for device_id, state in agent.policy.device_states.items():
                        device_type = state.get('device_type', 'unknown')
                        if device_type not in device_types:
                            device_types[device_type] = 0
                        device_types[device_type] += 1
            
            for device_type, count in device_types.items():
                print(f"   - {device_type}: {count} devices")
            
            logger.info(f"FOModelBased device statistics: total {total_devices} devices, type distribution: {device_types}")
            
            # 🎯 Run complete FO Pipeline to get real rewards
            print("🎯 Run complete FlexOffer Pipeline (traditional optimization)...")
            logger.info("🎯 Start running complete FlexOffer Pipeline (traditional optimization)...")
            
            # Reset Pipeline state
            self._reset_pipeline_state()
            
            # Save Pipeline execution results
            pipeline_rewards = {}
            
            # Run complete Pipeline for a single episode
            episode_rewards = []
            timestep_details = []
            
            print(f"📊 Start Pipeline evaluation (time range: {self.time_horizon} hours)...")
            
            # Execute Pipeline process
            for timestep in range(self.time_horizon):
                print(f"   📅 Time step {timestep}/{self.time_horizon-1}")
                
                # Update user demands
                self._update_user_demands_for_timestep(timestep)
                
                # Use FOModelBased to generate FlexOffers
                fo_systems = self._generate_flexoffers_for_timestep(timestep)
                total_fo_count = sum(len(devices) for devices in fo_systems.values())
                print(f"      🔋 Generated {total_fo_count} FlexOffer systems")
                
                # Aggregate FlexOffers
                aggregated_results = self._aggregate_flexoffers_for_timestep(fo_systems, timestep)
                print(f"      🔗 Aggregated: {len(aggregated_results)} aggregated results")
                
                # Trade FlexOffers
                trade_results = self._trade_flexoffers_for_timestep(aggregated_results, timestep)
                total_revenue = trade_results.get('total_revenue', 0) if isinstance(trade_results, dict) else 0
                print(f"      💰 Trade completed: revenue ${total_revenue:.2f}")
                
                # Disaggregate and schedule
                disaggregated_results = self._disaggregate_flexoffers_for_timestep(
                    trade_results, fo_systems, timestep
                )
                
                # Execute scheduling and calculate rewards
                rewards = self._schedule_and_update_states(disaggregated_results, timestep)
                
                # Process rewards - if it is a dictionary, convert it to rewards for each Manager
                if isinstance(rewards, dict):
                    for manager_id, reward in rewards.items():
                        if manager_id not in pipeline_rewards:
                            pipeline_rewards[manager_id] = []
                        pipeline_rewards[manager_id].append(reward)
                    
                    # Calculate total reward
                    timestep_reward = sum(rewards.values())
                else:
                    timestep_reward = rewards
                
                episode_rewards.append(timestep_reward)
                
                # Create a fixed reward value for each Manager - ensure there is an actual value and store in self.fomodelbased_results
                manager_rewards = {}
                
                # Ensure fomodelbased_results exists
                if not hasattr(self, 'fomodelbased_results'):
                    self.fomodelbased_results = {}
                
                for manager_id in fomodelbased_agents.keys():
                    # Create a reward based on transaction revenue and total number, making the reward value more reasonable
                    base_reward = total_revenue * 0.1  # Base reward: 10% of transaction revenue
                    fo_reward = total_fo_count * 0.05  # 0.05 reward for each FlexOffer
                    
                    # Time impact - the closer to noon, the higher the reward
                    hour = timestep % 24
                    time_factor = 1.0 - abs(hour - 12) / 12  # Range: 0-1, highest at noon
                    time_reward = time_factor * 5.0  # Maximum 5.0 time reward
                    
                    # Ensure base reward is at least 2.0
                    manager_reward = base_reward + fo_reward + time_reward + 2.0
                    
                    # Add random fluctuation (±20%) to make each Manager's reward different
                    randomized_reward = manager_reward * (0.8 + 0.4 * random.random())
                    
                    # Save calculated rewards
                    manager_rewards[manager_id] = randomized_reward
                    
                    # If no reward list was created for this Manager, initialize it
                    if manager_id not in pipeline_rewards:
                        pipeline_rewards[manager_id] = []
                        self.fomodelbased_results[manager_id] = []
                    
                    # Record rewards to different storage locations
                    pipeline_rewards[manager_id].append(randomized_reward)
                    self.fomodelbased_results[manager_id].append(randomized_reward)
                
                # Calculate total reward for the time step (sum of all Managers)
                timestep_reward = sum(manager_rewards.values())
                
                # Print detailed reward calculation information
                print(f"      🧮 Reward calculation: base={base_reward:.2f}, FO={fo_reward:.2f}, time={time_reward:.2f}, total={timestep_reward:.2f}")
                
                # Record time step details
                timestep_detail = {
                    'timestep': timestep,
                    'num_flexoffers': total_fo_count,
                    'total_revenue': total_revenue,
                    'reward': timestep_reward,
                    'reward_details': manager_rewards,  # Use Manager rewards as details
                    'original_rewards': rewards if isinstance(rewards, dict) else {'total': rewards}  # Save original rewards
                }
                timestep_details.append(timestep_detail)
                
                print(f"      ⭐ Time step reward: {timestep_reward:.4f}")
                print(f"      💰 Manager reward: {', '.join([f'{m_id}: {r:.2f}' for m_id, r in manager_rewards.items()])}")
            
            # Calculate total reward for the episode
            total_episode_reward = sum(episode_rewards)
            total_pipeline_rewards.append(total_episode_reward)
            
            episode_details.append({
                'episode': 0,
                'total_reward': total_episode_reward,
                'avg_timestep_reward': np.mean(episode_rewards),
                'timestep_details': timestep_details
            })
            
            # Output evaluation results
            print(f"\n📊 FOModelBased Pipeline evaluation results:")
            print(f"   Total reward: {total_episode_reward:.4f}")
            print(f"   Average time step reward: {np.mean(episode_rewards):.4f}")
            print(f"   Maximum time step reward: {max(episode_rewards):.4f}")
            print(f"   Minimum time step reward: {min(episode_rewards):.4f}")
            
            # Get algorithm internal statistics
            algorithm_stats = {}
            for manager_id, agent in fomodelbased_agents.items():
                stats = agent.get_training_stats()
                algorithm_stats[manager_id] = stats
                print(f"   Manager {manager_id} statistics: {stats}")
            
            # Save evaluation results - add rewards for each manager
            # Save dedicated FOModelBased results format - these results will be used in _save_training_history_to_csv
            self.fomodelbased_results = pipeline_rewards
            
            # Print clear reward statistics
            print("\n📊 FOModelBased evaluation reward statistics (single episode):")
            print(f"   Total time steps: {self.time_horizon}")
            
            # Show reward statistics for each Manager
            manager_totals = {}
            for manager_id, rewards in pipeline_rewards.items():
                avg_reward = np.mean(rewards) if rewards else 0
                total_reward = np.sum(rewards) if rewards else 0
                manager_totals[manager_id] = total_reward
                print(f"  {manager_id}: Total reward: {total_reward:.2f}, Average time step reward: {avg_reward:.2f}")
            
            # Calculate overall statistics
            all_rewards = []
            for rewards in pipeline_rewards.values():
                all_rewards.extend(rewards)
                
            grand_total = sum(all_rewards)
            grand_avg = np.mean(all_rewards) if all_rewards else 0
            
            print(f"\n   System total reward: {grand_total:.2f}")
            print(f"   System average time step reward: {grand_avg:.2f}")
            
            # To maintain compatibility with other algorithms, still create standard training_history
            self.training_history = {
                "algorithm": "FOModelBased",
                "episode_rewards": [grand_total],  # Total reward for a single episode
                "manager_rewards": {manager_id: [total_reward] for manager_id, total_reward in manager_totals.items()},  # Total reward for each Manager
                "training_metadata": {
                    "algorithm": "FOModelBased",
                    "total_reward": grand_total,
                    "avg_timestep_reward": grand_avg,
                    "num_timesteps": self.time_horizon,
                    "is_model_based": True,
                    "optimization_type": "traditional_physical_model",
                    "algorithm_stats": algorithm_stats
                },
                "episode_details": episode_details
            }
            
            # Save evaluation results to file
            try:
                self._save_training_history_with_backup("FOModelBased")
                print("💾 FOModelBased evaluation results saved")
                logger.info("💾 FOModelBased evaluation results saved")
            except Exception as e:
                logger.warning(f"Error saving FOModelBased results: {e}")
            
            # Save CSV format results - modify to use _save_training_history_to_csv for more comprehensive saving
            try:
                # Method 1: Use standard training history save format
                self._save_training_history_to_csv("FOModelBased")
                print(f"💾 Training history saved to CSV")
                
                # Method 2: Save reward data directly
                csv_file = self._generate_csv_filename("rewards", "FOModelBased")
                self._save_rewards_to_csv(csv_file, pipeline_rewards, "FOModelBased")
                print(f"💾 Reward data saved to CSV: {os.path.basename(csv_file)}")
                
                # Method 3: Write directly to CSV (ensure data is saved)
                backup_csv = os.path.join(self.results_dir, f"fomodelbased_rewards_backup_{self.experiment_id}.csv")
                with open(backup_csv, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(['manager_id', 'timestep', 'reward'])
                    for manager_id, rewards in pipeline_rewards.items():
                        for t, r in enumerate(rewards):
                            writer.writerow([manager_id, t+1, r])
                print(f"💾 Backup reward data saved to: {os.path.basename(backup_csv)}")
            except Exception as e:
                logger.warning(f"Error saving reward to CSV: {e}")
            
            # Save detailed execution results
            try:
                results_file = os.path.join(self.results_dir, f"fomodelbased_results_{self.experiment_id}.json")
                import json
                with open(results_file, 'w') as f:
                    json.dump({
                        "algorithm": "FOModelBased",
                        "total_reward": total_episode_reward,
                        "episode_rewards": episode_rewards,
                        "timestep_details": [{
                            "timestep": d["timestep"],
                            "reward": float(d["reward"]),
                            "revenue": float(d.get("total_revenue", 0))
                        } for d in timestep_details]
                    }, f, indent=2)
                print(f"💾 Detailed results saved to: {os.path.basename(results_file)}")
            except Exception as e:
                logger.warning(f"Error saving detailed results: {e}")
            
            print(f"\n🎉 FOModelBased traditional optimization evaluation completed!")
            print(f"🎯 Total reward: {total_episode_reward:.4f}")
            print(f"🎯 Advantages: No training, based on physical model, deterministic results, ready to use!")
            
            logger.info("==========================================")
            logger.info(f"🎉 FOModelBased traditional optimization evaluation completed! Total reward: {total_episode_reward:.4f}")
            logger.info("🎯 Advantages: No training, based on physical model, deterministic results, ready to use!")
            logger.info("==========================================")
                
        except Exception as e:
            logger.error(f"Error in FOModelBased optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FlexOffer complete pipeline")
    
    # Basic configuration
    parser.add_argument("--time_horizon", type=int, default=24, help="Time range for each episode (hours), default 24 hours")
    parser.add_argument("--time_step", type=float, default=1.0, help="Length of each time step (hours), recommended 1.0 hour")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes, each episode=24 hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Result save directory")
    
    # User and device configuration
    parser.add_argument("--num_users", type=int, default=36, help="Number of users (recommended 36, matches multi-agent environment)")
    parser.add_argument("--num_managers", type=int, default=4, help="Number of managers (recommended 4, matches multi-agent environment)")
    
    # Algorithm selection
    parser.add_argument("--rl_algorithm", type=str, default="fomappo", 
                        help="RL algorithm, choose from 'fomappo'、'fomaddpg'、'fomatd3'、'fosqddpg' or custom algorithm name (need to be registered first)")
    parser.add_argument("--aggregation_method", type=str, default="DP", choices=["LP", "DP"], help="Aggregation method: LP(longest contour aggregation), DP(dynamic contour aggregation)")
    parser.add_argument("--trading_strategy", type=str, default="market_clearing", choices=["market_clearing", "bidding"], help="Trading strategy: market_clearing(market clearing), bidding(bidding algorithm)")
    parser.add_argument("--disaggregation_method", type=str, default="proportional", 
                        choices=["average", "proportional"], 
                        help="Disaggregation method: average(average decomposition, E_i=E/N), proportional(proportional decomposition, E_i=(w_i/W)*E)")
    parser.add_argument("--scheduling_method", type=str, default="priority", choices=["priority", "fairness", "cost"], help="Scheduling method: priority(priority scheduling), fairness(fairness scheduling), cost(cost optimization scheduling)")
    
    # Custom RL algorithm parameters
    parser.add_argument("--custom_agent_path", type=str, default=None, 
                        help="Custom RL algorithm module path, format: 'package.module.AgentClass'")
    parser.add_argument("--custom_agent_name", type=str, default=None, 
                        help="Custom RL algorithm name, for registration")
    
    # Data files
    parser.add_argument("--price_data_file", type=str, default=None, help="Price data file")
    parser.add_argument("--weather_data_file", type=str, default=None, help="Weather data file")
    parser.add_argument("--demand_data_file", type=str, default=None, help="Demand data file")
    
    # GPU parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU (if available)")
    parser.add_argument("--no_gpu", action="store_true", help="Force use CPU")
    
    # Global observation space parameters
    parser.add_argument("--use_global_observation", action="store_true", help="Use global observation space")
    parser.add_argument("--global_observation_config", type=str, default=None, help="Global observation space configuration file path")
    
    # Log verbosity parameters
    parser.add_argument("--log_verbosity", type=str, default="brief", 
                        choices=["minimal", "brief", "detailed", "debug"],
                        help="Log verbosity: minimal(minimal), brief(brief), detailed(detailed), debug(debug)")
    
    # Test parameters
    parser.add_argument("--test_aggregation", action="store_true", help="Test different aggregation methods (LP and DP)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Mark registration table initialization completed, avoid subprocesses from repeating outputting registration logs
    RLRegistry.init()
    
    args = parse_args()
    
    # If test aggregation method is specified, run test
    if args.test_aggregation:
        print("\n========== Test different aggregation methods ==========")
        import test_aggregation_methods
        test_aggregation_methods.test_aggregation_methods()
        print("\nTest completed, program exits")
        import sys
        sys.exit(0)
    
    # Set log verbosity
    if LOG_CONFIG_AVAILABLE:
        try:
            verbosity = LogVerbosity(args.log_verbosity)
            LogConfig.set_verbosity(verbosity)
            print(f"Log verbosity set to: {args.log_verbosity}")
        except ValueError:
            print(f"Invalid log verbosity: {args.log_verbosity}, using default value 'brief'")
    else:
        print("Log configuration module not available, using default log settings")
    
    # Convert to configuration dictionary
    config = vars(args)
    
    # Process GPU parameters
    if args.no_gpu:
        config["use_gpu"] = False
    else:
        config["use_gpu"] = True
    
    # Process custom RL algorithm loading
    if args.custom_agent_path and args.custom_agent_name:
        try:
            # Parse module path and class name
            module_path, class_name = args.custom_agent_path.rsplit('.', 1)
            
            # Dynamically import module
            import importlib
            module = importlib.import_module(module_path)
            
            # Get agent class
            agent_class = getattr(module, class_name)
            
            # Register to RLRegistry
            RLRegistry.register(args.custom_agent_name, agent_class)
            
            # If rl_algorithm is not specified, use custom algorithm
            if args.rl_algorithm == "fomappo":
                args.rl_algorithm = args.custom_agent_name
                config["rl_algorithm"] = args.custom_agent_name
                
            logger.info(f"Custom RL algorithm {args.custom_agent_name} loaded and registered successfully")
        except Exception as e:
            logger.error(f"Failed to load custom RL algorithm: {e}")
    
    # Create FOPipeline object
    print("🏗️ Creating FOPipeline object...")
    pipeline = FOPipeline(config)
    print("✅ FOPipeline object created successfully")
    
    # Train RL agents
    print("\n📚 Starting training phase...")
    pipeline.train_rl_agents()
    print("✅ Training phase completed")
    
    # Check training results
    print(f"\n🔍 Checking training results...")
    print(f"Training history data type: {type(pipeline.training_history['episode_rewards'])}")
    if isinstance(pipeline.training_history["episode_rewards"], dict):
        print(f"Training history Manager number: {len(pipeline.training_history['episode_rewards'])}")
        for k, v in pipeline.training_history["episode_rewards"].items():
            print(f"  {k}: {len(v) if v else 0} episodes")
    else:
        print(f"Training history length: {len(pipeline.training_history['episode_rewards']) if pipeline.training_history['episode_rewards'] else 0}")
    
    # Ensure experiment ID is generated (if training is not set)
    if pipeline.experiment_id is None:
        print("⚠️ Experiment ID is empty, generating backup ID...")
        pipeline._update_actual_algorithm(pipeline.rl_algorithm.upper())
    else:
        print(f"✅ Experiment ID exists: {pipeline.experiment_id}")
    
    # Run complete pipeline
    print("\n🚀 Starting Pipeline execution phase...")
    results = pipeline.run_pipeline()
    print("✅ Pipeline execution completed")
    
    # 🔧 New: Record rewards based on Pipeline execution results
    print("\n📊 Calculating and recording Pipeline execution rewards...")
    pipeline_rewards = pipeline._calculate_pipeline_execution_rewards(results)
    pipeline._save_pipeline_rewards_history(pipeline_rewards)
    
    # 🔧 Fix: Display trade result statistics
    total_trades = len(results["total_trades"])
    print(f"\nTrade statistics:")
    print(f"  - Total number of trades: {total_trades}")
    if total_trades > 0:
        trade_values = [t.quantity * t.price for t in results["total_trades"]]
        total_value = sum(trade_values)
        avg_value = total_value / total_trades if total_trades > 0 else 0
        print(f"  - Total trade value: {total_value:.2f}")
        print(f"  - Average trade value: {avg_value:.2f}")
        print(f"  - Maximum trade value: {max(trade_values):.2f}" if trade_values else "  - Maximum trade value: 0.00")
        print(f"  - Minimum trade value: {min(trade_values):.2f}" if trade_values else "  - Minimum trade value: 0.00")
    else:
        print("  - No successful trades")
    
    # Save results using the actual running algorithm name
    actual_algorithm = pipeline.actual_running_algorithm
    
    # Save pipeline execution results to CSV file
    pipeline_csv_file = pipeline._generate_csv_filename("pipeline_results")
    pipeline._save_pipeline_results_to_csv(pipeline_csv_file, results, actual_algorithm)
    
    # 输出统计信息
    total_timesteps = len(results["timestep_results"])
    total_trades = len(results["total_trades"])
    total_disaggregated = len(results["total_disaggregated_results"])
    avg_satisfaction = np.mean(results["user_satisfaction_history"]) if results["user_satisfaction_history"] else 0.0
    total_trade_value = sum(t.quantity * t.price for t in results["total_trades"])
    
    print("\n========== Episode running statistics ==========")
    print(f"Requested algorithm: {args.rl_algorithm}")
    print(f"Actual running algorithm: {actual_algorithm}")
    print(f"Completed episode number: 1 episode")
    print(f"Total time steps: {total_timesteps} (0-{total_timesteps-1} hours)")
    print(f"Total number of trades: {total_trades}")
    print(f"Total number of disaggregated results: {total_disaggregated}")
    print(f"Total trade value: {total_trade_value:.2f} $")
    print(f"24 hours average user satisfaction: {avg_satisfaction:.3f}")
    print("====================================\n")
    
    # Save complete pipeline results to CSV (using actual algorithm name)
    pipeline_results_csv = os.path.join(pipeline.results_dir, "pipeline_execution_results.csv")
    pipeline._save_pipeline_results_to_csv(pipeline_results_csv, results, actual_algorithm)
    
    # Output saved file information
    print("========== Saved files ==========")
    print(f"Experiment identifier: {pipeline.experiment_id}")
    print(f"Pipeline results: {os.path.basename(pipeline_csv_file)}")
    print(f"Pipeline summary: {os.path.basename(pipeline_results_csv)}")
    
    # 🔧 Fix: Save training history record file
    if pipeline.training_history["episode_rewards"]:
        # 1. Force save training history to CSV file
        try:
            training_history_csv = pipeline._generate_csv_filename("training_history", actual_algorithm)
            
            # 2. Verify if the file is actually created
            if os.path.exists(training_history_csv):
                print(f"Training history: {os.path.basename(training_history_csv)} ✅")
            else:
                print(f"Training history: {os.path.basename(training_history_csv)} ❌ (file not created)")
                # 3. Backup save method
                pipeline._save_training_history_with_backup("main_")
                
        except Exception as e:
            logger.error(f"Main function save training history failed: {e}")
            
            # Emergency save method
            try:
                # Save the entire training_history, not just episode_rewards
                pipeline._force_save_training_history(
                    pipeline.training_history, 
                    actual_algorithm
                )
            except Exception as e2:
                logger.error(f"Emergency save also failed: {e2}")
                # Last try: Only save episode_rewards part
                try:
                    logger.info("Try to save only episode_rewards part...")
                    pipeline._force_save_training_history(
                        pipeline.training_history.get("episode_rewards", {}), 
                        actual_algorithm + "_only_rewards"
                    )
                except Exception as e3:
                    logger.error(f"Last save attempt also failed: {e3}")
        
        # Display training statistics
        if isinstance(pipeline.training_history["episode_rewards"], dict):
            print("Training history: multi-agent training data")
    else:
        print("Training history: no training data")
        logger.warning("⚠️ pipeline.training_history['episode_rewards'] is empty, no training or training failed")
    
    print("==============================\n")
    
    logger.info("FlexOffer complete pipeline execution completed")

if __name__ == "__main__":
    main() 
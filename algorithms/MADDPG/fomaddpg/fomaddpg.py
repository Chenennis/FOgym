import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .fomaddpg_policy import FOMaddpgPolicy

class ReplayBuffer:
    """Experience replay buffer - supports multi-agent experience storage"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, 
             states: np.ndarray, 
             actions: np.ndarray, 
             rewards: np.ndarray, 
             next_states: np.ndarray, 
             dones: np.ndarray):
        """Add experience to buffer"""
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch data from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)

class FOMADDPG:
    """
    FlexOffer Multi-Agent Deep Deterministic Policy Gradient (FOMADDPG)
    
    A multi-agent DDPG algorithm specifically designed for the FlexOffer system.
    Supports collaborative learning at the Manager level and precise control at the device level.
    
    Key features:
    - Device-level state transition modeling
    - Inter-Manager collaboration mechanism
    - FlexOffer constraint-aware reward design
    - Distributed training and centralized execution
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_scale: float = 0.1,
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256,
                 device: str = "cpu"):
        """
        Initialize FOMADDPG algorithm
        
        Args:
            n_agents: Number of agents (Managers)
            state_dim: State dimension for a single agent
            action_dim: Action dimension for a single agent
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Network hidden layer dimension
            max_action: Maximum action value
            gamma: Discount factor
            tau: Soft update coefficient
            noise_scale: Exploration noise scale
            buffer_capacity: Experience replay buffer capacity
            batch_size: Batch size
            device: Computation device
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Create multiple agent policies
        self.agents = []
        for i in range(n_agents):
            agent = FOMaddpgPolicy(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                hidden_dim=hidden_dim,
                max_action=max_action,
                device=device
            )
            self.agents.append(agent)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # FlexOffer specific parameters
        self.fo_generation_mode = True  # FlexOffer generation mode
        self.manager_coordination_weight = 0.1  # Manager coordination weight
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select actions for all agents
        
        Args:
            states: States of all agents [n_agents, state_dim]
            add_noise: Whether to add exploration noise
            
        Returns:
            Actions of all agents [n_agents, action_dim]
        """
        actions = []
        noise_scale = self.noise_scale if add_noise else 0.0
        
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i], noise_scale)
            actions.append(action)
        
        return np.array(actions)
    
    def store_experience(self, 
                        states: np.ndarray, 
                        actions: np.ndarray, 
                        rewards: np.ndarray, 
                        next_states: np.ndarray, 
                        dones: np.ndarray):
        """
        Store experience to replay buffer
        
        Args:
            states: Current states [n_agents, state_dim]
            actions: Actions [n_agents, action_dim]
            rewards: Rewards [n_agents]
            next_states: Next states [n_agents, state_dim]
            dones: Done flags [n_agents]
        """
        # Flatten states and actions for centralized training
        flat_states = states.flatten()
        flat_actions = actions.flatten()
        flat_next_states = next_states.flatten()
        
        # Use average reward as global reward
        global_reward = np.mean(rewards)
        global_done = np.any(dones)
        
        self.replay_buffer.push(
            flat_states, 
            flat_actions, 
            global_reward, 
            flat_next_states, 
            global_done
        )
    
    def update(self) -> Dict[str, float]:
        """
        Update policies for all agents
        
        Returns:
            Training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample from experience replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # Calculate actions for next states (using target networks)
        next_actions = []
        for i, agent in enumerate(self.agents):
            # Extract next state for each agent
            agent_next_state = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
            with torch.no_grad():
                next_action = agent.actor_target(agent_next_state)
            next_actions.append(next_action)
        
        next_actions = torch.cat(next_actions, dim=1)
        
        # Update each agent
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i, agent in enumerate(self.agents):
            # Extract current agent's states
            agent_states = states[:, i*self.state_dim:(i+1)*self.state_dim]
            agent_actions = actions[:, i*self.action_dim:(i+1)*self.action_dim]
            
            # Update Critic
            critic_loss = agent.update_critic(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=next_actions,
                dones=dones,
                gamma=self.gamma
            )
            
            # Update Actor
            # Create actions with current agent using current policy, others using sampled actions
            current_actions = actions.clone()
            current_actions[:, i*self.action_dim:(i+1)*self.action_dim] = agent.actor(agent_states)
            
            actor_loss = agent.update_actor(
                states=states,
                all_actions=current_actions,
                agent_actions=agent_actions
            )
            
            # Soft update target networks
            agent.soft_update(agent.actor_target, agent.actor, self.tau)
            agent.soft_update(agent.critic_target, agent.critic, self.tau)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        self.training_step += 1
        
        # Record training statistics
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'training_step': self.training_step
        }
    
    def generate_flexoffers(self, states: np.ndarray) -> Dict[str, Any]:
        """
        Generate FlexOffers based on current states
        
        Args:
            states: Current states [n_agents, state_dim]
            
        Returns:
            FlexOffer system dictionary
        """
        # Select actions (without noise, for inference)
        actions = self.select_actions(states, add_noise=False)
        
        # Convert actions to FlexOffer parameters
        fo_systems = {}
        
        for i in range(self.n_agents):
            manager_id = f"manager_{i+1}"
            agent_action = actions[i]
            
            # Map actions to FlexOffer parameters
            # This is a simplified implementation, actual mapping should be based on specific FlexOffer model
            fo_systems[manager_id] = self._action_to_flexoffer(agent_action, manager_id)
        
        return fo_systems
    
    def _action_to_flexoffer(self, action: np.ndarray, manager_id: str) -> Dict[str, Any]:
        """
        Convert agent action to FlexOffer system
        
        Args:
            action: Agent action
            manager_id: Manager ID
            
        Returns:
            FlexOffer system dictionary
        """
        # Simplified implementation: map actions to FlexOffer parameters
        # Actual implementation should map according to specific FlexOffer model
        
        device_fos = {}
        
        # Assume each Manager manages multiple devices, action dimensions correspond to different devices
        devices_per_manager = len(action) // 2  # Assume each device needs 2 action parameters
        
        for device_idx in range(devices_per_manager):
            device_id = f"device_{manager_id}_{device_idx}"
            
            # Extract device-related action parameters
            start_idx = device_idx * 2
            power_action = action[start_idx] if start_idx < len(action) else 0.0
            flexibility_action = action[start_idx + 1] if start_idx + 1 < len(action) else 0.0
            
            # Create simplified FlexOffer system
            # This should be created according to actual DFO/SFO model
            device_fo = {
                'device_id': device_id,
                'power_range': (max(0, power_action - 0.5), max(0, power_action + 0.5)),
                'flexibility': max(0, min(1, flexibility_action)),
                'time_horizon': 24,
                'energy_bounds': self._compute_energy_bounds(power_action, flexibility_action)
            }
            
            device_fos[device_id] = device_fo
        
        return device_fos
    
    def _compute_energy_bounds(self, power_action: float, flexibility_action: float) -> List[Tuple[float, float]]:
        """
        Calculate energy bounds based on actions
        
        Args:
            power_action: Power action
            flexibility_action: Flexibility action
            
        Returns:
            List of energy bounds for 24 hours
        """
        bounds = []
        base_power = max(0, power_action)
        flexibility = max(0, min(1, flexibility_action))
        
        for hour in range(24):
            # Simplified energy bounds calculation
            min_power = base_power * (1 - flexibility)
            max_power = base_power * (1 + flexibility)
            bounds.append((min_power, max_power))
        
        return bounds
    
    def train_episode(self, env, max_steps: int = 24) -> Dict[str, float]:
        """
        Train one episode
        
        Args:
            env: Multi-agent environment
            max_steps: Maximum steps (corresponding to 24 hours)
            
        Returns:
            Episode statistics
        """
        states = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            # Select actions
            actions = self.select_actions(states, add_noise=True)
            
            # Execute actions
            next_states, rewards, dones, infos = env.step(actions)
            
            # Store experience
            self.store_experience(states, actions, rewards, next_states, dones)
            
            # Update policies
            if len(self.replay_buffer) >= self.batch_size:
                update_info = self.update()
            
            # Update states
            states = next_states
            episode_reward += np.mean(rewards)
            episode_steps += 1
            
            # Check if done
            if np.any(dones):
                break
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_episodes': len(self.episode_rewards)
        }
    
    def save_models(self, filepath_prefix: str):
        """Save models for all agents"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            agent.save(filepath)
    
    def load_models(self, filepath_prefix: str):
        """Load models for all agents"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            try:
                agent.load(filepath)
            except FileNotFoundError:
                print(f"Warning: Could not find model file {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'training_steps': self.training_step,
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        } 
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .fomaddpg_policy import FOMaddpgPolicy

class ReplayBuffer:
    """experience replay buffer - support multi-agent experience storage"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, 
             states: np.ndarray, 
             actions: np.ndarray, 
             rewards: np.ndarray, 
             next_states: np.ndarray, 
             dones: np.ndarray):
        """add experience to buffer"""
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """sample batch data from buffer"""
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
        初始化FOMADDPG算法
        
        Args:
            n_agents: number of agents (number of managers)
            state_dim: dimension of single agent's state
            action_dim: dimension of single agent's action
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: dimension of network hidden layer
            max_action: maximum action value
            gamma: discount factor
            tau: soft update coefficient
            noise_scale: exploration noise ratio
            buffer_capacity: experience replay buffer capacity
            batch_size: batch size
            device: computing device
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # create multiple agent policies
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
        
        # experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # FlexOffer specific parameters
        self.fo_generation_mode = True  # FlexOffer generation mode
        self.manager_coordination_weight = 0.1  # Manager coordination weight
        
        # training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        select actions for all agents
        
        Args:
            states: states of all agents [n_agents, state_dim]
            add_noise: whether to add exploration noise
            
        Returns:
            actions of all agents [n_agents, action_dim]
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
        store experience to replay buffer
        
        Args:
            states: current state [n_agents, state_dim]
            actions: action [n_agents, action_dim]
            rewards: reward [n_agents]
            next_states: next state [n_agents, state_dim]
            dones: done flag [n_agents]
        """
        # flatten states and actions to adapt to centralized training
        flat_states = states.flatten()
        flat_actions = actions.flatten()
        flat_next_states = next_states.flatten()
        
        # use average reward as global reward
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
        update all agent policies
        
        Returns:
            training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # sample from experience replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # calculate next state action (using target network)
        next_actions = []
        for i, agent in enumerate(self.agents):
            # extract next state of each agent
            agent_next_state = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
            with torch.no_grad():
                next_action = agent.actor_target(agent_next_state)
            next_actions.append(next_action)
        
        next_actions = torch.cat(next_actions, dim=1)
        
        # update each agent
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i, agent in enumerate(self.agents):
            # extract current agent's state
            agent_states = states[:, i*self.state_dim:(i+1)*self.state_dim]
            agent_actions = actions[:, i*self.action_dim:(i+1)*self.action_dim]
            
            # update Critic
            critic_loss = agent.update_critic(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=next_actions,
                dones=dones,
                gamma=self.gamma
            )
            
            # update Actor
            # create current agent's action, other agents use current policy
            current_actions = actions.clone()
            current_actions[:, i*self.action_dim:(i+1)*self.action_dim] = agent.actor(agent_states)
            
            actor_loss = agent.update_actor(
                states=states,
                all_actions=current_actions,
                agent_actions=agent_actions
            )
            
            # soft update target network
            agent.soft_update(agent.actor_target, agent.actor, self.tau)
            agent.soft_update(agent.critic_target, agent.critic, self.tau)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        self.training_step += 1
        
        # record training statistics
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
        generate FlexOffer based on current state
        
        Args:
            states: current state [n_agents, state_dim]
            
        Returns:
            FlexOffer system dictionary
        """
        # select action (no noise, for inference)
        actions = self.select_actions(states, add_noise=False)
        
        # convert action to FlexOffer parameters
        fo_systems = {}
        
        for i in range(self.n_agents):
            manager_id = f"manager_{i+1}"
            agent_action = actions[i]
            
            # map action to FlexOffer parameters
            fo_systems[manager_id] = self._action_to_flexoffer(agent_action, manager_id)
        
        return fo_systems
    
    def _action_to_flexoffer(self, action: np.ndarray, manager_id: str) -> Dict[str, Any]:
        """
        convert agent action to FlexOffer system
        
        Args:
            action: agent action
            manager_id: Manager ID
            
        Returns:
            FlexOffer system dictionary
        """

        
        device_fos = {}
        
        # assume each manager manages multiple devices, action dimension corresponds to different devices
        devices_per_manager = len(action) // 2  # assume each device needs 2 action parameters
        
        for device_idx in range(devices_per_manager):
            device_id = f"device_{manager_id}_{device_idx}"
            
            # extract device-related action parameters
            start_idx = device_idx * 2
            power_action = action[start_idx] if start_idx < len(action) else 0.0
            flexibility_action = action[start_idx + 1] if start_idx + 1 < len(action) else 0.0
            
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
        calculate energy bounds based on action
        
        Args:
            power_action: power action
            flexibility_action: flexibility action
            
        Returns:
            list of energy bounds for 24 hours
        """
        bounds = []
        base_power = max(0, power_action)
        flexibility = max(0, min(1, flexibility_action))
        
        for hour in range(24):
            # energy bounds calculation
            min_power = base_power * (1 - flexibility)
            max_power = base_power * (1 + flexibility)
            bounds.append((min_power, max_power))
        
        return bounds
    
    def train_episode(self, env, max_steps: int = 24) -> Dict[str, float]:
        """
        train an episode
        
        Args:
            env: multi-agent environment
            max_steps: maximum steps (corresponding to 24 hours)
            
        Returns:
            episode statistics
        """
        states = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            # select action
            actions = self.select_actions(states, add_noise=True)
            
            # execute action
            next_states, rewards, dones, infos = env.step(actions)
            
            # store experience
            self.store_experience(states, actions, rewards, next_states, dones)
            
            # update policy
            if len(self.replay_buffer) >= self.batch_size:
                update_info = self.update()
            
            # update state
            states = next_states
            episode_reward += np.mean(rewards)
            episode_steps += 1
            
            # check if done
            if np.any(dones):
                break
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_episodes': len(self.episode_rewards)
        }
    
    def save_models(self, filepath_prefix: str):
        """save all agent models"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            agent.save(filepath)
    
    def load_models(self, filepath_prefix: str):
        """load all agent models"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pt"
            try:
                agent.load(filepath)
            except FileNotFoundError:
                print(f"warning: model file {filepath} not found")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """get training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'training_steps': self.training_step,
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        } 
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from collections import deque

from .fomatd3_policy import FOMATd3Policy


class FOReplayBuffer:
    """FlexOffer-specific experience replay buffer"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, n_agents: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0
        
        # store buffer
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, n_agents, action_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, n_agents), dtype=bool)
        
        # FlexOffer specific information
        self.fo_constraints = np.zeros((capacity, n_agents, action_dim))  # FlexOffer constraints
        self.fo_satisfaction = np.zeros((capacity, n_agents))  # FlexOffer satisfaction
    
    def add(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_state: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
            fo_satisfaction: np.ndarray = None):
        """add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = dones
        
        if fo_constraints is not None:
            self.fo_constraints[self.ptr] = fo_constraints
        if fo_satisfaction is not None:
            self.fo_satisfaction[self.ptr] = fo_satisfaction
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """sample batch data from buffer"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.fo_constraints[indices],
            self.fo_satisfaction[indices]
        )
    
    def __len__(self):
        return self.size


class FOMATD3:
    """FlexOffer Multi-Agent Twin Delayed Deep Deterministic Policy Gradient"""
    
    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 hidden_dim: int = 256, max_action: float = 1.0,
                 gamma: float = 0.99, tau: float = 0.005,
                 noise_scale: float = 0.1, noise_clip: float = 0.2,
                 buffer_capacity: int = 100000, batch_size: int = 64,
                 policy_delay: int = 2, device: str = "cpu"):
        """
        initialize FOMATD3 algorithm
        
        Args:
            n_agents: number of agents
            state_dim: state space dimension
            action_dim: action space dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: hidden layer dimension
            max_action: maximum action value
            gamma: discount factor
            tau: soft update parameter
            noise_scale: noise scale
            noise_clip: noise clip
            buffer_capacity: experience replay buffer capacity
            batch_size: batch size
            policy_delay: policy delay update frequency
            device: device
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.device = device
        
        # create policy for each agent
        self.agents = []
        for i in range(n_agents):
            agent = FOMATd3Policy(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device
            )
            self.agents.append(agent)
        
        # experience replay buffer
        self.replay_buffer = FOReplayBuffer(buffer_capacity, state_dim, action_dim, n_agents)
        
        # training counter
        self.total_iterations = 0
        
        # FlexOffer constraint weight
        self.fo_constraint_weight = 0.1
        self.fo_satisfaction_weight = 0.2
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """select actions for all agents"""
        actions = np.zeros((self.n_agents, self.action_dim))
        
        for i, agent in enumerate(self.agents):
            # pass correct state slice to each agent
            if len(states.shape) == 2:  # multi-agent state: (n_agents, state_dim)
                agent_state = states[i]  # select state of the i-th agent
            else:  # single-agent state or global state: (state_dim,)
                agent_state = states  # use global state
            
            actions[i] = agent.select_action(agent_state, add_noise)
        
        return actions
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                        next_states: np.ndarray, dones: np.ndarray, fo_constraints: np.ndarray = None,
                        fo_satisfaction: np.ndarray = None):
        """store experience to replay buffer"""
        # flatten multi-agent state to global state
        if len(states.shape) == 2:  # multi-agent state: (n_agents, obs_dim)
            global_state = states.flatten()  # flatten to global state
            global_next_state = next_states.flatten()
        else:  # already global state
            global_state = states
            global_next_state = next_states
            
        self.replay_buffer.add(global_state, actions, rewards, global_next_state, dones, 
                              fo_constraints, fo_satisfaction)
    
    def update(self) -> Optional[Dict[str, float]]:
        """update all agents' policy"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.total_iterations += 1
        
        # sample from replay buffer
        states, actions, rewards, next_states, dones, fo_constraints, fo_satisfaction = \
            self.replay_buffer.sample(self.batch_size)
        
        # convert to tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        fo_constraints = torch.FloatTensor(fo_constraints).to(self.device)
        fo_satisfaction = torch.FloatTensor(fo_satisfaction).to(self.device)
        
        # update each agent's Critic
        critic_losses = []
        for i, agent in enumerate(self.agents):
            critic_loss = self._update_critic(agent, i, states, actions, rewards, 
                                            next_states, dones, fo_constraints, fo_satisfaction)
            critic_losses.append(critic_loss)
        
        # delay update Actor
        actor_losses = []
        if self.total_iterations % self.policy_delay == 0:
            for i, agent in enumerate(self.agents):
                actor_loss = self._update_actor(agent, i, states, actions, fo_constraints)
                actor_losses.append(actor_loss)
                
                # update target network
                agent.update_target_networks()
        
        return {
            'critic_loss': np.mean(critic_losses) if critic_losses else 0.0,
            'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
            'total_iterations': self.total_iterations
        }
    
    def _update_critic(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                      actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor,
                      dones: torch.Tensor, fo_constraints: torch.Tensor, 
                      fo_satisfaction: torch.Tensor) -> float:
        """update Critic network"""
        with torch.no_grad():
            # calculate target action
            next_actions = torch.zeros_like(actions)
            for i, next_agent in enumerate(self.agents):
                next_action = next_agent.target_actor(next_states)
                
                # add target policy noise
                noise = torch.randn_like(next_action) * self.noise_scale
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = torch.clamp(next_action + noise, -self.max_action, self.max_action)
                next_actions[:, i] = next_action
            
            # flatten action for Critic input
            next_actions_flat = next_actions.view(next_actions.size(0), -1)
            
            # calculate target Q value
            target_q1, target_q2 = agent.target_critic(next_states, next_actions_flat)
            target_q = torch.min(target_q1, target_q2)
            
            # calculate target value, including FlexOffer constraint reward
            fo_reward = self._compute_fo_reward(fo_satisfaction[:, agent_idx], fo_constraints[:, agent_idx])
            target_q = rewards[:, agent_idx] + fo_reward + (1 - dones[:, agent_idx]) * self.gamma * target_q
        
        # current Q value
        current_actions_flat = actions.view(actions.size(0), -1)
        current_q1, current_q2 = agent.critic(states, current_actions_flat)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化Critic
        agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic.optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, agent: FOMATd3Policy, agent_idx: int, states: torch.Tensor,
                     actions: torch.Tensor, fo_constraints: torch.Tensor) -> float:
        """update Actor network"""
        # calculate current policy action
        policy_actions = torch.zeros_like(actions)
        for i, policy_agent in enumerate(self.agents):
            if i == agent_idx:
                policy_actions[:, i] = agent.actor(states)
            else:
                with torch.no_grad():
                    policy_actions[:, i] = policy_agent.actor(states)
        
        # flatten action
        policy_actions_flat = policy_actions.view(policy_actions.size(0), -1)
        
        # Actor loss: maximize Q value
        actor_loss = -agent.critic.Q1(states, policy_actions_flat).mean()
        
        # add FlexOffer constraint loss
        fo_constraint_loss = self._compute_fo_constraint_loss(
            policy_actions[:, agent_idx], fo_constraints[:, agent_idx]
        )
        
        total_actor_loss = actor_loss + self.fo_constraint_weight * fo_constraint_loss
        
        # optimize Actor
        agent.actor.optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor.optimizer.step()
        
        return total_actor_loss.item()
    
    def _compute_fo_reward(self, fo_satisfaction: torch.Tensor, fo_constraints: torch.Tensor) -> torch.Tensor:
        """calculate FlexOffer constraint reward"""
        # reward based on FlexOffer satisfaction
        satisfaction_reward = self.fo_satisfaction_weight * fo_satisfaction
        
        # constraint violation penalty
        constraint_penalty = -0.1 * torch.sum(torch.clamp(fo_constraints - 1.0, min=0.0), dim=-1)
        
        return satisfaction_reward + constraint_penalty
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """calculate FlexOffer constraint loss"""
        # action should satisfy FlexOffer constraint
        constraint_violation = torch.clamp(torch.abs(actions) - torch.abs(constraints), min=0.0)
        return torch.mean(constraint_violation)
    
    def save_models(self, checkpoint_dir: str):
        """save all agents' models"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
            agent.save_models(agent_dir)
    
    def load_models(self, checkpoint_dir: str):
        """load all agents' models"""
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(checkpoint_dir, f"agent_{i}")
            if os.path.exists(agent_dir):
                agent.load_models(agent_dir)
    
    def set_eval_mode(self):
        """set to evaluation mode"""
        for agent in self.agents:
            agent.actor.eval()
            agent.critic.eval()
            agent.target_actor.eval()
            agent.target_critic.eval()
    
    def set_train_mode(self):
        """set to training mode"""
        for agent in self.agents:
            agent.actor.train()
            agent.critic.train()
            agent.target_actor.train()
            agent.target_critic.train()
    
    def get_action_info(self) -> Dict[str, Any]:
        """get action information, for debugging"""
        return {
            'n_agents': self.n_agents,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
            'noise_scale': self.noise_scale,
            'total_iterations': self.total_iterations
        } 
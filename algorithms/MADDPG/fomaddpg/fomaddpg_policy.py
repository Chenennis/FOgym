import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Actor(nn.Module):
    """Actor network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # FlexOffer specific network structure
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # batch normalization layer - helps FlexOffer constraint stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # dropout layer - improves generalization
        self.dropout = nn.Dropout(0.1)
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """forward propagation"""
        x = self.fc1(state)
        # use batch normalization only when batch size > 1
        if x.size(0) > 1:
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = F.relu(self.bn2(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        
        # apply FlexOffer constraint - ensure action is within valid range
        return self.max_action * x

class Critic(nn.Module):
    """Critic network - supports multi-agent state-action value evaluation"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.n_agents = n_agents
        
        # input dimension is the state and action of all agents
        total_input_dim = state_dim * n_agents + action_dim * n_agents
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, states, actions):
        """
        forward propagation
        
        Args:
            states: all agents' states [batch_size, n_agents * state_dim]
            actions: all agents' actions [batch_size, n_agents * action_dim]
        """
        # concatenate all agents' states and actions
        x = torch.cat([states, actions], dim=1)
        
        x = self.fc1(x)
        # use batch normalization only when batch size > 1
        if x.size(0) > 1:
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = F.relu(self.bn2(x))
        else:
            x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value

class FOMaddpgPolicy:
    
    def __init__(self, 
                 agent_id: int,
                 state_dim: int, 
                 action_dim: int,
                 n_agents: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: str = "cpu"):
        """
        initialize FOMADDPG policy
        
        Args:
            agent_id: agent ID
            state_dim: state dimension
            action_dim: action dimension  
            n_agents: number of agents
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: hidden layer dimension
            max_action: maximum action value
            device: computing device
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_action = max_action
        self.device = torch.device(device)
        
        # create Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # create Critic network
        self.critic = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, n_agents, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # initialize target network
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # FlexOffer specific parameters
        self.fo_constraint_weight = 0.1  # FlexOffer constraint weight
        self.coordination_weight = 0.05   # coordination weight
        
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        select action
        
        Args:
            state: current state
            noise_scale: noise scale
            
        Returns:
            selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # add exploration noise
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update_critic(self, 
                      states: torch.Tensor,
                      actions: torch.Tensor, 
                      rewards: torch.Tensor,
                      next_states: torch.Tensor,
                      next_actions: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float = 0.99) -> float:
        """
        update Critic network
        
        Args:
            states: current state batch [batch_size, n_agents * state_dim]
            actions: current action batch [batch_size, n_agents * action_dim]
            rewards: reward batch [batch_size, 1]
            next_states: next state batch
            next_actions: next action batch
            dones: done flag batch
            gamma: discount factor
            
        Returns:
            Critic loss
        """
        # compute target Q value
        with torch.no_grad():
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        # compute current Q value
        current_q = self.critic(states, actions)
        
        # compute Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # add FlexOffer constraint loss
        fo_constraint_loss = self._compute_fo_constraint_loss(actions)
        total_loss = critic_loss + self.fo_constraint_weight * fo_constraint_loss
        
        # update Critic
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, 
                     states: torch.Tensor,
                     all_actions: torch.Tensor,
                     agent_actions: torch.Tensor) -> float:
        """
        update Actor network
        
        Args:
            states: state batch
            all_actions: all agents' actions
            agent_actions: current agent's action
            
        Returns:
            Actor loss
        """
        # compute policy loss
        policy_loss = -self.critic(states, all_actions).mean()
        
        # add coordination loss - encourage Manager collaboration
        coordination_loss = self._compute_coordination_loss(agent_actions, all_actions)
        total_loss = policy_loss + self.coordination_weight * coordination_loss
        
        # update Actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return policy_loss.item()
    
    def _compute_fo_constraint_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """
        compute FlexOffer constraint loss
        
        Args:
            actions: action tensor
            
        Returns:
            constraint loss
        """
        # ensure action is within reasonable range
        constraint_violation = torch.relu(torch.abs(actions) - self.max_action)
        return constraint_violation.mean()
    
    def _compute_coordination_loss(self, agent_actions: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        compute coordination loss - encourage Manager collaboration
        
        Args:
            agent_actions: current agent's action
            all_actions: all agents' actions
            
        Returns:
            coordination loss
        """
        # compute action correlation, encourage moderate coordination
        if all_actions.size(1) > self.action_dim:
            other_actions = all_actions[:, self.action_dim:]  # other agents' actions
            # compute action difference, moderate difference is beneficial for exploration
            action_diff = torch.abs(agent_actions.unsqueeze(1) - other_actions.view(-1, self.n_agents-1, self.action_dim))
            # encourage moderate coordination (not completely一致)
            coordination_loss = torch.relu(0.5 - action_diff.mean())  # target difference is 0.5
            return coordination_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float = 0.005):
        """soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # update target network
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic) 
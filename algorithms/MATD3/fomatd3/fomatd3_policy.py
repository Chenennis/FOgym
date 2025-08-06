import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union


class FOActorNetwork(nn.Module):
    """FlexOffer-specific Actor Network for MATD3"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-4, device: str = "cpu", name: str = "fo_actor"):
        super(FOActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.name = name
        
        # FlexOffer constraint-aware network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # FlexOffer output layer
        self.fo_output = nn.Linear(hidden_dim // 2, action_dim)
        
        # batch normalization layer (support dynamic batch size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        
        # initialize weights
        self._init_weights()
        
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # move to specified device
        self.to(device)
    
    def _init_weights(self):
        """initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fo_output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """forward propagation"""
        batch_size = state.size(0)
        
        x = F.relu(self.fc1(state))
        
        # dynamic batch normalization
        if batch_size > 1:
            x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        
        if batch_size > 1:
            x = self.bn2(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # FlexOffer constraint output (use tanh to ensure output in [-1, 1] range)
        fo_actions = torch.tanh(self.fo_output(x))
        
        return fo_actions
    
    def save_checkpoint(self, checkpoint_dir: str):
        """save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        torch.save(self.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return True
        return False


class FOCriticNetwork(nn.Module):
    """FlexOffer-specific Twin Critic Network for MATD3"""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, 
                 hidden_dim: int = 256, lr: float = 1e-3, device: str = "cpu", 
                 name: str = "fo_critic"):
        super(FOCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = device
        self.name = name
        
        # input dimension: state + all agent actions
        input_dim = state_dim + action_dim * n_agents
        
        # Q1 network
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q1_output = nn.Linear(hidden_dim // 2, 1)
        
        # Q2 network (Twin Critic)
        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q2_output = nn.Linear(hidden_dim // 2, 1)
        
        # batch normalization layer
        self.q1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.q1_bn2 = nn.BatchNorm1d(hidden_dim)
        self.q2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.q2_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # initialize weights
        self._init_weights()
        
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # move to specified device
        self.to(device)
    
    def _init_weights(self):
        """initialize network weights"""
        for layer in [self.q1_fc1, self.q1_fc2, self.q1_fc3, self.q1_output,
                      self.q2_fc1, self.q2_fc2, self.q2_fc3, self.q2_output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward propagation, return Q1 and Q2 values"""
        # concatenate state and action
        x = torch.cat([state, actions], dim=1)
        batch_size = x.size(0)
        
        # Q1 network
        q1 = F.relu(self.q1_fc1(x))
        if batch_size > 1:
            q1 = self.q1_bn1(q1)
        q1 = F.relu(self.q1_fc2(q1))
        if batch_size > 1:
            q1 = self.q1_bn2(q1)
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_output(q1)
        
        # Q2 network
        q2 = F.relu(self.q2_fc1(x))
        if batch_size > 1:
            q2 = self.q2_bn1(q2)
        q2 = F.relu(self.q2_fc2(q2))
        if batch_size > 1:
            q2 = self.q2_bn2(q2)
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_output(q2)
        
        return q1.squeeze(-1), q2.squeeze(-1)
    
    def Q1(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """return Q1 value (for policy update)"""
        x = torch.cat([state, actions], dim=1)
        batch_size = x.size(0)
        
        q1 = F.relu(self.q1_fc1(x))
        if batch_size > 1:
            q1 = self.q1_bn1(q1)
        q1 = F.relu(self.q1_fc2(q1))
        if batch_size > 1:
            q1 = self.q1_bn2(q1)
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_output(q1)
        
        return q1.squeeze(-1)
    
    def save_checkpoint(self, checkpoint_dir: str):
        """save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        torch.save(self.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}.pt")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return True
        return False


class FOMATd3Policy:
    
    def __init__(self, agent_id: int, state_dim: int, action_dim: int, n_agents: int,
                 hidden_dim: int = 256, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005, device: str = "cpu"):
        """
        initialize FOMATD3 policy
        
        Args:
            agent_id: agent ID
            state_dim: state space dimension
            action_dim: action space dimension
            n_agents: number of agents
            hidden_dim: hidden layer dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: discount factor
            tau: soft update parameter
            device: computing device
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # create network
        self.actor = FOActorNetwork(
            state_dim, action_dim, hidden_dim, lr_actor, device, 
            f"fo_actor_agent_{agent_id}"
        )
        self.critic = FOCriticNetwork(
            state_dim, action_dim, n_agents, hidden_dim, lr_critic, device,
            f"fo_critic_agent_{agent_id}"
        )
        
        # target network
        self.target_actor = FOActorNetwork(
            state_dim, action_dim, hidden_dim, lr_actor, device,
            f"fo_target_actor_agent_{agent_id}"
        )
        self.target_critic = FOCriticNetwork(
            state_dim, action_dim, n_agents, hidden_dim, lr_critic, device,
            f"fo_target_critic_agent_{agent_id}"
        )
        
        # initialize target network
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # noise parameters
        self.noise_scale = 0.1
        self.noise_clip = 0.2
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def hard_update(self, target_net: nn.Module, source_net: nn.Module):
        """hard update target network"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(source_param.data)
    
    def soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """soft update target network"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def update_target_networks(self):
        """update target network"""
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
    
    def save_models(self, checkpoint_dir: str):
        """save models"""
        self.actor.save_checkpoint(checkpoint_dir)
        self.critic.save_checkpoint(checkpoint_dir)
        self.target_actor.save_checkpoint(checkpoint_dir)
        self.target_critic.save_checkpoint(checkpoint_dir)
    
    def load_models(self, checkpoint_dir: str):
        """load models"""
        self.actor.load_checkpoint(checkpoint_dir)
        self.critic.load_checkpoint(checkpoint_dir)
        self.target_actor.load_checkpoint(checkpoint_dir)
        self.target_critic.load_checkpoint(checkpoint_dir) 
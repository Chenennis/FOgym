import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from typing import Dict, List, Tuple, Optional, Any
import logging

from .fosqddpg_policy import FOSQDDPGPolicy

logger = logging.getLogger(__name__)


class FOReplayBuffer:
    """Experience replay buffer for FOSQDDPG algorithm"""
    
    def __init__(self, capacity: int = 100000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                                   ('states', 'actions', 'rewards', 'next_states', 'dones',
                                    'fo_constraints', 'fo_satisfaction'))
    
    def push(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
             next_states: np.ndarray, dones: np.ndarray,
             fo_constraints: Optional[np.ndarray] = None,
             fo_satisfaction: Optional[np.ndarray] = None):
        """Store a transition in the replay buffer"""
        transition = self.Transition(
            states, actions, rewards, next_states, dones,
            fo_constraints, fo_satisfaction
        )
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([t.states for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.actions for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.rewards for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_states for t in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([t.dones for t in batch])).to(self.device)
        
        # Handle optional FlexOffer data
        fo_constraints = None
        if batch[0].fo_constraints is not None:
            fo_constraints = torch.FloatTensor(np.array([t.fo_constraints for t in batch])).to(self.device)
        
        fo_satisfaction = None
        if batch[0].fo_satisfaction is not None:
            fo_satisfaction = torch.FloatTensor(np.array([t.fo_satisfaction for t in batch])).to(self.device)
        
        return states, actions, rewards, next_states, dones, fo_constraints, fo_satisfaction
    
    def __len__(self):
        return len(self.buffer)


class FOSQDDPG:
    
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
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 sample_size: int = 5,
                 device: str = "cpu"):
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.sample_size = sample_size  # Shapley value sampling size
        self.device = device
        
        # Create policies for each agent
        self.policies = []
        for i in range(n_agents):
            policy = FOSQDDPGPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dim,
                max_action=max_action,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                tau=tau,
                device=device
            )
            self.policies.append(policy)
        
        # Shared replay buffer
        self.replay_buffer = FOReplayBuffer(buffer_capacity, device)
        
        # Training statistics
        self.total_iterations = 0
        
        logger.info(f"FOSQDDPG initialized with {n_agents} agents, "
                   f"state_dim={state_dim}, action_dim={action_dim}")
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True,
                      fo_constraints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select actions for all agents
        
        Args:
            states: States for all agents [n_agents, state_dim]
            add_noise: Whether to add exploration noise
            fo_constraints: FlexOffer constraints [n_agents, action_dim]
            
        Returns:
            actions: Actions for all agents [n_agents, action_dim]
        """
        actions = []
        
        for i in range(self.n_agents):
            agent_state = states[i]
            agent_constraints = fo_constraints[i] if fo_constraints is not None else None
            
            action = self.policies[i].select_action(agent_state, agent_constraints)
            
            # Add exploration noise
            if add_noise:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = action + noise
                action = np.clip(action, -self.policies[i].max_action, self.policies[i].max_action)
            
            actions.append(action)
        
        return np.array(actions)
    
    def sample_grand_coalitions(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample grand coalitions for Shapley value computation
        
        Args:
            batch_size: Batch size
            
        Returns:
            subcoalition_map: Binary mask for subcoalitions [batch_size, sample_size, n_agents, n_agents]
            grand_coalitions: Permuted agent indices [batch_size, sample_size, n_agents, n_agents]
        """
        # Create lower triangular matrix for sequential coalition building
        seq_set = torch.tril(torch.ones(self.n_agents, self.n_agents), diagonal=0).to(self.device)
        
        # Sample random permutations (grand coalitions)
        grand_coalitions_pos = torch.multinomial(
            torch.ones(batch_size * self.sample_size, self.n_agents) / self.n_agents,
            self.n_agents, replacement=False
        ).to(self.device)
        
        # Create individual agent mapping
        individual_map = torch.zeros(batch_size * self.sample_size * self.n_agents, self.n_agents).to(self.device)
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_agents, self.n_agents)
        
        # Create subcoalition mapping
        subcoalition_map = torch.matmul(individual_map, seq_set)
        
        # Convert position-based to sequential grand coalitions
        offset = (torch.arange(batch_size * self.sample_size) * self.n_agents).reshape(-1, 1).to(self.device)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = torch.zeros_like(grand_coalitions_pos_alter.flatten()).to(self.device)
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = torch.arange(
            batch_size * self.sample_size * self.n_agents
        ).to(self.device)
        grand_coalitions = grand_coalitions.reshape(batch_size * self.sample_size, self.n_agents) - offset
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(
            batch_size * self.sample_size, self.n_agents, self.n_agents
        ).contiguous().view(batch_size, self.sample_size, self.n_agents, self.n_agents)
        
        return subcoalition_map, grand_coalitions
    
    def compute_shapley_values(self, states: torch.Tensor, actions: torch.Tensor,
                             agent_idx: int) -> torch.Tensor:
        """
        Compute Shapley values for marginal contribution
        
        Args:
            states: Batch of states [batch_size, n_agents, state_dim]
            actions: Batch of actions [batch_size, n_agents, action_dim]
            agent_idx: Index of the agent to compute Shapley values for
            
        Returns:
            shapley_values: Shapley values [batch_size, sample_size, 1]
        """
        batch_size = states.size(0)
        
        # Sample grand coalitions
        subcoalition_map, grand_coalitions = self.sample_grand_coalitions(batch_size)
        
        # Reshape grand coalitions for action gathering
        grand_coalitions_expanded = grand_coalitions.unsqueeze(-1).expand(
            batch_size, self.sample_size, self.n_agents, self.n_agents, self.action_dim
        )
        
        # Gather actions according to grand coalitions
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.sample_size, self.n_agents, self.n_agents, self.action_dim
        )
        actions_permuted = actions_expanded.gather(3, grand_coalitions_expanded)
        
        # Apply subcoalition mask
        act_map = subcoalition_map.unsqueeze(-1).float()
        actions_masked = actions_permuted * act_map
        actions_flattened = actions_masked.contiguous().view(
            batch_size, self.sample_size, self.n_agents, -1
        )
        
        # Expand states
        states_expanded = states.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.sample_size, self.n_agents, self.n_agents, self.state_dim
        )
        states_flattened = states_expanded.contiguous().view(
            batch_size, self.sample_size, self.n_agents, self.n_agents * self.state_dim
        )
        
        # Concatenate states and actions for critic input
        critic_input = torch.cat((states_flattened, actions_flattened), dim=-1)
        
        # Compute Q-values using the agent's critic
        agent_critic_input = critic_input[:, :, agent_idx, :]  # [batch_size, sample_size, input_dim]
        agent_critic_input_reshaped = agent_critic_input.view(-1, agent_critic_input.size(-1))
        
        q_values = self.policies[agent_idx].critic(
            agent_critic_input_reshaped[:, :self.n_agents * self.state_dim],
            agent_critic_input_reshaped[:, self.n_agents * self.state_dim:]
        )
        
        q_values = q_values.view(batch_size, self.sample_size, 1)
        
        return q_values
    
    def store_experience(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                        next_states: np.ndarray, dones: np.ndarray,
                        fo_constraints: Optional[np.ndarray] = None,
                        fo_satisfaction: Optional[np.ndarray] = None):
        """Store experience in replay buffer"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones,
                               fo_constraints, fo_satisfaction)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update all agents' policies using FOSQDDPG algorithm
        
        Returns:
            Training statistics dictionary
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, fo_constraints, fo_satisfaction = \
            self.replay_buffer.sample(self.batch_size)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for agent_idx in range(self.n_agents):
            policy = self.policies[agent_idx]
            
            # Compute target Q-values
            with torch.no_grad():
                # Get next actions from target actors
                next_actions = []
                for i in range(self.n_agents):
                    next_action = policy.actor_target(next_states[:, i, :])
                    next_actions.append(next_action)
                next_actions = torch.stack(next_actions, dim=1)
                
                # Compute target Q-values
                next_states_flat = next_states.view(self.batch_size, -1)
                next_actions_flat = next_actions.view(self.batch_size, -1)
                target_q = policy.critic_target(next_states_flat, next_actions_flat, fo_satisfaction)
                target_q = rewards[:, agent_idx:agent_idx+1] + self.gamma * target_q * (~dones[:, agent_idx:agent_idx+1])
            
            # Compute current Q-values
            states_flat = states.view(self.batch_size, -1)
            actions_flat = actions.view(self.batch_size, -1)
            current_q = policy.critic(states_flat, actions_flat, fo_satisfaction)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q)
            
            # Update critic
            policy.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), 1.0)
            policy.critic_optimizer.step()
            
            # Compute Shapley values for actor loss
            shapley_values = self.compute_shapley_values(states, actions, agent_idx)
            shapley_mean = shapley_values.mean(dim=1)  # [batch_size, 1]
            
            # Actor loss (negative Shapley values for gradient ascent)
            actor_loss = -shapley_mean.mean()
            
            # Add FlexOffer constraint loss if available
            if fo_constraints is not None:
                predicted_actions = policy.actor(states[:, agent_idx, :], fo_constraints[:, agent_idx, :])
                constraint_loss = F.mse_loss(predicted_actions, actions[:, agent_idx, :])
                actor_loss = actor_loss + 0.1 * constraint_loss
            
            # Update actor
            policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
            policy.actor_optimizer.step()
            
            # Update target networks
            policy.update_target_networks()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        self.total_iterations += 1
        
        return {
            'actor_loss': total_actor_loss / self.n_agents,
            'critic_loss': total_critic_loss / self.n_agents,
            'total_iterations': self.total_iterations
        }
    
    def save_models(self, filepath_prefix: str):
        """Save all agent models"""
        for i, policy in enumerate(self.policies):
            policy.save_models(f"{filepath_prefix}_agent_{i}")
        
        logger.info(f"FOSQDDPG models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load all agent models"""
        for i, policy in enumerate(self.policies):
            policy.load_models(f"{filepath_prefix}_agent_{i}")
        
        logger.info(f"FOSQDDPG models loaded with prefix: {filepath_prefix}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'n_agents': self.n_agents,
            'total_iterations': self.total_iterations,
            'buffer_size': len(self.replay_buffer),
            'sample_size': self.sample_size
        } 
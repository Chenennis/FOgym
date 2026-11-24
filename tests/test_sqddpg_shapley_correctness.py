"""Numerical correctness tests for SQDDPG Shapley marginal contributions."""
import torch
from torch import nn

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG


class AdditiveCoalitionCritic(nn.Module):
    """Q(a_0, a_1) = 2*a_0 + 3*a_1, with zero-action coalition baseline."""

    def forward(self, states, actions, fo_satisfaction=None):
        return 2.0 * actions[:, 0:1] + 3.0 * actions[:, 1:2]


def test_sqddpg_shapley_returns_marginal_contribution_for_additive_critic():
    """Every permutation must give the same exact marginal in an additive game."""
    torch.manual_seed(23)
    agent = FOSQDDPG(
        n_agents=2, state_dim=1, action_dim=1, hidden_dim=16,
        batch_size=2, sample_size=8, device="cpu",
    )
    for policy in agent.policies:
        policy.critic = AdditiveCoalitionCritic()

    states = torch.zeros(2, 2, 1)
    actions = torch.tensor([[[2.0], [5.0]], [[-1.0], [4.0]]])

    # With Q = 2*a_0 + 3*a_1, Shapley values are exactly 2*a_0
    # and 3*a_1 for every sampled agent ordering.
    expected = [
        torch.tensor([4.0, -2.0]),
        torch.tensor([15.0, 12.0]),
    ]
    for agent_idx in range(2):
        values = agent.compute_shapley_values(states, actions, agent_idx)
        assert values.shape == (2, agent.sample_size, 1)
        torch.testing.assert_close(
            values.squeeze(-1),
            expected[agent_idx].unsqueeze(1).expand(-1, agent.sample_size),
        )

"""SQDDPG interaction-game and actor-credit regression tests."""
import numpy as np
import torch
from torch import nn
from unittest.mock import patch

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG


class InteractionCritic(nn.Module):
    """Q(a0,a1)=2*a0+3*a1+4*a0*a1."""

    def forward(self, states, actions, fo_satisfaction=None):
        a0, a1 = actions[:, :1], actions[:, 1:2]
        return 2 * a0 + 3 * a1 + 4 * a0 * a1


def test_sqddpg_shapley_interaction_game_marginals():
    torch.manual_seed(29)
    algo = FOSQDDPG(
        n_agents=2, state_dim=1, action_dim=1,
        hidden_dim=16, batch_size=2, sample_size=64, device="cpu",
    )
    for policy in algo.policies:
        policy.critic = InteractionCritic()

    states = torch.zeros(1, 2, 1)
    actions = torch.tensor([[[2.0], [5.0]]])
    # The first agent contributes 4 if it arrives first, 44 if it
    # arrives second. The second contributes 15 or 55 respectively.
    expected = {0: (4.0, 44.0), 1: (15.0, 55.0)}
    for agent_idx, allowed in expected.items():
        values = algo.compute_shapley_values(states, actions, agent_idx)
        assert values.shape == (1, algo.sample_size, 1)
        assert all(
            any(abs(v.item() - option) < 1e-5 for option in allowed)
            for v in values.flatten()
        )
        # With 64 independent samples, both possible orders should appear.
        assert any(abs(v.item() - allowed[0]) < 1e-5 for v in values.flatten())
        assert any(abs(v.item() - allowed[1]) < 1e-5 for v in values.flatten())


def test_sqddpg_actor_update_uses_differentiable_shapley_credit():
    torch.manual_seed(31)
    np.random.seed(31)
    algo = FOSQDDPG(
        n_agents=2, state_dim=2, action_dim=1, hidden_dim=16,
        batch_size=2, sample_size=2, device="cpu",
    )
    for i in range(2):
        states = np.full((2, 2), 0.2 * (i + 1), dtype=np.float32)
        actions = np.full((2, 1), 0.1 * (i + 1), dtype=np.float32)
        algo.store_experience(
            states, actions, np.array([1., 2.], dtype=np.float32),
            states + 0.1, np.array([False, i == 1]),
        )

    original = algo.compute_shapley_values
    seen = []

    def spy(states, actions, agent_idx):
        seen.append((agent_idx, actions.requires_grad))
        return original(states, actions, agent_idx)

    with patch.object(algo, "compute_shapley_values", side_effect=spy):
        result = algo.update()

    assert result is not None
    assert len(seen) == algo.n_agents
    assert {idx for idx, _ in seen} == set(range(algo.n_agents))
    assert all(differentiable for _, differentiable in seen)

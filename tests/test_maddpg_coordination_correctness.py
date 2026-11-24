"""MADDPG coordination must exclude the current manager, not manager 1."""
import torch
import pytest

from algorithms.MADDPG.fomaddpg.fomaddpg_policy import FOMaddpgPolicy


@pytest.mark.parametrize("agent_id,expected", [
    (0, 0.10),
    (1, 0.10),
    (2, 0.10),
])
def test_coordination_compares_with_other_managers_only(agent_id, expected):
    policy = FOMaddpgPolicy(agent_id=agent_id, state_dim=3,
                            action_dim=5, n_agents=3, hidden_dim=16)
    # All managers share the same action; coordination loss is 0.5,
    # regardless of the identity of the manager being optimized.
    action = torch.full((2, 5), expected, requires_grad=True)
    combined = torch.cat([action.detach()] * 3, dim=1)
    loss = policy._compute_coordination_loss(action, combined)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    assert action.grad is not None


def test_coordination_excludes_own_action_when_manager_is_not_first():
    policy = FOMaddpgPolicy(agent_id=1, state_dim=3,
                            action_dim=5, n_agents=3, hidden_dim=16)
    actions = torch.cat([
        torch.zeros(2, 5),          # manager 1
        torch.full((2, 5), 0.4),    # manager 2 (self)
        torch.full((2, 5), 0.8),    # manager 3
    ], dim=1)
    own_actions = actions[:, 5:10].clone().requires_grad_(True)
    result = policy._compute_coordination_loss(own_actions, actions)
    # Differences from managers 1 and 3 are both 0.4.
    torch.testing.assert_close(result, torch.tensor(0.1), rtol=0, atol=1e-6)

"""MADDPG must use deterministic target networks and report real training failures."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter
from algorithms.MADDPG.fomaddpg.fomaddpg_policy import FOMaddpgPolicy


def test_maddpg_target_networks_are_deterministic_and_bn_buffers_updated():
    torch.manual_seed(13)
    policy = FOMaddpgPolicy(
        agent_id=0, state_dim=3, action_dim=5, n_agents=2,
        hidden_dim=16, device="cpu",
    )
    assert not policy.actor_target.training
    assert not policy.critic_target.training
    states = torch.randn(8, 6)
    actions = torch.randn(8, 10)
    with torch.no_grad():
        q1 = policy.critic_target(states, actions)
        q2 = policy.critic_target(states, actions)
    torch.testing.assert_close(q1, q2, rtol=0, atol=0)

    with torch.no_grad():
        policy.actor.bn1.running_mean.fill_(4.0)
        policy.actor_target.bn1.running_mean.zero_()
    policy.soft_update(policy.actor_target, policy.actor, tau=0.25)
    torch.testing.assert_close(
        policy.actor_target.bn1.running_mean,
        torch.ones_like(policy.actor_target.bn1.running_mean),
        rtol=0, atol=0,
    )
    assert not policy.actor_target.training


def test_maddpg_deterministic_selection_disables_dropout():
    torch.manual_seed(17)
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    obs = {mid: np.full(3, 0.2, dtype=np.float32)
           for mid in ("manager_1", "manager_2")}
    first, _, _ = adapter.select_actions(obs, deterministic=True)
    second, _, _ = adapter.select_actions(obs, deterministic=True)
    for mid in obs:
        np.testing.assert_array_equal(first[mid], second[mid])


def test_maddpg_failed_optimizer_does_not_report_zero_loss_success():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=1, device="cpu",
    )
    adapter.fomaddpg.store_experience(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 5), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.zeros(2, dtype=bool),
    )
    with patch.object(adapter.fomaddpg, "update", side_effect=RuntimeError("optimizer failure")):
        with pytest.raises(RuntimeError, match="optimizer failure"):
            adapter.train_on_batch()
    assert adapter.training_iterations == 0


def test_maddpg_insufficient_replay_is_not_reported_as_update():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    assert adapter.train_on_batch() is None
    assert adapter.training_iterations == 0

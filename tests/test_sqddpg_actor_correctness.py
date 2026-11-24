"""Regression tests for SQDDPG actor learning correctness."""
import numpy as np
import torch

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG


def test_sqddpg_actor_parameters_change_after_update():
    """An update must propagate a nonzero gradient to at least one actor parameter."""
    torch.manual_seed(17)
    np.random.seed(17)
    agent = FOSQDDPG(
        n_agents=2, state_dim=3, action_dim=2,
        hidden_dim=16, batch_size=4, sample_size=2, device="cpu",
    )
    for i in range(4):
        states = np.full((2, 3), 0.1 * (i + 1), dtype=np.float32)
        actions = np.full((2, 2), 0.05 * (i + 1), dtype=np.float32)
        next_states = states + 0.02
        rewards = np.array([1.0, 0.5], dtype=np.float32)
        dones = np.array([False, i == 3])
        agent.store_experience(states, actions, rewards, next_states, dones)

    before = [
        [p.detach().clone() for p in policy.actor.parameters()]
        for policy in agent.policies
    ]
    result = agent.update()
    assert result is not None
    for policy, initial_params in zip(agent.policies, before):
        assert any(
            not torch.equal(old, new)
            for old, new in zip(initial_params, policy.actor.parameters())
        ), "SQDDPG update did not change actor parameters"

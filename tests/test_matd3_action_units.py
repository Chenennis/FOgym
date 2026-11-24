import numpy as np
import pytest
import torch

from algorithms.MATD3.fomatd3.fomatd3_adapter import (
    FOMATD3Adapter, map_normalized_fo_actions,
)


def test_matd3_manager_specific_dimensions_and_replay_padding():
    adapter = FOMATD3Adapter(
        state_dim=7, action_dim=10, num_agents=2,
        observation_dims={"manager_1": 3, "manager_2": 7},
        action_dims={"manager_1": 5, "manager_2": 10},
        hidden_dim=16, buffer_capacity=8, batch_size=2,
    )
    obs = {
        "manager_1": np.array([1, 2, 3], dtype=np.float32),
        "manager_2": np.arange(7, dtype=np.float32),
    }
    actions, _, _ = adapter.select_actions(obs, deterministic=True)
    assert actions["manager_1"].shape == (5,)
    assert actions["manager_2"].shape == (10,)
    next_obs = {k: v + 1 for k, v in obs.items()}
    adapter.collect_step(
        obs, actions, {"manager_1": 2., "manager_2": 3.},
        {"manager_1": False, "manager_2": True}, {},
        next_obs=next_obs,
    )
    data = adapter.replay_buffer
    np.testing.assert_array_equal(data["states"][0, 0, 3:], np.zeros(4))
    np.testing.assert_array_equal(data["actions"][0, 0, 5:], np.zeros(5))
    np.testing.assert_array_equal(data["next_states"][0, 0, :3], next_obs["manager_1"])
    with pytest.raises(ValueError):
        adapter.select_actions({**obs, "manager_1": np.ones(4)}, deterministic=True)


def test_matd3_physical_mapping_is_differentiable_and_matches_execution():
    raw = torch.tensor([[-1., 0., 1., -1., 0.]], requires_grad=True)
    physical = map_normalized_fo_actions(raw)
    np.testing.assert_allclose(
        physical.detach().numpy()[0], [-1., 0., 1., 1., 1.05], atol=1e-6,
    )
    physical.sum().backward()
    assert raw.grad is not None
    assert raw.grad.abs().sum() > 0
    adapter = FOMATD3Adapter(state_dim=3, action_dim=5, num_agents=1, hidden_dim=16)
    np.testing.assert_allclose(
        adapter._map_action_to_fo_params(raw.detach().numpy()[0]),
        physical.detach().numpy()[0], atol=1e-6,
    )


def test_matd3_critic_and_actor_use_physical_action_units():
    adapter = FOMATD3Adapter(state_dim=3, action_dim=5, num_agents=1, hidden_dim=16)
    agent = adapter.agents["manager_1"]
    state = torch.zeros((4, 3))
    own_action = torch.zeros((4, 5))
    captured = []
    orig_target = agent.critic_target.forward
    orig_q1 = agent.critic.Q1

    def capture_target(x):
        captured.append(x.detach().clone())
        return orig_target(x)

    def capture_q1(x):
        captured.append(x.detach().clone())
        return orig_q1(x)

    agent.critic_target.forward = capture_target
    agent.critic.Q1 = capture_q1
    agent.update_critic(
        state, own_action, torch.zeros(4), state,
        torch.zeros((4, 5)), torch.zeros(4),
        target_noise=0.,
    )
    np.testing.assert_allclose(
        captured[0][:, -5:].numpy(),
        np.tile(np.array([0., 0., .55, 1.5, 1.05]), (4, 1)), atol=1e-6,
    )
    agent.update_actor(state, state)
    assert torch.all(captured[1][:, -3:] >= 0.1)
    assert torch.all(captured[1][:, -3:] <= 2.0)

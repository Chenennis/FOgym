"""MADDPG uses padded tensors internally but physical per-manager dimensions externally."""
import numpy as np
import pytest

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter


def test_maddpg_heterogeneous_manager_action_and_transition_shapes():
    obs_dims = {"manager_1": 3, "manager_2": 5}
    act_dims = {"manager_1": 5, "manager_2": 10}
    adapter = FOMAddpgAdapter(
        state_dim=5, action_dim=10, num_agents=2,
        observation_dims=obs_dims, action_dims=act_dims,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    obs = {mid: np.arange(dim, dtype=np.float32)
           for mid, dim in obs_dims.items()}
    actions, _, _ = adapter.select_actions(obs, deterministic=True)
    assert {mid: action.shape[0] for mid, action in actions.items()} == act_dims
    future = {mid: values + 1 for mid, values in obs.items()}
    adapter.collect_step(
        obs=obs, next_obs=future, actions=actions,
        rewards={mid: 1.0 for mid in obs},
        dones={mid: True for mid in obs}, infos={},
    )
    states, joint_actions, reward, next_states, done = adapter.fomaddpg.replay_buffer.buffer[0]
    np.testing.assert_array_equal(states.reshape(2, 5)[0, 3:], [0, 0])
    np.testing.assert_array_equal(next_states.reshape(2, 5)[0, 3:], [0, 0])
    np.testing.assert_array_equal(joint_actions.reshape(2, 10)[0, 5:], np.zeros(5))
    assert done
    assert np.isfinite(reward)


def test_maddpg_rejects_invalid_observation_length():
    adapter = FOMAddpgAdapter(
        state_dim=5, action_dim=10, num_agents=2,
        observation_dims={"manager_1": 3, "manager_2": 5},
        action_dims={"manager_1": 5, "manager_2": 10},
        hidden_dim=16, batch_size=2, device="cpu",
    )
    obs = {"manager_1": np.zeros(4), "manager_2": np.zeros(5)}
    with pytest.raises(ValueError, match="manager_1"):
        adapter.select_actions(obs, deterministic=True)

"""SQDDPG must support manager-specific observation and action widths."""
import numpy as np

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_heterogeneous_manager_actions_and_replay_shapes():
    adapter = FOSQDDPGAdapter(
        state_dim=7, action_dim=10, num_agents=2, hidden_dim=16,
        batch_size=2, sample_size=2, device="cpu",
        observation_dims={"manager_1": 4, "manager_2": 7},
        action_dims={"manager_1": 5, "manager_2": 10},
    )
    obs = {
        "manager_1": np.array([1., 2., 3., 4.], dtype=np.float32),
        "manager_2": np.arange(7, dtype=np.float32),
    }
    next_obs = {
        "manager_1": np.array([2., 3., 4., 5.], dtype=np.float32),
        "manager_2": np.arange(7, dtype=np.float32) + 1,
    }

    actions, _, _ = adapter.select_actions(obs, deterministic=True)
    assert actions["manager_1"].shape == (5,)
    assert actions["manager_2"].shape == (10,)

    adapter.collect_step(
        obs=obs, actions=actions, rewards={"manager_1": 1., "manager_2": 2.},
        next_obs=next_obs, dones={"manager_1": False, "manager_2": True},
        infos={}, timestep=0,
    )
    transition = adapter.fosqddpg.replay_buffer.buffer[-1]
    assert transition.states.shape == (2, 7)
    assert transition.next_states.shape == (2, 7)
    assert transition.actions.shape == (2, 10)
    np.testing.assert_array_equal(transition.states[0, :4], obs["manager_1"])
    np.testing.assert_array_equal(transition.next_states[0, :4], next_obs["manager_1"])
    np.testing.assert_array_equal(transition.states[0, 4:], np.zeros(3))
    np.testing.assert_array_equal(transition.actions[0, 5:], np.zeros(5))

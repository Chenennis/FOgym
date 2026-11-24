import numpy as np

from algorithms.MATD3.fomatd3.fomatd3_adapter import FOMATD3Adapter


def _adapter():
    return FOMATD3Adapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, buffer_capacity=8, batch_size=2,
        device="cpu",
    )


def _step(adapter, t, done=False):
    obs = {
        "manager_1": np.array([t, 1., 2.], dtype=np.float32),
        "manager_2": np.array([t, 3., 4.], dtype=np.float32),
    }
    next_obs = {
        "manager_1": np.array([t + 1, 1., 2.], dtype=np.float32),
        "manager_2": np.array([t + 1, 3., 4.], dtype=np.float32),
    }
    actions = {mid: np.full(5, 0.25, dtype=np.float32) for mid in obs}
    rewards = {"manager_1": float(t + 10), "manager_2": float(t + 20)}
    dones = {mid: done for mid in obs}
    adapter.collect_step(
        obs=obs, actions=actions, rewards=rewards, dones=dones,
        next_obs=next_obs, infos={},
    )


def test_matd3_stores_each_actual_transition_including_final_step():
    adapter = _adapter()
    _step(adapter, 0)
    _step(adapter, 1, done=True)
    buffer = adapter.replay_buffer

    assert buffer["size"] == 2
    np.testing.assert_allclose(buffer["states"][0, :, 0], [0, 0])
    np.testing.assert_allclose(buffer["next_states"][0, :, 0], [1, 1])
    np.testing.assert_allclose(buffer["next_states"][1, :, 0], [2, 2])
    np.testing.assert_allclose(buffer["rewards"][0], [10, 20])
    assert not buffer["dones"][0].any()
    assert buffer["dones"][1].all()


def test_matd3_episode_reset_preserves_off_policy_replay():
    adapter = _adapter()
    _step(adapter, 0, done=True)
    before = {key: adapter.replay_buffer[key].copy() for key in
              ("states", "next_states", "actions", "rewards", "dones")}
    adapter.reset_buffers()
    assert adapter.replay_buffer["size"] == 1
    assert adapter.replay_buffer["ptr"] == 1
    for key, value in before.items():
        np.testing.assert_array_equal(adapter.replay_buffer[key], value)
    assert adapter._prev_states is None

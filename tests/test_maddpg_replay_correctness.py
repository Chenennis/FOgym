"""MADDPG replay must preserve exact per-step transitions, including terminal steps."""
import numpy as np

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter


def test_maddpg_replay_uses_actual_next_state_and_keeps_terminal_transition():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    ids = ["manager_1", "manager_2"]
    obs0 = {mid: np.full(3, i + 1, dtype=np.float32) for i, mid in enumerate(ids)}
    obs1 = {mid: np.full(3, i + 11, dtype=np.float32) for i, mid in enumerate(ids)}
    actions = {mid: np.zeros(5, dtype=np.float32) for mid in ids}
    rewards = {mid: 10.0 for mid in ids}
    dones = {mid: True for mid in ids}
    adapter.collect_step(
        obs=obs0, next_obs=obs1, actions=actions,
        rewards=rewards, dones=dones, infos={},
    )
    transitions = list(adapter.fomaddpg.replay_buffer.buffer)
    assert len(transitions) == 1
    states, stored_actions, reward, next_states, done = transitions[0]
    np.testing.assert_array_equal(states, np.stack(list(obs0.values())).flatten())
    np.testing.assert_array_equal(next_states, np.stack(list(obs1.values())).flatten())
    np.testing.assert_array_equal(stored_actions, np.zeros(10, dtype=np.float32))
    assert bool(done)


def test_maddpg_replay_does_not_link_separate_episodes():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    ids = ["manager_1", "manager_2"]
    acts = {mid: np.zeros(5, dtype=np.float32) for mid in ids}
    r = {mid: 1.0 for mid in ids}
    for value in (1.0, 100.0):
        current = {mid: np.full(3, value, dtype=np.float32) for mid in ids}
        following = {mid: np.full(3, value + 1, dtype=np.float32) for mid in ids}
        adapter.collect_step(
            obs=current, next_obs=following, actions=acts, rewards=r,
            dones={mid: True for mid in ids}, infos={},
        )
    assert len(adapter.fomaddpg.replay_buffer) == 2
    transitions = list(adapter.fomaddpg.replay_buffer.buffer)
    assert transitions[0][3][0] == 2.0
    assert transitions[1][0][0] == 100.0

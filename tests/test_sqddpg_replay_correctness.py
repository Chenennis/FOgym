"""Regression tests for SQDDPG replay transition alignment."""
import numpy as np

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_collect_step_stores_each_transition_with_true_next_state():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=2, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    adapter.reset_episode()
    obs0 = {
        "manager_1": np.array([1., 2., 3.], dtype=np.float32),
        "manager_2": np.array([4., 5., 6.], dtype=np.float32),
    }
    obs1 = {
        "manager_1": np.array([11., 12., 13.], dtype=np.float32),
        "manager_2": np.array([14., 15., 16.], dtype=np.float32),
    }
    obs2 = {
        "manager_1": np.array([21., 22., 23.], dtype=np.float32),
        "manager_2": np.array([24., 25., 26.], dtype=np.float32),
    }
    actions = {
        "manager_1": np.array([0.1, 0.2], dtype=np.float32),
        "manager_2": np.array([0.3, 0.4], dtype=np.float32),
    }
    rewards = {"manager_1": 10.0, "manager_2": 20.0}
    adapter.collect_step(
        obs=obs0, actions=actions, rewards=rewards,
        next_obs=obs1,
        dones={"manager_1": False, "manager_2": False},
        infos={}, timestep=0,
    )
    adapter.collect_step(
        obs=obs1, actions=actions, rewards=rewards,
        next_obs=obs2,
        dones={"manager_1": True, "manager_2": True},
        infos={}, timestep=1,
    )
    transitions = list(adapter.fosqddpg.replay_buffer.buffer)
    assert len(transitions) == 2, "Every environment step, including terminal steps, must be stored"
    np.testing.assert_array_equal(transitions[0].states, np.stack(list(obs0.values())))
    np.testing.assert_array_equal(transitions[0].next_states, np.stack(list(obs1.values())))
    np.testing.assert_array_equal(transitions[1].states, np.stack(list(obs1.values())))
    np.testing.assert_array_equal(transitions[1].next_states, np.stack(list(obs2.values())))
    np.testing.assert_array_equal(transitions[1].dones, [True, True])

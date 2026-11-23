"""SQDDPG reward scaling must preserve a stationary Bellman reward definition."""
import numpy as np

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_reward_scale_is_fixed_across_agents_and_steps():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    inputs = [
        np.array([-100.0, -50.0], dtype=np.float32),
        np.array([-100.0, 100.0], dtype=np.float32),
        np.array([100.0, 100.0], dtype=np.float32),
    ]
    outputs = [adapter._normalize_rewards(x) for x in inputs]
    for raw, scaled in zip(inputs, outputs):
        np.testing.assert_allclose(scaled, raw / 1000.0, atol=1e-8)
    assert outputs[0][0] == outputs[1][0]
    assert outputs[1][1] == outputs[2][1]


def test_sqddpg_reward_statistics_reflect_collected_samples():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    obs = {mid: np.zeros(3, dtype=np.float32) for mid in adapter.agent_ids}
    acts = {mid: np.zeros(5, dtype=np.float32) for mid in adapter.agent_ids}
    for reward in ({"manager_1": -100., "manager_2": 100.},
                   {"manager_1": 0., "manager_2": 200.}):
        adapter.collect_step(
            obs=obs, actions=acts, rewards=reward, next_obs=obs,
            dones={mid: False for mid in adapter.agent_ids},
            infos={}, timestep=0,
        )
    stats = adapter.reward_normalizer
    assert stats["count"] == 4
    np.testing.assert_allclose(stats["running_mean"], 50.0)
    np.testing.assert_allclose(stats["running_var"], 12500.0)

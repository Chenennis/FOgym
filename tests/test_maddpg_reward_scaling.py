"""MADDPG reward scaling must preserve differences above an arbitrary clipping threshold."""
import numpy as np

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter


def test_maddpg_reward_scaling_preserves_order_and_raw_statistics():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    obs = {mid: np.zeros(3, dtype=np.float32)
           for mid in ("manager_1", "manager_2")}
    actions = {mid: np.array([0., 0., .55, 1.5, 1.], dtype=np.float32)
               for mid in obs}
    for raw in (100.0, 200.0):
        adapter.collect_step(
            obs=obs, next_obs=obs, actions=actions,
            rewards={mid: raw for mid in obs},
            dones={mid: False for mid in obs}, infos={},
        )
    values = [transition[2] for transition in adapter.fomaddpg.replay_buffer.buffer]
    assert values[1] > values[0], "Clipping makes distinct rewards indistinguishable"
    np.testing.assert_allclose(values, [0.1, 0.2])
    for mid in obs:
        assert adapter.manager_stats[mid]["total_reward"] == 300.0
        assert adapter.manager_stats[mid]["best_reward"] == 200.0

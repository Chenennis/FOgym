"""MATD3 pipeline capacity and loss-reporting regressions."""
import numpy as np

from algorithms.MATD3.fomatd3.fomatd3_adapter import FOMATD3Adapter
from algorithms.MATD3.fomatd3.fomatd3_training_methods import matd3_replay_capacity


def test_matd3_replay_capacity_uses_actual_training_horizon():
    assert matd3_replay_capacity(11, 24, 128) == 264
    assert matd3_replay_capacity(1, 24, 128) == 128
    assert matd3_replay_capacity(100000, 24, 128) == 100000


def test_matd3_train_reports_actual_aggregate_actor_and_critic_loss():
    adapter = FOMATD3Adapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, buffer_capacity=8, batch_size=2, policy_delay=2,
    )
    adapter.training_improvement_config["buffer_warm_up"] = 2
    for t in range(3):
        obs = {mid: np.array([t + 1., i + 1., 3.], dtype=np.float32)
               for i, mid in enumerate(adapter.agent_ids)}
        next_obs = {mid: val + .1 for mid, val in obs.items()}
        actions = {mid: np.array([0., 0., .55, 1.5, 1.05], dtype=np.float32)
                   for mid in adapter.agent_ids}
        adapter.collect_step(
            obs, actions, {mid: 100. for mid in adapter.agent_ids},
            {mid: False for mid in adapter.agent_ids}, {},
            next_obs=next_obs,
        )
    result1 = adapter.train_on_batch()
    assert result1["status"] == "updated"
    assert result1["actor_updated"] is False
    assert result1["policy_loss"] == 0.
    assert result1["value_loss"] == np.mean([
        result1[f"{mid}_critic_loss"] for mid in adapter.agent_ids
    ])
    result2 = adapter.train_on_batch()
    assert result2["actor_updated"] is True
    assert np.isfinite(result2["policy_loss"])
    assert result2["policy_loss"] == np.mean([
        result2[f"{mid}_actor_loss"] for mid in adapter.agent_ids
    ])
    assert result2["value_loss"] == np.mean([
        result2[f"{mid}_critic_loss"] for mid in adapter.agent_ids
    ])
    assert adapter.training_iterations == 2

"""MATD3 real FOgym rollout, physical actions, replay and delayed updates."""
import numpy as np
import torch

from algorithms.MATD3.fomatd3.fomatd3_adapter import FOMATD3Adapter
from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv


def _changed(before, network):
    return any(not torch.equal(b, p) for b, p in zip(before, network.parameters()))


def test_matd3_real_24_step_rollout_with_consecutive_updates():
    torch.manual_seed(71)
    np.random.seed(71)
    env = MultiAgentFlexOfferEnv(
        data_dir="data", time_horizon=24, data_config="36users",
        aggregation_method="LP", trading_method="market_clearing",
        disaggregation_method="proportional",
    )
    obs, _ = env.reset(seed=71)
    mids = list(env.manager_agents)
    obs_dims = {mid: len(obs[mid]) for mid in mids}
    act_dims = {mid: env.action_spaces[mid].shape[0] for mid in mids}
    adapter = FOMATD3Adapter(
        state_dim=max(obs_dims.values()), action_dim=max(act_dims.values()),
        num_agents=len(mids), observation_dims=obs_dims, action_dims=act_dims,
        hidden_dim=16, buffer_capacity=64, batch_size=8, policy_delay=2,
        device="cpu",
    )
    adapter.training_improvement_config["buffer_warm_up"] = 8
    actors_before = {mid: [p.detach().clone() for p in agent.actor.parameters()]
                     for mid, agent in adapter.agents.items()}
    critics_before = {mid: [p.detach().clone() for p in agent.critic.parameters()]
                      for mid, agent in adapter.agents.items()}
    updates = []

    for t in range(24):
        actions, _, _ = adapter.select_actions(obs, deterministic=True)
        assert set(actions) == set(mids)
        for mid in mids:
            space = env.action_spaces[mid]
            assert actions[mid].shape == space.shape
            assert np.isfinite(actions[mid]).all()
            assert np.all(actions[mid] >= space.low - 1e-6)
            assert np.all(actions[mid] <= space.high + 1e-6)

        next_obs, rewards, dones, truncated, infos = env.step(actions)
        terminal = {mid: bool(dones[mid] or truncated[mid]) for mid in mids}
        adapter.collect_step(
            obs=obs, next_obs=next_obs, actions=actions,
            rewards=rewards, dones=terminal, infos=infos,
        )
        index = (adapter.replay_buffer["ptr"] - 1) % adapter.replay_buffer["capacity"]
        data = adapter.replay_buffer
        np.testing.assert_allclose(
            data["next_states"][index], adapter._stack_observations(next_obs)
        )
        np.testing.assert_allclose(
            data["actions"][index], adapter._stack_actions(actions)
        )
        np.testing.assert_allclose(
            data["rewards"][index], [rewards[mid] for mid in mids]
        )
        np.testing.assert_array_equal(
            data["dones"][index], [terminal[mid] for mid in mids]
        )
        assert np.isfinite(data["states"][index]).all()
        assert np.isfinite(data["next_states"][index]).all()
        assert bool(dones["__all__"]) == (t == 23)

        if t in (11, 23):
            info = adapter.train_on_batch()
            assert info["total_iterations"] == len(updates) + 1
            assert all(np.isfinite(info[f"{mid}_critic_loss"]) for mid in mids)
            updates.append(info)
        obs = next_obs

    assert adapter.replay_buffer["size"] == 24
    assert len(updates) == 2
    assert updates[0]["actor_updated"] is False
    assert updates[1]["actor_updated"] is True
    for mid, agent in adapter.agents.items():
        assert _changed(actors_before[mid], agent.actor)
        assert _changed(critics_before[mid], agent.critic)

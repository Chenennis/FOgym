"""Real FOgym MADDPG rollout: physical bounds, replay alignment and two updates."""
import numpy as np
import torch

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter
from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv


def test_maddpg_real_24_step_episode_with_multiple_updates():
    torch.manual_seed(67)
    np.random.seed(67)
    env = MultiAgentFlexOfferEnv(
        data_dir="data", time_horizon=24, data_config="36users",
        aggregation_method="LP", trading_method="market_clearing",
        disaggregation_method="proportional",
    )
    obs, _ = env.reset(seed=67)
    manager_ids = list(env.manager_agents)
    observation_dims = {mid: len(obs[mid]) for mid in manager_ids}
    action_dims = {mid: env.action_spaces[mid].shape[0] for mid in manager_ids}
    adapter = FOMAddpgAdapter(
        state_dim=max(observation_dims.values()),
        action_dim=max(action_dims.values()),
        num_agents=len(manager_ids),
        observation_dims=observation_dims, action_dims=action_dims,
        hidden_dim=16, batch_size=8, device="cpu",
    )
    actor_before = [
        [p.detach().clone() for p in policy.actor.parameters()]
        for policy in adapter.fomaddpg.agents
    ]
    critic_before = [
        [p.detach().clone() for p in policy.critic.parameters()]
        for policy in adapter.fomaddpg.agents
    ]
    updates = []
    for timestep in range(24):
        actions, _, _ = adapter.select_actions(obs, deterministic=True)
        assert set(actions) == set(manager_ids)
        for mid in manager_ids:
            space = env.action_spaces[mid]
            assert actions[mid].shape == space.shape
            assert np.isfinite(actions[mid]).all()
            assert np.all(actions[mid] >= space.low - 1e-6)
            assert np.all(actions[mid] <= space.high + 1e-6)

        next_obs, rewards, dones, truncated, infos = env.step(actions)
        adapter.collect_step(
            obs=obs, next_obs=next_obs, actions=actions,
            rewards=rewards, dones=dones, infos=infos,
        )
        states, stored_actions, reward, next_states, done = adapter.fomaddpg.replay_buffer.buffer[-1]
        expected_next = adapter._stack_observations(next_obs).flatten()
        expected_actions = adapter._stack_actions(actions).flatten()
        np.testing.assert_allclose(next_states, expected_next)
        np.testing.assert_allclose(stored_actions, expected_actions)
        assert np.isfinite(states).all()
        assert np.isfinite(reward)
        assert bool(done) == (timestep == 23)
        assert bool(dones["__all__"]) == (timestep == 23)
        if timestep in (11, 23):
            result = adapter.train_on_batch()
            assert result is not None
            assert np.isfinite(result["actor_loss"])
            assert np.isfinite(result["critic_loss"])
            updates.append(result)
        obs = next_obs

    assert len(adapter.fomaddpg.replay_buffer) == 24
    assert len(updates) == 2
    assert adapter.get_training_stats()["training_iterations"] == 2
    for i, policy in enumerate(adapter.fomaddpg.agents):
        assert any(not torch.equal(before, after)
                   for before, after in zip(actor_before[i], policy.actor.parameters()))
        assert any(not torch.equal(before, after)
                   for before, after in zip(critic_before[i], policy.critic.parameters()))

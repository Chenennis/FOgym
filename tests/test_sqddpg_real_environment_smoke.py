"""SQDDPG end-to-end smoke test with actual FOgym manager interactions."""
import numpy as np
import torch

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter
from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv


def test_sqddpg_real_environment_collects_and_updates_parameters():
    torch.manual_seed(47)
    np.random.seed(47)

    env = MultiAgentFlexOfferEnv(
        data_dir="data", time_horizon=24, data_config="36users",
        aggregation_method="LP", trading_method="market_clearing",
        disaggregation_method="proportional",
    )
    obs, _ = env.reset(seed=47)
    manager_ids = list(env.manager_agents)
    obs_dims = {mid: len(obs[mid]) for mid in manager_ids}
    act_dims = {mid: env.action_spaces[mid].shape[0] for mid in manager_ids}
    adapter = FOSQDDPGAdapter(
        state_dim=max(obs_dims.values()), action_dim=max(act_dims.values()),
        num_agents=len(manager_ids), observation_dims=obs_dims, action_dims=act_dims,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )

    for timestep in range(2):
        actions, _, _ = adapter.select_actions(obs, deterministic=True)
        for mid in manager_ids:
            assert actions[mid].shape == env.action_spaces[mid].shape
            assert np.isfinite(actions[mid]).all()
            assert np.all(actions[mid] >= env.action_spaces[mid].low - 1e-5)
            assert np.all(actions[mid] <= env.action_spaces[mid].high + 1e-5)

        next_obs, rewards, dones, truncated, infos = env.step(actions)
        adapter.collect_step(
            obs=obs, actions=actions, rewards=rewards, dones=dones,
            next_obs=next_obs, infos=infos, timestep=timestep,
        )
        obs = next_obs

    assert len(adapter.fosqddpg.replay_buffer) == 2
    actors_before = [
        [param.detach().clone() for param in policy.actor.parameters()]
        for policy in adapter.fosqddpg.policies
    ]
    critics_before = [
        [param.detach().clone() for param in policy.critic.parameters()]
        for policy in adapter.fosqddpg.policies
    ]
    info = adapter.train_on_batch()
    assert info is not None
    assert adapter.get_training_stats()["training_iterations"] == 1
    assert np.isfinite(info["actor_loss"])
    assert np.isfinite(info["critic_loss"])
    for idx, policy in enumerate(adapter.fosqddpg.policies):
        assert any(
            not torch.equal(old, new)
            for old, new in zip(actors_before[idx], policy.actor.parameters())
        )
        assert any(
            not torch.equal(old, new)
            for old, new in zip(critics_before[idx], policy.critic.parameters())
        )

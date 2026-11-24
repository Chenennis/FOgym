"""Real 24-step SQDDPG episode, replay alignment and consecutive updates."""
import numpy as np
import torch

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter
from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv


def test_sqddpg_real_24_step_episode_with_multiple_updates():
    torch.manual_seed(53)
    np.random.seed(53)
    env = MultiAgentFlexOfferEnv(
        data_dir="data", time_horizon=24, data_config="36users",
        aggregation_method="LP", trading_method="market_clearing",
        disaggregation_method="proportional",
    )
    obs, _ = env.reset(seed=53)
    manager_ids = list(env.manager_agents)
    observation_dims = {mid: len(obs[mid]) for mid in manager_ids}
    action_dims = {mid: env.action_spaces[mid].shape[0] for mid in manager_ids}
    adapter = FOSQDDPGAdapter(
        state_dim=max(observation_dims.values()),
        action_dim=max(action_dims.values()),
        num_agents=len(manager_ids),
        observation_dims=observation_dims, action_dims=action_dims,
        hidden_dim=16, batch_size=8, sample_size=2, device="cpu",
    )
    actor_before = [
        [p.detach().clone() for p in policy.actor.parameters()]
        for policy in adapter.fosqddpg.policies
    ]
    critic_before = [
        [p.detach().clone() for p in policy.critic.parameters()]
        for policy in adapter.fosqddpg.policies
    ]
    updates = []
    for t in range(24):
        actions, _, _ = adapter.select_actions(obs, deterministic=True)
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        adapter.collect_step(
            obs=obs, next_obs=next_obs, actions=actions,
            rewards=rewards, dones=dones, infos=infos, timestep=t,
        )
        transition = adapter.fosqddpg.replay_buffer.buffer[-1]
        assert transition.states.shape == (
            len(manager_ids), adapter.state_dim
        )
        assert transition.next_states.shape == transition.states.shape
        assert np.isfinite(transition.next_states).all()
        assert np.isfinite(transition.rewards).all()
        assert bool(dones["__all__"]) == (t == 23)
        assert all(bool(dones[mid]) == (t == 23) for mid in manager_ids)
        assert all(bool(transition.dones[i]) == (t == 23)
                   for i in range(len(manager_ids)))

        if t in (11, 23):
            info = adapter.train_on_batch()
            assert info is not None
            assert np.isfinite(info["actor_loss"])
            assert np.isfinite(info["critic_loss"])
            updates.append(info)

        obs = next_obs

    assert len(adapter.fosqddpg.replay_buffer) == 24
    assert len(updates) == 2
    assert adapter.get_training_stats()["training_iterations"] == 2
    for i, policy in enumerate(adapter.fosqddpg.policies):
        assert any(
            not torch.equal(before, after)
            for before, after in zip(actor_before[i], policy.actor.parameters())
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(critic_before[i], policy.critic.parameters())
        )

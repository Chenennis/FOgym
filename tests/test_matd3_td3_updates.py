import numpy as np
import pytest
import torch

from algorithms.MATD3.fomatd3.fomatd3_adapter import (
    FOMATD3Adapter, FOTd3Agent,
)


def _parameters(network):
    return [p.detach().clone() for p in network.parameters()]


def _different(before, after):
    return any(not torch.equal(a, b) for a, b in zip(before, after))


def test_matd3_clipped_double_q_and_terminal_mask_numerically():
    torch.manual_seed(7)
    agent = FOTd3Agent(
        agent_id="manager_1", obs_dim=3, action_dim=5,
        global_state_dim=3, hidden_dim=16, lr_critic=0., gamma=0.9,
    )
    with torch.no_grad():
        for param in agent.critic.parameters():
            param.zero_()
        for param in agent.critic_target.parameters():
            param.zero_()
        agent.critic_target.fc3_q1.bias.fill_(4.)
        agent.critic_target.fc3_q2.bias.fill_(2.)
    states = torch.zeros((2, 3))
    actions = torch.zeros((2, 5))
    loss = agent.update_critic(
        states, actions, torch.ones(2), states, actions,
        torch.tensor([0., 1.]), target_noise=0.,
    )
    # min(4,2)=2: nonterminal target 1 + .9*2 = 2.8;
    # terminal target 1. Two zero-valued online Q heads.
    expected = 2 * ((2.8 ** 2 + 1.) / 2.)
    assert loss == pytest.approx(expected, rel=1e-5)


def test_matd3_reward_scaling_is_fixed_across_batches_and_preserves_sign():
    adapter = FOMATD3Adapter(state_dim=3, action_dim=5, num_agents=1, hidden_dim=16)
    first = adapter._normalize_rewards(np.array([-50., 0., 50.]))
    adapter._normalize_rewards(np.array([-10000., 20000.]))
    again = adapter._normalize_rewards(np.array([-50., 0., 50.]))
    np.testing.assert_allclose(first, [-0.05, 0., 0.05])
    np.testing.assert_allclose(again, first)


def test_matd3_critic_every_update_actor_and_targets_delayed():
    torch.manual_seed(11)
    np.random.seed(11)
    adapter = FOMATD3Adapter(
        state_dim=3, action_dim=5, num_agents=1, hidden_dim=16,
        buffer_capacity=8, batch_size=2, policy_delay=2,
    )
    adapter.training_improvement_config["buffer_warm_up"] = 2
    for t in range(3):
        obs = {"manager_1": np.array([t + 1., 2., 3.], dtype=np.float32)}
        next_obs = {"manager_1": obs["manager_1"] + .1}
        action = {"manager_1": np.array([0., 0., .55, 1.5, 1.05], dtype=np.float32)}
        adapter.collect_step(
            obs, action, {"manager_1": 100. + t},
            {"manager_1": False}, {}, next_obs=next_obs,
        )
    agent = adapter.agents["manager_1"]
    actor_before = _parameters(agent.actor)
    critic_before = _parameters(agent.critic)
    actor_target_before = _parameters(agent.actor_target)
    critic_target_before = _parameters(agent.critic_target)

    result1 = adapter.train_on_batch()
    assert result1["actor_updated"] is False
    assert _different(critic_before, _parameters(agent.critic))
    assert not _different(actor_before, _parameters(agent.actor))
    assert not _different(actor_target_before, _parameters(agent.actor_target))
    assert not _different(critic_target_before, _parameters(agent.critic_target))

    result2 = adapter.train_on_batch()
    assert result2["actor_updated"] is True
    assert _different(actor_before, _parameters(agent.actor))
    assert _different(actor_target_before, _parameters(agent.actor_target))
    assert _different(critic_target_before, _parameters(agent.critic_target))

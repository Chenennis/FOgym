"""Regression tests for SQDDPG target-network normalization state."""
import torch

from algorithms.SQDDPG.fosqddpg.fosqddpg_policy import FOSQDDPGPolicy


def test_sqddpg_target_networks_remain_in_eval_mode():
    policy = FOSQDDPGPolicy(
        state_dim=3, action_dim=2, n_agents=2, hidden_dim=16, device="cpu"
    )
    assert not policy.actor_target.training
    assert not policy.critic_target.training

    actor_before = policy.actor_target.bn1.running_mean.detach().clone()
    critic_before = policy.critic_target.bn1.running_mean.detach().clone()
    with torch.no_grad():
        policy.actor_target(torch.full((8, 3), 5.0))
        policy.critic_target(torch.full((8, 6), 5.0), torch.zeros(8, 4))
    torch.testing.assert_close(policy.actor_target.bn1.running_mean, actor_before)
    torch.testing.assert_close(policy.critic_target.bn1.running_mean, critic_before)


def test_sqddpg_target_batchnorm_running_stats_are_synchronized():
    policy = FOSQDDPGPolicy(
        state_dim=3, action_dim=2, n_agents=2, hidden_dim=16,
        tau=0.25, device="cpu"
    )
    with torch.no_grad():
        policy.actor.bn1.running_mean.fill_(4.0)
        policy.actor_target.bn1.running_mean.zero_()
        policy.critic.bn2.running_var.fill_(5.0)
        policy.critic_target.bn2.running_var.fill_(1.0)

    policy.update_target_networks()

    torch.testing.assert_close(
        policy.actor_target.bn1.running_mean,
        torch.full_like(policy.actor_target.bn1.running_mean, 1.0),
    )
    torch.testing.assert_close(
        policy.critic_target.bn2.running_var,
        torch.full_like(policy.critic_target.bn2.running_var, 2.0),
    )

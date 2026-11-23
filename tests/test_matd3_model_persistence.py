"""MATD3 checkpoints must preserve target networks and optimizer states."""
import pytest
import torch

from algorithms.MATD3.fomatd3.fomatd3_adapter import FOMATD3Adapter


def _adapter():
    return FOMATD3Adapter(state_dim=3, action_dim=5, num_agents=2,
                          hidden_dim=16, buffer_capacity=8, batch_size=2,
                          device="cpu")


def test_matd3_missing_checkpoint_fails_loudly(tmp_path):
    adapter = _adapter()
    with pytest.raises(FileNotFoundError):
        adapter.load_models(str(tmp_path / "does_not_exist"))


def test_matd3_checkpoint_restores_targets_and_optimizer_state(tmp_path):
    torch.manual_seed(17)
    adapter = _adapter()
    agent = adapter.agents["manager_1"]
    states = torch.randn(3, adapter.global_state_dim)
    obs = torch.randn(3, adapter.state_dim)
    agent.update_actor(states, obs)
    agent.update_critic(
        states, torch.zeros(3, adapter.action_dim), torch.ones(3),
        states, torch.zeros(3, adapter.action_dim), torch.zeros(3),
        target_noise=0.,
    )
    agent.soft_update_targets()
    adapter.total_iterations = 7
    adapter.noise_scale = 0.13

    expected = {}
    for mid, item in adapter.agents.items():
        expected[mid] = {
            key: {name: tensor.detach().clone() for name, tensor in
                  getattr(item, key).state_dict().items()}
            for key in ("actor", "critic", "actor_target", "critic_target")
        }
    path = str(tmp_path / "nested" / "checkpoint")
    adapter.save_models(path)

    for item in adapter.agents.values():
        with torch.no_grad():
            for network in (item.actor, item.critic, item.actor_target, item.critic_target):
                for param in network.parameters():
                    param.add_(10.)
    adapter.total_iterations = 0
    adapter.noise_scale = 0.3
    adapter.load_models(path)
    assert adapter.total_iterations == 7
    assert adapter.noise_scale == pytest.approx(0.13)
    for mid, item in adapter.agents.items():
        for key in expected[mid]:
            for name, tensor in getattr(item, key).state_dict().items():
                torch.testing.assert_close(tensor, expected[mid][key][name])
    assert len(adapter.agents["manager_1"].actor_optimizer.state) > 0
    assert len(adapter.agents["manager_1"].critic_optimizer.state) > 0

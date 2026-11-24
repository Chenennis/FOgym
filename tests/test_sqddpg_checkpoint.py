"""SQDDPG checkpoints must restore target networks and optimizer training state."""
import numpy as np
import torch

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def make_adapter():
    return FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )


def test_sqddpg_checkpoint_restores_training_state(tmp_path):
    torch.manual_seed(127)
    np.random.seed(127)
    original = make_adapter()
    for i in range(3):
        obs = {mid: np.full(3, i, dtype=np.float32) for mid in original.agent_ids}
        actions = {mid: np.zeros(5, dtype=np.float32) for mid in original.agent_ids}
        original.collect_step(
            obs=obs, next_obs=obs, actions=actions,
            rewards={"manager_1": 10.0, "manager_2": 20.0},
            dones={mid: False for mid in original.agent_ids},
            infos={}, timestep=i,
        )
    assert original.train_on_batch() is not None
    original.current_episode = 3
    original.noise_scale = 0.07
    original.fosqddpg.noise_scale = 0.07
    original.save_models(str(tmp_path))

    restored = make_adapter()
    restored.load_models(str(tmp_path))
    assert restored.current_episode == original.current_episode
    assert restored.training_iterations == original.training_iterations
    assert restored.fosqddpg.total_iterations == original.fosqddpg.total_iterations
    assert len(restored.fosqddpg.replay_buffer) == len(original.fosqddpg.replay_buffer)
    assert restored.noise_scale == original.noise_scale
    assert restored.reward_normalizer == original.reward_normalizer

    for before, after in zip(original.fosqddpg.policies, restored.fosqddpg.policies):
        for attribute in ("actor", "critic", "actor_target", "critic_target"):
            left = getattr(before, attribute).state_dict()
            right = getattr(after, attribute).state_dict()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        assert len(after.actor_optimizer.state) > 0
        assert len(after.critic_optimizer.state) > 0
        assert not after.actor_target.training
        assert not after.critic_target.training
    assert restored.train_on_batch() is not None


def test_sqddpg_checkpoint_missing_files_raise(tmp_path):
    restored = make_adapter()
    import pytest
    with pytest.raises(FileNotFoundError):
        restored.load_models(str(tmp_path))

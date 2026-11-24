"""MADDPG target and actor updates must use executed FlexOffer action units."""
from unittest.mock import patch

import numpy as np
import torch

from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG, map_normalized_fo_actions
from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter


def test_maddpg_fo_action_mapping_matches_adapter_and_keeps_gradient():
    adapter = FOMAddpgAdapter(
        state_dim=3, action_dim=10, num_agents=2,
        hidden_dim=16, batch_size=2, device="cpu",
    )
    raw = torch.tensor([[-0.4, 0.3, 0.0, 0.2, -0.7,
                         0.1, -0.2, 0.6, -0.4, 0.8]], requires_grad=True)
    mapped = map_normalized_fo_actions(raw)
    np.testing.assert_allclose(
        mapped.detach().numpy()[0],
        adapter._map_action_to_fo_params(raw.detach().numpy()[0]),
        rtol=1e-6, atol=1e-6,
    )
    mapped.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_maddpg_critic_target_and_actor_receive_physical_actions():
    torch.manual_seed(23)
    algorithm = FOMADDPG(
        n_agents=2, state_dim=3, action_dim=5, hidden_dim=16,
        batch_size=2, device="cpu",
    )
    algorithm.action_dims = [5, 5]
    for i in range(2):
        algorithm.store_experience(
            np.full((2, 3), i * 0.1, dtype=np.float32),
            np.array([[0., 0., .55, 1.5, 1.],
                      [0., 0., .55, 1.5, 1.]], dtype=np.float32),
            np.array([1., 1.], dtype=np.float32),
            np.full((2, 3), (i + 1) * 0.1, dtype=np.float32),
            np.array([False, False]),
        )
    actor_before = [p.detach().clone() for p in algorithm.agents[0].actor.parameters()]
    seen = {}
    original_critic = algorithm.agents[0].update_critic
    original_actor = algorithm.agents[0].update_actor

    def inspect_critic(*args, **kwargs):
        seen["next_actions"] = kwargs["next_actions"].detach().clone()
        return original_critic(*args, **kwargs)

    def inspect_actor(*args, **kwargs):
        seen["actor_actions"] = kwargs["all_actions"].detach().clone()
        return original_actor(*args, **kwargs)

    with patch.object(algorithm.agents[0], "update_critic", side_effect=inspect_critic), \
         patch.object(algorithm.agents[0], "update_actor", side_effect=inspect_actor):
        result = algorithm.update()

    assert np.isfinite(result["actor_loss"])
    assert np.isfinite(result["critic_loss"])
    for name in ("next_actions", "actor_actions"):
        blocks = seen[name].reshape(-1, 2, 5)
        assert torch.all(blocks[..., 2] >= 0.1)
        assert torch.all(blocks[..., 2] <= 1.0)
        assert torch.all(blocks[..., 3] >= 1.0)
        assert torch.all(blocks[..., 3] <= 2.0)
        assert torch.all(blocks[..., 4] >= 0.1)
        assert torch.all(blocks[..., 4] <= 2.0)
    assert any(not torch.equal(before, after)
               for before, after in zip(actor_before, algorithm.agents[0].actor.parameters()))

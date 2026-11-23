"""SQDDPG Bellman targets must use the same physical action encoding as replay."""
import numpy as np
import torch
from unittest.mock import patch

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG, map_normalized_fo_actions
from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_target_critic_receives_environment_action_units():
    torch.manual_seed(41)
    np.random.seed(41)
    algo = FOSQDDPG(
        n_agents=2, state_dim=3, action_dim=5,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    for i in range(2):
        states = np.full((2, 3), 0.1 * (i + 1), dtype=np.float32)
        # Executed FO actions: start/end in [-1,1], min [0.1,1],
        # max [1,2], priority [0.1,2].
        actions = np.tile(
            np.array([0., 0., 0.55, 1.5, 1.05], dtype=np.float32), (2, 1)
        )
        algo.store_experience(
            states, actions, np.array([1., 1.], dtype=np.float32),
            states + 0.1, np.array([False, False]),
        )

    received = []
    critic = algo.policies[0].critic_target
    original_forward = critic.forward

    def capture(states, actions, fo_satisfaction=None):
        received.append(actions.detach().clone())
        return original_forward(states, actions, fo_satisfaction)

    with patch.object(critic, "forward", side_effect=capture):
        result = algo.update()

    assert result is not None
    assert received, "Target critic was not evaluated"
    joint_actions = received[0].reshape(-1, 2, 5)
    assert torch.all(joint_actions[..., 2] >= 0.1)
    assert torch.all(joint_actions[..., 2] <= 1.0)
    assert torch.all(joint_actions[..., 3] >= 1.0)
    assert torch.all(joint_actions[..., 3] <= 2.0)
    assert torch.all(joint_actions[..., 4] >= 0.1)
    assert torch.all(joint_actions[..., 4] <= 2.0)


def test_sqddpg_actor_and_adapter_share_physical_mapping_and_gradient():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=10, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2,
        action_dims={"manager_1": 5, "manager_2": 10},
    )
    raw = torch.tensor(
        [-0.4, 0.2, -0.5, 0.1, 0.6, 0.7, -0.2, 0.3, -0.8, -0.1],
        dtype=torch.float32, requires_grad=True
    )
    mapped = map_normalized_fo_actions(raw)
    np.testing.assert_allclose(
        mapped.detach().numpy(),
        adapter._map_action_to_fo_params(raw.detach().numpy()),
        rtol=1e-6, atol=1e-6,
    )
    mapped.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad.abs() > 0)

    # Padded tail dimensions must remain zero when an agent only owns one device.
    raw_batch = torch.zeros(2, 10)
    physical = map_normalized_fo_actions(raw_batch)
    physical[0, 5:] = 0
    torch.testing.assert_close(physical[0, 5:], torch.zeros(5))

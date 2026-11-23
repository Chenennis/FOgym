"""SQDDPG Bellman targets must use the same physical action encoding as replay."""
import numpy as np
import torch
from unittest.mock import patch

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG


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

"""SQDDPG must never report a failed update as successful training."""
import numpy as np
import pytest
from unittest.mock import patch

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_failed_update_is_not_silently_swallowed():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    for i in range(2):
        state = np.ones((2, 3), dtype=np.float32) * i
        actions = np.zeros((2, 5), dtype=np.float32)
        adapter.fosqddpg.store_experience(
            state, actions, np.ones(2, dtype=np.float32),
            state + 1, np.zeros(2, dtype=bool),
        )

    with patch.object(adapter.fosqddpg, "update", side_effect=RuntimeError("simulated optimizer failure")):
        with pytest.raises(RuntimeError, match="simulated optimizer failure"):
            adapter.train_on_batch()

    assert adapter.training_iterations == 0
    assert adapter.total_iterations == 0


def test_sqddpg_training_stats_count_only_real_updates():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
    )
    # No transitions means no optimizer steps, not a successful training run.
    assert adapter.train_on_batch() is None
    assert adapter.get_training_stats()["training_iterations"] == 0

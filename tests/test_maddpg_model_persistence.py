"""MADDPG model save/load should fail visibly rather than claim success."""
from pathlib import Path

import pytest

from algorithms.MADDPG.fomaddpg.fomaddpg_adapter import FOMAddpgAdapter


def test_maddpg_load_missing_model_raises(tmp_path):
    adapter = FOMAddpgAdapter(state_dim=3, action_dim=5, num_agents=2,
                              hidden_dim=16, batch_size=2, device="cpu")
    with pytest.raises(FileNotFoundError):
        adapter.load_models(str(tmp_path / "absent"))


def test_maddpg_save_and_reload_policy_weights(tmp_path):
    adapter = FOMAddpgAdapter(state_dim=3, action_dim=5, num_agents=2,
                              hidden_dim=16, batch_size=2, device="cpu")
    import torch
    expected = [
        [tensor.detach().clone() for tensor in policy.actor.parameters()]
        for policy in adapter.fomaddpg.agents
    ]
    prefix = str(tmp_path / "nested" / "checkpoint")
    adapter.save_models(prefix)
    assert all(Path(f"{prefix}_agent_{i}.pt").is_file() for i in range(2))
    for policy in adapter.fomaddpg.agents:
        with torch.no_grad():
            for parameter in policy.actor.parameters():
                parameter.add_(1.0)
    adapter.load_models(prefix)
    for actual, original in zip(adapter.fomaddpg.agents, expected):
        for param, saved in zip(actual.actor.parameters(), original):
            torch.testing.assert_close(param, saved)

"""MAPPO D2 regression: buffer observations and PPO action/log-prob pairing.

These tests specify the required collection contract. They intentionally fail
against the uncorrected adapter, which lacks next_obs and raw_actions inputs.
The environment executes mapped actions; PPO must store the policy-sampled
actions and the log probabilities of those same actions.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _collect_step_method():
    source = (Path(__file__).resolve().parents[1] / "algorithms" /
              "MAPPO" / "fomappo" / "fomappo_adapter.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    adapter = next(x for x in module.body
                   if isinstance(x, ast.ClassDef) and x.name == "FOMAPPOAdapter")
    method = next(x for x in adapter.body
                  if isinstance(x, ast.FunctionDef) and x.name == "collect_step")
    namespace = {"np": np, "Dict": Dict, "Tuple": Tuple, "Any": Any,
                 "Optional": Optional, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["collect_step"]


class _RecordingBuffer:
    def __init__(self):
        self.step = 0
        self.collected_steps = 0
        self.recorded = []

    def insert(self, **kwargs):
        self.recorded.append(kwargs)
        self.step = (self.step + 1) % 2


def _adapter():
    return SimpleNamespace(
        device=None, num_agents=1, action_dim=1,
        actual_obs_dim=1, buffer=_RecordingBuffer(), trainer=None,
        args=SimpleNamespace(use_reward_normalization=False,
                             recurrent_N=1, hidden_size=2, episode_length=2),
    )


def _step_kwargs():
    return dict(
        obs={"manager_1": np.array([1.0], dtype=np.float32)},
        next_obs={"manager_1": np.array([3.0], dtype=np.float32)},
        actions={"manager_1": np.array([0.8], dtype=np.float32)},
        raw_actions={"manager_1": np.array([-0.2], dtype=np.float32)},
        rewards={"manager_1": 1.0},
        dones={"manager_1": False},
        infos={"manager_1": {}},
        action_log_probs={"manager_1": np.array([-0.4], dtype=np.float32)},
        values={"manager_1": np.array([0.5], dtype=np.float32)},
    )


def test_insert_stores_next_observation_not_current_observation():
    adapter = _adapter()
    params = _step_kwargs()
    params.pop("raw_actions")
    _collect_step_method()(adapter, **params)
    inserted = adapter.buffer.recorded[0]
    np.testing.assert_allclose(inserted["obs"][0, 0], params["next_obs"]["manager_1"])
    np.testing.assert_allclose(inserted["share_obs"][0, 0],
                               params["next_obs"]["manager_1"])


def test_insert_pairs_raw_policy_action_with_its_log_probability():
    adapter = _adapter()
    params = _step_kwargs()
    _collect_step_method()(adapter, **params)
    inserted = adapter.buffer.recorded[0]
    np.testing.assert_allclose(inserted["actions"][0, 0],
                               params["raw_actions"]["manager_1"])
    np.testing.assert_allclose(inserted["action_log_probs"][0, 0],
                               params["action_log_probs"]["manager_1"])


def test_heterogeneous_manager_observations_are_padded_for_shared_buffer():
    """Collector must use the same zero-padded dimension as select_actions."""
    buffer = _RecordingBuffer()
    adapter = SimpleNamespace(
        device=None, num_agents=4, action_dim=3,
        actual_obs_dim=4, buffer=buffer, trainer=None,
        args=SimpleNamespace(use_reward_normalization=False,
                             recurrent_N=1, hidden_size=2, episode_length=2),
    )
    manager_ids = [f"manager_{i}" for i in range(1, 5)]
    obs = {
        mid: np.arange(dim, dtype=np.float32)
        for mid, dim in zip(manager_ids, [2, 4, 3, 1])
    }
    next_obs = {
        mid: np.arange(dim, dtype=np.float32) + 10
        for mid, dim in zip(manager_ids, [2, 4, 3, 1])
    }
    _collect_step_method()(
        adapter, obs=obs, next_obs=next_obs,
        actions={mid: np.full(3, 0.5, dtype=np.float32) for mid in manager_ids},
        raw_actions={mid: np.full(3, 0.2, dtype=np.float32) for mid in manager_ids},
        rewards={mid: 1.0 for mid in manager_ids},
        dones={mid: False for mid in manager_ids},
        infos={mid: {} for mid in manager_ids},
        action_log_probs={
            mid: np.array([-0.5], dtype=np.float32) for mid in manager_ids
        },
        values={mid: np.array([0.1], dtype=np.float32) for mid in manager_ids},
    )
    assert len(buffer.recorded) == 1
    recorded = buffer.recorded[0]
    assert recorded["obs"].shape == (1, 4, 4)
    assert recorded["share_obs"].shape == (1, 4, 4)
    np.testing.assert_allclose(recorded["obs"][0, 0], [10, 11, 0, 0])
    np.testing.assert_allclose(recorded["obs"][0, 1], [10, 11, 12, 13])
    np.testing.assert_allclose(recorded["obs"][0, 2], [10, 11, 12, 0])
    np.testing.assert_allclose(recorded["obs"][0, 3], [10, 0, 0, 0])

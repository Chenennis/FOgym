"""Regression: MAPPO return computation must accept a full wrapped rollout.

A ring write index of zero after episode_length inserts is not an empty
buffer. The adapter must not inject synthetic transitions during GAE setup.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _load_compute_returns():
    src = (Path(__file__).resolve().parents[1] / "algorithms" / "MAPPO" /
           "fomappo" / "fomappo_adapter.py")
    module = ast.parse(src.read_text(encoding="utf-8"))
    cls = next(x for x in module.body
               if isinstance(x, ast.ClassDef) and x.name == "FOMAPPOAdapter")
    fn = next(x for x in cls.body
              if isinstance(x, ast.FunctionDef) and x.name == "compute_returns")
    ns = {"np": np, "torch": torch,
          "logger": logging.getLogger(__name__),
          "Dict": Dict, "Any": Any, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    return ns["compute_returns"]


class _ReturnBuffer:
    def __init__(self):
        self.step = 0
        self.collected_steps = 2
        self.rewards = np.array([[[[1.0]]], [[[2.0]]]], dtype=np.float32)
        self.share_obs = np.array([[[[1.]]], [[[2.]]], [[[3.]]]], dtype=np.float32)
        self.rnn_states_critic = np.zeros((3, 1, 1, 1, 2), dtype=np.float32)
        self.masks = np.ones((3, 1, 1, 1), dtype=np.float32)
        self.value_preds = np.zeros((3, 1, 1, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((2, 1, 1, 1), dtype=np.float32)
        self.calls = 0

    def compute_returns(self, next_values, normalizer):
        self.calls += 1
        self.returns[0] = 1.0
        self.returns[1] = 2.0


class _Policy:
    def get_values(self, obs, states, masks):
        return torch.zeros((obs.shape[0], 1), dtype=torch.float32)


def test_full_wrapped_rollout_computes_returns_without_dummy_transitions():
    buffer = _ReturnBuffer()
    dummy_calls = []
    adapter = SimpleNamespace(
        buffer=buffer, policy=_Policy(), device=torch.device("cpu"),
        trainer=SimpleNamespace(value_normalizer=None),
        args=SimpleNamespace(episode_length=2),
        _training_started=True,
        _add_dummy_data_to_buffer=lambda: dummy_calls.append(True),
    )
    success = _load_compute_returns()(adapter)
    assert success is True
    assert buffer.calls == 1
    assert dummy_calls == [], "A full rollout must never be padded with dummy data"

"""Regression for MAPPO D1: a complete circular rollout must train.

The test extracts the repository's actual FOMAPPO.train method without importing
optional onpolicy dependencies. It uses a minimal, deterministic PPO batch and
checks a real torch actor parameter update (not a reported loss alone).
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def _load_train_method():
    source = (Path(__file__).resolve().parents[1]
              / "algorithms" / "MAPPO" / "fomappo" / "fomappo.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    trainer_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FOMAPPO"
    )
    method = next(
        node for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    namespace = {"np": np, "torch": torch, "nn": nn,
                 "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["train"]


class _Policy:
    def __init__(self):
        self.actor = nn.Linear(1, 1, bias=False)
        self.critic = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.actor.weight.fill_(0.1)
            self.critic.weight.fill_(0.1)
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.1)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.1)

    def evaluate_actions(self, share_obs, obs, rnn_actor, rnn_critic,
                         actions, masks, active_masks):
        values = self.critic(share_obs)
        log_probs = self.actor(obs)
        entropy = log_probs.mean() * 0 + torch.tensor(0.1)
        return values, log_probs, entropy


class _FullCircularBuffer:
    """Two-step rollout: the ring index has wrapped to zero after filling."""

    def __init__(self):
        self.step = 0
        self.episode_length = 2
        self.collected_steps = 2
        self.rewards = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
        self.advantages = np.ones((2, 1, 1), dtype=np.float32)
        self.returns = np.array([[[1.0]], [[2.0]], [[0.0]]], dtype=np.float32)
        self.value_preds = np.zeros((3, 1, 1), dtype=np.float32)
        self.updated = False

    def feed_forward_generator(self, advantages, num_mini_batch):
        obs = np.array([[1.0], [2.0]], dtype=np.float32)
        zeros = np.zeros((2, 1), dtype=np.float32)
        ones = np.ones((2, 1), dtype=np.float32)
        yield (
            obs, obs, zeros, zeros, zeros,
            zeros, np.array([[1.0], [2.0]], dtype=np.float32),
            ones, ones, zeros, np.array([[1.0], [0.5]], dtype=np.float32),
            None,
        )

    def after_update(self):
        self.updated = True


def test_full_rollout_trains_after_buffer_step_wraps_to_zero():
    policy = _Policy()
    buffer = _FullCircularBuffer()
    args = SimpleNamespace(
        use_advantage_normalization=False,
        episode_length=2,
        ppo_epoch=1,
        num_mini_batch=1,
        clip_param=0.2,
        use_clipped_value_loss=False,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        use_max_grad_norm=True,
        max_grad_norm=10.0,
    )
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy, args=args, device=torch.device("cpu"),
        _use_popart=False, _use_valuenorm=False, value_normalizer=None,
    )
    original_weight = policy.actor.weight.detach().clone()
    _load_train_method()(trainer)
    assert buffer.updated, "PPO skipped the full rollout after the ring index wrapped"
    assert not torch.equal(original_weight, policy.actor.weight.detach()), (
        "PPO failed to update the actor after a complete rollout"
    )


def test_empty_rollout_is_not_trainable_even_when_ring_index_is_zero():
    policy = _Policy()
    buffer = _FullCircularBuffer()
    buffer.collected_steps = 0
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy,
        args=SimpleNamespace(episode_length=2),
        device=torch.device("cpu"),
    )
    original = policy.actor.weight.detach().clone()
    _load_train_method()(trainer)
    assert not buffer.updated
    assert torch.equal(original, policy.actor.weight.detach())


def test_failed_ppo_update_does_not_report_success_or_consume_rollout():
    """A failed optimizer pass must not mark the entire rollout as trained."""
    policy = _Policy()
    buffer = _FullCircularBuffer()
    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")
    policy.evaluate_actions = fail_evaluation
    args = SimpleNamespace(
        episode_length=2, ppo_epoch=1, num_mini_batch=1,
        use_advantage_normalization=False, clip_param=0.2,
        use_clipped_value_loss=False, value_loss_coef=0.5,
        entropy_coef=0.0, use_max_grad_norm=True, max_grad_norm=10.0,
    )
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy, args=args, device=torch.device("cpu"),
        _use_popart=False, _use_valuenorm=False, value_normalizer=None,
    )
    with pytest.raises(RuntimeError, match="PPO update failed"):
        _load_train_method()(trainer)
    assert not buffer.updated
    assert buffer.collected_steps == 2

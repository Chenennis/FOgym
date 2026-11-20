"""Numerical GAE and bootstrap checks against FOgym's actual MAPPO buffer.

These tests evaluate the unmodified SharedReplayBuffer.compute_returns method,
not a reimplementation of GAE.
"""
from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from test_mappo_real_buffer import _load_shared_buffer_class


def _buffer(last_step_done):
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=1, hidden_size=2, recurrent_N=1,
        gamma=0.9, gae_lambda=0.8, use_gae=True, use_popart=False,
        use_valuenorm=False, use_proper_time_limits=False,
        algorithm_name="rmappo",
    )
    space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    buffer = _load_shared_buffer_class()(args, 1, space, space, space)
    buffer.rewards[:, 0, 0, 0] = [1.0, 2.0]
    buffer.value_preds[:2, 0, 0, 0] = [0.5, 0.25]
    buffer.masks[:, 0, 0, 0] = [1.0, 1.0, 0.0 if last_step_done else 1.0]
    return buffer


def test_gae_terminal_transition_does_not_bootstrap():
    buffer = _buffer(last_step_done=True)
    buffer.compute_returns(np.array([[[0.75]]], dtype=np.float32))
    # Last delta: 2 - 0.25 = 1.75 (no bootstrap).
    # First delta: 1 + 0.9*0.25 - 0.5 = 0.725.
    # First GAE: 0.725 + 0.9*0.8*1.75 = 1.985.
    np.testing.assert_allclose(buffer.returns[:2, 0, 0, 0],
                               [2.485, 2.0], rtol=1e-6)
    np.testing.assert_allclose(
        buffer.returns[:2, 0, 0, 0] - buffer.value_preds[:2, 0, 0, 0],
        [1.985, 1.75], rtol=1e-6
    )


def test_gae_nonterminal_transition_uses_final_bootstrap():
    buffer = _buffer(last_step_done=False)
    buffer.compute_returns(np.array([[[0.75]]], dtype=np.float32))
    # Last delta: 2 + 0.9*0.75 - 0.25 = 2.425.
    # First GAE: 0.725 + 0.9*0.8*2.425 = 2.471.
    np.testing.assert_allclose(buffer.returns[:2, 0, 0, 0],
                               [2.971, 2.675], rtol=1e-6)

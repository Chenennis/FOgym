"""Integration test for MAPPO D2 using the repository's real SharedReplayBuffer.

Verifies observation indexing, PPO action/log-prob pairing, rewards,
terminal masks, and the circular write-index behavior over a complete rollout.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from test_mappo_buffer_correctness import _collect_step_method


def _load_shared_buffer_class():
    """Load the real buffer class without importing onpolicy's eager package init."""
    source = (Path(__file__).resolve().parents[1] / "algorithms" / "MAPPO"
              / "onpolicy" / "utils" / "shared_buffer.py")
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in parsed.body
               if isinstance(node, ast.ClassDef) and node.name == "SharedReplayBuffer")

    def get_shape_from_obs_space(space):
        return space.shape

    def get_shape_from_act_space(space):
        return space.shape[0]

    namespace = {
        "np": np, "torch": torch,
        "get_shape_from_obs_space": get_shape_from_obs_space,
        "get_shape_from_act_space": get_shape_from_act_space,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["SharedReplayBuffer"]


def test_real_buffer_retains_aligned_two_step_rollout():
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=1, hidden_size=2,
        recurrent_N=1, gamma=0.99, gae_lambda=0.95, use_gae=True,
        use_popart=False, use_valuenorm=False,
        use_proper_time_limits=False, algorithm_name="rmappo",
        use_reward_normalization=False,
    )
    space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    buffer = _load_shared_buffer_class()(args, num_agents=1, obs_space=space,
                                cent_obs_space=space, act_space=space)
    adapter = SimpleNamespace(
        num_agents=1, action_dim=1, actual_obs_dim=1,
        args=args, buffer=buffer, trainer=None,
    )
    collect = _collect_step_method()

    for t in range(2):
        collect(
            adapter,
            obs={"manager_1": np.array([10.0 + t], dtype=np.float32)},
            next_obs={"manager_1": np.array([11.0 + t], dtype=np.float32)},
            actions={"manager_1": np.array([0.8], dtype=np.float32)},
            raw_actions={"manager_1": np.array([-0.2 - t], dtype=np.float32)},
            action_log_probs={"manager_1": np.array([-0.4 - t], dtype=np.float32)},
            values={"manager_1": np.array([0.5 + t], dtype=np.float32)},
            rewards={"manager_1": 1.0 + t},
            dones={"manager_1": t == 1},
            infos={"manager_1": {}},
        )

    assert buffer.step == 0
    assert buffer.collected_steps == 2
    np.testing.assert_allclose(buffer.obs[:, 0, 0, 0], [10., 11., 12.])
    np.testing.assert_allclose(buffer.share_obs[:, 0, 0, 0], [10., 11., 12.])
    np.testing.assert_allclose(buffer.actions[:, 0, 0, 0], [-0.2, -1.2])
    np.testing.assert_allclose(buffer.action_log_probs[:, 0, 0, 0],
                               [-0.4, -1.4])
    np.testing.assert_allclose(buffer.value_preds[:2, 0, 0, 0], [0.5, 1.5])
    np.testing.assert_allclose(buffer.rewards[:, 0, 0, 0], [1., 2.])
    np.testing.assert_allclose(buffer.masks[:, 0, 0, 0], [1., 1., 0.])
    np.testing.assert_allclose(buffer.active_masks[:, 0, 0, 0], [1., 1., 0.])

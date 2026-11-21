"""MAPPO PPO numerical invariants and multi-dimensional Gaussian action contract."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from test_mappo_real_buffer import _load_shared_buffer_class
from test_mappo_buffer_correctness import _collect_step_method
from test_mappo_training_correctness import (
    _FullCircularBuffer, _Policy, _load_train_method,
)


def test_multidimensional_box_buffer_stores_one_joint_action_log_probability():
    """FixedNormal.log_probs sums over action dimensions: shape [B, 1]."""
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=1, hidden_size=2, recurrent_N=1,
        gamma=0.99, gae_lambda=0.95, use_gae=True, use_popart=False,
        use_valuenorm=False, use_proper_time_limits=False,
        algorithm_name="rmappo",
    )
    obs_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    buffer = _load_shared_buffer_class()(args, 1, obs_space, obs_space, action_space)
    assert buffer.actions.shape == (2, 1, 1, 3)
    assert buffer.action_log_probs.shape == (2, 1, 1, 1), (
        "Joint log probability should not be duplicated once per action dimension"
    )


@pytest.mark.parametrize(
    "ratios, expected_policy_loss",
    [
        ((1.5, 0.5), -0.2),  # Positive advantage clips high ratio; negative clips low ratio.
        ((0.5, 1.5), 0.5),   # Unfavorable changes are not clipped away.
    ],
)
def test_ppo_clipping_with_positive_and_negative_advantages(ratios, expected_policy_loss):
    policy = _Policy()
    buffer = _FullCircularBuffer()
    buffer.returns[:2, 0, 0] = [1.0, -1.0]

    def deterministic_generator(advantages, num_mini_batch):
        obs = np.array([[1.0], [2.0]], dtype=np.float32)
        zeros = np.zeros((2, 1), dtype=np.float32)
        ones = np.ones((2, 1), dtype=np.float32)
        yield (
            obs, obs, zeros, zeros, zeros, zeros,
            buffer.returns[:2].reshape(2, 1),
            ones, ones, zeros, advantages.reshape(2, 1), None,
        )

    buffer.feed_forward_generator = deterministic_generator

    def fixed_ratios(share_obs, obs, rnn_actor, rnn_critic,
                     actions, masks, available_actions=None, active_masks=None):
        current_log_probs = torch.log(
            torch.tensor(ratios, dtype=torch.float32).reshape(-1, 1)
        )
        # Keep the actor in the graph so backward() exercises the real PPO path.
        current_log_probs = current_log_probs + 0.0 * policy.actor(obs)
        values = policy.critic(share_obs)
        return values, current_log_probs, torch.tensor(0.0)

    policy.evaluate_actions = fixed_ratios
    args = SimpleNamespace(
        use_advantage_normalization=False, episode_length=2,
        ppo_epoch=1, num_mini_batch=1, clip_param=0.2,
        use_clipped_value_loss=False, value_loss_coef=0.0,
        entropy_coef=0.0, use_max_grad_norm=True, max_grad_norm=10.0,
    )
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy, args=args, device=torch.device("cpu"),
        _use_popart=False, _use_valuenorm=False, value_normalizer=None,
    )
    info = _load_train_method()(trainer)
    assert info["num_updates"] == 1
    assert info["policy_loss"] == pytest.approx(expected_policy_loss, abs=1e-6)


def test_multidimensional_action_collection_preserves_joint_log_probs():
    """Exercise adapter collection with a real 3-D-action SharedReplayBuffer."""
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=1, hidden_size=2, recurrent_N=1,
        gamma=0.99, gae_lambda=0.95, use_gae=True, use_popart=False,
        use_valuenorm=False, use_proper_time_limits=False,
        algorithm_name="rmappo", use_reward_normalization=False,
    )
    observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    buffer = _load_shared_buffer_class()(
        args, 1, observation_space, observation_space, action_space
    )
    adapter = SimpleNamespace(
        num_agents=1, action_dim=3, actual_obs_dim=1,
        args=args, buffer=buffer, trainer=None,
    )
    for t in range(2):
        _collect_step_method()(
            adapter,
            obs={"manager_1": np.array([t], dtype=np.float32)},
            next_obs={"manager_1": np.array([t + 1], dtype=np.float32)},
            actions={"manager_1": np.array([0.5, 0.5, 0.5], dtype=np.float32)},
            raw_actions={"manager_1": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
            action_log_probs={"manager_1": np.array([-1.5 - t], dtype=np.float32)},
            values={"manager_1": np.array([0.1], dtype=np.float32)},
            rewards={"manager_1": 1.0},
            dones={"manager_1": t == 1},
            infos={"manager_1": {}},
        )
    assert buffer.step == 0
    assert buffer.collected_steps == 2
    assert buffer.actions.shape == (2, 1, 1, 3)
    assert buffer.action_log_probs.shape == (2, 1, 1, 1)
    np.testing.assert_allclose(buffer.action_log_probs[:, 0, 0, 0], [-1.5, -2.5])
    np.testing.assert_allclose(buffer.actions[0, 0, 0], [0.1, 0.2, 0.3])


def test_real_multidimensional_rollout_can_run_gae_and_ppo_update():
    """Real buffer + real collector + real GAE + real PPO trainer, mock tiny policy."""
    args = SimpleNamespace(
        episode_length=2, n_rollout_threads=1, hidden_size=2, recurrent_N=1,
        gamma=0.9, gae_lambda=0.95, use_gae=True,
        use_popart=False, use_valuenorm=False, use_proper_time_limits=False,
        algorithm_name="rmappo", use_reward_normalization=False,
        use_advantage_normalization=False,
        ppo_epoch=1, num_mini_batch=1, clip_param=0.2,
        use_clipped_value_loss=False, value_loss_coef=0.5,
        entropy_coef=0.0, use_max_grad_norm=True, max_grad_norm=10.0,
    )
    obs_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    buffer = _load_shared_buffer_class()(args, 1, obs_space, obs_space, act_space)
    adapter = SimpleNamespace(
        num_agents=1, action_dim=3, actual_obs_dim=1,
        args=args, buffer=buffer, trainer=None,
    )
    collect = _collect_step_method()
    for t in range(2):
        collect(
            adapter,
            obs={"manager_1": np.array([float(t + 1)], dtype=np.float32)},
            next_obs={"manager_1": np.array([float(t + 2)], dtype=np.float32)},
            actions={"manager_1": np.full(3, 0.6, dtype=np.float32)},
            raw_actions={"manager_1": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
            # Match the initial mock actor's log-prob (weight=0.1, obs=t+1),
            # so PPO starts at ratio=1 rather than at a clipped plateau.
            action_log_probs={"manager_1": np.array([0.1 * (t + 1)], dtype=np.float32)},
            values={"manager_1": np.array([0.1], dtype=np.float32)},
            rewards={"manager_1": float(t + 1)},
            dones={"manager_1": t == 1},
            infos={"manager_1": {}},
        )

    buffer.compute_returns(np.array([[[0.0]]], dtype=np.float32))
    policy = _Policy()
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy, args=args, device=torch.device("cpu"),
        _use_popart=False, _use_valuenorm=False, value_normalizer=None,
    )
    original_actor = policy.actor.weight.detach().clone()
    info = _load_train_method()(trainer)
    assert info["num_updates"] == 1
    assert info["ratio"] == pytest.approx(1.0, abs=1e-6)
    assert not torch.equal(policy.actor.weight.detach(), original_actor)
    assert buffer.collected_steps == 0
    assert buffer.step == 0

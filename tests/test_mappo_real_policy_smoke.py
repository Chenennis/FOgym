"""Smoke test using the repository's real FOActor, FOCritic and Gaussian policy.

Load only the required source modules, avoiding the bundled onpolicy package's
unrelated eager imports (which require optional environment dependencies).
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from test_mappo_buffer_correctness import _collect_step_method
from test_mappo_real_buffer import _load_shared_buffer_class
from test_mappo_training_correctness import _load_train_method


def _load_real_policy(monkeypatch):
    base = Path(__file__).resolve().parents[1] / "algorithms" / "MAPPO"
    packages = {
        "onpolicy": base / "onpolicy",
        "onpolicy.algorithms": base / "onpolicy" / "algorithms",
        "onpolicy.algorithms.r_mappo": base / "onpolicy" / "algorithms" / "r_mappo",
        "onpolicy.algorithms.r_mappo.algorithm": base / "onpolicy" / "algorithms" / "r_mappo" / "algorithm",
        "onpolicy.algorithms.utils": base / "onpolicy" / "algorithms" / "utils",
        "onpolicy.utils": base / "onpolicy" / "utils",
    }
    for name, directory in packages.items():
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, package)
    path = base / "fomappo" / "fomappo_policy.py"
    name = "_test_real_fomappo_policy"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.FOMAPPOPolicy


def test_real_fo_policy_samples_and_re_evaluates_joint_gaussian_actions(monkeypatch):
    torch.manual_seed(7)
    policy_type = _load_real_policy(monkeypatch)
    args = SimpleNamespace(
        lr=1e-3, critic_lr=1e-3, opti_eps=1e-5, weight_decay=0.0,
        hidden_size=16, gain=0.01, use_orthogonal=True,
        use_policy_active_masks=False, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, recurrent_N=1, algorithm_name="rmappo",
        use_popart=False, use_feature_normalization=False, stacked_frames=1,
        layer_N=1, use_ReLU=True, use_device_attention=False,
        use_manager_coordination=False,
    )
    obs_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    policy = policy_type(args, obs_space, obs_space, act_space, torch.device("cpu"))
    observation = torch.tensor([[0.2, -0.1, 0.3, 0.4], [0.1, 0.2, -0.1, 0.2]])
    rnn = torch.zeros((2, 1, 16))
    masks = torch.ones((2, 1))
    values, actions, old_log_probs, _, _ = policy.get_actions(
        observation, observation, rnn, rnn, masks
    )
    assert actions.shape == (2, 3)
    assert old_log_probs.shape == (2, 1)
    assert values.shape == (2, 1)

    new_values, new_log_probs, entropy = policy.evaluate_actions(
        observation, observation, rnn, rnn, actions, masks,
        available_actions=None, active_masks=torch.ones((2, 1)),
    )
    torch.testing.assert_close(new_log_probs, old_log_probs.detach(), atol=1e-5, rtol=1e-5)
    assert entropy.ndim == 0
    actor_before = [p.detach().clone() for p in policy.actor.parameters()]
    critic_before = [p.detach().clone() for p in policy.critic.parameters()]
    objective = -new_log_probs.mean() + (new_values - 1.0).square().mean()
    policy.actor_optimizer.zero_grad()
    policy.critic_optimizer.zero_grad()
    objective.backward()
    policy.actor_optimizer.step()
    policy.critic_optimizer.step()
    assert any(not torch.equal(before, after) for before, after in
               zip(actor_before, policy.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in
               zip(critic_before, policy.critic.parameters()))


@pytest.mark.parametrize("use_valuenorm", [False, True])
def test_real_policy_and_real_buffer_complete_ppo_training_update(monkeypatch, use_valuenorm):
    """One full PPO update using real sampled actions, Gaussian log-probs and GAE."""
    torch.manual_seed(11)
    policy_type = _load_real_policy(monkeypatch)
    args = SimpleNamespace(
        lr=1e-3, critic_lr=1e-3, opti_eps=1e-5, weight_decay=0.0,
        hidden_size=16, gain=0.01, use_orthogonal=True,
        use_policy_active_masks=False, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, recurrent_N=1, algorithm_name="rmappo",
        use_popart=False, use_feature_normalization=False, stacked_frames=1,
        layer_N=1, use_ReLU=True, use_device_attention=False,
        use_manager_coordination=False, episode_length=2, n_rollout_threads=1,
        gamma=0.9, gae_lambda=0.95, use_gae=True, use_valuenorm=use_valuenorm,
        use_proper_time_limits=False, use_reward_normalization=False,
        use_advantage_normalization=False, ppo_epoch=1, num_mini_batch=1,
        clip_param=0.2, use_clipped_value_loss=False,
        value_loss_coef=0.5, entropy_coef=0.0, use_max_grad_norm=True,
        max_grad_norm=1.0,
    )
    obs_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    policy = policy_type(args, obs_space, obs_space, action_space, torch.device("cpu"))
    buffer = _load_shared_buffer_class()(args, 1, obs_space, obs_space, action_space)
    adapter = SimpleNamespace(
        num_agents=1, action_dim=3, actual_obs_dim=4,
        args=args, buffer=buffer, trainer=None,
    )
    observations = [
        np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float32),
        np.array([0.1, 0.2, -0.1, 0.2], dtype=np.float32),
        np.array([-0.2, 0.1, 0.2, 0.3], dtype=np.float32),
    ]
    for t in range(2):
        state = torch.as_tensor(observations[t]).unsqueeze(0)
        rnn = torch.zeros((1, 1, 16))
        masks = torch.ones((1, 1))
        with torch.no_grad():
            values, actions, old_log_probs, _, _ = policy.get_actions(
                state, state, rnn, rnn, masks
            )
        _collect_step_method()(
            adapter,
            obs={"manager_1": observations[t]},
            next_obs={"manager_1": observations[t + 1]},
            actions={"manager_1": actions[0].numpy()},
            raw_actions={"manager_1": actions[0].numpy()},
            action_log_probs={"manager_1": old_log_probs[0].numpy()},
            values={"manager_1": values[0].numpy()},
            rewards={"manager_1": float(t + 1)},
            dones={"manager_1": t == 1},
            infos={"manager_1": {}},
        )
    assert buffer.collected_steps == 2
    if use_valuenorm:
        from onpolicy.utils.valuenorm import ValueNorm
        value_normalizer = ValueNorm(1, device=torch.device("cpu"))
    else:
        value_normalizer = None
    buffer.compute_returns(np.zeros((1, 1, 1), dtype=np.float32), value_normalizer)
    trainer = SimpleNamespace(
        buffer=buffer, policy=policy, args=args, device=torch.device("cpu"),
        _use_popart=False, _use_valuenorm=use_valuenorm,
        value_normalizer=value_normalizer,
    )
    actor_before = [p.detach().clone() for p in policy.actor.parameters()]
    critic_before = [p.detach().clone() for p in policy.critic.parameters()]
    result = _load_train_method()(trainer)
    assert result["num_updates"] == 1
    assert np.isfinite(result["policy_loss"])
    assert np.isfinite(result["value_loss"])
    assert result["ratio"] == pytest.approx(1.0, abs=1e-4)
    assert any(not torch.equal(old, new) for old, new in
               zip(actor_before, policy.actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in
               zip(critic_before, policy.critic.parameters()))
    assert buffer.collected_steps == 0

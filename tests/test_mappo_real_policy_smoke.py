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
import torch
from gymnasium import spaces


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

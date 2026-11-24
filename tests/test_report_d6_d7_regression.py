"""Regression tests for original report issues D6 and D7."""
import ast
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algorithms.SQDDPG.fosqddpg.fosqddpg import FOReplayBuffer, FOSQDDPG


ROOT = Path(__file__).resolve().parents[1]


def test_d7_training_entrypoints_do_not_cap_requested_episodes():
    paths = (
        "algorithms/SQDDPG/fosqddpg/fosqddpg_training_methods.py",
        "algorithms/MADDPG/fomaddpg/fomaddpg_training_methods.py",
        "algorithms/MATD3/fomatd3/fomatd3_training_methods.py",
    )
    for path in paths:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        assignments = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "max_allowed_episodes"
                    for target in node.targets)
        ]
        assert len(assignments) == 1, path
        assert isinstance(assignments[0], ast.Attribute), path
        assert isinstance(assignments[0].value, ast.Name), path
        assert assignments[0].value.id == "pipeline", path
        assert assignments[0].attr == "num_episodes", path


def _buffer(seed=None):
    buffer = FOReplayBuffer(capacity=20, rng=random.Random(seed) if seed is not None else None)
    for i in range(12):
        value = np.array([float(i)], dtype=np.float32)
        buffer.push(value, value, value, value, np.array([False]))
    return buffer


def test_d6_replay_injected_rng_is_independent_of_global_random():
    left, right = _buffer(101), _buffer(101)
    first = left.sample(6)[0].numpy()
    random.seed(918)
    for _ in range(100):
        random.random()
    second = right.sample(6)[0].numpy()
    np.testing.assert_array_equal(first, second)


def test_d6_replay_default_rng_preserves_global_random_behavior():
    buffer = _buffer()
    random.seed(42)
    expected = random.sample(range(12), 5)
    random.seed(42)
    actual = buffer.sample(5)[0].numpy().flatten().astype(int).tolist()
    assert actual == expected


def _sample_actions(noise_rng):
    algorithm = object.__new__(FOSQDDPG)
    algorithm.n_agents = 1
    algorithm.noise_scale = 0.2
    algorithm.noise_rng = noise_rng
    algorithm.policies = [
        SimpleNamespace(select_action=lambda state, constraints: np.zeros(3),
                        max_action=1.0)
    ]
    return algorithm.select_actions(np.zeros((1, 2)), add_noise=True)


def test_d6_exploration_injected_generator_is_independent_of_global_numpy():
    first = _sample_actions(np.random.default_rng(23))
    np.random.seed(718)
    np.random.normal(size=100)
    second = _sample_actions(np.random.default_rng(23))
    np.testing.assert_array_equal(first, second)


def test_d6_default_noise_matches_original_numpy_source():
    np.random.seed(67)
    expected = np.random.normal(0, 0.2, size=3)
    np.random.seed(67)
    actual = _sample_actions(np.random).ravel()
    np.testing.assert_array_equal(actual, expected)

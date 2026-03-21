"""Unit tests for action projection and constraint enforcement (Step 2)."""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Patch import issue
import fo_common.config
fo_common.config.Config = type('Config', (), {})


def test_project_actions_within_bounds():
    """Test that actions within bounds are not modified."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }
    agent = ManagerAgent(
        manager_id="test", manager_config=manager_config,
        users=[], devices=[]
    )
    # Simulate 2 controllable devices (10-dim actions)
    agent.controllable_devices = ['dev_1', 'dev_2']

    # All within bounds
    actions = np.array([
        0.5, -0.5, 0.5, 1.5, 1.0,   # device 1: all within bounds
        -0.3, 0.8, 0.3, 1.2, 0.5     # device 2: all within bounds
    ])

    projected, score, count = agent._project_actions(actions)
    np.testing.assert_array_almost_equal(projected, actions)
    assert score == 0.0, f"Expected score=0, got {score}"
    assert count == 0, f"Expected count=0, got {count}"
    print("PASS: Actions within bounds are not modified")


def test_project_actions_out_of_bounds():
    """Test that out-of-bound actions are clipped and violations counted."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }
    agent = ManagerAgent(
        manager_id="test", manager_config=manager_config,
        users=[], devices=[]
    )
    agent.controllable_devices = ['dev_1']

    # Some out of bounds: start_flex=2.0 (max 1.0), e_min=-0.5 (min 0.1)
    actions = np.array([2.0, -0.5, -0.5, 1.5, 1.0])

    projected, score, count = agent._project_actions(actions)

    # Check clipping
    assert projected[0] == 1.0, f"start_flex should be clipped to 1.0, got {projected[0]}"
    assert projected[1] == -0.5, f"end_flex -0.5 is within bounds, got {projected[1]}"
    assert projected[2] == 0.1, f"e_min should be clipped to 0.1, got {projected[2]}"
    assert projected[3] == 1.5, f"e_max 1.5 is within bounds, got {projected[3]}"

    # Check violation metrics
    assert score > 0.0, f"Expected positive violation score, got {score}"
    assert count >= 2, f"Expected at least 2 violations, got {count}"
    print(f"PASS: Out-of-bound actions clipped (score={score:.3f}, count={count})")


def test_constraint_penalty_in_reward():
    """Test that constraint penalty is reflected in reward info."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }
    agent = ManagerAgent(
        manager_id="test", manager_config=manager_config,
        users=[], devices=[],
        reward_weights={'alpha': 0.4, 'beta': 0.3, 'delta': 0.1, 'lambda': 0.2}
    )

    pipeline_results = {
        'stats': {
            'total_cost': 10.0, 'total_revenue': 30.0,
            'avg_satisfaction': 0.7, 'num_trades': 5, 'num_flexoffers': 10,
        }
    }
    env_state = {'price': 0.15}

    # With no constraint penalty
    r_clean, info_clean = agent._calculate_pipeline_reward(
        pipeline_results, env_state, constraint_penalty=0.0
    )

    # With constraint penalty
    r_penalty, info_penalty = agent._calculate_pipeline_reward(
        pipeline_results, env_state, constraint_penalty=5.0
    )

    assert info_clean['constraint_penalty_raw'] == 0.0
    assert info_penalty['constraint_penalty_raw'] == 5.0
    assert info_penalty['constraint_reward'] < 0, "Constraint reward should be negative for violations"
    assert r_penalty < r_clean, \
        f"Reward with penalty ({r_penalty:.2f}) should be less than clean ({r_clean:.2f})"
    print(f"PASS: Constraint penalty reduces reward ({r_clean:.2f} → {r_penalty:.2f})")


def test_delta_zero_removes_constraint_effect():
    """Test that delta=0 makes constraint penalty have no effect."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }

    pipeline_results = {
        'stats': {
            'total_cost': 10.0, 'total_revenue': 30.0,
            'avg_satisfaction': 0.7, 'num_trades': 5, 'num_flexoffers': 10,
        }
    }
    env_state = {'price': 0.15}

    # Use two separate agents at the same episode_count and fixed seed
    # to avoid the internal randomness (learning_progress_reward)
    np.random.seed(42)
    agent1 = ManagerAgent(
        manager_id="test1", manager_config=manager_config,
        users=[], devices=[],
        reward_weights={'alpha': 0.4, 'beta': 0.3, 'delta': 0.0, 'lambda': 0.2}
    )
    # Set episode_count > 50 to avoid random exploration phase
    agent1.episode_count = 100
    r_clean, _ = agent1._calculate_pipeline_reward(
        pipeline_results, env_state, constraint_penalty=0.0
    )

    agent2 = ManagerAgent(
        manager_id="test2", manager_config=manager_config,
        users=[], devices=[],
        reward_weights={'alpha': 0.4, 'beta': 0.3, 'delta': 0.0, 'lambda': 0.2}
    )
    agent2.episode_count = 100
    r_with_penalty, _ = agent2._calculate_pipeline_reward(
        pipeline_results, env_state, constraint_penalty=10.0
    )

    # With delta=0, constraint penalty should have no effect
    assert abs(r_clean - r_with_penalty) < 0.01, \
        f"With delta=0, rewards should be equal: {r_clean:.2f} vs {r_with_penalty:.2f}"
    print(f"PASS: delta=0 removes constraint effect ({r_clean:.2f} ≈ {r_with_penalty:.2f})")


if __name__ == "__main__":
    test_project_actions_within_bounds()
    test_project_actions_out_of_bounds()
    test_constraint_penalty_in_reward()
    test_delta_zero_removes_constraint_effect()
    print("\nAll action projection tests passed!")

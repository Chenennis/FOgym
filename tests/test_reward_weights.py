"""Unit tests for configurable reward weights (Step 1)."""
import sys
import os
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Patch the import issue in unified_config.py
import fo_common.config
fo_common.config.Config = type('Config', (), {})  # stub


def test_environment_config_reward_fields():
    """Test that EnvironmentConfig has the paper-aligned reward weight fields."""
    from fo_common.unified_config import EnvironmentConfig
    config = EnvironmentConfig()
    assert config.reward_alpha == 0.4, f"Expected alpha=0.4, got {config.reward_alpha}"
    assert config.reward_beta == 0.3, f"Expected beta=0.3, got {config.reward_beta}"
    assert config.reward_delta == 0.1, f"Expected delta=0.1, got {config.reward_delta}"
    assert config.reward_lambda == 0.2, f"Expected lambda=0.2, got {config.reward_lambda}"
    print("PASS: EnvironmentConfig has correct default reward weight fields")


def test_manager_agent_accepts_reward_weights():
    """Test that ManagerAgent stores reward_weights correctly."""
    from fo_generate.multi_agent_env import ManagerAgent

    custom_weights = {'alpha': 0.5, 'beta': 0.0, 'delta': 0.2, 'lambda': 0.3}
    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }
    agent = ManagerAgent(
        manager_id="test_manager",
        manager_config=manager_config,
        users=[],
        devices=[],
        reward_weights=custom_weights
    )
    assert agent.reward_weights == custom_weights, \
        f"Expected {custom_weights}, got {agent.reward_weights}"
    print("PASS: ManagerAgent stores custom reward weights")


def test_manager_agent_default_reward_weights():
    """Test that ManagerAgent uses defaults when no weights provided."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }
    agent = ManagerAgent(
        manager_id="test_manager",
        manager_config=manager_config,
        users=[],
        devices=[]
    )
    assert agent.reward_weights['alpha'] == 0.4
    assert agent.reward_weights['beta'] == 0.3
    assert agent.reward_weights['delta'] == 0.1
    assert agent.reward_weights['lambda'] == 0.2
    print("PASS: ManagerAgent uses default reward weights when none provided")


def test_calculate_pipeline_reward_uses_weights():
    """Test that _calculate_pipeline_reward uses configured weights."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }

    # Test with beta=0 (w/o user satisfaction)
    weights_no_user = {'alpha': 0.4, 'beta': 0.0, 'delta': 0.1, 'lambda': 0.2}
    agent = ManagerAgent(
        manager_id="test_manager",
        manager_config=manager_config,
        users=[],
        devices=[],
        reward_weights=weights_no_user
    )

    # Create mock pipeline results
    pipeline_results = {
        'stats': {
            'total_cost': 10.0,
            'total_revenue': 30.0,
            'avg_satisfaction': 0.9,
            'num_trades': 5,
            'num_flexoffers': 10,
        }
    }
    env_state = {'price': 0.15}

    reward1, info1 = agent._calculate_pipeline_reward(pipeline_results, env_state)

    # Test with default weights
    weights_default = {'alpha': 0.4, 'beta': 0.3, 'delta': 0.1, 'lambda': 0.2}
    agent2 = ManagerAgent(
        manager_id="test_manager2",
        manager_config=manager_config,
        users=[],
        devices=[],
        reward_weights=weights_default
    )
    reward2, info2 = agent2._calculate_pipeline_reward(pipeline_results, env_state)

    # With beta=0, satisfaction should not contribute, so reward should differ
    # (unless satisfaction_reward happens to be 0, which is unlikely with avg_satisfaction=0.9)
    assert reward1 != reward2, \
        f"Rewards should differ when beta changes: {reward1} vs {reward2}"
    print(f"PASS: Reward weights affect output (beta=0: {reward1:.2f}, default: {reward2:.2f})")


def test_lambda_zero_coordination():
    """Test that lambda=0 removes coordination reward contribution."""
    from fo_generate.multi_agent_env import ManagerAgent

    manager_config = {
        'location_x': 0.0, 'location_y': 0.0,
        'coverage_area': 1.0, 'district_type': 'residential'
    }

    pipeline_results = {
        'stats': {
            'total_cost': 5.0,
            'total_revenue': 25.0,
            'avg_satisfaction': 0.7,
            'num_trades': 8,
            'num_flexoffers': 10,
        }
    }
    env_state = {'price': 0.15}

    # With lambda=0
    agent_no_coord = ManagerAgent(
        manager_id="m1", manager_config=manager_config,
        users=[], devices=[],
        reward_weights={'alpha': 0.4, 'beta': 0.3, 'delta': 0.1, 'lambda': 0.0}
    )
    r_no_coord, _ = agent_no_coord._calculate_pipeline_reward(pipeline_results, env_state)

    # With lambda=0.2
    agent_with_coord = ManagerAgent(
        manager_id="m2", manager_config=manager_config,
        users=[], devices=[],
        reward_weights={'alpha': 0.4, 'beta': 0.3, 'delta': 0.1, 'lambda': 0.2}
    )
    r_with_coord, _ = agent_with_coord._calculate_pipeline_reward(pipeline_results, env_state)

    # Trade success rate = 8/10 = 0.8 > 0.7, so coordination_reward > 0
    # With lambda=0, it shouldn't contribute
    assert r_no_coord < r_with_coord, \
        f"lambda=0 should give lower reward: {r_no_coord:.2f} vs {r_with_coord:.2f}"
    print(f"PASS: lambda=0 removes coordination contribution ({r_no_coord:.2f} < {r_with_coord:.2f})")


if __name__ == "__main__":
    test_environment_config_reward_fields()
    test_manager_agent_accepts_reward_weights()
    test_manager_agent_default_reward_weights()
    test_calculate_pipeline_reward_uses_weights()
    test_lambda_zero_coordination()
    print("\nAll reward weight tests passed!")

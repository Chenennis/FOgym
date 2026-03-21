"""Unit tests for experiment metrics tracker (Step 5)."""
import sys
import os
import numpy as np
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def test_metrics_tracker_basic():
    """Test basic metric recording and computation."""
    from fo_common.experiment_metrics import ExperimentMetricsTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentMetricsTracker(
            experiment_name="test_exp",
            results_dir=tmpdir,
            num_managers=4, num_users=36
        )

        # Record 10 episodes
        for i in range(10):
            tracker.start_episode()
            reward = 100.0 + i * 5.0
            tracker.record_episode(
                episode_num=i,
                total_reward=reward,
                manager_rewards={'m1': reward * 0.3, 'm2': reward * 0.7},
                reward_info={
                    'economic': 50.0 + i,
                    'user_satisfaction': 0.7 + i * 0.02,
                    'coordination': 30.0 + i,
                    'trade_success_rate': 0.6 + i * 0.03,
                    'net_benefit': 15.0 + i * 2,
                },
                constraint_violations=max(0, 5 - i),
                constraint_score=max(0.0, 2.0 - i * 0.2),
            )

        # Check computed metrics
        assert tracker.total_reward() > 0, "Total reward should be positive"
        assert tracker.user_satisfaction() > 0, "Satisfaction should be positive"
        assert tracker.trading_score() > 0, "Trading score should be positive"
        assert tracker.economic_benefit_dkk() > 0, "Economic benefit should be positive"
        assert tracker.constraint_violations_total() > 0, "Should have some violations"
        assert tracker.training_variance() > 0, "Variance should be nonzero"
        assert tracker.convergence_episode() > 0, "Convergence episode should be positive"
        print(f"Total reward: {tracker.total_reward():.2f}")
        print(f"User satisfaction: {tracker.user_satisfaction():.4f}")
        print(f"Economic (DKK): {tracker.economic_benefit_dkk():.2f}")
        print(f"Constraint violations: {tracker.constraint_violations_total()}")
        print(f"Training variance: {tracker.training_variance():.4f}")
        print(f"Convergence episode: {tracker.convergence_episode()}")
        print("PASS: Basic metric computation works")


def test_metrics_export_csv():
    """Test CSV export."""
    from fo_common.experiment_metrics import ExperimentMetricsTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentMetricsTracker(
            experiment_name="test_export",
            results_dir=tmpdir,
            num_managers=4, num_users=36
        )

        for i in range(5):
            tracker.start_episode()
            tracker.record_episode(
                episode_num=i, total_reward=100.0 + i,
                manager_rewards={'m1': 50.0},
                reward_info={
                    'economic': 50.0, 'user_satisfaction': 0.8,
                    'coordination': 30.0, 'trade_success_rate': 0.7,
                    'net_benefit': 20.0,
                },
                constraint_violations=1,
            )

        # Export episode CSV
        ep_path = tracker.export_episode_csv()
        assert os.path.exists(ep_path), f"Episode CSV not found at {ep_path}"

        # Check content
        with open(ep_path) as f:
            lines = f.readlines()
        assert len(lines) == 6, f"Expected 6 lines (header + 5 episodes), got {len(lines)}"
        print(f"Episode CSV: {ep_path} ({len(lines)} lines)")

        # Export summary CSV
        summary_path = tracker.export_summary_csv()
        assert os.path.exists(summary_path), f"Summary CSV not found at {summary_path}"

        with open(summary_path) as f:
            lines = f.readlines()
        assert len(lines) == 2, f"Expected 2 lines (header + 1 row), got {len(lines)}"
        print(f"Summary CSV: {summary_path} ({len(lines)} lines)")
        print("PASS: CSV export works correctly")


def test_per_user_normalization():
    """Test per-user normalized metrics."""
    from fo_common.experiment_metrics import ExperimentMetricsTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentMetricsTracker(
            experiment_name="test_peruser",
            results_dir=tmpdir,
            num_managers=10, num_users=90
        )
        tracker.start_episode()
        tracker.record_episode(
            episode_num=0, total_reward=900.0,
            manager_rewards={'m1': 90.0},
            reward_info={
                'economic': 50.0, 'user_satisfaction': 0.8,
                'coordination': 30.0, 'trade_success_rate': 0.7,
                'net_benefit': 180.0,
            },
        )

        per_user_r = tracker.per_user_reward()
        assert abs(per_user_r - 10.0) < 0.01, f"Expected ~10.0, got {per_user_r}"
        print(f"PASS: Per-user reward = {per_user_r:.2f} (900/90)")


if __name__ == "__main__":
    test_metrics_tracker_basic()
    test_metrics_export_csv()
    test_per_user_normalization()
    print("\nAll metrics tests passed!")

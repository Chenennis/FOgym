"""
Experiment Metrics Tracker for FOgym.

Tracks all paper-required metrics per episode and exports results as CSV.
Metrics correspond to Table 1 in the FOgym paper:
- Total Reward
- A & DisA (g_a): aggregation/disaggregation quality
- Trading (g_t): trading score
- Eco (g_e): economic benefit (DKK)
- User Sat. (g_u): user satisfaction

Plus auxiliary metrics:
- Constraint Violations
- Training Variance (std of last 100 episodes)
- Convergence Episode (first episode reaching 90% of final reward)
- Training Time per Episode
"""
import os
import time
import csv
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Economic values are already in DKK from the pipeline
DKK_PER_USD = 1.0  # No conversion needed - pipeline already uses DKK


class ExperimentMetricsTracker:
    """Tracks experiment metrics across episodes and exports results."""

    def __init__(self, experiment_name: str, results_dir: str = "results",
                 num_managers: int = 4, num_users: int = 36):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.num_managers = num_managers
        self.num_users = num_users

        # Per-episode records
        self.episode_rewards: List[float] = []
        self.episode_economic: List[float] = []
        self.episode_satisfaction: List[float] = []
        self.episode_coordination: List[float] = []
        self.episode_constraint_violations: List[int] = []
        self.episode_constraint_scores: List[float] = []
        self.episode_trade_success_rates: List[float] = []
        self.episode_net_benefits: List[float] = []
        self.episode_times: List[float] = []

        # Per-manager per-episode records
        self.manager_rewards: Dict[str, List[float]] = {}

        # Timing
        self._episode_start_time: Optional[float] = None

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "summary_tables"), exist_ok=True)

    def start_episode(self):
        """Call at the beginning of each episode."""
        self._episode_start_time = time.time()

    def record_episode(self, episode_num: int,
                       total_reward: float,
                       manager_rewards: Dict[str, float],
                       reward_info: Dict[str, float],
                       constraint_violations: int = 0,
                       constraint_score: float = 0.0):
        """Record metrics for one completed episode.

        Args:
            episode_num: episode number
            total_reward: total episode reward (summed across steps)
            manager_rewards: per-manager total rewards
            reward_info: aggregated reward components from pipeline
            constraint_violations: total constraint violation count
            constraint_score: total constraint violation score
        """
        # Timing
        elapsed = time.time() - self._episode_start_time if self._episode_start_time else 0.0
        self.episode_times.append(elapsed)

        # Core metrics
        self.episode_rewards.append(total_reward)
        self.episode_economic.append(reward_info.get('economic', 0.0))
        self.episode_satisfaction.append(reward_info.get('user_satisfaction', 0.0))
        self.episode_coordination.append(reward_info.get('coordination', 0.0))
        self.episode_trade_success_rates.append(reward_info.get('trade_success_rate', 0.0))
        self.episode_net_benefits.append(reward_info.get('net_benefit', 0.0))
        self.episode_constraint_violations.append(constraint_violations)
        self.episode_constraint_scores.append(constraint_score)

        # Per-manager rewards
        for mid, r in manager_rewards.items():
            if mid not in self.manager_rewards:
                self.manager_rewards[mid] = []
            self.manager_rewards[mid].append(r)

    # ──────────────── Computed Metrics ────────────────

    def total_reward(self) -> float:
        """Mean total reward over all episodes."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    def aggregation_disaggregation_score(self) -> float:
        """g_a: approximated from trade success rate (higher trade success → better aggregation)."""
        return float(np.mean(self.episode_trade_success_rates)) if self.episode_trade_success_rates else 0.0

    def trading_score(self) -> float:
        """g_t: mean trade success rate."""
        return float(np.mean(self.episode_trade_success_rates)) if self.episode_trade_success_rates else 0.0

    def economic_benefit_dkk(self) -> float:
        """g_e: mean net economic benefit in DKK."""
        return float(np.mean(self.episode_net_benefits)) * DKK_PER_USD if self.episode_net_benefits else 0.0

    def user_satisfaction(self) -> float:
        """g_u: mean user satisfaction."""
        return float(np.mean(self.episode_satisfaction)) if self.episode_satisfaction else 0.0

    def constraint_violations_total(self) -> int:
        """Total constraint violations across all episodes."""
        return int(np.sum(self.episode_constraint_violations))

    def training_variance(self, last_n: int = 100) -> float:
        """Std of last N episode rewards."""
        if len(self.episode_rewards) < 2:
            return 0.0
        tail = self.episode_rewards[-last_n:]
        return float(np.std(tail))

    def convergence_episode(self, threshold: float = 0.9) -> int:
        """First episode where rolling mean reaches threshold * final mean."""
        if len(self.episode_rewards) < 10:
            return len(self.episode_rewards)
        final_mean = np.mean(self.episode_rewards[-50:])
        target = threshold * final_mean
        window = 10
        for i in range(window, len(self.episode_rewards)):
            rolling_mean = np.mean(self.episode_rewards[i - window:i])
            if rolling_mean >= target:
                return i
        return len(self.episode_rewards)

    def mean_time_per_episode(self) -> float:
        """Mean training time per episode in seconds."""
        return float(np.mean(self.episode_times)) if self.episode_times else 0.0

    def per_user_reward(self) -> float:
        """Total reward normalized by number of users."""
        return self.total_reward() / max(self.num_users, 1)

    def per_user_economic_dkk(self) -> float:
        """Economic benefit per user in DKK."""
        return self.economic_benefit_dkk() / max(self.num_users, 1)

    # ──────────────── Export ────────────────

    def get_summary_dict(self) -> Dict[str, float]:
        """Get all metrics as a dict. Uses override values if set by pipeline."""
        # Use overrides from env_metrics if available, otherwise computed values
        trading = getattr(self, '_override_trading_score', None)
        user_sat = getattr(self, '_override_user_satisfaction', None)
        a_disa = getattr(self, '_override_a_disa_score', None)
        eco = getattr(self, '_override_eco_dkk', None)

        return {
            'experiment': self.experiment_name,
            'total_reward': round(self.total_reward(), 2),
            'a_disa_score': round(a_disa if a_disa is not None else self.aggregation_disaggregation_score(), 2),
            'trading_score': round(trading if trading is not None else self.trading_score(), 2),
            'eco_dkk': round(eco if eco is not None else self.economic_benefit_dkk(), 2),
            'user_satisfaction': round(user_sat if user_sat is not None else self.user_satisfaction(), 4),
            'constraint_violations': self.constraint_violations_total(),
            'training_variance': round(self.training_variance(), 4),
            'convergence_episode': self.convergence_episode(),
            'time_per_episode_s': round(self.mean_time_per_episode(), 2),
            'per_user_reward': round(self.per_user_reward(), 2),
            'per_user_eco_dkk': round(self.per_user_economic_dkk(), 2),
            'num_episodes': len(self.episode_rewards),
            'num_managers': self.num_managers,
            'num_users': self.num_users,
        }

    def export_episode_csv(self):
        """Export per-episode metrics to CSV."""
        filepath = os.path.join(self.results_dir, f"{self.experiment_name}.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'economic', 'user_satisfaction',
                'coordination', 'trade_success_rate', 'net_benefit',
                'constraint_violations', 'constraint_score', 'time_s'
            ])
            for i in range(len(self.episode_rewards)):
                writer.writerow([
                    i + 1,
                    round(self.episode_rewards[i], 4),
                    round(self.episode_economic[i], 4),
                    round(self.episode_satisfaction[i], 4),
                    round(self.episode_coordination[i], 4),
                    round(self.episode_trade_success_rates[i], 4),
                    round(self.episode_net_benefits[i], 4),
                    self.episode_constraint_violations[i],
                    round(self.episode_constraint_scores[i], 4),
                    round(self.episode_times[i], 2),
                ])
        logger.info(f"Episode metrics exported to {filepath}")
        return filepath

    def export_summary_csv(self, filepath: Optional[str] = None):
        """Export summary metrics to CSV (append mode for aggregating multiple experiments)."""
        if filepath is None:
            filepath = os.path.join(self.results_dir, "summary_tables", "summary.csv")

        summary = self.get_summary_dict()
        file_exists = os.path.exists(filepath)

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
        logger.info(f"Summary metrics appended to {filepath}")
        return filepath

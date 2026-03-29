#!/usr/bin/env python3
"""
FOgym Experiment Runner

Orchestrates all experimental configurations for the paper revision:
- EXP-1: Reward Weight Ablation (4 configs × SQDDPG+DMA)
- EXP-2: 10-Manager Full Benchmark (4 algorithms × 8 pipelines)

Usage:
    # Verify mode (quick test with 5 episodes)
    python run_experiments.py --verify

    # Run EXP-1 only
    python run_experiments.py --exp1

    # Run EXP-2 only
    python run_experiments.py --exp2

    # Run all experiments
    python run_experiments.py --all

    # Parallel execution (specify max workers)
    python run_experiments.py --all --parallel 4
"""
import subprocess
import argparse
import os
import sys
import time
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────── Configuration ────────────────

RESULTS_DIR = "results"
DEFAULT_EPISODES = 500
VERIFY_EPISODES = 5

# EXP-1: Reward Weight Ablation (SQDDPG + DMA pipeline)
EXP1_CONFIGS = {
    "exp1_ablation_A0_default": {
        "reward_alpha": 0.4, "reward_beta": 0.3,
        "reward_delta": 0.1, "reward_lambda": 0.2,
    },
    "exp1_ablation_A1_no_coord": {
        "reward_alpha": 0.4, "reward_beta": 0.3,
        "reward_delta": 0.1, "reward_lambda": 0.0,
    },
    "exp1_ablation_A2_no_constraint": {
        "reward_alpha": 0.4, "reward_beta": 0.3,
        "reward_delta": 0.0, "reward_lambda": 0.2,
    },
    "exp1_ablation_A3_no_user": {
        "reward_alpha": 0.4, "reward_beta": 0.0,
        "reward_delta": 0.1, "reward_lambda": 0.2,
    },
    "exp1_ablation_A4_no_eco": {
        "reward_alpha": 0.0, "reward_beta": 0.3,
        "reward_delta": 0.1, "reward_lambda": 0.2,
    },
}
EXP1_BASE = {
    "rl_algorithm": "fosqddpg",
    "aggregation_method": "DP",
    "trading_strategy": "market_clearing",
    "disaggregation_method": "average",
    "data_config": "4manager",
}

# EXP-2: 10-Manager Full Benchmark
EXP2_ALGORITHMS = ["fomappo", "fomaddpg", "fomatd3", "fosqddpg"]
EXP2_AGGREGATION = ["LP", "DP"]
EXP2_TRADING = ["market_clearing", "bidding"]
EXP2_DISAGGREGATION = ["average", "proportional"]

# Short codes for naming
AGG_CODE = {"LP": "LP", "DP": "DP"}
TRADE_CODE = {"market_clearing": "MC", "bidding": "BD"}
DISAGG_CODE = {"average": "A", "proportional": "P"}


def build_exp2_configs():
    """Generate all 32 EXP-2 configurations."""
    configs = {}
    for algo, agg, trade, disagg in product(
        EXP2_ALGORITHMS, EXP2_AGGREGATION, EXP2_TRADING, EXP2_DISAGGREGATION
    ):
        name = f"exp2_10m_{algo}_{AGG_CODE[agg]}_{TRADE_CODE[trade]}_{DISAGG_CODE[disagg]}"
        configs[name] = {
            "rl_algorithm": algo,
            "aggregation_method": agg,
            "trading_strategy": trade,
            "disaggregation_method": disagg,
            "data_config": "10manager",
        }
    return configs


# ──────────────── Execution ────────────────

def run_single_experiment(name: str, config: dict, num_episodes: int,
                          results_dir: str = RESULTS_DIR) -> dict:
    """Run a single experiment configuration via run_fo_pipeline.py."""
    cmd = [
        sys.executable, "run_fo_pipeline.py",
        "--num_episodes", str(num_episodes),
        "--results_dir", results_dir,
        "--log_verbosity", "brief",
        "--time_step", "1.0",
        "--use_gpu",
    ]

    for key, val in config.items():
        cmd.extend([f"--{key}", str(val)])

    # Pass experiment name for proper CSV naming
    cmd.extend(["--experiment_name", name])

    print(f"[START] {name} ({num_episodes} episodes)")
    start_time = time.time()

    result = {
        "name": name,
        "config": config,
        "num_episodes": num_episodes,
        "success": False,
        "elapsed_s": 0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600 * 4,  # 4h timeout
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        result["elapsed_s"] = round(time.time() - start_time, 1)
        result["success"] = proc.returncode == 0

        if proc.returncode != 0:
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
            print(f"[FAIL] {name} (exit={proc.returncode}, {result['elapsed_s']}s)")
        else:
            print(f"[DONE] {name} ({result['elapsed_s']}s)")

    except subprocess.TimeoutExpired:
        result["elapsed_s"] = round(time.time() - start_time, 1)
        result["error"] = "Timeout (4h)"
        print(f"[TIMEOUT] {name}")
    except Exception as e:
        result["elapsed_s"] = round(time.time() - start_time, 1)
        result["error"] = str(e)
        print(f"[ERROR] {name}: {e}")

    return result


def run_experiments(configs: dict, num_episodes: int, max_parallel: int = 1):
    """Run a set of experiment configurations."""
    results = []
    total = len(configs)
    print(f"\n{'='*60}")
    print(f"Running {total} experiments ({num_episodes} episodes each)")
    print(f"Parallel workers: {max_parallel}")
    print(f"{'='*60}\n")

    if max_parallel <= 1:
        for i, (name, config) in enumerate(configs.items(), 1):
            print(f"\n--- [{i}/{total}] ---")
            result = run_single_experiment(name, config, num_episodes)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(run_single_experiment, name, config, num_episodes): name
                for name, config in configs.items()
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    # Summary
    succeeded = sum(1 for r in results if r["success"])
    failed = total - succeeded
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"\n{'='*60}")
    print(f"Results: {succeeded}/{total} succeeded, {failed} failed")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")

    # Save run log
    log_path = os.path.join(RESULTS_DIR, "experiment_run_log.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Run log saved to {log_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FOgym Experiment Runner")
    parser.add_argument("--exp1", action="store_true", help="Run EXP-1 (ablation)")
    parser.add_argument("--exp2", action="store_true", help="Run EXP-2 (10-manager)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--verify", action="store_true",
                        help=f"Verify mode: run {VERIFY_EPISODES} episodes only")
    parser.add_argument("--episodes", type=int, default=None,
                        help=f"Override num_episodes (default: {DEFAULT_EPISODES})")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max parallel workers (default: 1)")
    args = parser.parse_args()

    num_episodes = args.episodes or (VERIFY_EPISODES if args.verify else DEFAULT_EPISODES)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "summary_tables"), exist_ok=True)

    configs = {}

    if args.exp1 or args.all:
        for name, exp_config in EXP1_CONFIGS.items():
            merged = {**EXP1_BASE, **exp_config}
            configs[name] = merged

    if args.exp2 or args.all:
        exp2_configs = build_exp2_configs()
        if args.verify:
            # In verify mode, only run 1 config per algorithm
            verify_configs = {}
            seen_algos = set()
            for name, config in exp2_configs.items():
                algo = config["rl_algorithm"]
                if algo not in seen_algos:
                    verify_configs[name] = config
                    seen_algos.add(algo)
            configs.update(verify_configs)
        else:
            configs.update(exp2_configs)

    if not configs:
        parser.print_help()
        print("\nError: specify --exp1, --exp2, --all, or --verify")
        sys.exit(1)

    run_experiments(configs, num_episodes, args.parallel)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import argparse
from itertools import product
from typing import List, Dict, Any, Tuple
import sys

# Deprecated
try:
    # try to import as package
    from .config import PipelineConfig, ModelBasedConfig
    from .model_based_pipeline import ModelBasedPipeline, run_pipeline
except (ImportError, SystemError):
    # import when running script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import PipelineConfig, ModelBasedConfig
    from model_based_pipeline import ModelBasedPipeline, run_pipeline

# set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelBasedComparison')


def run_algorithm_comparison(
    num_timesteps: int = 24,
    aggregation_methods: List[str] = None,
    trading_methods: List[str] = None,
    disaggregation_methods: List[str] = None,
    num_managers: int = 4,
    users_per_manager: List[int] = None,
    output_dir: str = "results/modelbased_comparison"
) -> Dict[str, Any]:
    """
    run algorithm comparison
    
    Args:
        num_timesteps: time steps
        aggregation_methods: aggregation methods to compare
        trading_methods: trading methods to compare
        disaggregation_methods: disaggregation methods to compare
        num_managers: number of managers
        users_per_manager: number of users per manager
        output_dir: output directory
        
    Returns:
        comparison results dictionary
    """
    # set default values
    if aggregation_methods is None:
        aggregation_methods = ["LP", "DP"]
    
    if trading_methods is None:
        trading_methods = ["bidding", "market-clearing"]
    
    if disaggregation_methods is None:
        disaggregation_methods = ["proportional", "average"]
        
    if users_per_manager is None:
        users_per_manager = [6, 10, 8, 12] 
    
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info(f"start algorithm comparison, experiment ID: {experiment_id}")
    logger.info(f"aggregation methods: {aggregation_methods}")
    logger.info(f"trading methods: {trading_methods}")
    logger.info(f"disaggregation methods: {disaggregation_methods}")
    logger.info(f"number of managers: {num_managers}")
    logger.info(f"user distribution: {users_per_manager} (total {sum(users_per_manager)} users)")
    logger.info(f"time steps: {num_timesteps}")
    
    # generate all algorithm combinations
    combinations = list(product(aggregation_methods, trading_methods, disaggregation_methods))
    logger.info(f"total {len(combinations)} algorithm combinations")
    
    # store results
    results = {
        "experiment_id": experiment_id,
        "num_timesteps": num_timesteps,
        "num_managers": num_managers,
        "users_per_manager": users_per_manager,
        "combinations": [],
        "rewards": [],
        "manager_rewards": [],
        "completion_times": []
    }
    
    # run each combination
    for i, (agg, trade, disagg) in enumerate(combinations):
        combo_name = f"{agg}_{trade}_{disagg}"
        logger.info(f"running combination {i+1}/{len(combinations)}: {combo_name}")
        
        # configure
        config = PipelineConfig(
            aggregation_method=agg,
            trading_method=trade,
            disaggregation_method=disagg,
            num_managers=num_managers,
            users_per_manager=users_per_manager,
            results_dir=os.path.join(experiment_dir, combo_name)
        )
        
        start_time = datetime.now()
        pipeline_results = run_pipeline(
            config_path=None, 
            num_timesteps=num_timesteps,
            aggregation_method=agg,
            trading_method=trade,
            disaggregation_method=disagg,
            save_results=True
        )
        end_time = datetime.now()
        completion_time = (end_time - start_time).total_seconds()
        
        # extract results
        total_reward = sum(pipeline_results['total_rewards'])
        manager_rewards = {
            manager_id: sum(rewards)
            for manager_id, rewards in pipeline_results['manager_rewards'].items()
        }
        
        # add to results
        results["combinations"].append(combo_name)
        results["rewards"].append(total_reward)
        results["manager_rewards"].append(manager_rewards)
        results["completion_times"].append(completion_time)
        
        logger.info(f"combination {combo_name} completed, total reward: {total_reward:.4f}, time: {completion_time:.2f} seconds")
    
    # save results
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"results saved to: {results_file}")
    
    # generate comparison charts
    generate_comparison_charts(results, experiment_dir)
    
    return results


def generate_comparison_charts(results: Dict[str, Any], output_dir: str):
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # prepare data
    combinations = results["combinations"]
    rewards = results["rewards"]
    completion_times = results["completion_times"]
    
    # chart 1: total reward comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combinations, rewards)
    plt.xlabel('algorithm combinations')
    plt.ylabel('total reward')
    plt.title('total reward comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # add value labels
    for bar, reward in zip(bars, rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{reward:.4f}',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    plt.savefig(os.path.join(output_dir, "rewards_comparison.png"))
    plt.close()
    
    # chart 2: completion time comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combinations, completion_times)
    plt.xlabel('algorithm combinations')
    plt.ylabel('completion time (seconds)')
    plt.title('completion time comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # add value labels
    for bar, time in zip(bars, completion_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{time:.2f}s',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()
    
    # chart 3: manager reward comparison
    plt.figure(figsize=(14, 8))
    
    # prepare manager reward data
    manager_ids = set()
    for manager_reward in results["manager_rewards"]:
        manager_ids.update(manager_reward.keys())
    manager_ids = sorted(manager_ids)
    
    # collect rewards for each manager
    manager_data = {manager_id: [] for manager_id in manager_ids}
    for manager_reward in results["manager_rewards"]:
        for manager_id in manager_ids:
            manager_data[manager_id].append(manager_reward.get(manager_id, 0.0))
    
    # plot rewards for each manager
    width = 0.8 / len(manager_ids)  # bar width
    x = np.arange(len(combinations))
    
    for i, manager_id in enumerate(manager_ids):
        plt.bar(
            x + i * width - 0.4 + width/2,
            manager_data[manager_id],
            width,
            label=f'Manager {manager_id}'
        )
    
    plt.xlabel('algorithm combinations')
    plt.ylabel('manager reward')
    plt.title('manager reward comparison')
    plt.xticks(x, combinations, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "manager_rewards_comparison.png"))
    plt.close()
    
    logger.info(f"comparison charts saved to: {output_dir}")


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="run ModelBased algorithm comparison")
    parser.add_argument("--timesteps", type=int, default=24, help="time steps")
    parser.add_argument("--output", type=str, default="results/modelbased_comparison", help="output directory")
    parser.add_argument("--aggregation", type=str, nargs='+', default=["LP", "DP"], 
                        help="aggregation algorithms to compare, can specify multiple")
    parser.add_argument("--trading", type=str, nargs='+', default=["bidding", "market-clearing"], 
                        help="trading algorithms to compare, can specify multiple")
    parser.add_argument("--disaggregation", type=str, nargs='+', default=["proportional", "average"], 
                        help="disaggregation algorithms to compare, can specify multiple")
    parser.add_argument("--managers", type=int, default=4, help="number of managers")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="number of users per manager, comma separated")
    
    args = parser.parse_args()
    
    # parse number of users
    try:
        users_per_manager = [int(n) for n in args.users.split(",")]
    except:
        users_per_manager = [6, 10, 8, 12]  # default value
    
    # print config
    print(f"algorithm comparison config:")
    print(f"- aggregation algorithms: {args.aggregation}")
    print(f"- trading algorithms: {args.trading}")
    print(f"- disaggregation algorithms: {args.disaggregation}")
    print(f"- number of managers: {args.managers}")
    print(f"- user distribution: {users_per_manager}")
    print(f"- time steps: {args.timesteps}")
    print(f"- output directory: {args.output}")
    
    # run comparison
    results = run_algorithm_comparison(
        num_timesteps=args.timesteps,
        aggregation_methods=args.aggregation,
        trading_methods=args.trading,
        disaggregation_methods=args.disaggregation,
        num_managers=args.managers,
        users_per_manager=users_per_manager,
        output_dir=args.output
    )
    
    # print results summary
    print("\nalgorithm comparison results summary:")
    best_idx = np.argmax(results["rewards"])
    best_combo = results["combinations"][best_idx]
    best_reward = results["rewards"][best_idx]
    
    print(f"best algorithm combination: {best_combo}")
    print(f"best total reward: {best_reward:.4f}")
    print(f"detailed results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    # run main function 
    main()

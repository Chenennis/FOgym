#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline run script

This script can be run directly to start the ModelBased FlexOffer Pipeline.
It solves the problem of relative imports and is suitable for standalone running.
"""

import os
import sys
import argparse

# ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# import required modules
from model_based_pipeline import run_pipeline

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="run ModelBased FlexOffer Pipeline")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--timesteps", type=int, default=24, help="time steps")
    parser.add_argument("--aggregation", type=str, default="LP", choices=["LP", "DP"], help="aggregation method")
    parser.add_argument("--trading", type=str, default="bidding", choices=["bidding", "market-clearing"], help="trading method")
    parser.add_argument("--disaggregation", type=str, default="proportional", choices=["proportional", "average"], help="disaggregation method")
    parser.add_argument("--managers", type=int, default=4, help="number of managers")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="number of users per manager, comma separated")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducibility")
    
    args = parser.parse_args()
    
    # print config information
    print(f"running ModelBased Pipeline:")
    print(f"- aggregation method: {args.aggregation}")
    print(f"- trading method: {args.trading}")
    print(f"- disaggregation method: {args.disaggregation}")
    print(f"- time steps: {args.timesteps}")
    if args.seed is not None:
        print(f"- random seed: {args.seed}")
    
    # run pipeline and pass parameters
    results = run_pipeline(
        config_path=args.config,
        num_timesteps=args.timesteps,
        aggregation_method=args.aggregation,
        trading_method=args.trading,
        disaggregation_method=args.disaggregation,
        seed=args.seed
    )
    
    # print results
    total_reward = sum(results.get('total_rewards', []))
    print(f"\nModelBased Pipeline completed!")
    print(f"total reward: {total_reward:.4f}") 
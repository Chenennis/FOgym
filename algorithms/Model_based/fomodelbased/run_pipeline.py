#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline run script
"""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_based_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ModelBased FlexOffer Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Configuration file path")
    parser.add_argument("--timesteps", type=int, default=24, help="Total timesteps")
    parser.add_argument("--aggregation", type=str, default="LP", choices=["LP", "DP"], help="Aggregation method")
    parser.add_argument("--trading", type=str, default="bidding", choices=["bidding", "market-clearing"], help="Trading method")
    parser.add_argument("--disaggregation", type=str, default="proportional", choices=["proportional", "average"], help="Disaggregation method")
    parser.add_argument("--managers", type=int, default=4, help="Manager number")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="Users per manager, comma separated")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Running ModelBased Pipeline:")
    print(f"- Aggregation method: {args.aggregation}")
    print(f"- Trading method: {args.trading}")
    print(f"- Disaggregation method: {args.disaggregation}")
    print(f"- Time steps: {args.timesteps}")
    if args.seed is not None:
        print(f"- Random seed: {args.seed}")
    
    results = run_pipeline(
        config_path=args.config,
        num_timesteps=args.timesteps,
        aggregation_method=args.aggregation,
        trading_method=args.trading,
        disaggregation_method=args.disaggregation,
        seed=args.seed
    )
    
    total_reward = sum(results.get('total_rewards', []))
    print(f"\nModelBased Pipeline finished!")
    print(f"Total reward: {total_reward:.4f}") 
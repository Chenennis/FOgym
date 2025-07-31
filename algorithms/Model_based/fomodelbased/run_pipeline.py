#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline运行脚本

这个脚本可以直接运行，用于启动ModelBased FlexOffer Pipeline。
它解决了相对导入的问题，适合单独运行。
"""

import os
import sys
import argparse

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入所需模块
from model_based_pipeline import run_pipeline

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行ModelBased FlexOffer Pipeline")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--timesteps", type=int, default=24, help="运行的时间步数")
    parser.add_argument("--aggregation", type=str, default="LP", choices=["LP", "DP"], help="聚合方法")
    parser.add_argument("--trading", type=str, default="bidding", choices=["bidding", "market-clearing"], help="交易方法")
    parser.add_argument("--disaggregation", type=str, default="proportional", choices=["proportional", "average"], help="分解方法")
    parser.add_argument("--managers", type=int, default=4, help="Manager数量")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="每个Manager的用户数，逗号分隔")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于保证实验可重复性")
    
    args = parser.parse_args()
    
    # 输出配置信息
    print(f"运行ModelBased Pipeline:")
    print(f"- 聚合方法: {args.aggregation}")
    print(f"- 交易方法: {args.trading}")
    print(f"- 分解方法: {args.disaggregation}")
    print(f"- 时间步数: {args.timesteps}")
    if args.seed is not None:
        print(f"- 随机种子: {args.seed}")
    
    # 运行pipeline并传递参数
    results = run_pipeline(
        config_path=args.config,
        num_timesteps=args.timesteps,
        aggregation_method=args.aggregation,
        trading_method=args.trading,
        disaggregation_method=args.disaggregation,
        seed=args.seed
    )
    
    # 输出结果
    total_reward = sum(results.get('total_rewards', []))
    print(f"\nModelBased Pipeline运行完成!")
    print(f"总奖励: {total_reward:.4f}") 
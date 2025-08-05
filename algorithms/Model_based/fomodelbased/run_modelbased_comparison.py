#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline比较工具

用于比较不同算法组合（聚合、交易、分解）的性能。
"""

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

# 处理导入方式
try:
    # 尝试作为包的一部分导入
    from .config import PipelineConfig, ModelBasedConfig
    from .model_based_pipeline import ModelBasedPipeline, run_pipeline
except (ImportError, SystemError):
    # 直接运行脚本时的导入方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import PipelineConfig, ModelBasedConfig
    from model_based_pipeline import ModelBasedPipeline, run_pipeline

# 设置日志
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
    运行算法比较
    
    Args:
        num_timesteps: 时间步数
        aggregation_methods: 要比较的聚合方法列表
        trading_methods: 要比较的交易方法列表
        disaggregation_methods: 要比较的分解方法列表
        num_managers: Manager数量
        users_per_manager: 每个Manager的用户数
        output_dir: 输出目录
        
    Returns:
        比较结果字典
    """
    # 设置默认值
    if aggregation_methods is None:
        aggregation_methods = ["LP", "DP"]
    
    if trading_methods is None:
        trading_methods = ["bidding", "market-clearing"]
    
    if disaggregation_methods is None:
        disaggregation_methods = ["proportional", "average"]
        
    if users_per_manager is None:
        users_per_manager = [6, 10, 8, 12]  # 总共36个用户
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建实验ID
    experiment_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info(f"开始算法比较，实验ID: {experiment_id}")
    logger.info(f"聚合方法: {aggregation_methods}")
    logger.info(f"交易方法: {trading_methods}")
    logger.info(f"分解方法: {disaggregation_methods}")
    logger.info(f"Manager数量: {num_managers}")
    logger.info(f"用户分布: {users_per_manager} (总计 {sum(users_per_manager)} 个用户)")
    logger.info(f"时间步数: {num_timesteps}")
    
    # 生成所有算法组合
    combinations = list(product(aggregation_methods, trading_methods, disaggregation_methods))
    logger.info(f"共 {len(combinations)} 种算法组合")
    
    # 存储结果
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
    
    # 运行每种组合
    for i, (agg, trade, disagg) in enumerate(combinations):
        combo_name = f"{agg}_{trade}_{disagg}"
        logger.info(f"运行组合 {i+1}/{len(combinations)}: {combo_name}")
        
        # 配置
        config = PipelineConfig(
            aggregation_method=agg,
            trading_method=trade,
            disaggregation_method=disagg,
            num_managers=num_managers,
            users_per_manager=users_per_manager,
            results_dir=os.path.join(experiment_dir, combo_name)
        )
        
        # 运行pipeline
        # 使用run_pipeline函数而非直接创建ModelBasedPipeline，确保算法设置正确传递
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
        
        # 提取结果
        total_reward = sum(pipeline_results['total_rewards'])
        manager_rewards = {
            manager_id: sum(rewards)
            for manager_id, rewards in pipeline_results['manager_rewards'].items()
        }
        
        # 添加到结果
        results["combinations"].append(combo_name)
        results["rewards"].append(total_reward)
        results["manager_rewards"].append(manager_rewards)
        results["completion_times"].append(completion_time)
        
        logger.info(f"组合 {combo_name} 完成，总奖励: {total_reward:.4f}，用时: {completion_time:.2f}秒")
    
    # 保存结果
    results_file = os.path.join(experiment_dir, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存到: {results_file}")
    
    # 生成比较图表
    generate_comparison_charts(results, experiment_dir)
    
    return results


def generate_comparison_charts(results: Dict[str, Any], output_dir: str):
    """
    生成比较图表
    
    Args:
        results: 比较结果
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    combinations = results["combinations"]
    rewards = results["rewards"]
    completion_times = results["completion_times"]
    
    # 图1：总奖励比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combinations, rewards)
    plt.xlabel('算法组合')
    plt.ylabel('总奖励')
    plt.title('不同算法组合的总奖励比较')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 添加数值标签
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
    
    # 图2：完成时间比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combinations, completion_times)
    plt.xlabel('算法组合')
    plt.ylabel('完成时间 (秒)')
    plt.title('不同算法组合的完成时间比较')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 添加数值标签
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
    
    # 图3：Manager奖励比较
    plt.figure(figsize=(14, 8))
    
    # 准备Manager奖励数据
    manager_ids = set()
    for manager_reward in results["manager_rewards"]:
        manager_ids.update(manager_reward.keys())
    manager_ids = sorted(manager_ids)
    
    # 为每个Manager收集奖励
    manager_data = {manager_id: [] for manager_id in manager_ids}
    for manager_reward in results["manager_rewards"]:
        for manager_id in manager_ids:
            manager_data[manager_id].append(manager_reward.get(manager_id, 0.0))
    
    # 绘制每个Manager的奖励
    width = 0.8 / len(manager_ids)  # 柱状图宽度
    x = np.arange(len(combinations))
    
    for i, manager_id in enumerate(manager_ids):
        plt.bar(
            x + i * width - 0.4 + width/2,
            manager_data[manager_id],
            width,
            label=f'Manager {manager_id}'
        )
    
    plt.xlabel('算法组合')
    plt.ylabel('Manager奖励')
    plt.title('不同算法组合下各Manager奖励比较')
    plt.xticks(x, combinations, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "manager_rewards_comparison.png"))
    plt.close()
    
    logger.info(f"比较图表已保存到: {output_dir}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行ModelBased算法比较")
    parser.add_argument("--timesteps", type=int, default=24, help="运行的时间步数")
    parser.add_argument("--output", type=str, default="results/modelbased_comparison", help="输出目录")
    parser.add_argument("--aggregation", type=str, nargs='+', default=["LP", "DP"], 
                        help="要比较的聚合算法，可指定多个")
    parser.add_argument("--trading", type=str, nargs='+', default=["bidding", "market-clearing"], 
                        help="要比较的交易算法，可指定多个")
    parser.add_argument("--disaggregation", type=str, nargs='+', default=["proportional", "average"], 
                        help="要比较的分解算法，可指定多个")
    parser.add_argument("--managers", type=int, default=4, help="Manager数量")
    parser.add_argument("--users", type=str, default="6,10,8,12", help="每个Manager的用户数，逗号分隔")
    
    args = parser.parse_args()
    
    # 解析用户数量
    try:
        users_per_manager = [int(n) for n in args.users.split(",")]
    except:
        users_per_manager = [6, 10, 8, 12]  # 默认值
    
    # 显示配置
    print(f"算法比较配置:")
    print(f"- 聚合算法: {args.aggregation}")
    print(f"- 交易算法: {args.trading}")
    print(f"- 分解算法: {args.disaggregation}")
    print(f"- Manager数量: {args.managers}")
    print(f"- 用户分布: {users_per_manager}")
    print(f"- 时间步数: {args.timesteps}")
    print(f"- 输出目录: {args.output}")
    
    # 运行比较
    results = run_algorithm_comparison(
        num_timesteps=args.timesteps,
        aggregation_methods=args.aggregation,
        trading_methods=args.trading,
        disaggregation_methods=args.disaggregation,
        num_managers=args.managers,
        users_per_manager=users_per_manager,
        output_dir=args.output
    )
    
    # 显示结果摘要
    print("\n算法比较结果摘要:")
    best_idx = np.argmax(results["rewards"])
    best_combo = results["combinations"][best_idx]
    best_reward = results["rewards"][best_idx]
    
    print(f"最佳算法组合: {best_combo}")
    print(f"最佳总奖励: {best_reward:.4f}")
    print(f"详细结果已保存到: {args.output}")
    
    return results


if __name__ == "__main__":
    # 运行主函数
    main()

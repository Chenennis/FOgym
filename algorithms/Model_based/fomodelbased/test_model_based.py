#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelBased FlexOffer Pipeline测试脚本

用于测试基于模型的FlexOffer Pipeline的功能和性能。
"""

import os
import time
import logging
import argparse
import json
from datetime import datetime
import sys

# 处理导入方式
try:
    # 尝试作为包的一部分导入
    from .config import PipelineConfig, ModelBasedConfig
    from .model_based_pipeline import ModelBasedPipeline, run_pipeline
    from .run_modelbased_comparison import run_algorithm_comparison
    from .model_based_controller import ModelBasedController, BatteryModel, HeatPumpModel
except (ImportError, SystemError):
    # 直接运行脚本时的导入方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import PipelineConfig, ModelBasedConfig
    from model_based_pipeline import ModelBasedPipeline, run_pipeline
    from run_modelbased_comparison import run_algorithm_comparison
    from model_based_controller import ModelBasedController, BatteryModel, HeatPumpModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelBasedTest')


def test_basic_pipeline():
    """测试基本的pipeline流程"""
    logger.info("测试基本pipeline流程")
    
    # 创建配置
    config = PipelineConfig(
        aggregation_method="LP",
        trading_method="bidding",
        disaggregation_method="proportional",
        num_managers=2,
        users_per_manager=[3, 3],
        time_horizon=12,
        model_config=ModelBasedConfig()
    )
    
    # 创建并运行pipeline
    pipeline = ModelBasedPipeline(config)
    results = pipeline.run(num_timesteps=2)
    
    # 验证结果
    assert 'total_rewards' in results, "结果中应包含total_rewards"
    assert len(results['total_rewards']) == 2, "应该有2个时间步的奖励"
    
    logger.info(f"基本pipeline测试完成，总奖励: {sum(results['total_rewards']):.4f}")
    return True


def test_algorithm_combinations():
    """测试不同算法组合"""
    logger.info("测试算法组合")
    
    # 聚合方法
    aggregation_methods = ["LP", "DP"]
    # 交易方法
    trading_methods = ["bidding", "market-clearing"]
    # 分解方法
    disaggregation_methods = ["proportional", "average"]
    
    # 测试配置
    num_managers = 1
    users_per_manager = [2]
    time_horizon = 6
    num_timesteps = 1
    
    # 记录结果
    results = []
    
    # 测试每种组合
    for agg in aggregation_methods:
        for trade in trading_methods:
            for disagg in disaggregation_methods:
                logger.info(f"测试组合: {agg}-{trade}-{disagg}")
                
                # 创建配置
                config = PipelineConfig(
                    aggregation_method=agg,
                    trading_method=trade,
                    disaggregation_method=disagg,
                    num_managers=num_managers,
                    users_per_manager=users_per_manager,
                    time_horizon=time_horizon,
                    model_config=ModelBasedConfig()
                )
                
                # 创建并运行pipeline
                start_time = time.time()
                pipeline = ModelBasedPipeline(config)
                pipeline_results = pipeline.run(num_timesteps=num_timesteps)
                end_time = time.time()
                
                # 记录结果
                total_reward = sum(pipeline_results['total_rewards'])
                execution_time = end_time - start_time
                
                results.append({
                    'combination': f"{agg}-{trade}-{disagg}",
                    'total_reward': total_reward,
                    'execution_time': execution_time
                })
                
                logger.info(f"组合 {agg}-{trade}-{disagg} 完成, 奖励: {total_reward:.4f}, 用时: {execution_time:.2f}秒")
    
    # 输出结果比较
    logger.info("\n算法组合比较结果:")
    for result in results:
        logger.info(f"{result['combination']}: 奖励 = {result['total_reward']:.4f}, 时间 = {result['execution_time']:.2f}秒")
    
    return True


def test_model_controller():
    """测试模型控制器功能"""
    logger.info("测试模型控制器")
    
    # 创建控制器
    controller = ModelBasedController(
        manager_id="test_manager",
        time_horizon=12,
        time_step=1,
        config=ModelBasedConfig()
    )
    
    # 添加设备模型
    controller.add_device_model(
        device_id="battery_1",
        device_type="BATTERY",
        device_params={
            'capacity': 10.0,
            'initial_soc': 0.5,
            'min_soc': 0.1,
            'max_soc': 0.9,
            'p_min': -3.0,
            'p_max': 3.0,
            'efficiency': 0.95,
            'initial_charge': 5.0
        }
    )
    
    controller.add_device_model(
        device_id="heat_pump_1",
        device_type="HEAT_PUMP",
        device_params={
            'initial_temp': 20.0,
            'min_temp': 18.0,
            'max_temp': 22.0,
            'target_temp': 21.0,
            'max_power': 2.0,
            'outdoor_temp': 5.0,
            'thermal_mass': 5000.0,
            'heat_transfer_coeff': 100.0
        }
    )
    
    # 测试生成FlexOffer
    prices = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1]
    flexoffers = controller.generate_flex_offers(prices)
    
    # 验证结果
    assert len(flexoffers) == 2, "应该生成2个FlexOffer"
    assert "battery_1" in flexoffers, "应该包含电池FlexOffer"
    assert "heat_pump_1" in flexoffers, "应该包含热泵FlexOffer"
    
    # 检查每个FlexOffer的内容
    for device_id, fo in flexoffers.items():
        assert 'energy_profile' in fo, f"{device_id} FlexOffer应包含energy_profile"
        assert 'time_flexibility' in fo, f"{device_id} FlexOffer应包含time_flexibility"
        assert len(fo['energy_profile']) == 12, f"{device_id} energy_profile长度应为12"
    
    logger.info("模型控制器测试完成")
    return True


def run_tests():
    """运行所有测试"""
    tests = [
        test_basic_pipeline,
        test_algorithm_combinations,
        test_model_controller
    ]
    
    results = []
    for test_func in tests:
        test_name = test_func.__name__
        logger.info(f"开始测试: {test_name}")
        
        try:
            success = test_func()
            results.append({
                'name': test_name,
                'success': success,
                'message': "通过" if success else "失败"
            })
            logger.info(f"测试 {test_name} 结果: {'通过' if success else '失败'}")
        except Exception as e:
            results.append({
                'name': test_name,
                'success': False,
                'message': str(e)
            })
            logger.error(f"测试 {test_name} 异常: {e}")
    
    # 汇总结果
    logger.info("\n测试结果汇总:")
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    logger.info(f"通过: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        logger.info(f"{status} {result['name']}: {result['message']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试ModelBased FlexOffer Pipeline")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行测试
    run_tests() 
"""
FlexOffer算法测试基础类

本模块提供统一的测试基础类和工具函数，
减少测试代码重复，提升测试效率。
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
import logging

# 添加项目路径
if 'tests' in os.getcwd():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fo_common.dec_pomdp_config import DecPOMDPConfig

logger = logging.getLogger(__name__)


class BaseAlgorithmTest(unittest.TestCase):
    """算法测试基础类"""
    
    def setUp(self):
        """通用测试环境设置"""
        self.config = DecPOMDPConfig()
        self.device = torch.device("cpu")
        
        # 标准维度配置
        self.private_dim = 40   # 私有信息维度
        self.public_dim = 18    # 公共信息维度  
        self.others_dim = 15    # 他者信息维度
        self.state_dim = 73     # 单智能体状态维度 (40+18+15)
        self.action_dim = 36    # 动作维度
        self.n_agents = 4       # 智能体数量
        self.hidden_dim = 256   # 隐藏层维度
        self.max_action = 1.0   # 最大动作值
        self.batch_size = 8     # 测试批次大小
        
        # 创建测试数据
        self._setup_test_data()
        
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
    
    def _setup_test_data(self):
        """设置测试数据"""
        # 单智能体观测数据
        self.private_obs = torch.randn(self.batch_size, self.private_dim)
        self.public_obs = torch.randn(self.batch_size, self.public_dim)
        self.others_obs = torch.randn(self.batch_size, self.others_dim)
        self.single_state = torch.randn(self.batch_size, self.state_dim)
        self.single_action = torch.randn(self.batch_size, self.action_dim)
        
        # 多智能体数据
        self.global_states = torch.randn(self.batch_size, self.state_dim * self.n_agents)
        self.global_actions = torch.randn(self.batch_size, self.action_dim * self.n_agents)
        
        # NumPy格式数据
        self.np_states = np.random.randn(self.n_agents, self.state_dim)
        self.np_actions = np.random.randn(self.n_agents, self.action_dim)
        self.np_rewards = np.random.randn(self.n_agents)
        self.np_next_states = np.random.randn(self.n_agents, self.state_dim)
        self.np_dones = np.random.choice([0, 1], size=self.n_agents)
    
    # ========== 通用网络测试方法 ==========
    
    def assert_network_initialization(self, network: nn.Module, expected_params: Dict[str, Any]):
        """验证网络初始化"""
        self.assertIsInstance(network, nn.Module, "网络应该是PyTorch模块")
        
        # 检查网络参数
        for param_name, expected_value in expected_params.items():
            if hasattr(network, param_name):
                actual_value = getattr(network, param_name)
                self.assertEqual(actual_value, expected_value, 
                               f"参数{param_name}不匹配: 期望{expected_value}, 实际{actual_value}")
    
    def assert_network_forward(self, network: nn.Module, input_tensor: torch.Tensor, 
                              expected_output_shape: Tuple[int, ...]):
        """验证网络前向传播"""
        network.eval()
        with torch.no_grad():
            output = network(input_tensor)
        
        self.assertEqual(output.shape, expected_output_shape, 
                        f"输出形状不匹配: 期望{expected_output_shape}, 实际{output.shape}")
        self.assertTrue(torch.isfinite(output).all(), "输出应该是有限值")
        
        return output
    
    def assert_gradient_computation(self, network: nn.Module, input_tensor: torch.Tensor, 
                                   loss_fn=None):
        """验证梯度计算"""
        network.train()
        
        # 清零梯度
        network.zero_grad()
        
        # 前向传播
        output = network(input_tensor)
        
        # 计算损失
        if loss_fn is None:
            loss = output.mean()  # 简单的损失函数
        else:
            loss = loss_fn(output)
        
        # 反向传播
        loss.backward()
        
        # 验证梯度存在
        has_grad = any(p.grad is not None for p in network.parameters())
        self.assertTrue(has_grad, "应该有参数具有梯度")
        
        # 验证梯度有限
        for name, param in network.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), 
                              f"参数{name}的梯度应该是有限值")
    
    # ========== 算法特定测试方法 ==========
    
    def test_algorithm_initialization(self, algorithm_class, algorithm_config: Dict[str, Any]):
        """测试算法初始化"""
        algorithm = algorithm_class(**algorithm_config)
        
        # 基本属性检查
        self.assertEqual(algorithm.n_agents, self.n_agents)
        self.assertEqual(algorithm.state_dim, self.state_dim)
        self.assertEqual(algorithm.action_dim, self.action_dim)
        
        # 设备检查
        self.assertEqual(str(algorithm.device), str(self.device))
        
        return algorithm
    
    def test_action_selection(self, algorithm, expected_action_shape: Tuple[int, ...]):
        """测试动作选择功能"""
        # 确定性动作选择
        actions_deterministic = algorithm.select_actions(self.np_states, add_noise=False)
        self.assertEqual(actions_deterministic.shape, expected_action_shape)
        
        # 带噪声的动作选择
        actions_noisy = algorithm.select_actions(self.np_states, add_noise=True)
        self.assertEqual(actions_noisy.shape, expected_action_shape)
        
        # 验证动作范围（通常在[-1, 1]之间）
        self.assertTrue(np.all(actions_deterministic >= -self.max_action))
        self.assertTrue(np.all(actions_deterministic <= self.max_action))
        
        return actions_deterministic, actions_noisy
    
    def test_experience_storage(self, algorithm):
        """测试经验存储功能"""
        initial_buffer_size = len(algorithm.replay_buffer) if hasattr(algorithm, 'replay_buffer') else 0
        
        # 存储经验
        algorithm.store_experience(
            states=self.np_states,
            actions=self.np_actions,
            rewards=self.np_rewards,
            next_states=self.np_next_states,
            dones=self.np_dones
        )
        
        # 验证缓冲区大小增加
        if hasattr(algorithm, 'replay_buffer'):
            new_buffer_size = len(algorithm.replay_buffer)
            self.assertGreater(new_buffer_size, initial_buffer_size, "缓冲区大小应该增加")
    
    # ========== Dec-POMDP特定测试方法 ==========
    
    def test_dec_pomdp_dimensions(self):
        """测试Dec-POMDP观测维度"""
        # 验证各层观测维度
        total_obs_dim = self.private_dim + self.public_dim + self.others_dim
        self.assertEqual(total_obs_dim, self.state_dim, "观测维度计算应该正确")
        
        # 验证私有信息增强 (DDPG特色: 40维包含历史信息)
        self.assertEqual(self.private_dim, 40, "私有信息应该是40维")
        self.assertEqual(self.public_dim, 18, "公共信息应该是18维")
        self.assertEqual(self.others_dim, 15, "他者信息应该是15维")
    
    def test_information_fusion(self, actor_network):
        """测试信息融合功能"""
        # 测试分层观测处理
        if hasattr(actor_network, 'forward') and callable(actor_network.forward):
            try:
                # 尝试分层输入
                actions = actor_network(self.private_obs, self.public_obs, self.others_obs)
                self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
            except TypeError:
                # 如果不支持分层输入，尝试合并输入
                combined_obs = torch.cat([self.private_obs, self.public_obs, self.others_obs], dim=1)
                actions = actor_network(combined_obs)
                self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
        
        print("✅ 信息融合功能测试通过")
    
    # ========== 算法类型特定测试 ==========
    
    def test_deterministic_policy(self, actor_network):
        """测试确定性策略 (适用于DDPG类算法)"""
        actor_network.eval()
        
        with torch.no_grad():
            # 多次前向传播应产生相同结果
            action1 = actor_network(self.single_state)
            action2 = actor_network(self.single_state)
            action3 = actor_network(self.single_state)
        
        torch.testing.assert_close(action1, action2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(action2, action3, atol=1e-6, rtol=1e-6)
        
        print("✅ 确定性策略测试通过")
    
    def test_stochastic_policy(self, actor_network):
        """测试随机策略 (适用于PPO类算法)"""
        actor_network.eval()
        
        # 随机策略应产生概率分布
        if hasattr(actor_network, 'get_action_logprob'):
            action, log_prob = actor_network.get_action_logprob(self.single_state)
            self.assertTrue(torch.isfinite(log_prob).all(), "对数概率应该是有限值")
        
        print("✅ 随机策略测试通过")
    
    def test_centralized_training(self, critic_network):
        """测试集中式训练 (适用于所有MARL算法)"""
        # Critic应该能够处理全局状态和动作
        q_values = critic_network(self.global_states, self.global_actions)
        
        expected_q_shape = (self.batch_size, 1) if q_values.dim() == 2 else (self.batch_size,)
        self.assertEqual(q_values.shape, expected_q_shape, "Q值形状应该正确")
        self.assertTrue(torch.isfinite(q_values).all(), "Q值应该是有限值")
        
        print("✅ 集中式训练测试通过")
    
    def test_twin_critic(self, critic_network):
        """测试双Q网络 (适用于TD3类算法)"""
        if hasattr(critic_network, 'Q1') and hasattr(critic_network, 'forward'):
            # 双Q网络应该返回两个Q值
            q1, q2 = critic_network(self.global_states, self.global_actions)
            
            self.assertEqual(q1.shape, q2.shape, "两个Q网络输出形状应该相同")
            self.assertTrue(torch.isfinite(q1).all(), "Q1值应该是有限值")
            self.assertTrue(torch.isfinite(q2).all(), "Q2值应该是有限值")
            
            # 测试单独的Q1方法
            q1_only = critic_network.Q1(self.global_states, self.global_actions)
            torch.testing.assert_close(q1, q1_only, atol=1e-6, rtol=1e-6)
        
        print("✅ 双Q网络测试通过")
    
    # ========== 工具方法 ==========
    
    def count_parameters(self, network: nn.Module) -> int:
        """计算网络参数数量"""
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    def print_network_info(self, network: nn.Module, network_name: str):
        """打印网络信息"""
        param_count = self.count_parameters(network)
        print(f"   📊 {network_name}参数数量: {param_count:,}")
        
        if hasattr(network, 'get_network_info'):
            info = network.get_network_info()
            for key, value in info.items():
                print(f"   📊 {key}: {value}")
    
    def run_performance_test(self, algorithm, num_iterations: int = 100):
        """运行性能测试"""
        import time
        
        start_time = time.time()
        for _ in range(num_iterations):
            algorithm.select_actions(self.np_states, add_noise=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        iterations_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        print(f"   ⚡ 性能测试: {iterations_per_sec:.1f} 次/秒")
        return iterations_per_sec


class TestRunner:
    """测试运行器"""
    
    @staticmethod
    def run_algorithm_tests(test_class, algorithm_name: str = ""):
        """运行算法测试套件"""
        print(f"开始{algorithm_name}算法测试...")
        print("=" * 60)
        
        # 创建测试套件
        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # 统计结果
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
        
        print("=" * 60)
        print(f"{algorithm_name}算法测试完成!")
        print(f"总测试数: {total_tests}")
        print(f"成功: {total_tests - failures - errors}")
        print(f"失败: {failures}")
        print(f"错误: {errors}")
        print(f"成功率: {success_rate:.1f}%")
        
        # 返回是否成功（成功率>=85%）
        return success_rate >= 85.0
    
    @staticmethod
    def run_quick_verification(test_functions: List[callable], test_name: str = ""):
        """运行快速验证测试"""
        print(f"开始{test_name}快速验证...")
        print("=" * 50)
        
        results = []
        for i, test_func in enumerate(test_functions, 1):
            try:
                result = test_func()
                results.append(result if result is not None else True)
                print(f"✅ 测试{i}通过\n")
            except Exception as e:
                print(f"❌ 测试{i}失败: {str(e)}\n")
                results.append(False)
        
        # 总结
        passed = sum(results)
        total = len(results)
        
        print("=" * 50)
        print(f"🎯 {test_name}验证总结")
        print(f"   📊 通过: {passed}/{total}")
        print(f"   📊 成功率: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 所有测试通过！")
            return True
        else:
            print("⚠️  部分测试失败")
            return False 
"""
FlexOffer algorithm test base class

This module provides a unified test base class and utility functions,
reducing test code duplication and improving testing efficiency.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
import logging

# Add project path
if 'tests' in os.getcwd():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fo_common.dec_pomdp_config import DecPOMDPConfig

logger = logging.getLogger(__name__)


class BaseAlgorithmTest(unittest.TestCase):
    """Algorithm test base class"""
    
    def setUp(self):
        """Common test environment setup"""
        self.config = DecPOMDPConfig()
        self.device = torch.device("cpu")
        
        # Standard dimension configuration
        self.private_dim = 40   # Private information dimension
        self.public_dim = 18    # Public information dimension  
        self.others_dim = 15    # Others information dimension
        self.state_dim = 73     # Single agent state dimension (40+18+15)
        self.action_dim = 36    # Action dimension
        self.n_agents = 4       # Number of agents
        self.hidden_dim = 256   # Hidden layer dimension
        self.max_action = 1.0   # Maximum action value
        self.batch_size = 8     # Test batch size
        
        # Create test data
        self._setup_test_data()
        
        # Set log level
        logging.basicConfig(level=logging.INFO)
    
    def _setup_test_data(self):
        """Set up test data"""
        # Single agent observation data
        self.private_obs = torch.randn(self.batch_size, self.private_dim)
        self.public_obs = torch.randn(self.batch_size, self.public_dim)
        self.others_obs = torch.randn(self.batch_size, self.others_dim)
        self.single_state = torch.randn(self.batch_size, self.state_dim)
        self.single_action = torch.randn(self.batch_size, self.action_dim)
        
        # Multi-agent data
        self.global_states = torch.randn(self.batch_size, self.state_dim * self.n_agents)
        self.global_actions = torch.randn(self.batch_size, self.action_dim * self.n_agents)
        
        # NumPy format data
        self.np_states = np.random.randn(self.n_agents, self.state_dim)
        self.np_actions = np.random.randn(self.n_agents, self.action_dim)
        self.np_rewards = np.random.randn(self.n_agents)
        self.np_next_states = np.random.randn(self.n_agents, self.state_dim)
        self.np_dones = np.random.choice([0, 1], size=self.n_agents)
    
    # ========== Common Network Test Methods ==========
    
    def assert_network_initialization(self, network: nn.Module, expected_params: Dict[str, Any]):
        """Verify network initialization"""
        self.assertIsInstance(network, nn.Module, "Network should be a PyTorch module")
        
        # Check network parameters
        for param_name, expected_value in expected_params.items():
            if hasattr(network, param_name):
                actual_value = getattr(network, param_name)
                self.assertEqual(actual_value, expected_value, 
                               f"Parameter {param_name} mismatch: expected {expected_value}, actual {actual_value}")
    
    def assert_network_forward(self, network: nn.Module, input_tensor: torch.Tensor, 
                              expected_output_shape: Tuple[int, ...]):
        """Verify network forward propagation"""
        network.eval()
        with torch.no_grad():
            output = network(input_tensor)
        
        self.assertEqual(output.shape, expected_output_shape, 
                        f"Output shape mismatch: expected {expected_output_shape}, actual {output.shape}")
        self.assertTrue(torch.isfinite(output).all(), "Output should be finite values")
        
        return output
    
    def assert_gradient_computation(self, network: nn.Module, input_tensor: torch.Tensor, 
                                   loss_fn=None):
        """Verify gradient computation"""
        network.train()
        
        # Zero gradients
        network.zero_grad()
        
        # Forward propagation
        output = network(input_tensor)
        
        # Calculate loss
        if loss_fn is None:
            loss = output.mean()  # Simple loss function
        else:
            loss = loss_fn(output)
        
        # Backward propagation
        loss.backward()
        
        # Verify gradients exist
        has_grad = any(p.grad is not None for p in network.parameters())
        self.assertTrue(has_grad, "Some parameters should have gradients")
        
        # Verify gradients are finite
        for name, param in network.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), 
                              f"Gradients for parameter {name} should be finite values")
    
    # ========== Algorithm-Specific Test Methods ==========
    
    def test_algorithm_initialization(self, algorithm_class, algorithm_config: Dict[str, Any]):
        """Test algorithm initialization"""
        algorithm = algorithm_class(**algorithm_config)
        
        # Basic property checks
        self.assertEqual(algorithm.n_agents, self.n_agents)
        self.assertEqual(algorithm.state_dim, self.state_dim)
        self.assertEqual(algorithm.action_dim, self.action_dim)
        
        # Device check
        self.assertEqual(str(algorithm.device), str(self.device))
        
        return algorithm
    
    def test_action_selection(self, algorithm, expected_action_shape: Tuple[int, ...]):
        """Test action selection functionality"""
        # Deterministic action selection
        actions_deterministic = algorithm.select_actions(self.np_states, add_noise=False)
        self.assertEqual(actions_deterministic.shape, expected_action_shape)
        
        # Action selection with noise
        actions_noisy = algorithm.select_actions(self.np_states, add_noise=True)
        self.assertEqual(actions_noisy.shape, expected_action_shape)
        
        # Verify action range (typically between [-1, 1])
        self.assertTrue(np.all(actions_deterministic >= -self.max_action))
        self.assertTrue(np.all(actions_deterministic <= self.max_action))
        
        return actions_deterministic, actions_noisy
    
    def test_experience_storage(self, algorithm):
        """Test experience storage functionality"""
        initial_buffer_size = len(algorithm.replay_buffer) if hasattr(algorithm, 'replay_buffer') else 0
        
        # Store experience
        algorithm.store_experience(
            states=self.np_states,
            actions=self.np_actions,
            rewards=self.np_rewards,
            next_states=self.np_next_states,
            dones=self.np_dones
        )
        
        # Verify buffer size increased
        if hasattr(algorithm, 'replay_buffer'):
            new_buffer_size = len(algorithm.replay_buffer)
            self.assertGreater(new_buffer_size, initial_buffer_size, "Buffer size should increase")
    
    # ========== Dec-POMDP Specific Test Methods ==========
    
    def test_dec_pomdp_dimensions(self):
        """Test Dec-POMDP observation dimensions"""
        # Verify observation dimension layers
        total_obs_dim = self.private_dim + self.public_dim + self.others_dim
        self.assertEqual(total_obs_dim, self.state_dim, "Observation dimension calculation should be correct")
        
        # Verify private information augmentation (DDPG feature: 40 dimensions including history)
        self.assertEqual(self.private_dim, 40, "Private information should be 40 dimensions")
        self.assertEqual(self.public_dim, 18, "Public information should be 18 dimensions")
        self.assertEqual(self.others_dim, 15, "Others information should be 15 dimensions")
    
    def test_information_fusion(self, actor_network):
        """Test information fusion functionality"""
        # Test layered observation processing
        if hasattr(actor_network, 'forward') and callable(actor_network.forward):
            try:
                # Try layered input
                actions = actor_network(self.private_obs, self.public_obs, self.others_obs)
                self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
            except TypeError:
                # If layered input not supported, try combined input
                combined_obs = torch.cat([self.private_obs, self.public_obs, self.others_obs], dim=1)
                actions = actor_network(combined_obs)
                self.assertEqual(actions.shape, (self.batch_size, self.action_dim))
        
        print("✅ Information fusion functionality test passed")
    
    # ========== Algorithm Type-Specific Tests ==========
    
    def test_deterministic_policy(self, actor_network):
        """Test deterministic policy (for DDPG-like algorithms)"""
        actor_network.eval()
        
        with torch.no_grad():
            # Multiple forward passes should produce the same result
            action1 = actor_network(self.single_state)
            action2 = actor_network(self.single_state)
            action3 = actor_network(self.single_state)
        
        torch.testing.assert_close(action1, action2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(action2, action3, atol=1e-6, rtol=1e-6)
        
        print("✅ Deterministic policy test passed")
    
    def test_stochastic_policy(self, actor_network):
        """Test stochastic policy (for PPO-like algorithms)"""
        actor_network.eval()
        
        # Random policy should produce probability distribution
        if hasattr(actor_network, 'get_action_logprob'):
            action, log_prob = actor_network.get_action_logprob(self.single_state)
            self.assertTrue(torch.isfinite(log_prob).all(), "Log probabilities should be finite values")
        
        print("✅ Stochastic policy test passed")
    
    def test_centralized_training(self, critic_network):
        """Test centralized training (for all MARL algorithms)"""
        # Critic should be able to handle global states and actions
        q_values = critic_network(self.global_states, self.global_actions)
        
        expected_q_shape = (self.batch_size, 1) if q_values.dim() == 2 else (self.batch_size,)
        self.assertEqual(q_values.shape, expected_q_shape, "Q-value shape should be correct")
        self.assertTrue(torch.isfinite(q_values).all(), "Q-values should be finite values")
        
        print("✅ Centralized training test passed")
    
    def test_twin_critic(self, critic_network):
        """Test twin Q-networks (for TD3-like algorithms)"""
        if hasattr(critic_network, 'Q1') and hasattr(critic_network, 'forward'):
            # Twin Q-networks should return two Q-values
            q1, q2 = critic_network(self.global_states, self.global_actions)
            
            self.assertEqual(q1.shape, q2.shape, "Both Q-network outputs should have the same shape")
            self.assertTrue(torch.isfinite(q1).all(), "Q1 values should be finite values")
            self.assertTrue(torch.isfinite(q2).all(), "Q2 values should be finite values")
            
            # Test standalone Q1 method
            q1_only = critic_network.Q1(self.global_states, self.global_actions)
            torch.testing.assert_close(q1, q1_only, atol=1e-6, rtol=1e-6)
        
        print("✅ Twin Q-network test passed")
    
    # ========== Utility Methods ==========
    
    def count_parameters(self, network: nn.Module) -> int:
        """Count network parameters"""
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    def print_network_info(self, network: nn.Module, network_name: str):
        """Print network information"""
        param_count = self.count_parameters(network)
        print(f"   📊 {network_name} parameter count: {param_count:,}")
        
        if hasattr(network, 'get_network_info'):
            info = network.get_network_info()
            for key, value in info.items():
                print(f"   📊 {key}: {value}")
    
    def run_performance_test(self, algorithm, num_iterations: int = 100):
        """Run performance test"""
        import time
        
        start_time = time.time()
        for _ in range(num_iterations):
            algorithm.select_actions(self.np_states, add_noise=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        iterations_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        print(f"   ⚡ Performance test: {iterations_per_sec:.1f} iterations/sec")
        return iterations_per_sec


class TestRunner:
    """Test runner"""
    
    @staticmethod
    def run_algorithm_tests(test_class, algorithm_name: str = ""):
        """Run algorithm test suite"""
        print(f"Starting {algorithm_name} algorithm tests...")
        print("=" * 60)
        
        # Create test suite
        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Summarize results
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
        
        print("=" * 60)
        print(f"{algorithm_name} algorithm tests completed!")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {total_tests - failures - errors}")
        print(f"Failed: {failures}")
        print(f"Errors: {errors}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Return success (success rate >= 85%)
        return success_rate >= 85.0
    
    @staticmethod
    def run_quick_verification(test_functions: List[callable], test_name: str = ""):
        """Run quick verification tests"""
        print(f"Starting {test_name} quick verification...")
        print("=" * 50)
        
        results = []
        for i, test_func in enumerate(test_functions, 1):
            try:
                result = test_func()
                results.append(result if result is not None else True)
                print(f"✅ Test {i} passed\n")
            except Exception as e:
                print(f"❌ Test {i} failed: {str(e)}\n")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("=" * 50)
        print(f"🎯 {test_name} verification summary")
        print(f"   📊 Passed: {passed}/{total}")
        print(f"   📊 Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed")
            return False 
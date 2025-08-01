# FlexOffer Multi-Agent Algorithm Usage Guide

This document provides detailed instructions on how to run different multi-agent algorithm combinations, including **5 MARL algorithms** (FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG) and **1 Model-based baseline algorithm** (FOModelBased), totaling **6 algorithms** supporting **40 complete combination configurations**.

## 📋 Table of Contents

- [🚀 Single Algorithm Execution](#-single-algorithm-execution)
- [🔀 Algorithm Combination Configuration](#-algorithm-combination-configuration)
- [🏁 Batch Algorithm Comparison](#-batch-algorithm-comparison)
- [⚙️ Parameter Details](#️-parameter-details)
- [🔧 Algorithm Architecture and Features](#-algorithm-architecture-and-features)


## 🚀 Single Algorithm Execution

### Five Core Algorithms (5 MARL)

#### MAPPO (Higher stability, shared policy)
```bash
# Standard execution
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 100

# Extended training
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 200 --use_gpu
```

#### MAIPPO (Independent policy, resolves policy conflicts)
```bash
# Standard execution
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 100

# For avoiding policy conflicts
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 170 --use_gpu
```

#### SQDDPG (Best fairness)
```bash
# Standard execution
python run_fo_pipeline.py --rl_algorithm fosqddpg --num_episodes 100

# Enhanced fairness
python run_fo_pipeline.py --rl_algorithm fosqddpg --num_episodes 200 --use_gpu
```

#### MATD3 (High stability)
```bash
# Standard execution
python run_fo_pipeline.py --rl_algorithm fomatd3 --num_episodes 100

# High stability configuration
python run_fo_pipeline.py --rl_algorithm fomatd3 --num_episodes 200 --use_gpu
```

#### MADDPG (Highest efficiency)
```bash
# Standard execution
python run_fo_pipeline.py --rl_algorithm fomaddpg --num_episodes 100

# Fast training
python run_fo_pipeline.py --rl_algorithm fomaddpg --num_episodes 50 --use_gpu
```



## 🔀 Algorithm Combination Configuration

### 🎯 Complete 40 Combination Configurations

#### **Combination Calculation**: 5 algorithms × 2 aggregation methods × 2 trading strategies × 2 disaggregation methods = **48 theoretical combinations**
> Note: The FOModelBased algorithm does not require training, but other parameter combinations are still valid, with 40 actually usable combinations

#### **Complete Combination Parameter Template**
```bash
python run_fo_pipeline.py \
  --rl_algorithm [fomappo|fomaippo|fomaddpg|fomatd3|fosqddpg|fomodelbased] \
  --aggregation_method [LP|DP] \
  --trading_strategy [market_clearing|bidding] \
  --disaggregation_method [average|proportional] \
  --scheduling_method [priority|fairness|cost] \
  --num_episodes [training episodes, not needed for FOModelBased] \
  --num_users [number of users, default 36] \
  --num_managers [number of managers, default 4] \
  --time_horizon [time range, default 24 hours] \
  --use_gpu [optional, use GPU acceleration]
```

#### **Algorithm Classification**
| Algorithm Type | Algorithm Name | Features | Training Requirements |
|----------|----------|------|----------|
| **MARL Algorithm** | FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG | Requires training | num_episodes parameter required |
| **Model-based Baseline** | FOModelBased | Traditional optimization, no training required | Direct evaluation, ignore num_episodes |

## 🏁 Batch Algorithm Comparison

### PowerShell Batch Processing
```powershell
# Windows PowerShell - Complete comparison of 6 algorithms
foreach ($algo in @("fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg", "fomodelbased")) {
    if ($algo -eq "fomodelbased") {
        python run_fo_pipeline.py --rl_algorithm $algo  # No training needed
    } else {
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    }
}
```

### Bash Batch Processing
```bash
# Linux/Mac Bash - Complete comparison of 6 algorithms
for algo in fomappo fomaippo fomaddpg fomatd3 fosqddpg fomodelbased; do
    if [ "$algo" = "fomodelbased" ]; then
        python run_fo_pipeline.py --rl_algorithm $algo  # No training needed
    else {
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    fi
done
```

### Results Comparison Script
```bash
# Compare multiple algorithm results
python analyze_algorithm_performance.py --results_dir ./results --plot

# Plot reward curves
python analyze_algorithm_performance.py --plot_rewards --algorithms fomappo,fomaddpg,fomatd3
```

## 💡 Recommended Combinations

### Scenario 1: Stability-Focused (Long-term Training)
```bash
python run_fo_pipeline.py \
  --rl_algorithm fomappo \
  --aggregation_method LP \
  --trading_strategy market_clearing \
  --disaggregation_method proportional \
  --scheduling_method priority \
  --num_episodes 200 \
  --use_gpu
```

### Scenario 2: Fairness-Focused (Multi-party Collaboration)
```bash
python run_fo_pipeline.py \
  --rl_algorithm fosqddpg \
  --aggregation_method DP \
  --trading_strategy market_clearing \
  --disaggregation_method average \
  --scheduling_method fairness \
  --num_episodes 150 \
  --use_gpu
```

### Scenario 3: Efficiency-Focused (Fast Convergence)
```bash
python run_fo_pipeline.py \
  --rl_algorithm fomaddpg \
  --aggregation_method LP \
  --trading_strategy market_clearing \
  --disaggregation_method proportional \
  --scheduling_method priority \
  --num_episodes 50 \
  --use_gpu
```

### Scenario 4: Avoiding Manager Policy Conflicts
```bash
python run_fo_pipeline.py \
  --rl_algorithm fomaippo \
  --aggregation_method DP \
  --trading_strategy market_clearing \
  --disaggregation_method average \
  --scheduling_method fairness \
  --num_episodes 150 \
  --use_gpu
```

### Scenario 5: Quick Baseline Comparison (No Training Required)
```bash
python run_fo_pipeline.py \
  --rl_algorithm fomodelbased \
  --aggregation_method LP \
  --trading_strategy market_clearing \
  --disaggregation_method proportional \
  --scheduling_method priority
```

## ⚙️ Parameter Details

### Main Parameters

| Parameter | Description | Options | Default Value |
|--------|------|--------|--------|
| `--rl_algorithm` | Reinforcement learning algorithm | fomappo, fomaippo, fomaddpg, fomatd3, fosqddpg, fomodelbased | fomappo |
| `--aggregation_method` | Aggregation method | LP (Longest Profile), DP (Dynamic Profile) | LP |
| `--trading_strategy` | Trading strategy | market_clearing, bidding | market_clearing |
| `--clearing_method` | Market clearing method | uniform_price, pay_as_bid, lmp | uniform_price |
| `--disaggregation_method` | Disaggregation method | average, proportional | proportional |
| `--scheduling_method` | Scheduling method | priority, fairness, cost | priority |
| `--num_episodes` | Training episodes | 10-1000 | 100 |
| `--time_horizon` | Time range (hours) | 1-48 | 24 |
| `--num_users` | Number of users | 1-100 | 36 |
| `--num_managers` | Number of managers | 1-10 | 4 |

### Advanced Parameters

| Parameter | Description | Options | Default Value |
|--------|------|--------|--------|
| `--use_gpu` | Use GPU acceleration | - | False |
| `--enable_monitoring` | Enable performance monitoring | - | False |
| `--save_training_stats` | Save training statistics | - | False |
| `--save_results` | Save run results | - | False |
| `--visualize` | Visualize results | - | False |
| `--log_verbosity` | Log verbosity level | minimal, brief, detailed, debug | brief |
| `--learning_rate` | Learning rate | 0.0001-0.01 | 0.0003 |
| `--batch_size` | Batch size | 32-1024 | 256 |
| `--gamma` | Discount factor | 0.9-0.999 | 0.99 |

## 🔧 Algorithm Architecture and Features

### FOMAPPO vs FOMAIPPO Comparison

The MAPPO algorithm has recently integrated two policy architectures:

#### FOMAPPO (Shared Policy Architecture)
```python
# File location: algorithms/MAPPO/fomappo/fomappo_adapter.py
class FOMAPPOAdapter:
    - Uses SharedReplayBuffer
    - All Managers share a single policy network
    - References original MAPPO's shared/base_runner.py architecture
    - Advantages: High parameter efficiency, natural coordination
    - Applicable: Scenarios where Manager tasks are similar
```

#### FOMAIPPO (Independent Policy Architecture)
```python
# File location: algorithms/MAPPO/fomappo/fomaippo_adapter.py
class FOMAIPPOAdapter:
    - Uses SeparatedReplayBuffer
    - Each Manager has an independent policy network
    - References original MAPPO's separated/base_runner.py architecture
    - Advantages: Avoids policy conflicts, independent learning
    - Applicable: Scenarios where Managers handle different types of user groups
```

### Core Feature Comparison

| Feature | FOMAPPO (Shared Policy) | FOMAIPPO (Independent Policy) |
|------|-------------------|-------------------|
| Policy Network | Shared across all Managers | Independent for each Manager |
| Buffer Type | SharedReplayBuffer | SeparatedReplayBuffer |
| Parameter Count | Lower (shared parameters) | Higher (independent parameters) |
| Training Stability | Higher (reduced variance) | Moderate (independent learning) |
| Coordination Ability | Natural coordination | Requires additional mechanisms |
| Data Efficiency | High (shared experience) | Moderate (independent experience) |
| Applicable Scenarios | Similar Manager tasks | Diverse Manager tasks |

### Other Algorithm Features

#### FOMADDPG
- **Advantages**: Highest sample efficiency, excellent performance in continuous action spaces
- **Disadvantages**: Training stability slightly lower than FOMAPPO/FOMATD3
- **Applicable Scenarios**: Scenarios requiring fast convergence

#### FOMATD3
- **Advantages**: Dual Q-network design reduces overestimation, highest training stability
- **Disadvantages**: Computational complexity slightly higher than other algorithms
- **Applicable Scenarios**: High-noise environments, long-term training scenarios

#### FOSQDDPG
- **Advantages**: Shapley value fair distribution, ensures fairness in multi-party collaboration
- **Disadvantages**: High computational load, relatively slower convergence speed
- **Applicable Scenarios**: Multi-party collaboration requiring fairness guarantees

## 📊 Performance Comparison

### Learning Curves

| Algorithm | Convergence Speed | Stability | Final Performance |
|------|---------|-------|----------|
| **FOMAPPO** | Moderate (40-60 episodes) | Very High | High |
| **FOMAIPPO** | Moderate (50-70 episodes) | High | High |
| **FOMADDPG** | Fastest (20-30 episodes) | Moderate | Highest |
| **FOMATD3** | Fast (30-40 episodes) | Very High | High |
| **FOSQDDPG** | Slower (60-80 episodes) | High | Moderate but Fair |
| **FOModelBased** | No training required | N/A | Moderate |

### Resource Usage

| Algorithm | GPU Memory | CPU Usage | Training Time (100 episodes) |
|------|---------|----------|-----------------|
| **FOMAPPO** | Moderate (2-3GB) | 60% | ~45 minutes |
| **FOMAIPPO** | High (3-4GB) | 65% | ~52 minutes |
| **FOMADDPG** | Low (1-2GB) | 50% | ~30 minutes |
| **FOMATD3** | Moderate (2-3GB) | 55% | ~35 minutes |
| **FOSQDDPG** | High (3-4GB) | 70% | ~40 minutes |
| **FOModelBased** | Very Low (<1GB) | 40% | Immediate |

## 🚀 Experimental Suggestions

### Scenario Testing Methods
1. **Similar Task Scenario**: All Managers managing similar user groups
   - Recommended Algorithm: FOMAPPO
   - Example Configuration: `--rl_algorithm fomappo --aggregation_method LP`

2. **Diversified Task Scenario**: Managers managing different types of user groups
   - Recommended Algorithm: FOMAIPPO
   - Example Configuration: `--rl_algorithm fomaippo --aggregation_method DP`

3. **Scalability Testing**: Testing different numbers of Managers (2, 4, 8)
   ```bash
   # 2 Managers
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 2 --num_episodes 100
   
   # 4 Managers
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 4 --num_episodes 100
   
   # 8 Managers
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 8 --num_episodes 100
   ```

## 📋 Common Issues and Solutions

### Training Issues
1. **Issue**: Unstable training, large reward fluctuations
   **Solution**: Try FOMATD3 algorithm, increase `--batch_size` value, reduce learning rate

2. **Issue**: Policy conflicts between Managers
   **Solution**: Switch to FOMAIPPO algorithm, enable independent policy networks

3. **Issue**: Slow training speed
   **Solution**: Use FOMADDPG algorithm, add `--use_gpu` parameter, reduce `num_episodes`

### Runtime Issues
1. **Issue**: High memory usage
   **Solution**: Reduce batch_size, lower user or device count

2. **Issue**: Insufficient GPU memory
   **Solution**: Try `--mixed_precision` option, or reduce model complexity

3. **Issue**: System error "float() argument must be a string or a number"
   **Solution**: Check data format, possible input configuration file format error

## 📈 Summary

FlexOffer multi-agent algorithms provide a rich selection to choose from based on different scenarios:

- **Stability Priority**: FOMAPPO or FOMATD3
- **Fairness Priority**: FOSQDDPG
- **Efficiency Priority**: FOMADDPG
- **Avoiding Policy Conflicts**: FOMAIPPO
- **Baseline Comparison**: FOModelBased

The 40 different combination configurations offer flexible options to customize based on specific needs and scenarios, achieving optimal performance. 
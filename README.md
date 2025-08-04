# FlexOffer Multi-Agent Reinforcement Learning Trading System

## System Overview

This system (RLTRADE) is a complete platform for FlexOffer (flexibility offer) generation, aggregation, trading, and scheduling based on multi-agent deep reinforcement learning. The system integrates **five advanced multi-agent RL algorithms** and adopts a Manager-level collaborative architecture to implement an end-to-end energy management solution from device control to market trading.

## ✨ Core Features

### 🤖 Five Fully Integrated Algorithms
- **MAPPO**: FlexOffer-specialized multi-agent proximal policy optimization (shared policy)
- **MAIPPO**: FlexOffer multi-agent independent PPO (separate policy)
- **MADDPG**: FlexOffer multi-agent deep deterministic policy gradient  
- **MATD3**: FlexOffer multi-agent twin delayed DDPG
- **SQDDPG**: SQDDPG based on Shapley value fair credit assignment
- **FOModelBased**: Traditional model-based optimization benchmark (no training required)

### 🧠 Dec-POMDP Architecture
- **Decentralized Partially Observable Markov Decision Process**: Real multi-agent environment modeling
- **3-Layer Observation Architecture**: Private information (40-dim) + Public information (18-dim) + Others' information (15-dim)
- **Dynamic Observation Quality**: 5-level network quality dynamic adjustment, noise level 5-10%
- **Information Asymmetry Handling**: Information sharing restrictions between agents, simulating real distributed systems
- **Observation Function Z Design**: Probabilistic observation model, supporting uncertainty and communication delay


### 🔧 Device Ecosystem
- **5 Device Types**: Battery storage, heat pumps, electric vehicles, photovoltaics, dishwashers
- **118 Devices**: Distributed across 36 users, managed by 4 Managers
- **Device Deployment Rate**: Dishwashers (100%), Heat pumps (100%), Batteries (67%), EVs (39%), PV (22%)
- **Intelligent Control**: Each device type has specialized implementation and reward design

## 📊 System Architecture

```
FlexOffer System Four-Layer Architecture
┌──────────────────────────────────────────────────────────────────────────────────────────── ┐
│                        Multi-Algorithm Support Layer (6 algorithms)                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ FOMAPPO          │ FOMAIPPO               │ FOMADDPG   │ FOMATD3          │ FOSQDDPG        │            
│ Shared policy+   │ Independent policy+    │ Actor-     │ Dual Q-network+  │ Shapley value+  │                
│ Trust region     │ Conflict avoidance     │ Critic     │ Delayed updates  │ Fair allocation │                
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         Complete FlexOffer Process                                                         │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Generation Layer        │  Aggregation Layer       │  Trading Layer        │  Scheduling Layer            │
│  fo_generate/            │  fo_aggregate/           │  fo_trading/          │  fo_schedule/                │
│  Device MDP modeling     │  LP/DP aggregation       │  Market matching      │  Decomposition scheduling    │
│  Unified environment     │  Manager aggregation     │  Bilateral auction      │  Satisfaction assessment   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        Device Ecosystem                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Dishwashers(36)         │ Heat pumps(36)         │ Batteries(24)      │ EVs(14)            │ PV(8)               │
│ 100% deployment         │ 100% deployment        │ 67% deployment     │ 39% deployment     │ 22% deployment      │
│ User behavior modeling  │ Temperature control    │ SOC management     │ Charging strategy  │ Generation forecast │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🧠 Algorithm Feature Comparison

| Feature | MAPPO | MAIPPO | MADDPG | MATD3 | SQDDPG | 
|------|---------|----------|----------|---------|----------|--------------|
| **Algorithm Type** | Policy Gradient | Policy Gradient | Actor-Critic | Actor-Critic | Actor-Critic |
| **Policy Architecture** | Shared Policy | Independent Policy | Shared Policy | Shared Policy | Shared Policy 
| **Policy Update** | Batch+Trust Region | Batch+Trust Region | Continuous Policy Gradient | Delayed Policy Update | Continuous+Credit Assignment 
| **Value Estimation** | Advantage Function | Advantage Function | Single Q-Network | Dual Q-Network | Q-Network+Shapley 
| **Multi-Agent Collaboration** | Natural Coordination | Mechanism Required | Basic Collaboration | Basic Collaboration | **Fairness Guarantee** 
| **Policy Conflict Handling** | Weak | Strong | Weak | Weak | Moderate 
| **Credit Assignment** | Standard Method | Standard Method | Standard Method | Standard Method | **Shapley Value** 
| **Applicable Scenarios** | Similar Tasks | Diverse Tasks | Continuous Control | High-Noise Environment | Fair Collaboration 

## 🚀 Quick Start

### Installation Requirements
```bash
# Basic Dependencies
pip install torch numpy pandas matplotlib gymnasium

# Multi-Agent Environment
pip install pettingzoo supersuit

# Optional: GPU Support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Operation

#### 1. Using Default Configuration (Recommended for Beginners)
```bash
# FOMAPPO (Shared Policy, Most Stable)
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 100

# FOMAIPPO (Independent Policy, Avoids Conflicts)
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 100
```

#### 2. Custom Algorithm Combinations (40 Configuration Combinations)
```bash
# Complete Parameter Template: 6 algorithms × 2 aggregation × 2 trading × 2 decomposition = 48 theoretical combinations (40 actually available)
python run_fo_pipeline.py \
  --rl_algorithm [fomappo|fomaippo|fomaddpg|fomatd3|fosqddpg|fomodelbased] \
  --aggregation_method [LP|DP] \
  --trading_strategy [market_clearing|bidding] \
  --disaggregation_method [average|proportional] \
  --scheduling_method [priority|fairness|cost] \
  --num_episodes 100 \  # Not needed for FOModelBased
  --use_gpu
```

#### 3. Log Verbosity Control (New Feature)
```bash
# Minimal Mode - Only display key progress information
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity minimal

# Brief Mode - Merge repeated information into one line (Default)
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity brief

# Detailed Mode - Display all information
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity detailed

# Debug Mode - Display all debug information
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity debug
```

#### 4. Trading Algorithm Selection (New Feature)
```bash
# Using Market Clearing Algorithm (Default)
python run_fo_pipeline.py --rl_algorithm fomappo --trading_strategy market_clearing

# Using Bidding Algorithm
python run_fo_pipeline.py --rl_algorithm fomappo --trading_strategy bidding
```

### Batch Comparison Testing (Including Traditional Optimization Benchmark)
```bash
# Windows PowerShell - Complete Comparison of 6 Algorithms
foreach ($algo in @("fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg", "fomodelbased")) {
    if ($algo -eq "fomodelbased") {
        python run_fo_pipeline.py --rl_algorithm $algo  # No training needed
    } else {
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    }
}

# Linux/Mac Bash - Complete Comparison of 6 Algorithms
for algo in fomappo fomaippo fomaddpg fomatd3 fosqddpg fomodelbased; do
    if [ "$algo" = "fomodelbased" ]; then
        python run_fo_pipeline.py --rl_algorithm $algo  # No training needed
    else
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    fi
done
```


## 📁 Project Structure

```
RLtrade/
├── README.md                   # This document (system overview and basic usage)
├── SYSTEM_ARCHITECTURE.md      # Detailed system architecture documentation
├── ALGORITHM_GUIDE.md          # Algorithm usage and configuration guide
├── DEVELOPER_GUIDE.md          # Developer guide (logs, trading module, etc.)
├── run_fo_pipeline.py          # Main running script
├── algorithms/                 # Multi-agent algorithm implementations
│   ├── MAPPO/fomappo/         # FOMAPPO + FOMAIPPO algorithms
│   ├── MADDPG/fomaddpg/       # FOMADDPG algorithm
│   ├── MATD3/fomatd3/         # FOMATD3 algorithm
│   └── SQDDPG/fosqddpg/       # FOSQDDPG algorithm
├── fo_generate/               # FlexOffer generation module
├── fo_aggregate/              # FlexOffer aggregation module
├── fo_trading/                # FlexOffer trading module
├── fo_schedule/               # FlexOffer scheduling module
├── fo_common/                 # Common components
├── data/                      # Data files
└── results/                   # Training results
```

## 🛠️ Development and Debugging

### Debugging Tools
```bash
# System Diagnostics
python tests/test_components.py --verbose

# Performance Benchmarking  
python tests/benchmark_global_observation.py

# Algorithm Performance Comparison
python tests/run_tests.py --benchmark --algorithms fomappo,fomaippo,fosqddpg

# Visualization Analysis
python run_fo_pipeline.py --rl_algorithm fosqddpg --visualize --save_results
```

### Logging and Monitoring
```python
# Enable Detailed Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Performance Monitoring
python run_fo_pipeline.py --rl_algorithm fomappo \
    --enable_monitoring \
    --save_training_stats \
    --num_episodes 100
```

## 🎯 Summary

The RLTRADE system implements a complete FlexOffer multi-agent reinforcement learning solution with the following outstanding features:

- ✅ **Six Complete Algorithms**: 5 MARL algorithms (FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG)
- ✅ **40 Combination Configurations**: 5 algorithms × 2 aggregation methods × 2 trading strategies × 2 decomposition methods = 40 fully usable combinations
- ✅ **Policy Conflict Resolution**: MAIPPO independent policy architecture, avoiding policy conflicts between Managers
- ✅ **Traditional Optimization Benchmark**: ModelBased provides traditional optimization benchmark comparison without training
- ✅ **Complete FlexOffer Process**: End-to-end process of generation → aggregation → trading → scheduling
- ✅ **Experimental Validation**: Actual system validation with 4 Managers + 36 users + 118 devices


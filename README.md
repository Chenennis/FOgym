# FOgym: FlexOffer Multi-Agent Reinforcement Learning Trading Platform

## System Overview

FOgym is a complete platform for FlexOffer (flexibility offer) generation, aggregation, trading, and scheduling based on multi-agent deep reinforcement learning. The system integrates **five MARL algorithms** and **one model-based baseline**, adopting a Manager-level collaborative architecture to implement an end-to-end energy management solution from device control to market trading.

## Core Features

### Five MARL Algorithms
- **FOMAPPO**: Multi-agent proximal policy optimization (shared policy)
- **FOMAIPPO**: Multi-agent independent PPO (separate policy)
- **FOMADDPG**: Multi-agent deep deterministic policy gradient
- **FOMATD3**: Multi-agent twin delayed DDPG
- **FOSQDDPG**: Shapley value-based fair credit assignment

### Dec-POMDP Architecture
- **Decentralized Partially Observable Markov Decision Process**: Real multi-agent environment modeling
- **3-Layer Observation Architecture**: Private information (40-dim) + Public information (18-dim) + Others' information (15-dim)
- **Dynamic Observation Quality**: 5-level network quality dynamic adjustment, noise level 5-20%
- **Information Asymmetry Handling**: Information sharing restrictions between agents, simulating real distributed systems

### Device Ecosystem
- **5 Device Types**: Battery storage, heat pumps, electric vehicles, photovoltaics, dishwashers
- **Scalable Deployment**: Default 4-Manager (36 users, 118 devices) or 10-Manager configuration
- **Device Deployment Rate**: Dishwashers (100%), Heat pumps (100%), Batteries (67%), EVs (39%), PV (22%)
- **Intelligent Control**: Each device type has specialized MDP implementation and reward design

## System Architecture

```
FOgym Four-Layer Architecture
+------------------------------------------------------------------------------------+
|                    Multi-Algorithm Support Layer (5 MARL algorithms)                 |
+------------------------------------------------------------------------------------+
| FOMAPPO        | FOMAIPPO             | FOMADDPG   | FOMATD3        | FOSQDDPG    |
| Shared policy+ | Independent policy+  | Actor-     | Dual Q-network+| Shapley     |
| Trust region   | Conflict avoidance   | Critic     | Delayed updates| value+Fair  |
+------------------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------------------+
|                        Complete FlexOffer Process                                   |
+------------------------------------------------------------------------------------+
| Generation Layer    | Aggregation Layer    | Trading Layer    | Scheduling Layer    |
| fo_generate/        | fo_aggregate/        | fo_trading/      | fo_schedule/        |
| Device MDP modeling | LP/DP aggregation    | Market matching  | Decomposition       |
| Unified environment | Manager aggregation  | Bilateral auction| Satisfaction assess |
+------------------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------------------+
|                           Device Ecosystem                                          |
+------------------------------------------------------------------------------------+
| Dishwashers(36)     | Heat pumps(36)      | Batteries(24) | EVs(14)  | PV(8)      |
| 100% deployment     | 100% deployment     | 67% deployment| 39%      | 22%        |
| User behavior model | Temperature control | SOC management| Charging | Generation |
+------------------------------------------------------------------------------------+
```

## Algorithm Feature Comparison

| Feature | MAPPO | MAIPPO | MADDPG | MATD3 | SQDDPG |
|------|---------|----------|----------|---------|----------|
| **Algorithm Type** | Policy Gradient | Policy Gradient | Actor-Critic | Actor-Critic | Actor-Critic |
| **Policy Architecture** | Shared Policy | Independent Policy | Shared Policy | Shared Policy | Shared Policy |
| **Policy Update** | Batch+Trust Region | Batch+Trust Region | Continuous Policy Gradient | Delayed Policy Update | Continuous+Credit Assignment |
| **Value Estimation** | Advantage Function | Advantage Function | Single Q-Network | Dual Q-Network | Q-Network+Shapley |
| **Multi-Agent Collaboration** | Natural Coordination | Mechanism Required | Basic Collaboration | Basic Collaboration | **Fairness Guarantee** |
| **Credit Assignment** | Standard | Standard | Standard | Standard | **Shapley Value** |
| **Applicable Scenarios** | Similar Tasks | Diverse Tasks | Continuous Control | High-Noise Environment | Fair Collaboration |

## Quick Start

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

#### 2. Custom Algorithm Combinations
```bash
# Complete Parameter Template: 5 algorithms x 2 aggregation x 2 trading x 2 decomposition = 40 combinations
python run_fo_pipeline.py \
  --rl_algorithm [fomappo|fomaippo|fomaddpg|fomatd3|fosqddpg] \
  --aggregation_method [LP|DP] \
  --trading_strategy [market_clearing|bidding] \
  --disaggregation_method [average|proportional] \
  --scheduling_method [priority|fairness|cost] \
  --data_config [36users|4manager|10manager] \
  --num_episodes 100 \
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

### Batch Comparison Testing
```bash
# Windows PowerShell - Compare all 5 MARL algorithms
foreach ($algo in @("fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg")) {
    python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
}

# Linux/Mac Bash - Compare all 5 MARL algorithms
for algo in fomappo fomaippo fomaddpg fomatd3 fosqddpg; do
    python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
done
```


## Project Structure

```
FOgym/
├── README.md                   # This document (system overview and basic usage)
├── SYSTEM_ARCHITECTURE.md      # Detailed system architecture documentation
├── ALGORITHM_GUIDE.md          # Algorithm usage and configuration guide
├── DEVELOPER_GUIDE.md          # Developer guide (logs, module extension, etc.)
├── run_fo_pipeline.py          # Main pipeline script
├── run_experiments.py          # Experiment runner (ablation, scalability)
├── global_observation_config.json  # Observation space configuration
├── algorithms/                 # Multi-agent algorithm implementations
│   ├── MAPPO/fomappo/         # FOMAPPO + FOMAIPPO algorithms
│   ├── MADDPG/fomaddpg/       # FOMADDPG algorithm
│   ├── MATD3/fomatd3/         # FOMATD3 algorithm
│   └── SQDDPG/fosqddpg/      # FOSQDDPG algorithm
├── fo_generate/               # FlexOffer generation module (device MDP)
├── fo_aggregate/              # FlexOffer aggregation module (LP/DP)
├── fo_trading/                # FlexOffer trading module (market clearing/bidding)
├── fo_schedule/               # FlexOffer scheduling module (disaggregation)
├── fo_common/                 # Common components (config, observation, metrics)
├── data/                      # Data files (device configs, prices, weather)
├── tests/                     # Test files
└── results/                   # Training results (gitignored)
```

## Development and Debugging

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

## Summary

FOgym implements a complete FlexOffer multi-agent reinforcement learning solution:

- **Five MARL Algorithms**: FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG
- **40 Combination Configurations**: 5 algorithms x 2 aggregation methods x 2 trading strategies x 2 decomposition methods
- **Policy Conflict Resolution**: FOMAIPPO independent policy architecture avoids policy conflicts between Managers
- **Configurable Reward Weights**: Ablation-ready reward function with tunable alpha/beta/delta/lambda weights
- **Scalable Deployment**: Supports 4-Manager (36 users) and 10-Manager configurations
- **Complete FlexOffer Process**: End-to-end pipeline of generation -> aggregation -> trading -> scheduling
- **Experiment Runner**: Automated batch experiments with `run_experiments.py` for ablation and scalability studies


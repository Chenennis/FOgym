# FlexOffer多智能体强化学习交易系统

## 系统概述

本系统（RLTRADE）是一个基于多智能体深度强化学习的FlexOffer（灵活性报价）生成、聚合、交易和调度的完整平台。系统集成了**五种先进的多智能体RL算法**，采用Manager级别的协作架构，实现了从设备控制到市场交易的端到端能源管理解决方案。

## ✨ 核心特性

### 🤖 六种算法完整集成（5种MARL + 1种Model-based基准）
- **FOMAPPO**: FlexOffer专用多智能体近端策略优化（共享策略）
- **FOMAIPPO**: FlexOffer多智能体独立PPO（分离策略）
- **FOMADDPG**: FlexOffer多智能体深度确定性策略梯度  
- **FOMATD3**: FlexOffer多智能体双延迟DDPG
- **FOSQDDPG**: 基于Shapley值公平信用分配的SQDDPG
- **FOModelBased**: 传统基于模型的优化基准（无需训练）

### 🧠 Dec-POMDP架构突破
- **分布式部分可观测马尔可夫决策过程**: 真实多智能体环境建模
- **3层观测架构**: 私有信息(40维) + 公共信息(18维) + 他者信息(15维)
- **动态观测质量**: 5级网络质量动态调整，噪声水平5-10%
- **信息不对称处理**: 智能体间信息共享限制，模拟真实分布式系统
- **观测函数Z设计**: 概率观测模型，支持不确定性和通信延迟

### 🏗️ 完整FlexOffer流程
- **生成层**: 统一MDP环境和设备级建模
- **聚合层**: Manager-用户-设备三层架构
- **交易层**: 市场机制和智能撮合，支持Bidding和Market Clearing双算法
- **调度层**: 实时调度和满意度评估

### 🔧 设备生态系统
- **5种设备类型**: 电池储能、热泵、电动汽车、光伏、洗碗机
- **118个设备**: 分布在36个用户中，由4个Manager管理
- **设备部署率**: 洗碗机(100%)、热泵(100%)、电池(67%)、EV(39%)、PV(22%)
- **智能控制**: 每种设备都有专门的MDP实现和奖励设计

## 📊 系统架构

```
FlexOffer系统四层架构
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               多算法支持层（6种算法）                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ FOMAPPO    │ FOMAIPPO   │ FOMADDPG   │ FOMATD3   │ FOSQDDPG    │ FOModelBased              │
│ 共享策略+  │ 独立策略+  │ Actor-     │ 双Q网络+  │ Shapley值+  │ 传统优化+                 │
│ 信任域     │ 避免冲突   │ Critic     │ 延迟更新  │ 公平分配    │ 无需训练                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         FlexOffer完整流程                                │
├─────────────────────────────────────────────────────────────────────┤
│  生成层        │  聚合层        │  交易层        │  调度层              │
│  fo_generate/  │  fo_aggregate/ │  fo_trading/   │  fo_schedule/        │
│  设备MDP建模   │  LP/DP聚合     │  市场撮合      │  分解调度            │
│  统一环境      │  Manager聚合   │  双边拍卖      │  满意度评估          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                        设备生态系统                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 洗碗机(36个)  │ 热泵(36个)    │ 电池(24个)    │ EV(14个) │ 光伏(8个)    │
│ 100%部署率    │ 100%部署率    │ 67%部署率     │ 39%部署  │ 22%部署      │
│ 用户行为建模  │ 温度控制      │ SOC管理       │ 充电策略 │ 发电预测     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🧠 算法特性对比

| 特性 | FOMAPPO | FOMAIPPO | FOMADDPG | FOMATD3 | FOSQDDPG | FOModelBased |
|------|---------|----------|----------|---------|----------|--------------|
| **算法类型** | Policy Gradient | Policy Gradient | Actor-Critic | Actor-Critic | Actor-Critic | **Model-based** |
| **策略架构** | 共享策略 | 独立策略 | 共享策略 | 共享策略 | 共享策略 | **传统优化** |
| **策略更新** | 批量+信任域 | 批量+信任域 | 连续策略梯度 | 延迟策略更新 | 连续+信用分配 | **无需训练** |
| **价值估计** | 优势函数 | 优势函数 | 单Q网络 | 双Q网络 | Q网络+Shapley | **物理模型** |
| **训练稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| **样本效率** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| **多智能体协作** | 自然协调 | 需要机制 | 基础协作 | 基础协作 | **公平性保证** | **确定性协调** |
| **策略冲突处理** | 较弱 | **最强** | 较弱 | 较弱 | 中等 | **无冲突** |
| **信用分配** | 标准方法 | 标准方法 | 标准方法 | 标准方法 | **Shapley值** | **物理约束** |
| **适用场景** | 任务相似 | 任务差异大 | 连续控制 | 高噪声环境 | 公平协作 | **基准对比** |

## 🚀 快速开始

### 安装要求
```bash
# 基础依赖
pip install torch numpy pandas matplotlib gymnasium

# 多智能体环境
pip install pettingzoo supersuit

# 可选：GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 基本运行

#### 1. 使用默认配置（推荐新手）
```bash
# FOMAPPO（共享策略，最稳定）
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 100

# FOMAIPPO（独立策略，避免冲突）
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 100
```

#### 2. 自定义算法组合（40种组合配置）
```bash
# 完整参数模板：6种算法 × 2种聚合 × 2种交易 × 2种分解 = 48种理论组合（实际40种可用）
python run_fo_pipeline.py \
  --rl_algorithm [fomappo|fomaippo|fomaddpg|fomatd3|fosqddpg|fomodelbased] \
  --aggregation_method [LP|DP] \
  --trading_strategy [market_clearing|bidding] \
  --disaggregation_method [average|proportional] \
  --scheduling_method [priority|fairness|cost] \
  --num_episodes 100 \  # FOModelBased无需此参数
  --use_gpu
```

#### 3. 日志详细程度控制（新增特性）
```bash
# 最简模式 - 只显示关键进度信息
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity minimal

# 简略模式 - 合并重复信息到一行（默认）
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity brief

# 详细模式 - 显示所有信息
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity detailed

# 调试模式 - 显示所有调试信息
python run_fo_pipeline.py --rl_algorithm fomappo --log_verbosity debug
```

#### 4. 交易算法选择（新增特性）
```bash
# 使用Market Clearing算法（默认）
python run_fo_pipeline.py --rl_algorithm fomappo --trading_strategy market_clearing

# 使用Bidding算法
python run_fo_pipeline.py --rl_algorithm fomappo --trading_strategy bidding
```

### 批量对比测试（包含传统优化基准）
```bash
# Windows PowerShell - 完整6种算法对比
foreach ($algo in @("fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg", "fomodelbased")) {
    if ($algo -eq "fomodelbased") {
        python run_fo_pipeline.py --rl_algorithm $algo  # 无需训练
    } else {
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    }
}

# Linux/Mac Bash - 完整6种算法对比
for algo in fomappo fomaippo fomaddpg fomatd3 fosqddpg fomodelbased; do
    if [ "$algo" = "fomodelbased" ]; then
        python run_fo_pipeline.py --rl_algorithm $algo  # 无需训练
    else
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    fi
done
```

## 📈 验证结果和性能

### 最新实验数据

#### **多智能体协作效果**
- **4个Manager**: 管理36用户+118设备的大规模系统
- **协作学习**: Manager间信息共享和策略协调
- **训练稳定性**: 所有5种算法都收敛稳定

#### **FlexOffer生成效果**  
- **生成规模**: 每时间步106个设备级FlexOffer
- **聚合效率**: 26.5:1压缩比（106→4）
- **约束满足**: 100%符合设备物理约束

#### **市场交易效果**
- **交易成功率**: 67%（2/3时间步有成功交易）
- **交易总量**: 1,657 kWh，价值$213.05
- **价格发现**: 0.08-0.25 USD/kWh合理价格范围

#### **用户满意度效果**
- **能源分配**: 24/36用户获得能源配置
- **满意度提升**: 从0%提升到22.2%平均满意度
- **公平性**: FOSQDDPG算法确保Shapley值公平分配

## 📁 项目结构

```
RLtrade/
├── README.md                   # 本文档（系统概述和基本使用）
├── SYSTEM_ARCHITECTURE.md      # 系统架构详细文档
├── ALGORITHM_GUIDE.md          # 算法使用与配置指南
├── DEVELOPER_GUIDE.md          # 开发者指南（日志、交易模块等）
├── run_fo_pipeline.py          # 主运行脚本
├── algorithms/                 # 多智能体算法实现
│   ├── MAPPO/fomappo/         # FOMAPPO + FOMAIPPO算法
│   ├── MADDPG/fomaddpg/       # FOMADDPG算法
│   ├── MATD3/fomatd3/         # FOMATD3算法
│   └── SQDDPG/fosqddpg/       # FOSQDDPG算法
├── fo_generate/               # FlexOffer生成模块
├── fo_aggregate/              # FlexOffer聚合模块
├── fo_trading/                # FlexOffer交易模块
├── fo_schedule/               # FlexOffer调度模块
├── fo_common/                 # 通用组件
├── data/                      # 数据文件
└── results/                   # 训练结果
```

## 🛠️ 开发和调试

### 调试工具
```bash
# 系统诊断
python tests/test_components.py --verbose

# 性能基准测试  
python tests/benchmark_global_observation.py

# 算法性能对比
python tests/run_tests.py --benchmark --algorithms fomappo,fomaippo,fosqddpg

# 可视化分析
python run_fo_pipeline.py --rl_algorithm fosqddpg --visualize --save_results
```

### 日志和监控
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 性能监控
python run_fo_pipeline.py --rl_algorithm fomappo \
    --enable_monitoring \
    --save_training_stats \
    --num_episodes 100
```

## 🎯 总结

RLTRADE系统实现了完整的FlexOffer多智能体强化学习解决方案，具有以下突出特点：

- ✅ **六种完整算法**：5种MARL算法（FOMAPPO、FOMAIPPO、FOMADDPG、FOMATD3、FOSQDDPG）+ 1种Model-based基准（FOModelBased）
- ✅ **40种组合配置**：6种算法 × 2种聚合方法 × 2种交易策略 × 2种分解方法 = 40种完整可用组合
- ✅ **策略冲突解决**：FOMAIPPO独立策略架构，避免Manager间策略冲突
- ✅ **传统优化基准**：FOModelBased提供无需训练的传统优化基准对比
- ✅ **完整FlexOffer流程**：生成→聚合→交易→调度端到端流程
- ✅ **大规模验证**：4Manager + 36用户 + 118设备的实际系统验证
- ✅ **技术创新**：Shapley值公平分配、设备级MDP、洗碗机100%部署
- ✅ **公平性保证**：FOSQDDPG算法确保多方协作的公平信用分配
- ✅ **即时可用**：FOModelBased算法提供立即可用的传统优化基准

本系统为能源互联网、智能电网、多智能体系统等领域提供了完整的技术解决方案和研究平台。 
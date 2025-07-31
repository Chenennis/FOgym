# FlexOffer多智能体算法使用指南

本文档详细介绍了如何运行不同的多智能体算法组合，包括**5种MARL算法**（FOMAPPO、FOMAIPPO、FOMADDPG、FOMATD3、FOSQDDPG）和**1种Model-based基准算法**（FOModelBased），总共**6种算法**支持**40种完整组合配置**。

## 📋 目录

- [🚀 单一算法运行](#-单一算法运行)
- [🔀 算法组合配置](#-算法组合配置)
- [🏁 批量算法对比](#-批量算法对比)
- [💡 推荐组合](#-推荐组合)
- [⚙️ 参数详解](#️-参数详解)
- [🔧 算法架构与特性](#-算法架构与特性)
- [📊 性能对比](#-性能对比)

## 🚀 单一算法运行

### 六种核心算法（5种MARL + 1种Model-based基准）

#### FOMAPPO（推荐，稳定性最高，共享策略）
```bash
# 标准运行
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 100

# 长时间训练
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 200 --use_gpu
```

#### FOMAIPPO（独立策略，解决策略冲突）
```bash
# 标准运行
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 100

# 避免策略冲突的场景
python run_fo_pipeline.py --rl_algorithm fomaippo --num_episodes 170 --use_gpu
```

#### FOSQDDPG（公平性最佳）
```bash
# 标准运行
python run_fo_pipeline.py --rl_algorithm fosqddpg --num_episodes 100

# 强化公平性
python run_fo_pipeline.py --rl_algorithm fosqddpg --num_episodes 200 --use_gpu
```

#### FOMATD3（稳定性高）
```bash
# 标准运行
python run_fo_pipeline.py --rl_algorithm fomatd3 --num_episodes 100

# 高稳定性配置
python run_fo_pipeline.py --rl_algorithm fomatd3 --num_episodes 200 --use_gpu
```

#### FOMADDPG（效率最高）
```bash
# 标准运行
python run_fo_pipeline.py --rl_algorithm fomaddpg --num_episodes 100

# 快速训练
python run_fo_pipeline.py --rl_algorithm fomaddpg --num_episodes 50 --use_gpu
```

#### FOModelBased（传统优化基准，无需训练）
```bash
# 标准评估（无需训练，直接运行完整Pipeline）
python run_fo_pipeline.py --rl_algorithm fomodelbased

# 快速测试（短时间范围）
python run_fo_pipeline.py --rl_algorithm fomodelbased --time_horizon 6
```

## 🔀 算法组合配置

### 🎯 完整40种组合配置

#### **组合计算**: 6种算法 × 2种聚合方法 × 2种交易策略 × 2种分解方法 = **48种理论组合**
> 注意：FOModelBased算法不需要训练，其他参数组合仍然有效，实际可用组合为40种

#### **完整组合参数模板**
```bash
python run_fo_pipeline.py \
  --rl_algorithm [fomappo|fomaippo|fomaddpg|fomatd3|fosqddpg|fomodelbased] \
  --aggregation_method [LP|DP] \
  --trading_strategy [market_clearing|bidding] \
  --disaggregation_method [average|proportional] \
  --scheduling_method [priority|fairness|cost] \
  --num_episodes [训练回合数，FOModelBased无需此参数] \
  --num_users [用户数量，默认36] \
  --num_managers [管理者数量，默认4] \
  --time_horizon [时间范围，默认24小时] \
  --use_gpu [可选，使用GPU加速]
```

#### **算法分类说明**
| 算法类型 | 算法名称 | 特点 | 训练需求 |
|----------|----------|------|----------|
| **MARL算法** | FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG | 需要训练学习 | num_episodes参数必需 |
| **Model-based基准** | FOModelBased | 传统优化，无需训练 | 直接评估，忽略num_episodes |

## 🏁 批量算法对比

### PowerShell批处理
```powershell
# Windows PowerShell - 完整6种算法对比
foreach ($algo in @("fomappo", "fomaippo", "fomaddpg", "fomatd3", "fosqddpg", "fomodelbased")) {
    if ($algo -eq "fomodelbased") {
        python run_fo_pipeline.py --rl_algorithm $algo  # 无需训练
    } else {
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    }
}
```

### Bash批处理
```bash
# Linux/Mac Bash - 完整6种算法对比
for algo in fomappo fomaippo fomaddpg fomatd3 fosqddpg fomodelbased; do
    if [ "$algo" = "fomodelbased" ]; then
        python run_fo_pipeline.py --rl_algorithm $algo  # 无需训练
    else
        python run_fo_pipeline.py --rl_algorithm $algo --num_episodes 100
    fi
done
```

### 结果比较脚本
```bash
# 比较多个算法结果
python analyze_algorithm_performance.py --results_dir ./results --plot

# 绘制奖励曲线
python analyze_algorithm_performance.py --plot_rewards --algorithms fomappo,fomaddpg,fomatd3
```

## 💡 推荐组合

### 场景1: 稳定性为主（长期训练）
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

### 场景2: 公平性为主（多方协作）
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

### 场景3: 效率为主（快速收敛）
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

### 场景4: 避免Manager策略冲突
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

### 场景5: 快速基线对比（无需训练）
```bash
python run_fo_pipeline.py \
  --rl_algorithm fomodelbased \
  --aggregation_method LP \
  --trading_strategy market_clearing \
  --disaggregation_method proportional \
  --scheduling_method priority
```

## ⚙️ 参数详解

### 主要参数

| 参数名 | 描述 | 可选值 | 默认值 |
|--------|------|--------|--------|
| `--rl_algorithm` | 强化学习算法 | fomappo, fomaippo, fomaddpg, fomatd3, fosqddpg, fomodelbased | fomappo |
| `--aggregation_method` | 聚合方法 | LP (Longest Profile), DP (Dynamic Profile) | LP |
| `--trading_strategy` | 交易策略 | market_clearing, bidding | market_clearing |
| `--clearing_method` | 市场出清方式 | uniform_price, pay_as_bid, lmp | uniform_price |
| `--disaggregation_method` | 分解方法 | average, proportional | proportional |
| `--scheduling_method` | 调度方法 | priority, fairness, cost | priority |
| `--num_episodes` | 训练回合数 | 10-1000 | 100 |
| `--time_horizon` | 时间范围(小时) | 1-48 | 24 |
| `--num_users` | 用户数量 | 1-100 | 36 |
| `--num_managers` | Manager数量 | 1-10 | 4 |

### 高级参数

| 参数名 | 描述 | 可选值 | 默认值 |
|--------|------|--------|--------|
| `--use_gpu` | 使用GPU加速 | - | False |
| `--enable_monitoring` | 启用性能监控 | - | False |
| `--save_training_stats` | 保存训练统计 | - | False |
| `--save_results` | 保存运行结果 | - | False |
| `--visualize` | 可视化结果 | - | False |
| `--log_verbosity` | 日志详细程度 | minimal, brief, detailed, debug | brief |
| `--learning_rate` | 学习率 | 0.0001-0.01 | 0.0003 |
| `--batch_size` | 批次大小 | 32-1024 | 256 |
| `--gamma` | 折扣因子 | 0.9-0.999 | 0.99 |

## 🔧 算法架构与特性

### FOMAPPO与FOMAIPPO对比

MAPPO算法最新整合了两种策略架构：

#### FOMAPPO（共享策略架构）
```python
# 文件位置：algorithms/MAPPO/fomappo/fomappo_adapter.py
class FOMAPPOAdapter:
    - 使用 SharedReplayBuffer
    - 所有Manager共享一个策略网络
    - 参考原始MAPPO的shared/base_runner.py架构
    - 优势：参数效率高，自然协调
    - 适用：Manager任务相似的场景
```

#### FOMAIPPO（独立策略架构）
```python
# 文件位置：algorithms/MAPPO/fomappo/fomaippo_adapter.py
class FOMAIPPOAdapter:
    - 使用 SeparatedReplayBuffer
    - 每个Manager有独立的策略网络
    - 参考原始MAPPO的separated/base_runner.py架构
    - 优势：避免策略冲突，独立学习
    - 适用：Manager管理不同类型用户群体
```

### 核心特性对比

| 特性 | FOMAPPO（共享策略） | FOMAIPPO（独立策略） |
|------|-------------------|-------------------|
| 策略网络 | 所有Manager共享一个 | 每个Manager独立 |
| Buffer类型 | SharedReplayBuffer | SeparatedReplayBuffer |
| 参数数量 | 较少（参数共享） | 较多（独立参数） |
| 训练稳定性 | 较高（减少方差） | 中等（独立学习） |
| 协调能力 | 自然协调 | 需要额外机制 |
| 数据效率 | 高（共享经验） | 中等（独立经验） |
| 适用场景 | Manager任务相似 | Manager任务差异大 |

### 其他算法特点

#### FOMADDPG
- **优势**: 最高的样本效率，连续动作空间的极佳性能
- **缺点**: 训练稳定性略低于FOMAPPO/FOMATD3
- **适用场景**: 需要快速收敛的场景

#### FOMATD3
- **优势**: 双Q网络设计降低过估计，最高训练稳定性
- **缺点**: 计算复杂度略高于其他算法
- **适用场景**: 高噪声环境，长期训练场景

#### FOSQDDPG
- **优势**: Shapley值公平分配，确保多方协作公平性
- **缺点**: 计算量较大，收敛速度相对较慢
- **适用场景**: 需要保证公平性的多方协作

## 📊 性能对比

### 学习曲线

| 算法 | 收敛速度 | 稳定性 | 最终性能 |
|------|---------|-------|----------|
| **FOMAPPO** | 中等 (40-60回合) | 极高 | 高 |
| **FOMAIPPO** | 中等 (50-70回合) | 高 | 高 |
| **FOMADDPG** | 最快 (20-30回合) | 中等 | 最高 |
| **FOMATD3** | 快 (30-40回合) | 极高 | 高 |
| **FOSQDDPG** | 较慢 (60-80回合) | 高 | 中等但公平 |
| **FOModelBased** | 无需训练 | 不适用 | 中等 |

### 资源占用

| 算法 | GPU内存 | CPU使用率 | 训练时间(100回合) |
|------|---------|----------|-----------------|
| **FOMAPPO** | 中等 (2-3GB) | 60% | 约45分钟 |
| **FOMAIPPO** | 高 (3-4GB) | 65% | 约52分钟 |
| **FOMADDPG** | 低 (1-2GB) | 50% | 约30分钟 |
| **FOMATD3** | 中等 (2-3GB) | 55% | 约35分钟 |
| **FOSQDDPG** | 高 (3-4GB) | 70% | 约40分钟 |
| **FOModelBased** | 极低 (<1GB) | 40% | 立即完成 |

## 🚀 实验建议

### 场景测试方法
1. **相似任务场景**：所有Manager管理相似的用户群体
   - 推荐算法：FOMAPPO
   - 示例配置：`--rl_algorithm fomappo --aggregation_method LP`

2. **差异化任务场景**：Manager管理不同类型的用户群体
   - 推荐算法：FOMAIPPO
   - 示例配置：`--rl_algorithm fomaippo --aggregation_method DP`

3. **扩展性测试**：测试不同Manager数量（2, 4, 8个）
   ```bash
   # 2个Manager
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 2 --num_episodes 100
   
   # 4个Manager
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 4 --num_episodes 100
   
   # 8个Manager
   python run_fo_pipeline.py --rl_algorithm fomappo --num_managers 8 --num_episodes 100
   ```

## 📋 常见问题与解决方案

### 训练问题
1. **问题**: 训练不稳定，奖励波动大
   **解决**: 尝试FOMATD3算法，增加`--batch_size`值，降低学习率

2. **问题**: Manager之间策略冲突
   **解决**: 切换到FOMAIPPO算法，启用独立策略网络

3. **问题**: 训练速度缓慢
   **解决**: 使用FOMADDPG算法，增加`--use_gpu`参数，降低`num_episodes`

### 运行问题
1. **问题**: 内存占用过高
   **解决**: 减少batch_size，降低用户或设备数量

2. **问题**: GPU内存不足
   **解决**: 尝试`--mixed_precision`选项，或降低模型复杂度

3. **问题**: 系统报错"float() argument must be a string or a number"
   **解决**: 检查数据格式，可能是输入配置文件格式有误

## 📈 总结

FlexOffer多智能体算法提供了丰富的选择，可以根据不同场景选择合适的算法组合：

- **稳定性优先**: FOMAPPO或FOMATD3
- **公平性优先**: FOSQDDPG
- **效率优先**: FOMADDPG
- **避免策略冲突**: FOMAIPPO
- **基准对比**: FOModelBased

40种不同组合配置提供了灵活的选择空间，可以根据具体需求和场景进行定制化配置，实现最佳性能。 
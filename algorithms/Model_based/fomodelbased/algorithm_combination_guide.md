# ModelBased FlexOffer Pipeline 算法组合指南

本指南详细介绍如何在ModelBased FlexOffer Pipeline中组合不同的算法选项，以及如何通过命令行参数自由配置这些组合。

## 算法组合概述

ModelBased FlexOffer Pipeline支持在三个主要处理阶段使用不同的算法：

1. **聚合阶段 (Aggregation)**
   - **LP** (Longest Profile)：优先选择最长的能量轮廓，追求最大能量体积
   - **DP** (Dynamic Profile)：排除异常长的轮廓，优先考虑时间灵活性

2. **交易阶段 (Trading)**
   - **bidding**：基于投标机制，计算买卖价格并匹配订单
   - **market-clearing**：基于市场出清原理，优化社会福利

3. **分解阶段 (Disaggregation)**
   - **proportional**：按设备能源需求比例分配
   - **average**：在设备类型组内平均分配

这三个阶段的算法可以自由组合，形成 2×2×2=8 种不同的算法组合。

## 命令行参数详解

可以使用以下命令行参数控制算法选择：

```bash
python run_pipeline.py [--config CONFIG_FILE] [--timesteps N] 
                       [--aggregation {LP,DP}] 
                       [--trading {bidding,market-clearing}]
                       [--disaggregation {proportional,average}]
                       [--managers N] [--users USERS_LIST]
```

其中：

- `--config`：配置文件路径（可选，如不提供则使用默认配置）
- `--timesteps`：模拟的时间步数（默认为24小时）
- `--aggregation`：聚合算法选择，可为`LP`或`DP`（默认为`LP`）
- `--trading`：交易算法选择，可为`bidding`或`market-clearing`（默认为`bidding`）
- `--disaggregation`：分解算法选择，可为`proportional`或`average`（默认为`proportional`）
- `--managers`：Manager数量（默认为4）
- `--users`：每个Manager的用户数量，用逗号分隔（默认为`6,10,8,12`）

## 常用命令组合示例

以下是一些常用的命令组合示例：

### 1. 默认配置（LP聚合 + bidding交易 + proportional分解）

```bash
python run_pipeline.py
```

### 2. 使用DP聚合算法

```bash
python run_pipeline.py --aggregation DP
```

### 3. 使用market-clearing交易算法

```bash
python run_pipeline.py --trading market-clearing
```

### 4. 使用average分解算法

```bash
python run_pipeline.py --disaggregation average
```

### 5. 自定义组合：DP聚合 + market-clearing交易 + average分解

```bash
python run_pipeline.py --aggregation DP --trading market-clearing --disaggregation average
```

### 6. 更改模拟时长为48小时

```bash
python run_pipeline.py --timesteps 48
```

### 7. 更改Manager数量和用户分布

```bash
python run_pipeline.py --managers 3 --users 8,12,16
```

## 比较不同算法组合

要系统性地比较不同算法组合的性能，可以使用`run_modelbased_comparison.py`脚本：

```bash
python run_modelbased_comparison.py [--timesteps N] [--output OUTPUT_FILE]
```

该脚本会自动运行所有可能的算法组合，并生成比较结果。结果会保存为JSON文件和可视化图表。

## 结果分析

运行不同算法组合后，结果将保存在`results`目录下的对应实验ID文件夹中。每个实验会生成以下文件：

- `timestep_details.csv`：每个时间步的详细数据
- `manager_rewards.csv`：每个Manager的奖励数据
- `total_rewards.csv`：总奖励数据
- `config.json`：实验配置
- `statistics.json`：统计信息

可以通过比较不同算法组合的总奖励、Manager奖励分布、交易成功率等指标，评估不同算法组合的性能。

## 算法组合的选择建议

不同的算法组合适用于不同的场景：

1. **LP + bidding + proportional**：适用于能量需求稳定、价格波动小的场景
2. **DP + bidding + proportional**：适用于设备灵活性较高的场景
3. **LP + market-clearing + proportional**：适用于追求整体社会福利最大化的场景
4. **DP + market-clearing + average**：适用于设备类型多样、需要公平分配的场景

## 自定义配置文件

除了命令行参数，还可以通过配置文件更精细地控制算法参数：

```bash
python run_pipeline.py --config my_custom_config.json
```

配置文件示例：

```json
{
  "time_horizon": 24,
  "time_step": 1,
  "aggregation_method": "LP",
  "trading_method": "bidding",
  "disaggregation_method": "proportional",
  "num_managers": 4,
  "users_per_manager": [6, 10, 8, 12],
  "device_config_file": "data/device_config_36users.csv",
  "price_data_file": "data/grid_price.csv",
  "results_dir": "results",
  "model_config": {
    "time_horizon": 24,
    "time_step": 1,
    "optimization_type": "battery_type_0.55",
    "heat_pump_strategy": "simple",
    "use_convex_optimization": true
  }
}
```

## 调试和日志

要查看更详细的日志输出，可以在运行`test_model_based.py`时添加`--verbose`参数：

```bash
python test_model_based.py --verbose
```

这将显示各算法的详细运行信息，包括聚合率、分解精度等。 
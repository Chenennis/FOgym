# ModelBased FlexOffer Pipeline 使用指南

## 概述

ModelBased FlexOffer Pipeline 是一个基于物理模型的灵活性报价（FlexOffer）生成、聚合、交易和分解的完整流程系统。区别于基于强化学习的MARL方法，本系统采用传统的基于模型的控制方法，通过物理约束和优化算法来实现能源灵活性的管理。

## 特点

- **纯物理模型**：使用物理模型（如电池、热泵等）进行设备状态预测和控制，没有使用强化学习（RL）概念
- **完整FlexOffer流程**：涵盖生成、聚合、交易、分解和奖励计算的完整流程
- **多种算法组合**：支持多种聚合、交易和分解算法的自由组合
- **结果可比性**：生成与MARL方法可比较的奖励指标
- **设备多样性**：支持电池、热泵、电动汽车等多种设备类型
- **可扩展性**：易于添加新的设备模型和算法

## 基本使用

### 快速开始

从命令行运行基本ModelBased Pipeline有几种方法：

**方法1：使用专门的运行脚本（推荐）**

```bash
cd algorithms/Model_based/fomodelbased
python run_pipeline.py
```

**方法2：直接运行model_based_pipeline.py（可能会出现导入错误）**

```bash
cd algorithms/Model_based/fomodelbased
python model_based_pipeline.py
```

**方法3：将文件夹作为模块运行（适合包式导入）**

```bash
# 需要在项目根目录执行
python -m algorithms.Model_based.fomodelbased
```

默认情况下，这将使用LP聚合、bidding交易和proportional分解方法运行pipeline。

### 命令行选项

通过命令行参数自定义pipeline：

```bash
python run_pipeline.py \
  --config path/to/config.json \        # 配置文件（可选）
  --timesteps 24 \                      # 运行的时间步数
  --aggregation LP \                    # 聚合方法：LP或DP
  --trading bidding \                   # 交易方法：bidding或market-clearing
  --disaggregation proportional \       # 分解方法：proportional或average
  --managers 4 \                        # Manager数量
  --users 6,10,8,12                     # 每个Manager的用户数（逗号分隔）
```

### 算法比较

比较不同算法组合的性能：

```bash
python run_modelbased_comparison.py \
  --timesteps 24 \                      # 时间步数
  --managers 4 \                        # Manager数量
  --users 6,10,8,12 \                   # 每个Manager的用户数
  --output results/my_comparison \      # 输出目录
  --agg LP,DP \                         # 要比较的聚合方法
  --trade bidding,market-clearing \     # 要比较的交易方法
  --disagg proportional,average         # 要比较的分解方法
```

这将运行所有算法组合（2×2×2=8种组合）并生成比较结果和图表。

### 通过Python API使用

在代码中使用ModelBased Pipeline：

```python
import sys
import os
# 添加fomodelbased目录到Python路径
sys.path.append('/path/to/algorithms/Model_based/fomodelbased')

from config import PipelineConfig
from model_based_pipeline import ModelBasedPipeline

# 创建配置
config = PipelineConfig(
    aggregation_method="LP",
    trading_method="bidding",
    disaggregation_method="proportional",
    num_managers=4,
    users_per_manager=[6, 10, 8, 12],
    time_horizon=24
)

# 创建并运行pipeline
pipeline = ModelBasedPipeline(config)
results = pipeline.run(num_timesteps=24)

# 处理结果
total_reward = sum(results['total_rewards'])
print(f"总奖励: {total_reward}")
```

## 算法组合

ModelBased Pipeline支持多种算法组合，以下是可用的算法及其特点：

### 聚合方法

1. **线性规划（LP）**：
   - 简单地将各个FlexOffer的能量轮廓相加
   - 时间灵活性采用加权平均值
   - 计算速度快，适合大规模系统

2. **动态规划（DP）**：
   - 考虑时间灵活性，对FlexOffer进行加权聚合
   - 时间灵活性高的FlexOffer获得更高权重
   - 能更好地利用灵活性潜力

### 交易方法

1. **投标（Bidding）**：
   - 根据时间灵活性，寻找最佳调度以最大化收益
   - 计算每个时间步的投标/要价，与市场价格比较
   - 更加主动地适应价格波动

2. **市场出清（Market-clearing）**：
   - 尽量将用电安排在价格低的时段，发电安排在价格高的时段
   - 通过移动能量轮廓来优化调度
   - 更加直接地响应价格信号

### 分解方法

1. **比例分解（Proportional）**：
   - 根据设备在原始聚合中的占比进行分解
   - 保持各设备的相对贡献比例
   - 更加精确地反映各设备的实际需求

2. **平均分解（Average）**：
   - 按设备类型分组，然后平均分配
   - 每类设备内部均分能量分配
   - 更加公平地分配能源资源

## Pipeline架构

ModelBased Pipeline的完整流程包括以下步骤：

1. **FlexOffer生成**：
   - 使用物理设备模型（如电池、热泵）生成初始FlexOffer
   - 每个FlexOffer包含能量轮廓和时间灵活性

2. **聚合**：
   - 将同一Manager管理的设备FlexOffer聚合成一个聚合FlexOffer
   - 可选LP或DP聚合方法

3. **交易**：
   - 聚合FlexOffer在市场中进行交易
   - 可选bidding或market-clearing交易方法
   - 输出调度计划和收益

4. **分解**：
   - 将交易后的调度计划分解回各个设备
   - 可选proportional或average分解方法

5. **奖励计算**：
   - 基于用户满意度（调度与原始需求的相似度）
   - 基于交易收益
   - 综合计算最终奖励

## 设备模型

ModelBased Pipeline支持多种设备类型，每种设备都有其物理模型：

1. **电池（Battery）**：
   - 状态：荷电状态（SOC）
   - 控制：充放电功率
   - 约束：容量限制、充放电效率、最大功率

2. **热泵（Heat Pump）**：
   - 状态：室内温度
   - 控制：加热/制冷功率
   - 约束：温度范围、最大功率、热转换效率

3. **电动汽车（EV）**：
   - 状态：荷电状态（SOC）
   - 控制：充电功率
   - 约束：容量限制、充电效率、最大充电功率

## 结果分析

运行完成后，ModelBased Pipeline会生成以下结果：

1. **时间步详情**：每个时间步的奖励和配置信息
2. **Manager奖励**：每个Manager的奖励列表
3. **总奖励**：所有时间步的奖励总和
4. **配置信息**：运行时使用的配置参数
5. **统计信息**：包括设备类型分布、总奖励等统计数据

结果将保存在指定目录下，可以通过CSV文件和JSON文件进行访问和进一步分析。

## 与MARL方法比较

ModelBased Pipeline生成的奖励数据格式与MARL方法兼容，方便进行比较：

1. **奖励计算机制相同**：
   - 用户满意度（调度与原始需求的相似度）
   - 交易收益（市场交易获得的收入）

2. **相同配置条件**：
   - 同样的用户和设备数量
   - 相同的时间范围
   - 相同的价格数据

通过比较不同方法的奖励，可以评估传统Model-based方法与MARL方法的性能差异。 
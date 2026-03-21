# FOgym Revision — 实验设计文档

本文档详细描述revision需要补充的所有实验，包括实验目的、配置、运行步骤和结果记录格式。

---

## 实验总览

| 实验编号 | 实验名称 | 目的 | 对应审稿意见 |
|----------|----------|------|-------------|
| EXP-1 | Reward权重Ablation | 展示各reward分量的边际贡献 | R1 + Meta Review |
| EXP-2 | 10-Manager扩展 | 验证可扩展性 | R2 + Meta Review |

**预计需要的总实验runs**: 5 (EXP-1: 4组 + EXP-2: 1组)

---

## EXP-1: Reward权重Ablation Study

### 目的
1. 回应Reviewer 1: "sensitivity of the performance to the weighting coefficients (α,β,δ,λ)"
2. 回应Reviewer 1: "How does the system behave if the coordination weight is set to zero?"
3. 回应Meta Review: "show how changing the reward weights (α,β,δ,λ) affects the system"

### 实验设计

#### 算法选择
- **SQDDPG + DMA**（表现最好的组合）
- 只用一个算法组合即可，不需要每个算法都跑ablation

#### 环境配置
- 与原论文相同：4 managers, 36 users, 118 devices
- 训练episodes: 与原论文相同（1000 episodes）
- 其他超参数保持不变

#### 实验配置

| 配置ID | 名称 | α (经济) | β (用户) | δ (约束) | λ (协调) |
|--------|------|----------|----------|----------|----------|
| A0 | Default (原论文值) | [填入原值] | [填入原值] | [填入原值] | [填入原值] |
| A1 | w/o Coordination | [同A0] | [同A0] | [同A0] | **0** |
| A2 | w/o Constraint | [同A0] | [同A0] | **0** | [同A0] |
| A3 | w/o User Satisfaction | [同A0] | **0** | [同A0] | [同A0] |

> **注意**：请先在代码中确认α,β,δ,λ的默认值并填入上表A0行。

#### 运行步骤

```
Step 1: 确认默认权重值
  - 在代码config中找到 α, β, δ, λ 的默认值
  - 填入上表A0行

Step 2: 运行A0 (Default)
  - 如果已有这组数据（就是论文里SQDDPG-DMA的结果），可以直接复用
  - 否则需要告知我

Step 3: 运行A1 (λ=0)
  - 修改config: λ = 0，其余不变
  - 训练 + 评估
  - 这是最关键的一组：展示cross-stage feedback的贡献

Step 4: 运行A2 (δ=0)
  - 修改config: δ = 0，其余不变
  - 训练 + 评估
  - 注意：[NEW]由于有hard projection，即使δ=0（没有约束惩罚reward），
    agent仍然不会违反物理约束，但训练效率和performance会下降

Step 5: 运行A3 (β=0)
  - 修改config: β = 0，其余不变
  - 训练 + 评估
```

#### 记录指标

每组实验需要记录以下指标（与原论文Table 1一致）：

| 指标 | 含义 | 公式 |
|------|------|------|
| Total Reward | 总奖励 | 训练收敛后的平均episode reward |
| A & DisA (g_a) | 聚合/解聚评分 | Eq.15 |
| Trading (g_t) | 交易评分 | Eq.16 |
| Eco (g_e) | 经济收益 (DKK) | Eq.14 |
| User Sat. (g_u) | 用户满意度 | Eq.13 |
| Constraint Violations | 约束违反次数 | 训练过程中raw action超限的次数 |
| Training Variance | 训练方差 | 最后100 episode的reward标准差 |
| Convergence Episode | 收敛episode | 首次达到90% final reward的episode |

#### 结果表格模板

```
Table X: Ablation study on reward weight coefficients using SQDDPG-DMA.

| Config     | Reward | A&DisA | Trading | Eco(DKK) | User | Constr.Viol.↓ |
|------------|--------|--------|---------|----------|------|---------------|
| Default    |        |        |         |          |      |               |
| w/o Coord  |        |        |         |          |      |               |
| w/o Constr |        |        |         |          |      |               |
| w/o User   |        |        |         |          |      |               |
```

#### 预期结果方向

- **A1 (w/o Coord, λ=0)**: Trading score显著下降，因为agent无法感知下游市场反馈。
  Total reward也会下降。这证明cross-stage feedback的必要性。
- **A2 (w/o Constr, δ=0)**: Constraint violations增加（raw action层面），但由于[NEW]hard projection，
  物理约束不会被违反。Training可能更不稳定，收敛更慢。
- **A3 (w/o User, β=0)**: User satisfaction显著下降，经济指标可能上升（全力优化成本而忽视用户）。
  这展示了经济效率与用户满意度之间的trade-off。

---

## EXP-2: 10-Manager 扩展实验

### 目的
1. 回应Reviewer 1: "how the Cross-manager Information Layer scales when there are hundreds of managers"
2. 回应Meta Review: "Discuss the scalability issues and address what happens when there are more managers"
3. 用实际数据支撑可扩展性讨论

### 实验设计

#### 环境配置

| 参数 | 4-Manager (原) | 10-Manager (新) |
|------|----------------|-----------------|
| Managers | 4 | 10 |
| Users | 36 | ~90（按比例） |
| Devices | 118 | ~295（按比例） |
| 设备类型比例 | 保持相同 | 保持相同 |
| Episode length | 24h | 24h |
| Training episodes | 1000 | 1000 |

> **注意**：你说你已有10-Manager的数据。如果10M环境的用户/设备配置与上表不同，
> 请按实际配置填写。

#### Manager分配建议（10M配置）

如果按原有比例放大：
```
Manager 1: 6 users     →  保持或调整
Manager 2: 10 users    →  保持或调整
Manager 3: 8 users     →  保持或调整
Manager 4: 12 users    →  保持或调整
Manager 5-10: 新增，分配剩余users
```

#### 算法选择
- 主要用 **SQDDPG + DMA**（与EXP-1一致，方便对比）
- 如果时间允许，也可以跑 **MAPPO + DMA** 作为对比（on-policy vs off-policy在scalability上的差异）

#### 运行步骤

```
Step 1: 确认10-Manager环境配置
  - 确认用户数、设备数、manager分配
  - 确认observation维度变化：
    * o_m^h 维度从 3×state_dim → 9×state_dim（增加了200%）
    * 总obs维度 = dim(o_m^p) + dim(o_m^g) + dim(o_m^h)
    * 计算并记录4M和10M的obs维度差异

Step 2: 运行 SQDDPG + DMA (10-Manager)
  - 如果已有数据可直接使用
  - 记录训练过程和最终指标

Step 3: (可选) 运行 MAPPO + DMA (10-Manager)
  - 用于对比on-policy算法的scalability
```

#### 记录指标

| 指标 | 4M | 10M | 说明 |
|------|-----|------|------|
| Obs维度 (o_m^p) | | | Private层不变 |
| Obs维度 (o_m^g) | | | Public层不变 |
| Obs维度 (o_m^h) | | | Cross-manager层增大 |
| 总Obs维度 | | | 汇总 |
| Action维度 | | | = 5 × avg(|D_m|) |
| Total Reward | | | 绝对值 |
| Per-user Reward | | | Reward / N_users（公平比较） |
| Per-user Eco (DKK) | | | 人均经济收益 |
| User Satisfaction | | | 平均满意度 |
| Trading Score | | | 交易效率 |
| Training Time/Episode | | | 每episode训练时间 |
| Convergence Episode | | | 收敛所需episode |
| Training Variance | | | 最后100 ep方差 |

#### 结果表格模板

```
Table X: Scalability comparison between 4-manager and 10-manager configurations.

| Metric              | 4-Manager | 10-Manager | Change    |
|---------------------|-----------|------------|-----------|
| Obs dim (total)     |           |            |           |
| Action dim (avg)    |           |            |           |
| Total Reward        |           |            |           |
| Per-user Reward     |           |            | ↑↓ X%     |
| Per-user Eco (DKK)  |           |            | ↑↓ X%     |
| User Satisfaction   |           |            |           |
| Trading Score       |           |            |           |
| Time/Episode (s)    |           |            | ↑ X%      |
| Convergence Ep.     |           |            |           |
```

#### 预期结果方向

- **Obs维度增长**：o_m^h从 3×d → 9×d，总obs维度增加，但o_m^p和o_m^g不变
- **Per-user指标**：预期保持可比或略有下降（agent需要协调更多manager）
- **训练时间**：预期增加（更大的网络输入、更多agent需要计算Shapley值）
- **收敛速度**：可能需要更多episodes才能收敛
- **关键结论**：即使M从4→10，系统仍能有效运行，但观测维度和计算开销线性增长。
  对于M>>10，需要引入mean-field或attention等压缩手段。

---

## 实验执行优先级

```
1. [首先] 确认代码中reward权重默认值 α, β, δ, λ
2. [首先] 确认代码中是否有action projection（如无，需先实现）
3. [然后] 跑 EXP-1: A1 (λ=0) ← 审稿人最关心的一组
4. [然后] 跑 EXP-1: A2 (δ=0) 和 A3 (β=0)
5. [并行] 跑 EXP-2: 10-Manager（如已有数据则整理即可）
6. [最后] 整理结果，填入表格
```

---

## 实验完成后的论文写作计划

### EXP-1结果 → 添加到论文Section 6

位置：Main Results之后，Conclusion之前，新增一个小节 **"Ablation Study on Reward Components"**

内容结构：
1. 一句话介绍ablation目的
2. Table（上面的结果表格）
3. 3段分析（每组ablation一段）
4. 一句总结：各reward分量都有不可替代的作用，cross-stage feedback (r_o) 对end-to-end优化至关重要

预计篇幅：~0.5-0.75 column

### EXP-2结果 → 添加到论文Section 5.2 + Section 6

Section 5.2（Cross-manager Info Layer之后）：
- 讨论维度增长 + 10M实验结果简要提及
- 讨论更大规模的解决方案（mean-field, attention, graph）

Section 6（可选，如果空间够）：
- 完整的对比表格

预计篇幅：~0.3-0.5 column

---

## 实验结果记录区（跑完后填写）

### EXP-1 Results

A0 (Default): α=___, β=___, δ=___, λ=___
| Reward | A&DisA | Trading | Eco | User | Constr.Viol |
|--------|--------|---------|-----|------|-------------|
|        |        |         |     |      |             |

A1 (w/o Coord, λ=0):
| Reward | A&DisA | Trading | Eco | User | Constr.Viol |
|--------|--------|---------|-----|------|-------------|
|        |        |         |     |      |             |

A2 (w/o Constraint, δ=0):
| Reward | A&DisA | Trading | Eco | User | Constr.Viol |
|--------|--------|---------|-----|------|-------------|
|        |        |         |     |      |             |

A3 (w/o User, β=0):
| Reward | A&DisA | Trading | Eco | User | Constr.Viol |
|--------|--------|---------|-----|------|-------------|
|        |        |         |     |      |             |

### EXP-2 Results

4-Manager (from original paper):
| Obs_dim | Reward | Per-user Reward | Per-user Eco | User | Trading | Time/Ep |
|---------|--------|-----------------|--------------|------|---------|---------|
|         |        |                 |              |      |         |         |

10-Manager:
| Obs_dim | Reward | Per-user Reward | Per-user Eco | User | Trading | Time/Ep |
|---------|--------|-----------------|--------------|------|---------|---------|
|         |        |                 |              |      |         |         |

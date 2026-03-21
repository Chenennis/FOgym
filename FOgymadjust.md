# FOgym Revision — Design Decisions & Code Adjustments

## R3: Hard Constraint Enforcement Mechanism

### Design: Two-Layer Safety Mechanism

FOgym采用两层安全机制来保证约束满足：

#### Layer 1: Reward-based Soft Constraint (训练引导，已有)
现有的 $r_c(t)$ 惩罚项，在训练过程中引导agent远离约束边界。
这是一个"软"约束——agent可以违反，但会被惩罚。

#### Layer 2: Action Projection (硬约束，需确认/补充到代码中)
Agent输出的raw action在执行前，经过一个确定性的projection step，
强制裁剪到设备的物理可行域内。**无论agent输出什么，最终执行的action一定合法。**

具体projection规则（按设备类型）：

```
对每个设备 d，agent输出 raw_action_d:

1. Home Battery:
   - power = clip(raw_power, -p_max, p_max)          # 充放电功率限制
   - 如果执行后 SOC_next < SOC_min(0.1):
       power = 使SOC_next = SOC_min的最小功率        # 防止过放
   - 如果执行后 SOC_next > SOC_max(0.9):
       power = 使SOC_next = SOC_max的最大功率        # 防止过充

2. Heat Pump:
   - power = clip(raw_power, 0, P_max)               # 功率非负且不超上限
   - 如果执行后 T_next > T_max(26°C):
       power = 0                                       # 过热则停机
   - 如果执行后 T_next < T_min(18°C):
       power = 维持T_min所需的最小功率                 # 防止过冷

3. Electric Vehicle:
   - power = clip(raw_power, 0, p_charge_max)         # 只充不放，功率限制
   - 如果 EV未连接: power = 0                          # 未插电不能充
   - 如果执行后 SOC_next > SOC_max(0.9):
       power = 使SOC_next = SOC_max的功率              # 防止过充
   - 如果执行后 SOC_next < SOC_min(0.1):
       power = 0（并保持当前SOC）                      # 防止过放

4. Dishwasher:
   - 二值动作: action = 1 (启动) 或 0 (不启动)
   - 如果已在运行中: 忽略action，继续运行              # 不可中断
   - 如果未到ready状态: action = 0                     # 未准备好不能启动

5. PV System:
   - 无可控动作，被动发电                               # 无需projection
```

#### 设计关键点

1. **Projection独立于reward**：即使agent学会了"打擦边球"，projection也会强制裁剪，
   物理上不可能违反约束。

2. **Projection对训练的影响**：被clip的action会产生较低的reward（因为r_c惩罚 +
   实际执行效果偏离agent意图），agent会逐渐学会输出合法范围内的action。

3. **Projection是确定性的**：不引入额外随机性，不影响MARL训练的收敛性。

4. **FO参数的projection**：除了设备层面的物理约束，FO参数也需要projection：
   - e_min = clip(e_min, device_e_min, device_e_max)
   - e_max = clip(e_max, e_min, device_e_max)        # 保证 e_max >= e_min
   - t_start, t_end = clip到设备可用时间窗口内

### 需要检查/调整的代码位置

- [ ] 环境的 `step()` 函数中，在执行action之前是否有projection逻辑
- [ ] 每种设备的 `apply_action()` 或类似方法中是否有clip
- [ ] FO生成时，FO参数是否经过合法性检查

### 论文中对应的文字描述（建议添加在Section 5.1）

建议在 "Unified Action Space" 段落之后添加如下内容：

---

**Constraint Enforcement.** To guarantee that all device operations remain within
physical and safety bounds, FOgym employs a two-layer constraint enforcement
mechanism. The first layer operates during training through the reward penalty
$r_c$ (Eq. 13), which discourages agents from producing actions near constraint
boundaries by imposing negative rewards proportional to the magnitude of
violation. This soft constraint guides policy learning toward safe operating regions.

The second layer is a deterministic action projection step applied before
environment execution. Each agent's raw output is projected onto the device-specific
feasible region:
$$a_d^{\text{exec}} = \text{Proj}_{\mathcal{C}_d}(a_d^{\text{raw}})$$
where $\mathcal{C}_d$ denotes the feasible action set for device $d$, defined by
its operational constraints (e.g., SOC $\in$ [0.1, 0.9] for batteries,
temperature $\in$ [18°C, 26°C] for heat pumps, charging-only mode for EVs).
For continuous actions, this projection reduces to component-wise clipping;
for state-dependent constraints (e.g., preventing battery SOC from exceeding bounds),
the projection computes the maximum feasible action given the current device state.

This two-layer design ensures that (i) no physically invalid action is ever executed,
regardless of the learned policy, and (ii) agents cannot exploit safety boundaries
to maximize rewards, since the projection is applied independently of and prior to
the reward computation.

---

## R2: Scalability — 10-Manager Experiment + Discussion

### 实验设计
- 在现有4-manager基础上，增加10-manager配置
- 对比指标：total reward, 训练收敛速度, 每episode训练时间
- 用SQDDPG-DMA配置运行

### 论文中对应的文字描述（建议添加在Section 5.2 Cross-manager Info Layer之后）

---

**Scalability Analysis.** The cross-manager information layer $o_m^h = s_{-m}^a$
introduces an observation dimension that grows linearly with the number of
managers $M$, as each manager receives aggregated state information from all
$M-1$ peers. To empirically evaluate scalability, we extend our experiments
from 4 managers to 10 managers (with proportionally increased users and devices).
[此处插入实验结果的描述，如: Table X shows that FOgym maintains effective
coordination with 10 managers, achieving comparable per-user satisfaction
and economic benefit, though training time increases by approximately X%.]

For deployments with significantly larger $M$ (e.g., hundreds of managers),
the linear growth of observation dimensionality may become a bottleneck.
Several established techniques from the MARL literature can address this:
(i) \textit{mean-field approximation} \cite{meanfield}, which replaces
individual manager states with distributional statistics (e.g., mean and
variance), reducing the observation to a fixed-dimensional summary regardless
of $M$; (ii) \textit{attention mechanisms} \cite{attention_marl}, which allow
each manager to selectively attend to the most relevant peers rather than
observing all; and (iii) \textit{graph-based communication} \cite{graph_marl},
which restricts information exchange to topologically adjacent managers in
the distribution network. Integrating these mechanisms into FOgym is a
promising direction for future work.

---

## R1: Ablation Study — Reward Weights

### 实验配置

| 配置名 | α (经济) | β (用户) | δ (约束) | λ (协调) | 目的 |
|--------|----------|----------|----------|----------|------|
| Default | 原值 | 原值 | 原值 | 原值 | 基线 |
| w/o Coord (λ=0) | α | β | δ | **0** | 验证cross-stage feedback贡献 |
| w/o Constraint (δ=0) | α | β | **0** | λ | 验证约束惩罚的作用 |
| w/o User (β=0) | α | **0** | δ | λ | 验证用户满意度项的作用 |

用 SQDDPG + DMA 配置运行

### 结果表格结构

| Configuration | Reward | A&DisA | Trading | Eco(DKK) | User Sat. | Constraint Viol.↓ |
|---------------|--------|--------|---------|----------|-----------|-------------------|
| Default       |        |        |         |          |           |                   |
| w/o Coord     |        |        |         |          |           |                   |
| w/o Constraint|        |        |         |          |           |                   |
| w/o User      |        |        |         |          |           |                   |

### 论文中对应的文字描述（建议添加在Section 6，Main Results之后）

---

**Ablation Study on Reward Components.**
To analyze the sensitivity of FOgym to the reward weighting coefficients
and to quantify the marginal contribution of each reward component, we conduct
an ablation study by selectively disabling individual reward terms.
Table~\ref{tab:ablation} presents results using SQDDPG with the DMA
pipeline configuration.

[根据实际实验结果填写分析，预期方向如下:]

Removing the coordination reward ($\lambda=0$) leads to a significant
degradation in trading score and overall reward, as generation agents can
no longer perceive downstream market outcomes. This confirms that the
cross-stage feedback mechanism is essential for end-to-end optimization.

Disabling the constraint penalty ($\delta=0$) results in increased constraint
violations during training, though the hard projection layer prevents
actual safety breaches. However, the increased frequency of action clipping
degrades overall performance as agents spend more exploration budget on
infeasible regions.

Setting the user satisfaction weight to zero ($\beta=0$) substantially
reduces user satisfaction scores while improving economic metrics, indicating
a clear trade-off between cost optimization and user comfort that operators
must balance based on deployment priorities.

---

## R4: Stage Independence Clarification

### 建议添加位置：Section 5.2 开头

---

We note that while each module in the FO life-cycle is implemented as an
independently replaceable component (Section~\ref{sec:lifecycle}), the stages
are not optimized in isolation. The cross-stage feedback mechanism explicitly
couples generation decisions with downstream outcomes: the observation space
incorporates market signals and inter-manager states (Eq.~\ref{eq2}), while
the reward function propagates trading success and user satisfaction back to
the generation stage (Eq.~\ref{eq4}). This design ensures that upstream agents
learn to anticipate downstream consequences, achieving end-to-end coordination
despite the modular architecture.

---

## R5: Adversarial Setting Discussion

### 建议添加位置：Section 7 (Conclusion) 扩展future work

---

While FOgym currently operates under a cooperative paradigm where all managers
share a common reward signal, real-world deployments may involve managers with
competing interests. In such adversarial or mixed-motive settings, managers
could strategically manipulate bids or withhold flexibility to gain individual
advantage, potentially destabilizing the market. The Shapley-value-based credit
assignment in SQDDPG partially mitigates this by ensuring equitable reward
distribution, reducing incentives for free-riding. Extending FOgym to support
mixed cooperative-competitive settings, incorporating mechanism design principles
to ensure incentive compatibility, is an important direction for future work.

---

## R6: Generalizability Beyond FO

### 建议添加位置：Section 7 (Conclusion)

---

Although FOgym is designed around the FO model, its core components — the
Dec-POMDP formulation, cross-stage feedback mechanism, and modular pipeline
architecture — are not FO-specific. The framework can be adapted to other
flexibility representation models (e.g., demand response programs, virtual
power plant coordination) by replacing the FO-specific action space mapping
and constraint definitions while retaining the multi-stage optimization structure.

---

## R7: Appendix A Reference

### 建议修改位置：Section 4.1，在现有EV example之后

将现有的：
> "A concrete example illustrating how an EV participates in the complete FO life-cycle is provided in Appendix~\ref{appendix:example_lifecycle}."

无需修改，已经足够。或者可以加一句强调：

> "We strongly encourage readers to refer to the detailed end-to-end example
> in Appendix~\ref{appendix:example_lifecycle}, which illustrates how a single
> EV traverses all four stages of the FO life-cycle."

---

## Camera-Ready Housekeeping

- [ ] `\documentclass[sigconf, anonymous]{acmart}` → `\documentclass[sigconf]{acmart}`
- [ ] 更新作者信息
- [ ] 更新copyright/year
- [ ] 更新repository URL（去掉anonymous链接）

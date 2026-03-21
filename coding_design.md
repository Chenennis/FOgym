# FOgym Coding Design — 代码与论文一致性参考文档

本文档基于论文中的所有公式、定义和描述，精确列出FOgym系统的每个组件的设计规格。
用于对照检查代码实现是否与论文一致。标记 **[NEW]** 的部分是revision新增设计，需要在代码中补充。

---

## 1. 系统总体架构

### 1.1 环境配置（论文Section 6.1）

| 参数 | 值 |
|------|------|
| Manager数量 (M) | 4（现有），10（扩展实验） |
| 用户数量 | 36（4M配置） |
| 设备总数 | 118 |
| 时间步长 | 1小时 |
| Episode长度 (T) | 24步（一天） |
| 折扣因子 γ | 0.99 |

Manager分配（4M配置）：
- Manager 1: 6 users
- Manager 2: 10 users
- Manager 3: 8 users
- Manager 4: 12 users

每个用户平均管理 3.3 个设备。

### 1.2 Pipeline流程

```
Generation → Aggregation → Trading → Scheduling
    ↑                                      |
    └──── cross-stage feedback (obs+reward) ←┘
```

每个timestep t 的执行顺序（对应 Algorithm 1）：
```
1. 对每个设备 d：更新设备状态 s_d(t+1) 基于控制信号 C_d(t)
2. 更新市场价格 p_{t+1}、天气 w_{t+1}、用户满意度
3. 对每个manager m：构造三层observation o_m(t+1)
4. 计算reward r_m(t)
```

---

## 2. Dec-POMDP 公式化（论文Section 5.1, Eq.1）

```
⟨ M, S, {A_m}, {O_m}, T, R, γ ⟩
```

| 符号 | 含义 | 代码对应 |
|------|------|----------|
| M = {1,...,M} | Manager agent集合 | agent列表 |
| S | 全局状态空间（agent不可观测） | env内部state |
| A_m ⊆ R^{d_m} | Agent m的连续动作空间 | actor网络输出 |
| O_m ⊆ R^{o_m} | Agent m的观测空间 | obs向量 |
| T: S×A → Δ(S) | 状态转移函数 | env.step() |
| R: S×A → R | 共享reward函数 | reward计算 |
| γ ∈ [0,1] | 折扣因子 = 0.99 | config |

---

## 3. Observation Space 详细设计（论文Section 5.2, Eq.7）

### 总体结构
```
o_m(t) = [o_m^p(t), o_m^g(t), o_m^h(t)]
```

### 3.1 Private Information Layer: o_m^p(t)

```
o_m^p(t) = { s_m^d(t), u_m^p(t), a_m^h(t) }
```

| 分量 | 含义 | 具体内容 |
|------|------|----------|
| s_m^d(t) | 设备状态 | 见下方各设备state |
| u_m^p(t) | 用户偏好 | 充电完成时间、舒适温度范围等 |
| a_m^h(t) | 历史动作 | 上一时步的action（时序一致性） |

**各设备状态 s_m^d(t) 的具体维度：**

**Home Battery:**
| 维度 | 含义 | 范围 |
|------|------|------|
| SOC | 当前荷电状态 | [0.1, 0.9] |
| charge_rate | 充放电速率 | [-p_max, p_max], p∈[3.0, 7.0] kW |
| efficiency | 充放电效率 η | [0.8, 0.95] |
| temperature | 电池温度 | — |

**Heat Pump:**
| 维度 | 含义 | 范围 |
|------|------|------|
| T_current | 当前室温 | [18.0, 26.0] °C |
| T_setpoint | 用户设定温度 | 21 °C |
| COP | 性能系数 | [3.0, 5.0] |
| T_outdoor | 室外温度 | 来自天气数据 |

**Electric Vehicle:**
| 维度 | 含义 | 范围 |
|------|------|------|
| SOC | 当前荷电状态 | [0.1, 0.9] |
| is_connected | 是否插电 | {0, 1} |
| t_departure | 离开时间 | (0, 1) 归一化 |
| SOC_target | 目标SOC | 用户设定 |

**Dishwasher:**
| 维度 | 含义 | 范围 |
|------|------|------|
| is_ready | 是否已装载就绪 | {0, 1} |
| is_running | 是否在运行 | {0, 1} |
| progress | 运行进度 | [0, 1] |
| priority ε | 用户优先级 | {1,2,3,4,5} |
| time_delay t_l | 部署后延迟 | [0.5, 6.0] h |

**PV System:**
| 维度 | 含义 | 范围 |
|------|------|------|
| current_output | 当前发电功率 | [0, p_max], p∈[3.0, 10.0] kW |
| efficiency η_g | 发电效率 | [0.15, 0.22] |
| weather_dependent | 是否依赖天气 | True |

### 3.2 Public Information Layer: o_m^g(t)

```
o_m^g(t) = { t, m_p(t), w(t) }
```

| 分量 | 含义 | 具体内容 |
|------|------|----------|
| t | 时间特征 | hour_of_day (0-23), day_of_week (0-6) |
| m_p(t) | 市场价格 | 当前价格 + 预测价格 |
| w(t) | 天气状况 | 温度, 太阳辐照度 |

**注意**：所有manager共享相同的 o_m^g(t)。

### 3.3 Cross-manager Information Layer: o_m^h(t)

```
o_m^h(t) = s_{-m}^a
```

| 分量 | 含义 | 具体内容 |
|------|------|----------|
| s_{-m}^a | 其他所有manager的聚合状态 | 除m外所有manager的设备状态聚合信息 |

**代码检查要点**：
- [ ] 确认是传递所有其他manager的原始观测，还是传递聚合统计量
- [ ] 维度 = (M-1) × 每个manager的聚合状态维度
- [ ] 4M配置下：3 × state_dim_per_manager
- [ ] 10M配置下：9 × state_dim_per_manager（维度增大）

### 3.4 Observation维度汇总

需要在代码中确认总obs维度 = dim(o_m^p) + dim(o_m^g) + dim(o_m^h)

---

## 4. Action Space 详细设计（论文Section 5.1, Eq.2）

### 4.1 统一动作表示

每个manager m 的action为其管辖所有设备生成FO参数：
```
a_m(t) = { (f_d^ts, f_d^te, e_d^min, e_d^max, w_d) | d ∈ D_m }
```

| 参数 | 含义 | 范围 | 说明 |
|------|------|------|------|
| f_d^ts | 开始时间灵活性因子 | [-1, 1] | -1=最早可行时间, 1=最晚 |
| f_d^te | 结束时间灵活性因子 | [-1, 1] | 同上 |
| e_d^min | 最小能量因子 | [0.1, 1.0] | 乘以设备的基准能量 |
| e_d^max | 最大能量因子 | [1.0, 2.0] | 乘以设备的基准能量 |
| w_d | 优先级权重 | [0.1, 2.0] | 调度时的分配权重 |

**每个设备5维 → agent动作维度 = 5 × |D_m|**

### 4.2 **[NEW] 两层约束执行机制**

Agent输出的raw action在执行前必须经过两层约束处理：

#### Layer 1: Reward惩罚 r_c（软约束，训练引导）
- 已有机制，见Section 5中Eq.13
- 对超出设备限制的action施加负reward

#### Layer 2: Action Projection（硬约束，执行保障）

```python
def project_action(raw_action, device_state, device_params):
    """
    在env.step()中，action执行前调用。
    保证：无论agent输出什么，最终执行的action一定在物理可行域内。
    """
    projected = raw_action.copy()

    for device d in devices:
        if device_type == "Battery":
            # 1. 功率裁剪
            projected[d].power = clip(raw.power, -p_max, p_max)
            # 2. SOC安全检查
            soc_next = current_soc + projected[d].power * dt * η / capacity
            if soc_next < SOC_MIN (0.1):
                projected[d].power = (SOC_MIN - current_soc) * capacity / (dt * η)
            if soc_next > SOC_MAX (0.9):
                projected[d].power = (SOC_MAX - current_soc) * capacity / (dt * η)

        elif device_type == "HeatPump":
            # 1. 功率裁剪（非负）
            projected[d].power = clip(raw.power, 0, P_max)
            # 2. 温度安全检查
            T_next = T_current + (projected[d].power * COP - heat_loss) / C_h
            if T_next > T_MAX (26°C):
                projected[d].power = 0  # 过热停机
            if T_next < T_MIN (18°C):
                # 计算维持T_MIN所需最小功率
                projected[d].power = (T_MIN - T_current) * C_h + heat_loss) / COP
                projected[d].power = clip(above, 0, P_max)

        elif device_type == "EV":
            # 1. 连接检查
            if not is_connected:
                projected[d].power = 0
                continue
            # 2. 功率裁剪（只充不放）
            projected[d].power = clip(raw.power, 0, p_charge_max)
            # 3. SOC安全检查
            soc_next = current_soc + projected[d].power * dt * η / capacity
            if soc_next > SOC_MAX (0.9):
                projected[d].power = (SOC_MAX - current_soc) * capacity / (dt * η)

        elif device_type == "Dishwasher":
            # 二值动作 + 状态检查
            if is_running:
                projected[d].action = CONTINUE  # 不可中断
            elif not is_ready:
                projected[d].action = 0  # 未就绪
            else:
                projected[d].action = round(raw.action)  # 0或1

        elif device_type == "PV":
            pass  # 无可控动作

    return projected
```

**FO参数的projection：**
```python
def project_fo_params(raw_fo, device_params):
    """FO参数也需要projection到合法范围"""
    fo = raw_fo.copy()

    # 能量bounds
    fo.e_min = clip(raw_fo.e_min, device.e_min_absolute, device.e_max_absolute)
    fo.e_max = clip(raw_fo.e_max, fo.e_min, device.e_max_absolute)  # 保证 e_max >= e_min

    # 时间bounds
    fo.t_start = clip(raw_fo.t_start, device.earliest_start, device.latest_start)
    fo.t_end = clip(raw_fo.t_end, fo.t_start + min_duration, device.latest_end)

    # 优先级权重
    fo.w = clip(raw_fo.w, 0.1, 2.0)

    return fo
```

**代码检查要点**：
- [ ] `env.step()` 中是否在执行action前调用了projection
- [ ] 每种设备是否都有对应的clip/projection逻辑
- [ ] FO生成时的参数是否经过合法性检查
- [ ] projection后的action是否用于reward计算（应该用projected action计算r_e, r_u等，用raw action计算r_c惩罚）

---

## 5. Reward Function 详细设计（论文Section 5.2, Eq.10-13）

### 5.1 总体reward

```
r_m(t) = α·r_e(t) + β·r_u(t) + δ·r_c(t) + λ·r_o(t)
```

权重 α, β, δ, λ ∈ [0,1]

**代码检查要点**：
- [ ] 确认代码中这四个权重的默认值是多少（论文未给出具体数值，需要从代码中确认）
- [ ] 确认权重可以通过config修改（ablation实验需要）

### 5.2 经济效率 r_e(t)（Eq.11）

```
r_e(t) = -Σ_d [ C_d^e(a_t^d, m_p(t)) + C_d^η(a_t^d, s_t^d) ]
```

| 分量 | 含义 | 计算方式 |
|------|------|----------|
| C_d^e | 设备d的能量成本 | action_d(kWh) × market_price(t) |
| C_d^η | 设备d的效率损失 | 取决于设备类型和当前状态 |

注意：负号将cost转为reward（最小化cost = 最大化reward）。

### 5.3 用户满意度 r_u(t)（Eq.12）

```
r_u(t) = Σ_u [ w_u · u_sa(s_t^u, u_p) ]
```

| 设备类型 | 满意度计算 u_sa |
|----------|----------------|
| EV | min(EV_c^soc / EV_g^soc, 1.0)，即当前SOC/目标SOC |
| HeatPump | 1.0 - |T_current - T_setpoint| / T_range |
| Battery | SOC维护满意度 |
| Dishwasher | 完成度 + 及时性 |

### 5.4 约束满足 r_c(t)（Eq.13）

```
r_c(t) = -Σ_d max(0, |a_t^d| - a_lim,d) - Σ_d C_s^d
```

| 分量 | 含义 |
|------|------|
| max(0, \|a_t^d\| - a_lim,d) | 超出设备最大允许动作幅度的惩罚 |
| C_s^d | 设备特定安全约束违反惩罚 |

设备安全约束 C_s^d：
| 设备 | 约束 |
|------|------|
| Battery | SOC ∈ [0.1, 0.9] |
| HeatPump | T ∈ [18°C, 26°C] |
| EV | 必须连接才能充电 |

**[NEW] 代码检查要点**：
- [ ] r_c 应该基于 raw action（projection前）计算，这样才能惩罚agent输出违规动作
- [ ] 实际执行使用 projected action，所以物理上不会违规
- [ ] 这个区分很重要：如果r_c基于projected action计算，agent永远看不到惩罚，就不会学到避免违规

### 5.5 多智能体协调 r_o(t)（Eq.14）

```
r_o(t) = Σ_m [ s_m^p(t) + η_m^k(t) + q_m^f(t) ]
```

| 分量 | 含义 | 取值 |
|------|------|------|
| s_m^p(t) | 交易成功指示 | {0, 1} |
| η_m^k(t) | 竞标效率 | 相对于市场价格的经济效率 |
| q_m^f(t) | 通信质量 | 代理间信息共享的质量 |

---

## 6. 各设备的Reward子结构（论文Appendix D）

### 6.1 Battery reward

```
r_b(t) = α_b·r_b^e(t) + β_b·r_b^η(t) + γ_b·r_b^l(t)
```
- r_b^e: 经济（充电成本）
- r_b^η: 效率（效率损失）
- r_b^l: 维护（SOC维持在~0.6附近）

### 6.2 HeatPump reward

```
r_h(t) = α_h·r_h^e(t) + β_h·r_h^c(t)
```
- r_h^e: 经济（能耗成本）
- r_h^c: 舒适度（温度偏差）

### 6.3 EV reward

```
r_ev(t) = α_ev·r_ev^e(t) + β_ev·r_ev^s(t) + γ_ev·r_ev^a(t)
```
- r_ev^e: 经济（充电成本）
- r_ev^s: 充电完成状态
- r_ev^a: 用户满意度（可用性）

### 6.4 Dishwasher reward

```
r_w(t) = α_w·r_w^c(t) + β_w·r_w^p(t) + γ_w·r_w^u(t) - θ_w·r_w^e(t)
```
- r_w^c: 完成度
- r_w^p: 运行进度
- r_w^u: 用户满意度
- r_w^e: 经济（注意是减号）

### 6.5 PV reward

```
r_v(t) = α_v·η_v·r_v^g(t)·r_v^p(t)
```
- η_v: 发电效率
- r_v^g: 发电量
- r_v^p: 市场价格（发电时价格高则reward高）

---

## 7. MARL算法集成（论文Section 5.1）

### 7.1 共同接口

所有算法必须遵循：
```
输入：observation o_m(t) ∈ R^{o_m}
输出：action a_m(t) ∈ R^{d_m}（d_m = 5 × |D_m|）
```

### 7.2 MAPPO

| 属性 | 值 |
|------|------|
| 类型 | On-policy, 共享策略 |
| 范式 | CTCE (集中训练集中执行) |
| Actor | 共享策略网络 π_θ(a\|o)，多头输出（每种设备一个头） |
| Critic | 共享价值网络 V_φ(o) |
| 特有loss | L_f（FO约束损失）+ L_t（市场反馈损失） |

**Loss函数**：
```
L^π = L_p + λ_v·L_v - λ_e·H + λ_f·L_f + λ_t·L_t
```
- L_p: PPO clipped surrogate objective
- L_v: value function loss
- H: entropy bonus
- L_f = Σ_d max(0, |a_d^t - t̄_d|) + max(0, |a_d^e - ē_d|)  ← FO时间和能量约束违反
- L_t = -Σ_m s_m^p · η_m^k  ← 交易成功×竞标效率

**超参数**（Appendix Table 6）：
| Batch | lr_actor | lr_critic | γ | GAE_λ | clip | epochs | mini_batch | v_coef | e_coef |
|-------|----------|-----------|------|-------|------|--------|------------|--------|--------|
| 256-512 | 3e-4 | 3e-4 | 0.99 | 0.95 | 0.2 | 10 | 4 | 0.5 | 0.01 |

### 7.3 MAIPPO

| 属性 | 值 |
|------|------|
| 类型 | On-policy, 独立策略 |
| 范式 | DTDE |
| Actor | 每个manager独立 π_θ_m(a\|o) |
| Critic | 每个manager独立 V_φ_m(o) |
| 特点 | 独立replay buffer, 自适应学习率 |

**自适应学习率**：
```
α_m^{t+1} = max(α_m^t · l_d, α_min)
```
其中 l_d = 0.995, α_min = 1e-5

**超参数**（Appendix Table 7）：
| Batch | lr_actor | lr_critic | γ | GAE_λ | clip | epochs | mini_batch | v_coef | e_coef | lr_decay | lr_min |
|-------|----------|-----------|------|-------|------|--------|------------|--------|--------|----------|--------|
| 256-512 | 3e-4 | 3e-4 | 0.99 | 0.95 | 0.2 | 10 | 4 | 0.5 | 0.01-0.02 | 0.995 | 1e-5 |

### 7.4 MADDPG

| 属性 | 值 |
|------|------|
| 类型 | Off-policy, 确定性策略 |
| 范式 | CTDE |
| Actor | 确定性策略 μ_θ(o), batch normalization |
| Critic | Q_φ(o,a)，集中式（训练时可见全局） |
| 探索 | OU noise (θ_n=0.15, σ_n=0.2) |

**FO约束正则化**：
```
L_c = L_q + λ_c · Σ_d max(0, |a_d| - ā_d)^2
```

**Target网络软更新**：
```
θ' ← τ·θ + (1-τ)·θ',  τ = 0.01
```

**超参数**（Appendix Table 8）：
| Batch | lr_actor | lr_critic | γ | τ | Buffer | θ_noise | σ_noise |
|-------|----------|-----------|------|------|--------|---------|---------|
| 256 | 1e-3 | 2e-3 | 0.99 | 0.01 | 1e6 | 0.15 | 0.2 |

### 7.5 MATD3

| 属性 | 值 |
|------|------|
| 类型 | Off-policy, 确定性策略 |
| 范式 | CTDE |
| Actor | 确定性策略 μ_θ(o), delayed更新 (每2步) |
| Critic | Twin Q网络 Q_φ1, Q_φ2, 取min |
| 探索 | Target policy smoothing + clipped noise |

**Q值计算**：
```
Q^T(s,a) = min(Q_1^T(s,a), Q_2^T(s,a))
```

**Target action smoothing**：
```
a' = π'(s') + clip(ε, -c, c),  ε ~ N(0, σ)
```

**超参数**（Appendix Table 9）：
| Batch | lr_actor | lr_critic | γ | τ | policy_noise | clip_noise | policy_freq |
|-------|----------|-----------|------|------|-------------|------------|-------------|
| 256-512 | 1e-3 | 2e-3 | 0.99 | 0.005 | 0.2 | 0.5 | 2 |

### 7.6 SQDDPG

| 属性 | 值 |
|------|------|
| 类型 | Off-policy, Shapley-based credit |
| 范式 | CTDE |
| Actor | 确定性策略 |
| Critic | Shapley Q-value |
| 特点 | 公平性credit assignment |

**Shapley值计算**：
```
φ_i = Σ_{S⊆M\{i}} [ |S|!(|M|-|S|-1)! / |M|! ] × [v(S∪{i}) - v(S)]
```

**Coalition value**:
```
v(S) = Q_joint · Σ_{i∈S} c_i / Σ_{i∈M} c_i
```

**Critic loss**：
```
L^S = E[(Q(o,a) - (r + γQ'(o',a')))^2] + λ_s · |φ_m - φ̂_m|^2
```

**超参数**（Appendix Table 10）：
| Batch | lr | γ | τ | Shapley_iter | Shapley_batch | Credit_weight |
|-------|-----|------|------|-------------|---------------|---------------|
| 256-512 | 1e-3 | 0.99 | 0.01 | 100 | 64 | 0.8 |

---

## 8. 生命周期模块（论文Section 5.3）

### 8.1 Generation Module

```
输入：observations o_m(t)
输出：device-level FOs {F_d(t)}
```

F_d(t) = { fs_{d,i(t)} }，其中每个slice:
```
fs_{d,i(t)} = ( [t_s^i, t_e^i], [e_min^i, e_max^i], p_d, c_d )
```

### 8.2 Aggregation Module

```
输入：individual FOs {F_d(t) | d ∈ D_m}
输出：aggregated FO F_m^A(t)
```

两种策略：
- **LP (Longest Profile)**: 最大化profile长度和能量，适合能量需求预测准确的场景
- **DP (Dynamic Profile)**: 最大化时间灵活性，适合时间不确定的场景

F_m^A(t) = ( [t_s^m, t_e^m], [e_min^m, e_max^m], P_m )
- e_min^m = Σ_{d∈D_m} e_min,d
- e_max^m = Σ_{d∈D_m} e_max,d

### 8.3 Trading Module

```
输入：aggregated FOs {F_m^A(t)} from all managers
输出：successful trades T(t)
```

Bid: B_m(t) = (E_m^b(t), p_m^b(t), k_m(t))
- E_m^b ∈ [e_min^m, e_max^m]
- p_m^b = p_b^t × (1 ± φ_a ± ψ_r ± b_m)
  - φ_a: 市场动态因子
  - ψ_r ∈ [-1.5%, 1.5%]: 可控随机性
  - b_m: manager特定偏差
- k_m ∈ {0, 1}: 0=卖, 1=买

两种方法：
- **Market Clearing**: 供需曲线交叉确定出清价格
- **Bidding**: 去中心化竞标管理

### 8.4 Scheduling (Disaggregation) Module

```
输入：trade outcomes T(t)
输出：device control signals {C_d(t)}
```

C_d(t) = (P_d^c, t_d^s, t_d^e), P_d^c ∈ [p_min,d, p_max,d]

两种策略：
- **Average**: E_i = E / N（均分）
- **Proportional**: E_i = (w_i / W) × E（按权重分配）

---

## 9. 评估指标（论文Section 5.3）

### 9.1 User Satisfaction g_u

```
g_u = min(1, E_a / E_d)
```

### 9.2 Economic Benefit g_e

```
g_e = (R_t - C_t) × λ_c
```
单位：DKK

### 9.3 Aggregation & DisAggregation Score g_a

```
g_a = ω_1·Γ_c + ω_2·(1 - Γ_r/Γ_r^max) + ω_3·(1 - Γ_v) + ω_4·Γ_p
```
- Γ_c: 压缩比
- Γ_r: RMSE
- Γ_v: 变异系数
- Γ_p: 能量保持率

### 9.4 Trading Score g_t

```
g_t = μ_1·Γ_s + μ_2·Γ_m
```
- Γ_s = N_s / N_a（交易成功率）
- Γ_m: 市场出清效率

---

## 10. 代码一致性检查清单

### 最高优先级（论文核心公式）
- [ ] Observation三层结构是否与Eq.7一致
- [ ] Action维度是否为 5 × |D_m|（Eq.2）
- [ ] Reward四个分量及权重是否与Eq.10-14一致
- [ ] 各设备的reward子结构是否与Appendix一致

### 高优先级（算法集成）
- [ ] 5个MARL算法的loss函数是否包含FO特有项（L_f, L_t, L_c等）
- [ ] 超参数是否与Appendix表格一致
- [ ] MAPPO共享策略 vs MAIPPO独立策略是否正确实现
- [ ] SQDDPG的Shapley值计算是否正确

### **[NEW] 必须检查/修改**
- [ ] **Action Projection**：env.step()中是否在执行前clip action到可行域
- [ ] **r_c计算**：是否基于raw action（projection前）计算惩罚
- [ ] **FO参数projection**：FO生成时参数是否经过合法性裁剪
- [ ] **reward权重可配置**：α,β,δ,λ是否可通过config修改（ablation需要）

### 生命周期模块
- [ ] Aggregation LP/DP策略实现
- [ ] Trading Market Clearing/Bidding实现
- [ ] Scheduling Average/Proportional策略实现
- [ ] Pipeline数据流是否为 Generation→Aggregation→Trading→Scheduling

# FOgym System Architecture Design

## System Overview

FOgym is a complete energy trading platform that integrates five advanced multi-agent RL algorithms (FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG), adopting a Manager-level collaborative learning architecture to implement an end-to-end solution from device control to market trading.

## Four-Layer Modular Architecture

```
FOgym Four-Layer Architecture
+------------------------------------------------------------------------+
|                          RL Algorithm Layer                              |
+------------------------------------------------------------------------+
| MAPPO      | MAIPPO      | MADDPG      | MATD3      | SQDDPG          |
| Shared     | Independent | Actor-      | Dual Q-    | Shapley value+  |
| policy+    | policy+     | Critic      | network+   | Fair allocation |
| Trust      | Conflict    |             | Delayed    |                 |
| region     | avoidance   |             | updates    |                 |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                      FlexOffer Process Layer                            |
+------------------------------------------------------------------------+
| Generation     | Aggregation    | Trading        | Scheduling         |
| fo_generate/   | fo_aggregate/  | fo_trading/    | fo_schedule/       |
| Device MDP     | LP/DP          | Market match   | Decomposition      |
| Multi-agent    | Manager aggr   | Bilateral      | Satisfaction       |
| environment    |                | auction        | assessment         |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                      Infrastructure Layer                               |
+------------------------------------------------------------------------+
| Dec-POMDP      | Data           | Configuration  | Monitoring         |
| architecture   | management     | system         | logs               |
| Observation    | CSV loader     | Parameter      | Performance        |
| space design   | Model saving   | validation     | monitoring         |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                        Device Ecosystem                                 |
+------------------------------------------------------------------------+
| Dishwashers(36) | Heat pumps(36) | Batteries(24) | EVs(14) | PV(8)   |
| 100% deployment | 100% deployment| 67% deployment| 39%     | 22%     |
| User behavior   | Temperature    | SOC management| Charging| Forecast|
| modeling        | control        |               | strategy|         |
+------------------------------------------------------------------------+
```

## Algorithm Layer Detailed Design

### Five Integrated MARL Algorithms

The system implements five multi-agent algorithms specifically designed for the FlexOffer system:

#### Algorithm Comparison Table
| Algorithm | Type | Core Features | Advantages | Applicable Scenarios |
|------|------|----------|------|----------|
| **FOMAPPO** | Policy Gradient | Trust region + batch updates | Very high training stability | Long-term stable training |
| **FOMAIPPO** | Policy Gradient | Independent policy + conflict avoidance | Resolves policy conflicts | Diverse task scenarios |
| **FOMADDPG** | Actor-Critic | Deterministic policy gradient | Very high sample efficiency | Continuous control optimization |
| **FOMATD3** | Actor-Critic | Dual Q-network + delayed updates | Highest training stability | High-noise environments |
| **FOSQDDPG** | Actor-Critic | Shapley value fair allocation | Fairness guarantee | Multi-party collaboration scenarios |

### Algorithm Implementation Architecture

#### FOMAPPO vs FOMAIPPO Comparison

**FOMAPPO (Shared Policy Architecture)**:
```python
# File location: algorithms/MAPPO/fomappo/fomappo_adapter.py
class FOMAPPOAdapter:
    - Uses SharedReplayBuffer
    - All Managers share a single policy network
    - References the shared/base_runner.py architecture from the original MAPPO
    - Advantages: Parameter efficiency, natural coordination
    - Applicable: Manager tasks with similar characteristics
```

**FOMAIPPO (Independent Policy Architecture)**:
```python
# File location: algorithms/MAPPO/fomappo/fomaippo_adapter.py
class FOMAIPPOAdapter:
    - Uses SeparatedReplayBuffer
    - Each Manager has an independent policy network
    - References the separated/base_runner.py architecture from the original MAPPO
    - Advantages: Avoids policy conflicts, independent learning
    - Applicable: Managers managing different user groups
```

#### Other Algorithm Features

**FOMADDPG (Deterministic Policy)**:
```python
# File location: algorithms/MADDPG/fomaddpg/
- Deterministic policy gradient algorithm
- Suitable for continuous action spaces
- High sample efficiency, fast convergence
- Supports experience replay mechanism
```

**FOMATD3 (Dual Q-network)**:
```python
# File location: algorithms/MATD3/fomatd3/
- Dual Critic network reduces estimation variance
- Delayed policy updates improve stability
- Target policy smoothing reduces overestimation
- Suitable for high-noise environments
```

**FOSQDDPG (Shapley Value)**:
```python
# File location: algorithms/SQDDPG/fosqddpg/
- Shapley value calculation for fair contribution
- Multi-agent credit allocation mechanism
- Ensures fairness in multi-party collaboration
- Adaptive reward allocation
```

## Multi-level Reward Design System

### Device-Level Reward Mechanism

**Battery Energy Storage System (BatteryMDPDevice)**
```python
Total reward = 0.6 × economic reward + 0.2 × efficiency reward + 0.2 × SOC maintenance reward

# Reward components:
Economic reward = -action × price                    # Charging cost (negative value)
Efficiency reward = -efficiency_loss × price           # Efficiency loss penalty  
SOC maintenance reward = -|soc - 0.6| × 0.1             # Maintain optimal SOC (60%)
```

**Heat Pump System (HeatPumpMDPDevice)**
```python
Total reward = 0.4 × economic reward + 0.6 × comfort reward

# Comfort calculation:
if |temp - target| ≤ 0.5°C:     comfort_reward = 1.0
elif 0.5 < |temp - target| ≤ 2.0°C:  comfort_reward = 1.0 - (temp_diff - 0.5) / 1.5
else:                            comfort_reward = -temp_diff
```

**Electric Vehicle (EVMDPDevice)**
```python
Total reward = 0.3 × economic reward + 0.5 × charging completion reward + 0.2 × connectivity reward

# Charging completion reward:
if current_soc >= target_soc:   completion_reward = 2.0
else:                          completion_reward = current_soc / target_soc
```

**Dishwasher System (DishwasherMDPDevice)** (Innovative Feature)
```python
Total reward = task completion reward(100) + progress reward(10) + timing reward - energy cost - waiting penalty

# Timing reward:
if urgency > 0.8:   timing_reward = 20.0 × urgency
elif urgency < 0.3: timing_penalty = -5.0

# Waiting time penalty:
if wait_time > max_start_delay:   timeout_penalty = -50.0
```

**PV Generation (PVMDPDevice)**
```python
Total reward = power_generated × price    # Generation revenue

# Weather impact factor:
sunny: 1.0,  cloudy: 0.6,  rainy: 0.2,  snowy: 0.1
```

### Manager-Level Reward Aggregation

```python
# Manager total reward = weighted aggregation of device rewards + user preference adjustment
manager_reward = Σ(device_reward_i × user_preference_weight_i)

# User preference weight aggregation:
aggregated_preferences = {
    'economic': Σ(user_economic_pref) / num_users,      # Economic preference
    'comfort': Σ(user_comfort_pref) / num_users,        # Comfort preference
    'environmental': Σ(user_environmental_pref) / num_users  # Environmental preference
}

# Markov history enhancement:
markov_history = {
    'prev_actions': prev_device_actions,
    'prev_reward': previous_reward,
    'cumulative_cost': total_energy_cost,
    'cumulative_energy': total_energy_consumption,
    'user_satisfaction': aggregated_user_satisfaction
}
```

### System-Level Multi-agent Reward Coordination

```python
# Dec-POMDP observation enhanced reward
enhanced_reward = base_reward + collaboration_bonus + information_quality_bonus

# Collaboration bonus (FOSQDDPG specific):
shapley_value = calculate_shapley_value(agent_id, coalition, actions)
fairness_bonus = fairness_weight × shapley_value

# Information quality bonus:
network_quality_bonus = (1 - noise_level) × base_reward × 0.1

# Multi-Manager collaboration enhancement:
collaboration_score = calculate_collaboration_effectiveness(manager_actions)
system_bonus = collaboration_coefficient × collaboration_score
```

## Modular Algorithm Integration Architecture

### fo_generate/ - Generation Layer Algorithm Integration
```python
# Unified MDP environment architecture
├── FlexOfferEnv (unified_mdp_env.py)
│   ├── DeviceMDPInterface: Unified device interface
│   ├── BatteryMDPDevice: Battery MDP implementation
│   ├── HeatPumpMDPDevice: Heat pump MDP implementation
│   ├── EVMDPDevice: Electric vehicle MDP implementation
│   ├── DishwasherMDPDevice: Dishwasher MDP implementation (innovative)
│   ├── PVMDPDevice: PV MDP implementation
│   └── EnvironmentDynamics: Environment dynamics modeling

# Multi-agent environment architecture
├── MultiAgentFlexOfferEnv (multi_agent_env.py)
│   ├── ManagerAgent: Manager agent class
│   ├── Dec-POMDP observation space: 3-layer information architecture
│   ├── DynamicObservationQuality: Dynamic observation quality
│   └── Collaboration information mechanism: Information sharing between Managers
```

### fo_aggregate/ - Aggregation Layer Algorithm Framework
```python
# FlexOffer aggregation algorithm
├── FOAggregatorFactory (aggregator.py)
│   ├── LP aggregation (Longest Profile): Longest profile aggregation
│   ├── DP aggregation (Dynamic Profile): Dynamic profile aggregation
│   └── Parameterized configuration: SPT, PPT, TF threshold configuration

# Manager-User-Device three-layer architecture
├── Manager (manager.py)
│   ├── Manages multiple users and devices
│   ├── Geographical coverage and coverage
│   └── Aggregates FlexOffer generation
├── User: User preferences and device combinations
└── Device: Device parameters and state management
```

### fo_trading/ - Trading Layer Algorithm Framework
```python
# Trading algorithm architecture
├── TradingAlgorithmFactory (pool.py)
│   ├── MarketClearingAlgorithm: Market clearing mechanism
│   │   ├── uniform_price: Uniform price clearing
│   │   ├── pay_as_bid: Settlement based on bid price
│   │   └── lmp: Marginal pricing mechanism
│   ├── BiddingAlgorithm: Bidding strategy algorithm
│   └── TradingAlgorithm: Trading algorithm base class

# Market mechanism and model
├── WeatherModel: Weather impact modeling and prediction
├── DemandModel: Demand prediction and trend analysis  
├── TradingPool: Trading pool and smart matching engine
├── Bid: Bid data structure
├── ClearingResult: Market clearing result
└── Trade: Trade record and status management
```

### fo_schedule/ - Scheduling Layer Algorithm Architecture
```python
# FlexOffer decomposition algorithm framework
├── DisaggregationAlgorithmFactory (scheduler.py)
│   ├── AverageDisaggregationAlgorithm: Average decomposition E_i = E/N
│   ├── ProportionalDisaggregationAlgorithm: Proportional decomposition E_i = (w_i/W) × E
│   └── Algorithm registration mechanism: Supports dynamic addition of new algorithms

# Scheduling management architecture
├── ScheduleManager: Multi-Manager scheduling coordination
│   ├── Selection and switching of decomposition algorithms
│   ├── Performance monitoring and statistics
│   └── Dynamic user demand update
├── UserScheduler: User-level scheduling and satisfaction assessment
├── AggregatedResultDisaggregator: Aggregated result disaggregator
└── FlexOfferDisaggregator: FlexOffer disaggregator
```

## Dec-POMDP Observation Space Architecture

### Observation Space Design
```
O_i = [O_private_i, O_public, O_limited_others_i]

Where:
O_private_i: Manager i's complete private information (no noise)
O_public: Public environmental information (no noise, visible to all Managers)
O_limited_others_i: Limited aggregated information from other Managers (configurable noise)
```

### Observation Dimension Distribution
- **Private Information (O_private_i)**: 40 dimensions
  - Own device state: 25 dimensions
  - User preference aggregation: 5 dimensions
  - Own features: 5 dimensions
  - Markov history: 5 dimensions

- **Public Information (O_public)**: 18 dimensions
  - Time features: 5 dimensions
  - Price information: 5 dimensions
  - Weather information: 5 dimensions
  - Market basic information: 3 dimensions

- **Other Information (O_limited_others_i)**: 15 dimensions
  - Each other Manager: 5 dimensions
  - Includes: User proportion, device proportion, energy level, satisfaction level, activity status

### Dynamic Observation Quality
The system supports 5 levels of network quality dynamic adjustment, affecting observation noise and latency:

| Quality Level | Noise Level | Latency | Data Loss Rate |
|--------|---------|-----|----------|
| **Very High** | 5% | None | 0% |
| **High** | 7.5% | Low | 1% |
| **Medium** | 10% | Medium | 3% |
| **Low** | 15% | High | 5% |
| **Very Low** | 20% | Severe | 10% |

## Device Types and FlexOffer Pipeline

### Device Types and Parameter Settings

The system supports 5 device types, each of which can generate FlexOffer, providing energy flexibility.

#### 1. Battery Energy Storage System (Battery)

**Deployment Rate**: 67% (24/36 users)

**Parameter Settings**:
```python
BatteryParameters(
    battery_id=device_config['device_id'],
    soc_min=device_config.get('param1', 0.1),        # Minimum state of charge (SOC)
    soc_max=device_config.get('param2', 0.9),        # Maximum state of charge (SOC)
    p_min=-device_config['max_power'],               # Maximum discharge power (kW), negative value
    p_max=device_config['max_power'],                # Maximum charging power (kW), positive value
    efficiency=device_config['efficiency'],          # Charging/discharging efficiency (0.8-0.95)
    initial_soc=device_config['initial_state'],      # Initial SOC (0.1-0.9)
    battery_type="lithium-ion",                      # Battery type
    capacity_kwh=device_config['capacity']           # Battery capacity (kWh)
)
```

**Typical Values**:
- Capacity: 5-15 kWh
- Maximum Power: 3-7 kW
- Efficiency: 0.9 (90%)
- SOC Range: 0.1-0.9 (10%-90%)

**MDP Reward Function**:
- Economic Gain: 60% (Price arbitrage)
- Efficiency Maintenance: 20% (Avoid excessive charging/discharging)
- SOC Maintenance: 20% (Maintain optimal range)

#### 2. Heat Pump System (Heat Pump)

**Deployment Rate**: 100% (36/36 users)

**Parameter Settings**:
```python
HeatPumpParameters(
    room_id=device_config['device_id'],
    room_area=30.0,                                  # Room area (m²)
    room_volume=75.0,                                # Room volume (m³)
    temp_min=device_config.get('param1', 18.0),      # Minimum temperature (°C)
    temp_max=device_config.get('param2', 26.0),      # Maximum temperature (°C)
    initial_temp=device_config['initial_state'],     # Initial temperature (°C)
    cop=device_config['efficiency'],                 # Performance coefficient (COP)
    heat_loss_coef=device_config.get('param3', 0.1), # Heat loss coefficient
    primary_use_period="8:00-22:00",                 # Primary usage period
    secondary_use_period="22:00-8:00",               # Secondary usage period
    primary_target_temp=22.0,                        # Primary target temperature (°C)
    secondary_target_temp=19.0,                      # Secondary target temperature (°C)
    max_power=device_config['max_power']             # Maximum power (kW)
)
```

**Typical Values**:
- Maximum Power: 2-5 kW
- COP: 3.0-4.5
- Temperature Range: 18-26°C
- Heat Loss Coefficient: 0.1-0.2

**MDP Reward Function**:
- Comfort: 60% (Temperature maintained within target range)
- Economic Gain: 40% (Reduced energy costs)

#### 3. Electric Vehicle (EV)

**Deployment Rate**: 39% (14/36 users)

**Parameter Settings**:
```python
EVParameters(
    ev_id=device_config['device_id'],
    battery_capacity=device_config['capacity'],      # Battery capacity (kWh)
    soc_min=device_config.get('param1', 0.1),        # Minimum SOC
    soc_max=device_config.get('param2', 0.95),       # Maximum SOC
    max_charging_power=device_config['max_power'],   # Maximum charging power (kW)
    efficiency=device_config['efficiency'],          # Charging efficiency
    initial_soc=device_config['initial_state'],      # Initial SOC
    fast_charge_capable=True                         # Fast charging capability
)
```

**Typical Values**:
- Battery Capacity: 40-80 kWh
- Charging Power: 3.7-11 kW (Home use)
- Efficiency: 0.85-0.92
- Connection Time: 18:00-7:30

**MDP Reward Function**:
- Travel Assurance: 50% (Ensures travel needs are met)
- Economic Gain: 30% (Reduced charging costs)
- Battery Health: 20% (Optimize charging mode)

#### 4. Dishwasher (Dishwasher)

**Deployment Rate**: 100% (36/36 users)

**Parameter Settings**:
```python
DishwasherParameters(
    dishwasher_id=device_config['device_id'],
    total_energy=device_config.get('capacity', 3.0),  # Total energy demand (kWh)
    power_rating=device_config['max_power'],          # Rated power (kW)
    operation_hours=device_config.get('param1', 3.5), # Operation hours (h)
    min_start_delay=device_config.get('param2', 0.5), # Minimum start delay (h)
    max_start_delay=device_config.get('param3', 6.0), # Maximum start delay (h)
    efficiency=device_config['efficiency'],           # Energy efficiency
    can_interrupt=False                               # Cannot be interrupted
)
```

**Typical Values**:
- Total Energy Demand: 1-3 kWh/cycle
- Rated Power: 0.8-1.5 kW
- Operation Hours: 2-4 hours
- Start Delay: 0.5-6 hours

**MDP Reward Function**:
- Task Completion: 50% (Completed before deadline)
- Economic Gain: 30% (Run during low-price periods)
- User Preference: 20% (Approach user's preferred time)

#### 5. PV System (PV)

**Deployment Rate**: 22% (8/36 users)

**Parameter Settings**:
```python
PVParameters(
    pv_id=device_config['device_id'],
    max_power=device_config['max_power'],            # Maximum power (kW)
    efficiency=device_config['efficiency'],          # Conversion efficiency
    area=device_config.get('param3', 25.0),          # Panel area (m²)
    location="roof",                                 # Installation location
    tilt_angle=device_config.get('param1', 30.0),    # Tilt angle (°)
    azimuth_angle=device_config.get('param2', 180.0),# Azimuth angle (°)
    weather_dependent=True,                          # Weather dependent
    forecast_accuracy=0.8                            # Forecast accuracy
)
```

**Typical Values**:
- Maximum Power: 3-10 kW
- Efficiency: 0.15-0.22
- Panel Area: 15-35 m²
- Tilt Angle: 20-40°

**MDP Reward Function**:
- Self-consumption maximization: 50% (Self-generation)
- Revenue maximization: 40% (Excess energy sold)
- Forecast Accuracy: 10% (Improving forecast accuracy)

### FlexOffer Mathematical Definition and Structure

#### FlexOffer Mathematical Definition

A FlexOffer F is a time series defined by a series of time slices:

F = {S₁, S₂, ..., Sₙ}

Where each time slice Sᵢ is defined as:

Sᵢ = (tᵢ, [eᵢᵐⁱⁿ, eᵢᵐᵃˣ], dᵢ)

- tᵢ: Start time of the time slice
- eᵢᵐⁱⁿ: Minimum energy demand/supply (kWh)
- eᵢᵐᵃˣ: Maximum energy demand/supply (kWh)
- dᵢ: Duration of the time slice (minutes)

**Key Attributes**:
- Total energy range: E_min = ∑ᵢ eᵢᵐⁱⁿ, E_max = ∑ᵢ eᵢᵐᵃˣ
- Profile length: Number of non-zero energy time slices
- Time flexibility: ∑ᵢ(eᵢᵐᵃˣ - eᵢᵐⁱⁿ) / profile_length
- Power profile: pᵢᵐⁱⁿ = eᵢᵐⁱⁿ / (dᵢ/60), pᵢᵐᵃˣ = eᵢᵐᵃˣ / (dᵢ/60)

#### Device FlexOffer Generation Process

Device FlexOffer is generated by mapping actions from RL algorithms, with each device having 5 continuous action parameters:

```python
fo_params[device_id] = {
    'start_flex': np.clip(device_actions[0], -1.0, 1.0),  # Start time flexibility
    'end_flex': np.clip(device_actions[1], -1.0, 1.0),    # End time flexibility
    'energy_min_factor': np.clip(device_actions[2], 0.1, 1.0),  # Minimum energy factor
    'energy_max_factor': np.clip(device_actions[3], 1.0, 2.0),  # Maximum energy factor
    'priority_weight': np.clip(device_actions[4], 0.1, 2.0)     # Priority weight
}
```

### MARL and FlexOffer Pipeline Interaction Flow

#### Overall Flow

The interaction flow between MARL algorithms and the FlexOffer pipeline is as follows:

1. **Observation Collection**: MARL algorithms collect environment observations
2. **Action Generation**: MARL algorithms generate device control actions
3. **Action Mapping**: Actions are mapped to FlexOffer parameters
4. **FO Generation**: Generates FlexOffer for each device
5. **FO Aggregation**: Aggregates device FlexOffers
6. **FO Trading**: Trades aggregated FlexOffers in the market
7. **FO Decomposition**: Decomposes trading results to device level
8. **FO Scheduling**: Schedules devices to execute energy plans
9. **Reward Calculation**: Calculates rewards and feeds them back to MARL algorithms
10. **Policy Update**: MARL algorithms update their policies

#### FlexOffer Pipeline Detailed Explanation

##### Generation Layer (fo_generate/)

```python
# Action mapping to FlexOffer parameters
fo_params = _map_actions_to_fo_params(actions)

# Generate device FlexOffers
device_flexoffers = _generate_device_flexoffers(fo_params, env_state)
```

##### Aggregation Layer (fo_aggregate/)

```python
# Aggregate device FlexOffers
aggregated_results = _aggregate_flexoffers(device_flexoffers, env_state)

# Aggregation method: LP (Longest Profile) or DP (Dynamic Profile)
aggregation_method = getattr(self, 'aggregation_method', 'LP')
```

##### Trading Layer (fo_trading/)

```python
# Trade aggregated FlexOffers
trade_results = _trade_flexoffers(aggregated_results, env_state)

# Trading method: market_clearing or bidding
trading_method = env_state.get('trading_algorithm', 'market_clearing')
```

##### Scheduling Layer (fo_schedule/)

```python
# Decompose trading results
disaggregated_results = _disaggregate_flexoffers(trade_results, device_flexoffers, env_state)

# Schedule devices to execute
scheduled_results = _schedule_flexoffers(disaggregated_results, env_state)

# Decomposition method: average or proportional
disaggregation_method = getattr(self, 'disaggregation_method', 'proportional')
```

#### Key Performance Metrics

- **Aggregation Efficiency**: 26.5:1 compression ratio
- **Trading Success Rate**: 67%
- **User Satisfaction**: Average 22.2%
- **Energy Optimization**: Average 15% energy savings
- **Economic Benefits**: Average 12% electricity savings

## Data Flow and Interaction

### FlexOffer Generation Flow
1. **Device State Initialization**: Device parameter and initial state configuration
2. **MDP Environment Creation**: Creates a dedicated MDP environment for each device
3. **Multi-agent Environment Construction**: Manager-level multi-agent environment
4. **Reinforcement Learning Training**: Strategy network training and optimization
5. **FlexOffer Parameter Generation**: Maps RL actions to FO parameters
6. **FlexOffer Creation**: Generates device-level FlexOffers based on parameters

### FlexOffer Aggregation Flow
1. **Device-level FO Collection**: Managers collect all device FOs
2. **Aggregation Algorithm Selection**: LP or DP aggregation algorithm
3. **Feature Extraction**: Analyzes FlexOffer features
4. **Similarity Assessment**: Calculates FO similarity
5. **Group Aggregation**: Similar FOs grouped for aggregation
6. **Manager-level FO Creation**: Generates Manager-level FOs after aggregation

### Market Trading Flow
1. **Quote Generation**: Creates market quotes based on aggregated FOs
2. **Market Clearing**: Uses uniform_price or pay_as_bid mechanisms
3. **Price Discovery**: Determines transaction price points
4. **Trade Matching**: Buy/sell parties match
5. **Settlement**: Transaction records and results stored

### Scheduling Decomposition Flow
1. **Trade Result Reception**: Receives market trading results
2. **Decomposition Algorithm Selection**: average or proportional algorithm
3. **Energy Allocation Calculation**: Calculates energy allocation for each device
4. **Device Scheduling Generation**: Creates device-level execution plans
5. **Satisfaction Assessment**: Calculates user satisfaction metrics

## Performance Optimization

### Computational Optimization
1. **Parallel Training**: Manager strategies trained in parallel
2. **Batch Processing**: Large batches of experience replay
3. **GPU Acceleration**: Supports CUDA tensor calculations
4. **Agent Grouping**: Avoids unnecessary interaction calculations

### Memory Optimization
1. **Experience Replay Compression**: Efficiently stores converted samples
2. **Observation Space Optimization**: Dimension reduction and feature selection
3. **Cache Mechanism**: Caches repeated calculation results
4. **Progressive Training**: Incremental sample collection and updates



## System Integration Interface

### Python API
```python
# 1. Initialize the system
pipeline = FOPipeline(config)

# 2. Load data
pipeline.load_data('data/user_config.csv', 'data/device_config.csv')

# 3. Select algorithm
pipeline.set_algorithm('fomappo')

# 4. Configure pipeline
pipeline.configure(
    aggregation_method='LP',
    trading_strategy='market_clearing',
    disaggregation_method='proportional'
)

# 5. Train and run
pipeline.train(num_episodes=100)
results = pipeline.run_pipeline()

# 6. Result analysis
metrics = pipeline.calculate_metrics(results)
pipeline.visualize_results(results)
```

### Command Line Interface
```bash
# Basic run
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 100

# Advanced configuration
python run_fo_pipeline.py \
  --rl_algorithm fomappo \
  --aggregation_method LP \
  --trading_strategy market_clearing \
  --disaggregation_method proportional \
  --scheduling_method priority \
  --log_verbosity detailed \
  --use_gpu
```

## Summary

The FOgym system adopts a four-layer modular architecture to implement a complete energy management process from device control to market trading. Five MARL algorithms are integrated, providing flexible solutions for different scenarios:

1. **FOMAPPO/FOMAIPPO** provides shared policy/independent policy options
2. **FOMADDPG/FOMATD3** provide high-efficiency algorithms for continuous control scenarios
3. **FOSQDDPG** ensures fairness in multi-party collaboration through Shapley values

Dec-POMDP observation space design and multi-level reward mechanism jointly construct a real multi-agent distributed decision-making environment, enabling the system to cope with the complexity and uncertainty of the real world. 
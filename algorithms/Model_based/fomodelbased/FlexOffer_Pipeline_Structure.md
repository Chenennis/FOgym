# FlexOffer Pipeline Structure

## 1. User Structure

The FlexOffer pipeline follows a three-layer hierarchical structure:

### Manager-User-Device Architecture

- **4 Managers** with uneven user distribution: 
  - Manager 1: 6 users
  - Manager 2: 10 users
  - Manager 3: 8 users
  - Manager 4: 12 users

- **Total 36 Users** across all managers
  - Each user has multiple devices
  - User types: prosumer, consumer, producer
  - User preferences: economic, comfort, self-sufficient, environmental

- **Total 118 Devices** across all users
  - Average 3.3 devices per user
  - Device types and deployment rates:
    - Battery storage: 67% (24/36 users)
    - Heat pumps: 100% (36/36 users)
    - Electric vehicles: 39% (14/36 users)
    - Photovoltaic systems: 22% (8/36 users)
    - Dishwashers: 100% (36/36 users)

- **Device Distribution by Manager** (based on proportional allocation):
  - Manager 1 (6 users): ~20 devices
    * ~6 heat pumps (100% of users)
    * ~6 dishwashers (100% of users)
    * ~4 battery systems (67% of users)
    * ~2-3 electric vehicles (39% of users)
    * ~1-2 photovoltaic systems (22% of users)
  - Manager 2 (10 users): ~33 devices
    * ~10 heat pumps (100% of users)
    * ~10 dishwashers (100% of users)
    * ~7 battery systems (67% of users)
    * ~4 electric vehicles (39% of users)
    * ~2 photovoltaic systems (22% of users)
  - Manager 3 (8 users): ~26 devices
    * ~8 heat pumps (100% of users)
    * ~8 dishwashers (100% of users)
    * ~5-6 battery systems (67% of users)
    * ~3 electric vehicles (39% of users)
    * ~1-2 photovoltaic systems (22% of users)
  - Manager 4 (12 users): ~39 devices
    * ~12 heat pumps (100% of users)
    * ~12 dishwashers (100% of users)
    * ~8 battery systems (67% of users)
    * ~5 electric vehicles (39% of users)
    * ~3 photovoltaic systems (22% of users)

### Device Characteristics

Each device has specific parameters and generates FlexOffers based on its operational constraints:

**Battery Storage System**:
- Parameters: SOC bounds (0.1-0.9), capacity (5-15 kWh), charging/discharging rates (3-7 kW)
- MDP reward function: 60% economic, 20% efficiency, 20% SOC maintenance

**Heat Pump System**:
- Parameters: COP (3-5), temperature range (18-26°C), power limits
- MDP reward function: 40% economic, 60% comfort

**Electric Vehicle**:
- Parameters: Battery capacity (20-100 kWh), charging rate (3.7-11 kW), connection patterns
- MDP reward function: 50% economic, 30% battery health, 20% user satisfaction

**Photovoltaic System**:
- Parameters: Rated power (3-10 kW), efficiency, weather dependency
- MDP reward function: 70% self-consumption, 30% grid export

**Dishwasher**:
- Parameters: Energy needs, cycle time, user preferences
- MDP reward function: 100% completion + 10× progress + timing - cost
- Constraints: Non-interruptible, time preferences

## 2. Aggregation Module (fo_aggregate)

The aggregation module combines multiple device-level FlexOffers into manager-level aggregated FlexOffers.

### LP (Longest Profile) Algorithm

**Core Principle**: Focuses on producing aggregated FlexOffers with the maximum number of time slices, prioritizing longer profiles with higher energy content.

**Algorithm Flow**:
1. Find all FlexOffers with the maximum profile size
2. Select the FlexOffer with highest time flexibility (TF) as the initial FlexOffer
3. Add all other FlexOffers to the processing set
4. Iteratively perform binary aggregation by adding FlexOffers that improve RMSE and CV

**Mathematical Description**:
- Profile size: Number of time slices with non-zero energy
- Time flexibility (TF): Σ(e_max - e_min) / profile_length
- Binary aggregation: Combine two FlexOffers while maintaining time alignment
- Success criteria: Reduced RMSE (Root Mean Square Error) and CV (Coefficient of Variation)

**Advantages**:
- Captures maximum energy volume (98.6% FO participation rate)
- Good for scenarios prioritizing energy amount
- Faster processing time

**Disadvantages**:
- Results in low time flexibility in aggregated FlexOffer
- May consume flexibility of other FlexOffers
- Handling very long profiles can be challenging

### DP (Dynamic Profile) Algorithm

**Core Principle**: Excludes extremely long profiles (outliers) to produce more balanced aggregated FlexOffers with better time flexibility.

**Algorithm Flow**:
1. Calculate the upper fence for profile sizes using quartile method
2. Filter FlexOffer set to exclude outliers (FlexOffers with profile size > upper fence)
3. Select the FlexOffer with the longest profile and highest time flexibility
4. Iteratively perform binary aggregation similar to LP

**Mathematical Description**:
- Upper fence: Q3 + 1.5 × IQR (where IQR = Q3 - Q1)
- Outlier filtering: Remove FlexOffers with profile size > upper fence
- Binary aggregation with RMSE and CV optimization

**Advantages**:
- Produces more time-flexible aggregated FlexOffers
- Better aggregation quality through outlier filtering
- More suitable for similar-profile FlexOffer aggregation

**Disadvantages**:
- Lower FO participation rate (94.2%)
- Lower traded energy percentage (88.8%)
- May exclude high-energy long-profile FlexOffers

## 3. Trading Module (fo_trading)

The trading module handles the exchange of aggregated FlexOffers between managers in a market environment.

### Market Clearing Algorithm

**Core Principle**: Determines clearing price, clearing quantity, and matches bids based on supply-demand balance and maximum social welfare.

**Algorithm Flow**:
1. Collect buy and sell bids from all managers
2. Sort bids (buy bids by price descending, sell bids by price ascending)
3. Find the supply-demand intersection point (clearing price and quantity)
4. Match bids based on clearing results
5. Generate trades with matched bids

**Mathematical Description**:
- Clearing price: (p_buy + p_sell)/2 where supply meets demand
- Clearing quantity: min(supply_quantity, demand_quantity) at clearing point
- Social welfare: consumer_surplus + producer_surplus
- Consumer surplus: Σ(bid_price - clearing_price) × matched_quantity for buy bids
- Producer surplus: Σ(clearing_price - bid_price) × matched_quantity for sell bids

**Key Features**:
- Supports uniform_price, pay_as_bid, and lmp clearing methods
- Market efficiency optimization through welfare maximization
- Handles market imbalances with fallback mechanisms

### Bidding Algorithm

**Core Principle**: Enables market participants to express their energy buying/selling intentions and conditions.

**Algorithm Flow**:
1. Register market participants
2. Collect bids from participants
3. Validate and organize bids
4. Process bids (but doesn't execute clearing - this is done by Market Clearing)

**Mathematical Description**:
- Bid price calculation: base_price × (1 ± market_adj ± random_factor ± bias)
- Base price: Reference energy price (from grid)
- Market adjustment: Based on demand forecast and weather impact
- Random factor: Controlled randomness for price discovery (±1.5%)
- Bias: Manager-specific adjustment

**Key Features**:
- Support for multiple bid types: fixed, block, curve
- Market participation statistics
- Bid collection and management

## 4. Disaggregation Module (fo_schedule)

The disaggregation module distributes the energy allocated through trading back to individual devices.

### Average Disaggregation Algorithm

**Core Principle**: Distributes total energy equally among all participants without considering individual differences.

**Algorithm Flow**:
1. Calculate average allocation: E_i = E/N for each device
2. Distribute equal energy to all devices
3. Return disaggregated results with allocation metadata

**Mathematical Description**:
- E_i = E/N (where E is total energy, N is number of devices)
- Allocation ratio = 1/N for each device

**Advantages**:
- Simple and intuitive implementation
- No additional parameters needed
- Fair in terms of equal treatment

**Disadvantages**:
- Ignores device capabilities (power limits, battery capacity)
- May lead to resource waste or constraint violations
- Not optimal for heterogeneous device groups

### Proportional Disaggregation Algorithm

**Core Principle**: Distributes energy based on weighted contribution of each device according to a specified attribute (energy, capacity, priority).

**Algorithm Flow**:
1. Calculate total weight W = Σw_i across all devices
2. Compute weight ratio for each device: r_i = w_i/W
3. Allocate energy proportionally: E_i = r_i × E
4. Return disaggregated results with allocation metadata

**Mathematical Description**:
- E_i = (w_i/W) × E (where w_i is device weight, W is total weight)
- Weight can be based on device capacity, energy need, or priority

**Advantages**:
- Respects individual device capabilities
- Reduces risk of ineffective allocations
- More suitable for heterogeneous device groups

**Disadvantages**:
- Requires weight estimation in advance
- Slightly more complex implementation
- May favor larger devices consistently

## 5. System Integration

The complete FlexOffer pipeline integrates these modules into a cohesive workflow:

1. Device-level FlexOffers are generated from each device's state
2. FlexOffers are aggregated at the manager level using LP or DP algorithms
3. Managers trade aggregated FlexOffers using Bidding and Market Clearing algorithms
4. Trading results are disaggregated back to devices using Average or Proportional algorithms
5. Devices execute the resulting energy schedules

This multi-stage process enables efficient coordination of distributed energy resources while respecting device constraints, user preferences, and market conditions. 
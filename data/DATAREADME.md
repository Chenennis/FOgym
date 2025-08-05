# FlexOffer System Data Files Documentation

This document provides detailed explanations of all data file formats, structures, and usage methods for the FlexOffer multi-agent reinforcement learning system. All data files are in CSV format using UTF-8 encoding.

## 📋 Table of Contents

- [1. Data Files Overview](#1-data-files-overview)
- [2. Weather Data](#2-weather-data)
- [3. Electricity Price Data](#3-electricity-price-data)
- [4. User and Manager Configuration](#4-user-and-manager-configuration)
- [5. Device Configuration Data](#5-device-configuration-data)
- [6. Energy Demand Data](#6-energy-demand-data)
- [7. Battery System Data](#7-battery-system-data)
- [8. Heat Pump System Data](#8-heat-pump-system-data)
- [9. Uncertain Energy Data](#9-uncertain-energy-data)
- [10. Data Loading and Usage](#10-data-loading-and-usage)
- [11. Data Acquisition and Update Recommendations](#11-data-acquisition-and-update-recommendations)

## 1. Data Files Overview

| File Name | Description | Priority | Necessity |
|--------|------|--------|--------|
| weather_data.csv | Weather data (temperature, irradiance, wind speed) | High | Required |
| grid_price.csv | Danish grid electricity price data | High | Required |
| user_config_36users.csv | Configuration information for 36 users | High | Required |
| manager_config_36users.csv | Configuration information for 4 managers | High | Required |
| device_config_36users.csv | Configuration information for all devices | High | Required |
| scenario_config_36users.csv | Scenario configuration information | High | Required |
| user_demands.csv | User energy demand data | Medium | Required |
| battery_base_parameters.csv | Battery basic parameters | Medium | Required |
| battery_dfo_input.csv | Battery FlexOffer input parameters | Medium | Required |
| heat_pump_system.csv | Heat pump system parameters | Medium | Required |
| uncertain_energy_data.csv | Uncertain energy data | Low | Optional |

## 2. Weather Data

### File Name: weather_data.csv

### Data Dimensions: 4
- **timestamp**: Timestamp (ISO 8601 format)
- **temperature**: Temperature (°C)
- **solar_irradiance**: Solar irradiance (W/m²)
- **wind_speed**: Wind speed (m/s)

### Example Data
```csv
timestamp,temperature,solar_irradiance,wind_speed
2024-01-15T00:00:00,2.5,0,8.2
2024-01-15T01:00:00,2.1,0,7.8
2024-01-15T02:00:00,1.8,0,7.5
...
```

### Data Characteristics
- Danish winter temperatures typically range from -5°C to 10°C
- Summer temperatures typically range from 10°C to 25°C
- Solar irradiance is lower in winter (max 400-500 W/m²) and higher in summer (max 800-1000 W/m²)
- Wind speed typically ranges from 5-15 m/s

### Usage Scenarios
- PV generation forecasting
- Heat pump efficiency calculation
- Building heat loss estimation
- Renewable energy production forecasting

## 3. Electricity Price Data

### File Name: grid_price.csv

### Data Dimensions: 6
- **timestamp**: Timestamp (YYYY-MM-DD HH:MM:SS)
- **hour**: Hour (0-23)
- **day_type**: Day type (weekday/weekend)
- **price_dkk_kwh**: Danish Krone price (DKK/kWh)
- **price_usd_kwh**: US Dollar price (USD/kWh)
- **price_level**: Price level (low/rising/valley/high/peak/falling/medium)

### Example Data
```csv
timestamp,hour,day_type,price_dkk_kwh,price_usd_kwh,price_level
2024-12-06 00:00:00,0,weekday,0.85,0.12,low
2024-12-06 01:00:00,1,weekday,0.82,0.12,low
...
```

### Danish Electricity Price Patterns

#### Weekday Price Pattern
```
Time Period     | Price Feature   | Price Level   | USD/kWh Range
0:00-5:00      | Lower          | low          | 0.10-0.12
6:00-9:00      | Rising to high  | rising/high  | 0.16-0.23
10:00-16:00    | Price valley    | valley       | 0.13-0.15
17:00-21:00    | Peak           | peak         | 0.24-0.27
22:00-23:00    | Falling        | falling      | 0.15-0.19
```

#### Weekend Price Pattern
```
Time Period     | Price Feature                | Price Level   | USD/kWh Range
0:00-5:00      | Lower                       | low          | 0.10-0.11
6:00-9:00      | Slowly rising               | rising       | 0.12-0.16
10:00-16:00    | Medium-low                  | medium       | 0.15-0.17
17:00-21:00    | Peak (lower than weekdays)  | peak         | 0.22-0.25
22:00-23:00    | Falling                     | falling      | 0.14-0.18
```

### Price Level Explanation
- **low**: < 0.12 USD/kWh (night low price)
- **medium**: 0.12-0.16 USD/kWh (daytime average price)
- **high**: 0.16-0.20 USD/kWh (daytime high price)
- **peak**: > 0.20 USD/kWh (peak price)
- **rising**: Price rising period
- **falling**: Price falling period
- **valley**: Price valley period

### Electricity Price Priority System
1. **Priority 1**: `grid_price.csv` - Actual Danish electricity price data
2. **Priority 2**: Traditional `price_data.csv` file
3. **Priority 3**: Dynamic price forecasting (based on Danish price patterns)

### Usage Method
```python
from fo_generate.price_loader import PriceLoader
from datetime import datetime

# Initialize price loader
price_loader = PriceLoader("data")

# Get 24-hour price data
start_time = datetime.now()
price_data = price_loader.get_price_data(start_time, 24)

# Current price information
current_price = price_loader.get_current_price(datetime.now())
print(f"Current price: {current_price['price']:.4f} USD/kWh")
print(f"Price level: {current_price['price_level']}")
```

## 4. User and Manager Configuration

### Manager Configuration File: manager_config_36users.csv

#### Data Dimensions: 6
- **manager_id**: Manager ID
- **location_x**: X coordinate (km)
- **location_y**: Y coordinate (km)
- **coverage_area**: Coverage area (km²)
- **user_count**: Number of users
- **district_type**: District type ("residential", "commercial", "mixed")

#### Example Data
```csv
manager_id,location_x,location_y,coverage_area,user_count,district_type
manager_1,2.5,3.2,1.5,6,residential
manager_2,5.8,7.1,2.3,10,mixed
manager_3,8.2,4.6,1.8,8,residential
manager_4,11.5,9.3,3.1,12,commercial
```

### User Configuration File: user_config_36users.csv

#### Data Dimensions: 8
- **user_id**: User ID
- **manager_id**: Associated Manager ID
- **location_x**: X coordinate (km)
- **location_y**: Y coordinate (km)
- **user_type**: User type ("prosumer", "consumer", "producer")
- **economic_pref**: Economic preference (0-1)
- **comfort_pref**: Comfort preference (0-1)
- **environmental_pref**: Environmental preference (0-1)

#### Example Data
```csv
user_id,manager_id,location_x,location_y,user_type,economic_pref,comfort_pref,environmental_pref
user_1,manager_1,2.3,3.1,prosumer,0.3,0.4,0.3
user_2,manager_1,2.7,3.3,consumer,0.5,0.3,0.2
...
```

### Scenario Configuration File: scenario_config_36users.csv

#### Data Dimensions: 3
- **scenario_id**: Scenario ID
- **description**: Scenario description
- **parameters**: Scenario parameters (JSON format)

#### Usage Scenarios
- Define Manager-User hierarchy relationships
- Set user preference weights
- Configure geographical distribution
- Define user type distribution

## 5. Device Configuration Data

### File Name: device_config_36users.csv

### Data Dimensions: 10
- **device_id**: Device ID
- **user_id**: Associated user ID
- **device_type**: Device type ("battery", "heat_pump", "ev", "pv", "dishwasher")
- **capacity**: Capacity (kWh or other units)
- **max_power**: Maximum power (kW)
- **efficiency**: Efficiency (0-1)
- **initial_state**: Initial state (SOC, temperature, etc.)
- **param1**: Device-specific parameter 1
- **param2**: Device-specific parameter 2
- **param3**: Device-specific parameter 3

### Example Data
```csv
device_id,user_id,device_type,capacity,max_power,efficiency,initial_state,param1,param2,param3
battery_1,user_1,battery,10.0,5.0,0.95,0.5,0.1,0.9,
heatpump_1,user_1,heat_pump,0.0,3.0,3.5,20.0,18.0,26.0,0.1
ev_1,user_2,ev,60.0,7.0,0.9,0.3,0.1,0.95,
pv_1,user_1,pv,0.0,5.0,0.18,0.0,30.0,180.0,28.0
dishwasher_1,user_3,dishwasher,3.0,1.2,0.85,0.0,3.5,0.5,6.0
```

### Device Type Parameter Explanation

#### Battery Storage System
- **capacity**: Battery capacity (kWh)
- **max_power**: Maximum charge/discharge power (kW)
- **efficiency**: Charge/discharge efficiency (0.8-0.95)
- **initial_state**: Initial SOC (0.1-0.9)
- **param1**: Minimum SOC (0.1)
- **param2**: Maximum SOC (0.9)

#### Heat Pump System
- **capacity**: Not applicable (0.0)
- **max_power**: Maximum power (kW)
- **efficiency**: COP coefficient (3.0-4.5)
- **initial_state**: Initial temperature (°C)
- **param1**: Minimum temperature (°C)
- **param2**: Maximum temperature (°C)
- **param3**: Heat loss coefficient (0.1-0.2)

#### Electric Vehicle (EV)
- **capacity**: Battery capacity (kWh)
- **max_power**: Maximum charging power (kW)
- **efficiency**: Charging efficiency (0.85-0.92)
- **initial_state**: Initial SOC (0.1-0.9)
- **param1**: Minimum SOC (0.1)
- **param2**: Maximum SOC (0.95)

#### Photovoltaic System (PV)
- **capacity**: Not applicable (0.0)
- **max_power**: Maximum generation power (kW)
- **efficiency**: Conversion efficiency (0.15-0.22)
- **initial_state**: Not applicable (0.0)
- **param1**: Tilt angle (°)
- **param2**: Azimuth angle (°)
- **param3**: Panel area (m²)

#### Dishwasher
- **capacity**: Total energy demand (kWh)
- **max_power**: Rated power (kW)
- **efficiency**: Energy efficiency (0.8-0.9)
- **initial_state**: Not applicable (0.0)
- **param1**: Operation duration (h)
- **param2**: Minimum start delay (h)
- **param3**: Maximum start delay (h)

### Device Deployment Rate
- Heat pump systems: 100% (36/36 users)
- Dishwashers: 100% (36/36 users)
- Battery storage systems: 67% (24/36 users)
- Electric vehicles: 39% (14/36 users)
- Photovoltaic systems: 22% (8/36 users)

## 6. Energy Demand Data

### File Name: user_demands.csv

### Data Dimensions
- **user_id**: User ID
- **timestamp**: Timestamp
- **heating_demand**: Heating demand (kWh)
- **electricity_demand**: Electricity demand (kWh)
- **hot_water_demand**: Hot water demand (kWh)
- **ev_charging_need**: EV charging demand (kWh)

### Usage Scenarios
- Heat pump heating control
- EV charging planning
- User load forecasting
- Energy consumption pattern analysis

## 7. Battery System Data

### Basic Parameters File: battery_base_parameters.csv

#### Data Dimensions
- **battery_type**: Battery type
- **nominal_voltage**: Nominal voltage (V)
- **cycle_life**: Cycle life
- **energy_density**: Energy density (Wh/kg)
- **self_discharge_rate**: Self-discharge rate (%/month)
- **calendar_life**: Calendar life (years)
- **cost_per_kwh**: Cost per kilowatt-hour (USD/kWh)

### FlexOffer Input File: battery_dfo_input.csv

#### Data Dimensions
- **battery_id**: Battery ID
- **time_step**: Time step
- **min_power**: Minimum power (kW)
- **max_power**: Maximum power (kW)
- **min_energy**: Minimum energy (kWh)
- **max_energy**: Maximum energy (kWh)
- **flexibility_factor**: Flexibility factor (0-1)

## 8. Heat Pump System Data

### File Name: heat_pump_system.csv

### Data Dimensions
- **heat_pump_id**: Heat pump ID
- **heat_pump_type**: Heat pump type
- **cop_reference**: Reference COP value
- **thermal_capacity**: Thermal capacity (kW)
- **temperature_lift**: Temperature lift (°C)
- **min_part_load**: Minimum part load (%)
- **max_flow_temp**: Maximum flow temperature (°C)
- **defrost_energy**: Defrost energy consumption (kWh)

### Usage Scenarios
- Heat pump efficiency calculation
- Temperature control optimization
- Heating energy consumption prediction
- FlexOffer generation

## 9. Uncertain Energy Data

### File Name: uncertain_energy_data.csv

### Data Dimensions
- **timestamp**: Timestamp
- **energy_type**: Energy type
- **expected_value**: Expected value
- **uncertainty_low**: Low uncertainty boundary
- **uncertainty_high**: High uncertainty boundary
- **confidence_level**: Confidence level

### Usage Scenarios
- Robust optimization
- Risk assessment
- Uncertainty modeling
- Scenario generation

## 10. Data Loading and Usage

### Data Loaders
The FlexOffer system provides a series of data loaders for handling different types of data:

```python
from fo_generate.data_loader import DataLoader
from fo_generate.price_loader import PriceLoader
from fo_generate.weather_loader import WeatherLoader

# Initialize data loaders
data_loader = DataLoader("data")
price_loader = PriceLoader("data")
weather_loader = WeatherLoader("data")

# Load user configuration
users = data_loader.load_users("user_config_36users.csv")

# Load device configuration
devices = data_loader.load_devices("device_config_36users.csv")

# Load weather data
weather = weather_loader.load_weather_data()

# Load price data
prices = price_loader.load_price_data()
```

### Integration with FlexOffer Pipeline
```python
from run_fo_pipeline import FOPipeline

# Create FlexOffer Pipeline
pipeline = FOPipeline({
    'data_dir': 'data',
    'rl_algorithm': 'fomappo',
    'num_episodes': 100,
    'log_verbosity': 'brief'
})

# Run Pipeline
pipeline.run()
```

## 11. Data Acquisition and Update Recommendations

### Danish Data Sources
1. **Weather data**: DMI (Danish Meteorological Institute) - https://www.dmi.dk/
2. **Price data**: Energinet - https://www.energinet.dk/
3. **PV data**: PVGIS (European Commission) - https://re.jrc.ec.europa.eu/pvg_tools/en/

### Data Update Frequency
- Weather data: Hourly updates
- Price data: Hourly updates (day-ahead market)
- PV forecasts: Hourly updates, providing 24-hour forecasts
- Working day data: Annual updates

### Data Quality Requirements
- Timestamps must be continuous, without missing values
- Numerical ranges must be reasonable
- Forecast data needs to include uncertainty information
- All timestamps use UTC+1 (Danish time)

If external data is unavailable, the system will use built-in Danish weather and price models to generate simulated data. Simulated data is based on typical Danish climate and electricity market characteristics. 
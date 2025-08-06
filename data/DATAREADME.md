### Danish Data Sources
1. **Weather data**: DMI (Danish Meteorological Institute) - https://www.dmi.dk/
2. **Electricity price data**: Energinet - https://www.energinet.dk/
3. **Photovoltaic data**: PVGIS (European Commission) - https://re.jrc.ec.europa.eu/pvg_tools/en/

### Data Update Frequency
- Weather data: Updated hourly
- Electricity price data: Updated hourly
- Photovoltaic prediction: Updated hourly, providing 24-hour forecast

### Data Quality Requirements
- Timestamps must be continuous, with no missing values
- Numerical ranges must be reasonable
- Prediction data must include uncertainty information
- All timestamps use UTC+1 (Danish time)

### Device Type Parameter Description

#### Battery Storage System (Battery)
- **capacity**: Battery capacity (kWh)
- **max_power**: Maximum charge/discharge power (kW)
- **efficiency**: Charge/discharge efficiency (0.8-0.95)
- **initial_state**: Initial SOC (0.1-0.9)
- **param1**: Minimum SOC (0.1)
- **param2**: Maximum SOC (0.9)

#### Heat Pump System (Heat Pump)
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
- **capacity**: Total energy requirement (kWh)
- **max_power**: Rated power (kW)
- **efficiency**: Energy efficiency (0.8-0.9)
- **initial_state**: Not applicable (0.0)
- **param1**: Running time (h)
- **param2**: Minimum start delay (h)
- **param3**: Maximum start delay (h)


## Energy Demand Data

### File Name: user_demands.csv

### Data Dimensions
- **user_id**: User ID
- **timestamp**: Timestamp
- **heating_demand**: Heating demand (kWh)
- **electricity_demand**: Electricity demand (kWh)
- **hot_water_demand**: Hot water demand (kWh)
- **ev_charging_need**: EV charging demand (kWh)



## Detailed explanation of command line parameters

You can control algorithm selection using the following command line parameters:

```bash
python run_pipeline.py [--config CONFIG_FILE] [--timesteps N] 
                       [--aggregation {LP,DP}] 
                       [--trading {bidding,market-clearing}]
                       [--disaggregation {proportional,average}]
                       [--managers N] [--users USERS_LIST]
```

Where:

- `--config`: Configuration file path (optional, default configuration is used if not provided)
- `--timesteps`: Number of simulation time steps (default is 24 hours)
- `--aggregation`: Aggregation algorithm selection, can be `LP` or `DP` (default is `LP`)
- `--trading`: Trading algorithm selection, can be `bidding` or `market-clearing` (default is `bidding`)
- `--disaggregation`: Disaggregation algorithm selection, can be `proportional` or `average` (default is `proportional`)
- `--managers`: Number of Managers (default is 4)
- `--users`: Number of users for each Manager, comma-separated (default is `6,10,8,12`)

## Common command combinations examples

Here are some examples of commonly used command combinations:

### 1. Default configuration (LP aggregation + bidding trading + proportional disaggregation)

```bash
python run_pipeline.py
```

### 2. Using DP aggregation algorithm

```bash
python run_pipeline.py --aggregation DP
```

### 3. Using market-clearing trading algorithm

```bash
python run_pipeline.py --trading market-clearing
```

### 4. Using average disaggregation algorithm

```bash
python run_pipeline.py --disaggregation average
```

### 5. Custom combination: DP aggregation + market-clearing trading + average disaggregation

```bash
python run_pipeline.py --aggregation DP --trading market-clearing --disaggregation average
```

### 6. Change simulation duration to 48 hours

```bash
python run_pipeline.py --timesteps 48
```

### 7. Change number of Managers and user distribution

```bash
python run_pipeline.py --managers 3 --users 8,12,16
```


## Custom configuration file

In addition to command line parameters, you can also control algorithm parameters more precisely through a configuration file:

```bash
python run_pipeline.py --config my_custom_config.json
```

Configuration file example:

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

## Debugging and logging

To view more detailed log output, you can add the `--verbose` parameter when running `test_model_based.py`:

```bash
python test_model_based.py --verbose
```

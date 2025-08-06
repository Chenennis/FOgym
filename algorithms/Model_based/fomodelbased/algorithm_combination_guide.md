## Command Line Arguments Explained

You can use the following command line arguments to control algorithm selection:

```bash
python run_pipeline.py [--config CONFIG_FILE] [--timesteps N] 
                       [--aggregation {LP,DP}] 
                       [--trading {bidding,market-clearing}]
                       [--disaggregation {proportional,average}]
                       [--managers N] [--users USERS_LIST]
```

Where:

- `--config`: Configuration file path (optional, if not provided, default configuration is used)
- `--timesteps`: Number of time steps to simulate (default is 24 hours)
- `--aggregation`: Aggregation algorithm selection, can be `LP` or `DP` (default is `LP`)
- `--trading`: Trading algorithm selection, can be `bidding` or `market-clearing` (default is `bidding`)
- `--disaggregation`: Disaggregation algorithm selection, can be `proportional` or `average` (default is `proportional`)
- `--managers`: Number of Managers (default is 4)
- `--users`: Number of users per Manager, comma separated (default is `6,10,8,12`)

### Default Configuration (LP aggregation + bidding trading + proportional disaggregation)

```bash
python run_pipeline.py
```



In addition to command line arguments, you can also control algorithm parameters more precisely through a configuration file:

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

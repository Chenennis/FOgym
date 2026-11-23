# FOgym: A FlexOffer-Based Multi-Agent Reinforcement Learning Benchmark

FOgym is an open-source research environment for developing and evaluating multi-agent reinforcement learning (MARL) methods in cyber-physical energy systems. It models a **bottom-up FlexOffer (FO) workflow**, from device flexibility generation through aggregation and trading to device-level scheduling and disaggregation.

FOgym is designed to help researchers **run existing methods, compare alternatives under common configurations, and integrate new methods into individual pipeline components**. Its purpose is to provide a reusable testing environment, rather than a single fixed energy-management algorithm.

## Key Features

- **End-to-end FlexOffer workflow:** Generation, aggregation, trading, and scheduling/disaggregation in one environment.
- **Multi-agent learning:** Manager-level interaction with decentralized observations and coordinated energy management.
- **Built-in RL baselines:** FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, and FOSQDDPG.
- **Modular methods:** Select built-in aggregation, trading, and disaggregation strategies or integrate your own implementations.
- **Benchmark configurations:** A 4-manager / 36-user / 118-device setting and a 10-manager / 90-user / 328-device setting.
- **Research-oriented evaluation:** Reuse common environment settings and evaluation logic when comparing methods.

## System Overview

FOgym represents a bottom-up energy management system with managers, users, and flexible devices. Each stage exchanges structured FlexOffer-related information with the next stage.

```text
Devices and users
       |
       v
FO Generation (device-level flexibility and RL decisions)
       |
       v
FO Aggregation (combine and track source FlexOffers)
       |
       v
FO Trading (bids, market clearing and trades)
       |
       v
FO Scheduling / Disaggregation (allocate energy to devices)
       |
       v
Environment updates and evaluation
```

| Component | Role | Source directory |
| --- | --- | --- |
| FO Generation | Model devices and generate FlexOffers using RL-related decisions | `fo_generate/`, `algorithms/` |
| FO Aggregation | Combine compatible offers while tracking their sources | `fo_aggregate/` |
| FO Trading | Process bids and produce trading outcomes | `fo_trading/` |
| FO Scheduling / Disaggregation | Allocate aggregated or traded energy to individual devices | `fo_schedule/` |

The device ecosystem includes batteries, heat pumps, electric vehicles, photovoltaic generation, and dishwashers. See [System Architecture](SYSTEM_ARCHITECTURE.md) for the detailed environment design and data flow.

## Installation

Clone the repository and work from its root directory:

```powershell
git clone https://github.com/Chenennis/FOgym.git
cd FOgym
```

Create and activate a virtual environment (Windows PowerShell example):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The code uses PyTorch, NumPy, pandas, Gymnasium and other Python packages. A number of features also use plotting, progress reporting, and experiment-related dependencies. **This branch does not yet contain a complete, verified dependency-lock or requirements file.** Install the dependencies appropriate to the environment and training path you intend to use; Python 3.11 is the version used in recent development tests. GPU use is optional.

Before running an experiment, check available arguments:

```powershell
python run_fo_pipeline.py --help
```

## Quick Start

Run a short example using the built-in FOMAPPO algorithm and the 4-manager benchmark configuration:

```powershell
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 1 --time_horizon 24 --num_managers 4 --num_users 36 --data_config 36users --no_gpu --log_verbosity brief --results_dir results/quick_start --experiment_name quick_start
```

This command selects a 24-step episode using CPU execution. Once the environment is configured, increase `--num_episodes` for longer training runs.

To select another built-in RL algorithm, change the `--rl_algorithm` argument, for example:

```powershell
python run_fo_pipeline.py --rl_algorithm fomatd3 --num_episodes 1 --time_horizon 24 --num_managers 4 --num_users 36 --data_config 36users --no_gpu
```

## Supported Methods

| Stage | Built-in options |
| --- | --- |
| RL | `fomappo`, `fomaippo`, `fomaddpg`, `fomatd3`, `fosqddpg` |
| Aggregation | `LP`, `DP` |
| Trading | `market_clearing`, `bidding` |
| Disaggregation | `average`, `proportional` |
| Scheduling policy | `priority`, `fairness`, `cost` |

These choices can be combined through the main pipeline's command-line arguments. The scheduling policy is distinct from the disaggregation method.

## Benchmark Configurations

FOgym provides two main experiment scales:

| Setting | Managers | Users | Devices | Data configuration |
| --- | ---: | ---: | ---: | --- |
| Standard | 4 | 36 | 118 | `36users` / `4manager` |
| Larger-scale | 10 | 90 | 328 | `10manager` |

The pipeline exposes `--num_managers`, `--num_users`, `--data_config`, `--time_horizon`, and `--seed`. When changing benchmark scale, keep manager/user counts and the selected data configuration consistent.

FOgym uses FlexOffer and device-related data and configurations located under `data/`. Consult the source and configuration files for the particular dataset selection and any preprocessing assumptions.

## Extending FOgym

A central feature of FOgym is that researchers can replace or extend one module while reusing the remaining FlexOffer pipeline.

| Extension point | What you can develop |
| --- | --- |
| **FO Generation** | New RL methods and their observation/action/training integration |
| **FO Aggregation** | New FlexOffer combination and compatibility methods |
| **FO Trading** | New bidding, matching, pricing or clearing strategies |
| **FO Disaggregation** | New energy-allocation methods for individual devices |

The general process is to **implement a method, register or select it, connect it to the pipeline, and validate its behavior**. New methods may require updates to pipeline initialization or algorithm-selection code; FOgym does not automatically discover all external classes.

**Detailed instructions:** [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md) explains the required interfaces, source files, code templates, integration points, and validation checks.

## Project Structure

```text
FOgym/
├── README.md                  # Project overview and first steps
├── EXTENSION_GUIDE.md         # How to integrate new methods
├── SYSTEM_ARCHITECTURE.md     # Architecture and internal data flow
├── run_fo_pipeline.py         # Main pipeline entry point
├── run_experiments.py         # Experiment orchestration
├── algorithms/                # MARL implementations and adapters
├── fo_generate/               # Device models and FO generation
├── fo_aggregate/              # FO aggregation
├── fo_trading/                # Trading and market mechanisms
├── fo_schedule/               # Scheduling and disaggregation
├── fo_common/                 # Shared structures and utilities
├── data/                      # Experiment inputs and configurations
└── tests/                     # Correctness and integration tests
```

## Testing

Run the existing test suite from the repository root after installing test dependencies:

```powershell
python -m pytest tests/ -q
```

These tests cover existing implementations; developers integrating their own methods should add component-specific and end-to-end tests as described in the [Extension Guide](EXTENSION_GUIDE.md).

## Documentation

- [Extension Guide](EXTENSION_GUIDE.md): Integrate and validate new RL, aggregation, trading and disaggregation methods.
- [System Architecture](SYSTEM_ARCHITECTURE.md): Understand the environment, observations, rewards and FO pipeline.

## License

FOgym is distributed under the [MIT License](LICENSE).

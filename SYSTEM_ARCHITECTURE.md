# FOgym System Architecture

This document describes the major components, data flow, and implementation boundaries of FOgym. For an introduction and basic commands, see [README.md](README.md). To integrate your own method, see [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

## 1. Purpose and system model

FOgym is a research environment for studying manager-level multi-agent reinforcement learning (MARL) and bottom-up FlexOffer (FO) management. It combines a device ecosystem with a multi-stage pipeline:

```text
User preferences + device states + external signals
                        |
                        v
             Device-level FO generation
                        |
                        v
               Manager aggregation
                        |
                        v
             Energy trading / clearing
                        |
                        v
              FO disaggregation
                        |
                        v
          Device scheduling and state updates
                        |
                    next step
```

Managers coordinate collections of users and devices. RL agents interact with observations and actions defined by the environment. In the built-in MARL paths, manager-level decisions affect downstream FO generation and the subsequent pipeline stages. One experiment typically uses a 24-step horizon; timing and population sizes are configurable.

## 2. Repository components

| Directory / file | Responsibility |
| --- | --- |
| `fo_generate/` | Device models, environment logic and FlexOffer generation |
| `fo_aggregate/` | Device/user/manager associations and FO aggregation |
| `fo_trading/` | Bids, market clearing and trade records |
| `fo_schedule/` | Scheduling and device-level energy disaggregation |
| `fo_common/` | Shared data structures, configurations, algorithm bases and observation utilities |
| `algorithms/` | Built-in MARL models, policies, buffers and adapters |
| `run_fo_pipeline.py` | Experiment orchestration, module initialization and training paths |
| `run_experiments.py` | Batch experiment orchestration |
| `tests/` | Component-level and integration correctness checks |

The pipeline has four **energy-processing stages**: Generation, Aggregation, Trading and Scheduling/Disaggregation. The RL implementations are the learning/control layer used with this pipeline; they are not an additional energy-processing stage.

## 3. Device and user hierarchy

The default small-scale configuration uses **4 managers, 36 users and 118 devices**; a larger configuration uses **10 managers, 90 users and 328 devices**. Device types include battery storage, heat pumps, EVs, PV generation and dishwashers. The code in `fo_generate/` models device behavior, energy limits and device-specific flexibility.

Manager, user and device ownership matter throughout the pipeline. A manager should aggregate its own source offers, and disaggregation should map energy back to the corresponding devices. Inspect `fo_aggregate/manager.py`, `fo_aggregate/aggregator.py` and the per-timestep functions in `run_fo_pipeline.py` when modifying these relationships.

Typical device-level restrictions differ: batteries/EVs have state-of-charge and power constraints; heat pumps have comfort and thermal restrictions; dishwashers have operating windows and task-completion requirements; PV depends on production availability. Exact limits, default values and reward calculations should be read from the respective device model and configuration rather than inferred from the examples in this overview.

## 4. Learning and observation layer

FOgym includes five built-in MARL choices:

| Algorithm | General approach | Source |
| --- | --- | --- |
| FOMAPPO | Shared-policy multi-agent PPO | `algorithms/MAPPO/` |
| FOMAIPPO | Independent-policy PPO | `algorithms/MAPPO/` |
| FOMADDPG | Multi-agent deterministic actor–critic | `algorithms/MADDPG/` |
| FOMATD3 | Multi-agent twin-delayed deterministic actor–critic | `algorithms/MATD3/` |
| FOSQDDPG | Shapley-based credit assignment with deterministic policy learning | `algorithms/SQDDPG/` |

The environment follows a decentralized, partially observed multi-agent setting. Manager observations can combine private device/user information, shared external/market signals, and restricted information about other managers. Observation contents and dimensions depend on the active environment and configuration; they are **not** universally fixed at one static dimension across all managers or algorithms.

Relevant code includes `fo_generate/unified_mdp_env.py`, `fo_common/observation.py`, `global_observation_config.json`, and the algorithm-specific adapters. The configured observation processing may include noise or restricted information visibility. Read these implementations to determine which features are actually exposed in a particular experiment.

`fo_common/base_algorithm.py` defines the `BaseMARL` abstraction (network setup, optimizers, replay buffer, action selection, learning and checkpoint hooks), but the built-in implementations also use specialized training and buffer adapters. In `run_fo_pipeline.py`, manager-level algorithms have distinct training routines. A generic custom-agent path uses a different interface and should not be assumed equivalent; see the [Extension Guide](EXTENSION_GUIDE.md).

### Rewards and objectives

FOgym models economic incentives, user/device requirements, and coordination through device-specific and manager/system-level logic. The pipeline exposes configurable reward weights via `--reward_alpha`, `--reward_beta`, `--reward_delta` and `--reward_lambda`.

The exact calculation differs by device model and RL integration path. Reward equations and weight values in historical design notes should not be interpreted as a single, verified formula for every algorithm. For reproducible research, inspect the actual reward construction in the chosen path and record the parameters passed to the run.

## 5. FlexOffer processing pipeline

### 5.1 FO generation

The generation stage uses device states, user requirements and RL-related decisions to create device-level flexibility offers. FlexOffers contain time-profile and energy flexibility information needed for downstream processing. Implementation entry points include `fo_generate/unified_mdp_env.py`, `fo_generate/dfo.py`, `fo_generate/sfo.py` and `fo_generate/inference.py`.

### 5.2 FO aggregation

The aggregation stage accepts `FlexOffer` objects and returns `AggregatedFlexOffer` records. Each aggregated result contains an aggregated offer and `source_fo_ids`, which preserve the relationship to contributing FOs.

`fo_aggregate/aggregator.py` provides a `FOAggregator` base with `initialize(...)`, `aggregate(...)`, and processing functions. Built-in selections are LP (Longest Profile) and DP (Dynamic Profile). The current implementation also preserves offers that cannot be merged as standalone results rather than silently discarding them.

The pipeline integrates aggregation through `FOPipeline._setup_aggregators()` and per-timestep aggregation logic. Be careful to retain manager provenance, offer membership, compatibility and energy ranges when changing aggregation.

### 5.3 FO trading

Trading turns manager bids into market-clearing outcomes and trade records. The abstract `TradingAlgorithm` in `fo_trading/pool.py` defines `process_bids(bids)` and `generate_trades(clearing_results, bids)`. Built-in implementations cover `market_clearing` and `bidding`, selected through `TradingPool`.

Check prices, quantities, counterparty records, and trade consistency when modifying market behavior. See `FOPipeline._setup_trading_pool()` and `_trade_flexoffers_for_timestep()` for pipeline integration.

### 5.4 FO scheduling and disaggregation

The scheduling stage maps traded or aggregated energy to individual devices and updates execution state. `fo_schedule/scheduler.py` defines `DisaggregationRequest`, `DisaggregationResult` and the `DisaggregationAlgorithm` interface.

The main CLI offers `average` and `proportional` disaggregation, plus scheduling policies `priority`, `fairness`, and `cost`. A **disaggregation algorithm** and a **scheduling policy** are different selections. In addition to shape and energy accounting, allocation must satisfy applicable device restrictions.

## 6. End-to-end data flow and consistency

A conceptual single-step flow is:

1. Read device states, user settings and external signals.
2. Construct observations and obtain actions from the active policy.
3. Generate device-level FOs.
4. Group and aggregate eligible offers under their responsible managers.
5. Submit bids and process clearing/trading.
6. Disaggregate the relevant energy to devices.
7. Update devices, schedules and recorded metrics for the next step.

In the code, consult `run_fo_pipeline.py` functions `_aggregate_flexoffers_for_timestep()`, `_trade_flexoffers_for_timestep()`, `_disaggregate_flexoffers_for_timestep()`, and `_schedule_and_update_states()`. Algorithm-specific rollout paths can invoke components differently; this diagram is conceptual, not a guarantee that every training routine uses identical call sequences.

Cross-stage consistency checks include preserving manager ownership, maintaining source-FO provenance, avoiding dropped or duplicated energy offers, respecting trade quantities, and ensuring feasible device allocations.

## 7. Configuration and experiments

The main entry point is `run_fo_pipeline.py`. Some important parameters are:

| Parameter | Role |
| --- | --- |
| `--rl_algorithm` | Select built-in or integrated RL method |
| `--aggregation_method` | LP or DP |
| `--trading_strategy` | Market-clearing or bidding implementation |
| `--disaggregation_method` | Average or proportional allocation |
| `--scheduling_method` | Scheduling policy |
| `--data_config` | Dataset configuration (`36users`, `4manager`, `10manager`) |
| `--num_managers`, `--num_users` | Population sizes |
| `--num_episodes`, `--time_horizon`, `--seed` | Experiment duration and reproducibility |
| `--log_verbosity` | `minimal`, `brief`, `detailed`, `debug` |
| `--no_gpu` | Force CPU execution |

Use `python run_fo_pipeline.py --help` for the full argument list. For example commands and startup instructions, see [README.md](README.md). For new algorithm interfaces, registration and custom experiments, see [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

## 8. Architecture boundaries

- The pipeline's four stages are modular, but adding a new class does not automatically register it with all entry points.
- Aggregation, trading and disaggregation have distinct contracts, and all have selection logic in `run_fo_pipeline.py`.
- RL training paths are algorithm-specific. Action units, rollout semantics, terminal handling, replay/trajectory storage and save/load need to stay aligned.
- Device model extension involves coordinating device creation, environment integration, FO generation and tests, not merely declaring a new enum value.
- A valid benchmark comparison should keep datasets, seeds, population scale, metrics and other components consistent.

For detailed extension steps, refer to [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md).

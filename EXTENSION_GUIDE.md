# Extending FOgym

FOgym is a modular research environment for benchmarking methods across the FlexOffer (FO) workflow: **FO generation**, **FO aggregation**, **FO trading**, and **FO scheduling/disaggregation**. Researchers can develop a method for one stage while reusing the other stages and the same experimental environment.

This guide describes the extension points in the current `fix/environment-validation` implementation. It is intended for researchers who already know how to run the baseline pipeline; start with [README.md](README.md) if you are new to FOgym.

> **Important:** FOgym has module interfaces and algorithm registries, but it is **not an automatic plugin system**. Registering a class alone may not connect it to the main experiment pipeline. The steps below distinguish the module API from the additional pipeline integration work.

## Contents

1. [Extension workflow](#1-extension-workflow)
2. [FO generation: new RL methods](#2-fo-generation-new-rl-methods)
3. [FO aggregation: new aggregation methods](#3-fo-aggregation-new-aggregation-methods)
4. [FO trading: new trading strategies](#4-fo-trading-new-trading-strategies)
5. [FO disaggregation: new allocation methods](#5-fo-disaggregation-new-allocation-methods)
6. [Running controlled experiments](#6-running-controlled-experiments)
7. [Testing and validation](#7-testing-and-validation)
8. [Adding device models](#8-adding-device-models)
9. [Integration caveats](#9-integration-caveats)

## 1. Extension workflow

For any component, follow the same four stages:

1. **Implement:** Find the existing implementation and its input/output contract; add your method to the appropriate module.
2. **Register or select:** Add the method name to the relevant registry or selection function.
3. **Integrate:** Ensure `run_fo_pipeline.py` initializes and actually executes that method, including its training or processing path.
4. **Validate:** Test the component independently, then run the complete FO pipeline with the same benchmark data and settings as the baseline.

| Stage | Main source files | Existing choices |
| --- | --- | --- |
| FO generation / RL | `algorithms/`, `fo_common/base_algorithm.py`, `run_fo_pipeline.py` | FOMAPPO, FOMAIPPO, FOMADDPG, FOMATD3, FOSQDDPG |
| FO aggregation | `fo_aggregate/aggregator.py`, `fo_aggregate/manager.py` | LP, DP |
| FO trading | `fo_trading/pool.py` | `market_clearing`, `bidding` |
| FO disaggregation | `fo_schedule/scheduler.py` | `average`, `proportional` (CLI choices) |

**Key rule:** Verify the selected implementation at runtime. A successful command or an accepted algorithm name is not enough: some code paths fall back to an existing method.

## 2. FO generation: new RL methods

### What to extend

RL methods choose actions from environment observations and learn from collected experiences. Their actions must be compatible with the device environments and the subsequent FO generation and trading workflow. The built-in algorithms use manager-level multi-agent training paths; a new algorithm must be connected to the relevant rollout, action conversion, experience collection, update, and evaluation code.

**Relevant files:** `fo_common/base_algorithm.py`, `algorithms/`, and `run_fo_pipeline.py`. For concrete multi-agent integration patterns, inspect the existing `fomappo`, `fomaddpg`, `fomatd3`, and `fosqddpg` adapters and training routines rather than assuming they all share a single trainer.

### BaseMARL contract

`fo_common/base_algorithm.py` defines `BaseMARL`. Required abstract methods include:

| Method | Purpose |
| --- | --- |
| `_setup_networks()` | Initialize models |
| `_setup_optimizers()` | Initialize optimizers |
| `_setup_replay_buffer()` | Initialize learning storage |
| `select_actions(states, add_noise=True)` | Choose multi-agent actions |
| `update()` | Update the policy |
| `_add_algorithm_specific_save_data(save_dict)` | Add checkpoint data |
| `_load_algorithm_specific_data(checkpoint)` | Restore checkpoint data |

The class also provides `store_experience(states, actions, rewards, next_states, dones, **kwargs)`, `save_models(path)`, and `load_models(path)`. Implementations must preserve the shape, units, device placement, and ordering required by their actual training path. Subclassing `BaseMARL` does **not** automatically make a new method executable through the built-in manager-level training loop.

### Path A: integrate a manager-level MARL method (recommended for comparable MARL benchmarks)

1. Implement the new method and any action/observation or replay-buffer adapters under `algorithms/`.
2. Import and register the class in the startup code, for example:

   ```python
   # Illustrative registration; MyRLAgent must be implemented and imported.
   RLRegistry.register("my_rl", MyRLAgent)
   ```

3. Extend the algorithm initialization and training dispatch in `run_fo_pipeline.py`. Built-in algorithms have explicit, algorithm-specific code paths. A registry entry alone does not add a new manager-level training routine.
4. Validate manager count, observation/action dimensions, action normalization and physical units, transition alignment, terminal handling, and model persistence.
5. Run a short episode and confirm that the new rollout and update functions are called.

Use an existing built-in adapter and its corresponding training path as the implementation reference. Registration code must execute in the pipeline process **before** the algorithm is selected.

### Path B: dynamically load a custom agent (different execution interface)

The entry-point parser also accepts `--custom_agent_path` and `--custom_agent_name`. The module path must end in an importable class name. For example, after implementing `MyRLAgent` in `algorithms/my_rl_agent.py`:

```powershell
python run_fo_pipeline.py --custom_agent_path algorithms.my_rl_agent.MyRLAgent --custom_agent_name my_rl --rl_algorithm my_rl --num_episodes 1 --time_horizon 24 --no_gpu
```

**This command is an integration template, not a working example until the class is implemented.**

The current custom-agent code initializes agents using `state_dim`, `action_dim`, `max_action`, and `device`. It then attempts either `train(env, num_episodes=...)` or a path using `select_action(state, evaluate=False)`, optional `store_transition(...)`, and `update()`. Those methods are **not the same interface** as `BaseMARL.select_actions(...)`.

Moreover, this dynamic path operates through per-user environments and must **not** be treated as equivalent to the built-in manager-level MARL execution without additional adaptation and verification. Its current initialization logic can fall back to FOMAPPO on failure; inspect logs and the active algorithm after setup. If your research requires a comparable manager-level MARL benchmark, use Path A.

## 3. FO aggregation: new aggregation methods

### Interface and data flow

**Relevant files:** `fo_aggregate/aggregator.py`, `fo_aggregate/manager.py`, and `run_fo_pipeline.py`.

The aggregation module takes a list of `FlexOffer` objects and returns a list of `AggregatedFlexOffer` records. Each record includes the resulting FO and `source_fo_ids` for provenance. The current implementations are `LongestProfileAggregator` (LP) and `DynamicProfileAggregator` (DP).

`FOAggregator` exposes `aggregate(flex_offers)` and an abstract initializer:

```python
from fo_aggregate.aggregator import FOAggregator

class MyAggregator(FOAggregator):
    def initialize(self, flex_offers):
        # Return (PF, UF, fini, round_number).
        # PF: candidate/processed set; UF: unprocessed set;
        # fini: starting FlexOffer (or None).
        raise NotImplementedError
```

This is an **interface template**, not a runnable method. The inherited aggregation process also uses `process(...)`, `binary_aggregation(...)` and source-ID tracking. If your aggregation logic does not fit that process, override the relevant processing method or `aggregate(...)` and preserve the expected result structure.

### Connect the method

1. Implement and import `MyAggregator`.
2. Add a branch to `FOAggregatorFactory.create_aggregator(method, **kwargs)`, and update `get_available_methods()`.
3. Extend the `--aggregation_method` choices in `run_fo_pipeline.py`.
4. **Critically, update `FOPipeline._setup_aggregators()`**. The current implementation explicitly checks `["LP", "DP"]` and otherwise initializes DP even if the displayed method name is different.
5. Review other call sites that may still assume LP or DP.

Only after those changes can a command such as the following select a custom method:

```powershell
python run_fo_pipeline.py --aggregation_method MY_AGG --rl_algorithm fomappo --num_episodes 1 --time_horizon 24 --no_gpu
```

Check the **instantiated class**, not only the log line displaying the requested method.

### Required correctness checks

Maintain compatibility and power/time-flexibility constraints; preserve energy ranges, profile alignment, and provenance. In particular, every input FO must be accounted for in the results: rejected or unmerged FOs should remain available as standalone results instead of being silently dropped. Do not claim membership in an aggregate for an FO that was not accepted.

## 4. FO trading: new trading strategies

### Interface

**Relevant files:** `fo_trading/pool.py` and `run_fo_pipeline.py`.

Trading receives `Bid` objects and generates `ClearingResult` and `Trade` records. Subclass `TradingAlgorithm` and implement both abstract methods:

```python
from fo_trading.pool import TradingAlgorithm

class MyTrading(TradingAlgorithm):
    def __init__(self):
        super().__init__("my_trading")

    def process_bids(self, bids):
        # Return a list of ClearingResult objects.
        raise NotImplementedError

    def generate_trades(self, clearing_results, bids):
        # Return a list of Trade objects.
        raise NotImplementedError
```

The template must be implemented with valid bid, clearing, and trade semantics. Inspect `BiddingAlgorithm` or `MarketClearingAlgorithm` for concrete output construction.

### Register and integrate

`TradingAlgorithmFactory` provides `register_algorithm(name, class)`. If the class is placed in `fo_trading/my_trading.py`, register it before the pipeline initializes `TradingPool`:

```python
from fo_trading.my_trading import MyTrading
from fo_trading.pool import TradingAlgorithmFactory

TradingAlgorithmFactory.register_algorithm("my_trading", MyTrading)
```

Also extend the `--trading_strategy` choices in `run_fo_pipeline.py` and ensure its setup passes the custom name (and any required constructor parameters) to `TradingPool`. Registration in a separate Python process does not persist into a later CLI run.

After integration:

```powershell
python run_fo_pipeline.py --trading_strategy my_trading --rl_algorithm fomappo --num_episodes 1 --time_horizon 24 --no_gpu
```

Check that the new trading class processes bids and produces valid, consistent records that downstream scheduling can consume. Avoid changing settlement semantics or energy conservation implicitly.

## 5. FO disaggregation: new allocation methods

### Interface

**Relevant files:** `fo_schedule/scheduler.py` and `run_fo_pipeline.py`.

Disaggregation allocates energy from aggregated or traded results back to device-level records. `DisaggregationAlgorithm.disaggregate(request)` receives a `DisaggregationRequest` and returns a `DisaggregationResult`.

```python
from fo_schedule.scheduler import DisaggregationAlgorithm

class MyDisaggregation(DisaggregationAlgorithm):
    def __init__(self):
        super().__init__("my_disaggregation")

    def disaggregate(self, request):
        # Return a properly constructed DisaggregationResult.
        raise NotImplementedError
```

The request includes `aggregated_result`, `original_data`, `total_energy`, `time_step` and `metadata`. A result includes `disaggregated_data`, `algorithm_used`, `allocation_ratios`, `total_allocated_energy`, and `metadata`. The data and ratio lists must have matching lengths; allocated energy must be nonnegative.

### Register and integrate

Register the new method **in the pipeline process before scheduler initialization**:

```python
from fo_schedule.my_disaggregation import MyDisaggregation
from fo_schedule.scheduler import DisaggregationAlgorithmFactory

DisaggregationAlgorithmFactory.register_algorithm(
    "my_disaggregation", MyDisaggregation
)
```

Then extend `--disaggregation_method` choices in `run_fo_pipeline.py` and ensure the method name reaches both the scheduler and the disaggregation code path used by the pipeline. The module has additional aliases, but the CLI currently exposes only `average` and `proportional`. An unknown algorithm can fall back to a default in some disaggregation paths; verify actual selection.

After integration:

```powershell
python run_fo_pipeline.py --disaggregation_method my_disaggregation --rl_algorithm fomappo --num_episodes 1 --time_horizon 24 --no_gpu
```

Check conservation/consistency of allocated energy, device limits, applicable flexibility constraints, and the mapping between original devices and allocations. Disaggregation algorithms are separate from the `priority`, `fairness`, and `cost` scheduling policies selected by `--scheduling_method`.

## 6. Running controlled experiments

Use the same benchmark configuration, random seed, episode count, time horizon, and downstream modules when comparing methods. FOgym provides configurations for **4 managers / 36 users / 118 devices** and **10 managers / 90 users / 328 devices**. Select a matching dataset and explicitly set the associated sizes; inspect available configuration files before adapting scale-dependent code.

For a basic, **built-in** smoke test (no extension required), run from the repository root:

```powershell
python run_fo_pipeline.py --rl_algorithm fomappo --num_episodes 1 --time_horizon 24 --num_managers 4 --num_users 36 --data_config 36users --no_gpu --log_verbosity brief --results_dir results/docs_smoke_test --experiment_name docs_smoke_test
```

The custom method examples above are intended to be run **only after** implementing, registering, and integrating the named method. They are not expected to work on an unmodified checkout.

For comparisons, evaluate the appropriate measures for the changed component (e.g. user cost and constraint satisfaction, aggregation quality/provenance, trade consistency, or allocation feasibility), while holding other configuration choices fixed.

## 7. Testing and validation

Before running a full comparison, validate the new component and the end-to-end execution.

1. **Interface:** Verify required inputs, outputs, shapes, types, and error handling.
2. **Selection:** Confirm logs and instantiated objects identify the new class—not a fallback.
3. **Domain correctness:** Check energy accounting, flexibility windows, device feasibility, and provenance where relevant.
4. **Component test:** Add targeted tests in `tests/` for the algorithm and its boundary cases.
5. **Integration test:** Run a short 24-step pipeline; check the module's outputs are consumed by the next stage.
6. **Regression:** Re-run existing tests after integration.

```powershell
python -m pytest tests/ -q
```

For aggregation changes in particular, refer to `tests/test_aggregation_correctness.py` and add analogous tests for your implementation. Passing the repository's existing tests does **not** prove a new method is correct; those tests may not exercise your added code.

## 8. Adding device models

Device models are an additional extension area beyond the four algorithmic pipeline modules. A new device type requires more than implementing a `step()` function: its creation, observations and actions, constraints, FO generation, and downstream compatibility must all be connected.

1. Examine the existing implementations in `fo_generate/` (battery, heat pump, EV, PV and dishwasher) and the device creation/configuration code used in `run_fo_pipeline.py`.
2. Implement the new device dynamics and its operating limits using the same environment-facing conventions as the relevant existing device types.
3. Update the device type mapping and creation/configuration paths so that the new type is instantiated with valid parameters and included in the correct user/manager inventory.
4. Specify how the device's feasible energy and time flexibility are represented as a `FlexOffer` and how actions map to its physical control quantities.
5. Check observation/action shapes, energy units, scheduling feasibility, and source-device provenance. Add targeted tests and an end-to-end pipeline case.

The legacy developer guide contains schematic examples involving `DeviceMDPInterface`, `DeviceType`, and `device_factory`; those examples are **not** asserted to be drop-in implementations for the current fix branch. Use the actual model constructors and integration code in the branch you develop against.

## 9. Integration caveats

- **RL:** Built-in manager-level MARL and dynamically loaded custom agents have different training interfaces and execution paths.
- **Aggregation:** The factory and pipeline initialization both need to recognize new method names; otherwise the pipeline may select DP.
- **Trading:** Registration is necessary but the CLI selection and startup registration must also be updated.
- **Disaggregation:** Registration, CLI choices, and active scheduler/disaggregator selection must all agree; some code paths can fall back.
- **Examples:** Every `My...` class in this guide is a structural illustration and contains `NotImplementedError`; it must be implemented and tested before use.
- **Scope:** This guide covers algorithmic extension points, not a fully automated third-party plugin packaging API. Adding new device types may require coordinated changes to device models, configuration, device creation, and FO-generation handling; such work should be validated against the current code rather than copied from older illustrative guides.

For an overview of the system and its data flow, consult [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md). For installation and baseline usage, return to [README.md](README.md).

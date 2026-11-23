"""Pipeline disaggregation must not fabricate or over-allocate device energy."""
from types import SimpleNamespace

from run_fo_pipeline import FOPipeline
from fo_schedule.scheduler import AggregatedResultDisaggregator
from fo_trading.pool import Trade


def _pipeline():
    pipeline = FOPipeline.__new__(FOPipeline)
    pipeline.disaggregation_method = "proportional"
    pipeline.disaggregator = AggregatedResultDisaggregator(
        time_horizon=24, default_algorithm="proportional"
    )
    return pipeline


def _trade(number, quantity):
    return Trade(
        trade_id=f"trade_{number}", buyer_id="manager_1", seller_id="manager_2",
        energy_type="electricity", quantity=quantity, price=0.2, time_step=8
    )


def _systems():
    return {
        "manager_1": {
            "a": {"energy_min": 0.0, "energy_max": 1.0},
            "b": {"energy_min": 0.0, "energy_max": 2.0},
        },
        "manager_2": {},
    }


def test_no_allocation_to_nonexistent_buyer_devices():
    assert _pipeline()._disaggregate_flexoffers_for_timestep(
        [_trade(1, 1.0)], {"manager_2": {}}, 8
    ) == []


def test_disaggregation_rejects_more_than_total_device_capacity():
    assert _pipeline()._disaggregate_flexoffers_for_timestep(
        [_trade(1, 4.0)], _systems(), 8
    ) == []


def test_disaggregation_rejects_cumulative_overallocation_across_trades():
    assert _pipeline()._disaggregate_flexoffers_for_timestep(
        [_trade(1, 2.0), _trade(2, 2.0)], _systems(), 8
    ) == []


def test_feasible_trade_is_conserved_and_bounded_per_device():
    result = _pipeline()._disaggregate_flexoffers_for_timestep(
        [_trade(1, 2.0)], _systems(), 8
    )
    assert result
    assert abs(sum(row["allocated_energy"] for row in result) - 2.0) < 1e-9
    by_device = {row["device_id"]: row["allocated_energy"] for row in result}
    assert by_device["a"] <= 1.0 + 1e-9
    assert by_device["b"] <= 2.0 + 1e-9

"""Regression tests for Pipeline's dictionary-shaped trade disaggregation contract."""
import pytest

from fo_schedule.scheduler import AggregatedResultDisaggregator


@pytest.mark.parametrize("method", ["average", "proportional"])
def test_disaggregation_dict_trade_quantity_is_authoritative(method):
    disaggregator = AggregatedResultDisaggregator(time_horizon=24, default_algorithm=method)
    devices = [
        {"manager_id": "manager_1", "device_id": "a", "energy": 3.0},
        {"manager_id": "manager_1", "device_id": "b", "energy": 7.0},
    ]
    allocations = disaggregator.disaggregate(
        aggregated_result={"total_energy": 2.0, "trade_id": "trade_8"},
        original_data=devices,
        weighting_method=method,
        time_step=8,
    )
    assert len(allocations) == 2
    assert sum(a["allocated_energy"] for a in allocations) == pytest.approx(2.0)
    assert all(a["allocated_energy"] >= 0 for a in allocations)


@pytest.mark.parametrize("method", ["average", "proportional"])
def test_zero_trade_quantity_never_falls_back_to_device_energy(method):
    disaggregator = AggregatedResultDisaggregator(time_horizon=24, default_algorithm=method)
    devices = [
        {"manager_id": "manager_1", "device_id": "a", "energy": 3.0},
        {"manager_id": "manager_1", "device_id": "b", "energy": 7.0},
    ]
    allocations = disaggregator.disaggregate(
        aggregated_result={"total_energy": 0.0, "trade_id": "unmatched"},
        original_data=devices,
        weighting_method=method,
        time_step=8,
    )
    assert sum(a["allocated_energy"] for a in allocations) == pytest.approx(0.0)

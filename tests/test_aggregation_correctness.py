"""Regression tests for the aggregation path used by FOPipeline."""
from datetime import datetime, timedelta

import pytest

from fo_aggregate.aggregator import FOAggregatorFactory
from fo_common.flexoffer import FOFactory, FOSlice, FlexOffer


BASE = datetime(2026, 1, 1)


def make_fo(device, hour, energy_min, energy_max):
    return FOFactory.create_hourly_fo(
        device_id=device,
        device_type="battery",
        hour=hour,
        base_time=BASE,
        slices_per_hour=1,
        energy_profile=[(energy_min, energy_max)],
    )


@pytest.mark.parametrize("method", ["LP", "DP"])
def test_rejected_candidates_are_returned_without_loss_or_false_membership(method):
    # 95 kWh in one hour has a smaller RMSE to the 100-kW target than 96 kWh.
    first = make_fo("first", 8, 95.0, 95.0)
    second = make_fo("second", 8, 1.0, 1.0)
    results = FOAggregatorFactory.create_aggregator(method).aggregate([first, second])

    assert len(results) == 2
    assert sorted(tuple(r.source_fo_ids) for r in results) == [
        ("first_fo_h8",), ("second_fo_h8",)
    ]
    assert sum(r.aggregated_fo.total_energy_min for r in results) == pytest.approx(96.0)
    assert sum(r.aggregated_fo.total_energy_max for r in results) == pytest.approx(96.0)


@pytest.mark.parametrize("method", ["LP", "DP"])
def test_accepted_aggregation_conserves_slice_energy_and_provenance(method):
    first = make_fo("first", 8, 1.0, 2.0)
    second = make_fo("second", 8, 3.0, 4.0)
    results = FOAggregatorFactory.create_aggregator(method).aggregate([first, second])

    assert len(results) == 1
    result = results[0]
    assert set(result.source_fo_ids) == {first.fo_id, second.fo_id}
    assert result.aggregated_fo.total_energy_min == pytest.approx(4.0)
    assert result.aggregated_fo.total_energy_max == pytest.approx(6.0)
    assert result.aggregated_fo.slices[0].start_time == first.slices[0].start_time
    assert result.aggregated_fo.slices[0].end_time == first.slices[0].end_time


def test_binary_aggregation_rejects_different_hours():
    first = make_fo("first", 8, 1.0, 1.0)
    second = make_fo("second", 9, 2.0, 2.0)
    aggregator = FOAggregatorFactory.create_aggregator("DP")
    assert aggregator.binary_aggregation(first, second) is None


def test_binary_aggregation_rejects_different_slice_boundaries():
    first = make_fo("first", 8, 1.0, 1.0)
    second = make_fo("second", 8, 2.0, 2.0)
    second.slices[0].start_time += timedelta(minutes=15)
    second.slices[0].end_time += timedelta(minutes=15)
    aggregator = FOAggregatorFactory.create_aggregator("DP")
    assert aggregator.binary_aggregation(first, second) is None


def test_binary_aggregation_rejects_incompatible_slice_grids():
    first = make_fo("first", 8, 1.0, 1.0)
    second = FOFactory.create_hourly_fo(
        device_id="second", device_type="battery", hour=8,
        base_time=BASE, slices_per_hour=2,
        energy_profile=[(1.0, 1.0), (1.0, 1.0)]
    )
    aggregator = FOAggregatorFactory.create_aggregator("DP")
    assert aggregator.binary_aggregation(first, second) is None

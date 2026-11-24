"""Regression: aligned slices must conserve energy independently of slice IDs."""
from datetime import datetime, timedelta

import pytest

from fo_aggregate.aggregator import FOAggregatorFactory
from fo_common.flexoffer import FOSlice, FlexOffer


BASE = datetime(2026, 1, 1, 8)


def make_fo(device, ids, bounds):
    slices = []
    for position, (slice_id, (lower, upper)) in enumerate(zip(ids, bounds)):
        start = BASE + timedelta(minutes=30 * position)
        slices.append(FOSlice(
            slice_id=slice_id,
            start_time=start,
            end_time=start + timedelta(minutes=30),
            duration_minutes=30.0,
            energy_min=lower, energy_max=upper,
            device_id=device, device_type="battery",
        ))
    return FlexOffer(
        fo_id=f"{device}_fo", hour=8,
        start_time=BASE, end_time=BASE + timedelta(hours=1),
        device_id=device, device_type="battery", slices=slices
    )


def test_binary_aggregation_uses_aligned_slice_positions_not_ids():
    a = make_fo("a", [10, 20], [(1.0, 2.0), (3.0, 4.0)])
    b = make_fo("b", [100, 200], [(5.0, 6.0), (7.0, 8.0)])
    aggregate = FOAggregatorFactory.create_aggregator("DP").binary_aggregation(a, b)
    assert aggregate is not None
    assert [(s.energy_min, s.energy_max) for s in aggregate.slices] == pytest.approx(
        [(6.0, 8.0), (10.0, 12.0)]
    )
    assert aggregate.total_energy_min == pytest.approx(16.0)
    assert aggregate.total_energy_max == pytest.approx(20.0)


def test_binary_aggregation_rejects_reordered_slice_boundaries():
    a = make_fo("a", [10, 20], [(1.0, 2.0), (3.0, 4.0)])
    b = make_fo("b", [20, 10], [(5.0, 6.0), (7.0, 8.0)])
    b.slices.reverse()
    assert FOAggregatorFactory.create_aggregator("DP").binary_aggregation(a, b) is None

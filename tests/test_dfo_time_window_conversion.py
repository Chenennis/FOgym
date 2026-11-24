"""Regression tests for the repository's legacy dynamic DFO conversion contract.

The stand-alone pipeline currently assigns device DFO slices to a synthetic
hourly FO grid. This is a compatibility check, NOT proof of temporal dispatch
feasibility. Preserve this behavior until the environment's time semantics are
redesigned and validated end-to-end.
"""
from datetime import datetime, timedelta

import pytest

from fo_generate.dfo import DFOSlice, DFOSystem
from run_fo_pipeline import FOPipeline


def make_system(windows):
    system = DFOSystem(time_horizon=24, device_id="d1", device_type="battery")
    for index, (start, end) in enumerate(windows):
        system.add_slice(DFOSlice(
            time_step=index,
            energy_min=-1.0, energy_max=2.0, constraints=[],
            start_time=start, end_time=end,
        ))
    return system


def convert(system, timestep=8):
    return FOPipeline.__new__(FOPipeline)._convert_dfo_to_flexoffer(
        system, "d1", timestep
    )


def test_dynamic_multi_hour_dfo_remains_convertible():
    start = datetime(2026, 1, 1, 8, 7)
    result = convert(make_system([
        (start, start + timedelta(hours=1)),
        (start + timedelta(hours=1), start + timedelta(hours=3)),
    ]))
    assert result is not None
    assert len(result.slices) == 2
    assert result.total_energy_min == pytest.approx(-2)
    assert result.total_energy_max == pytest.approx(4)


def test_non_hour_aligned_windows_remain_convertible():
    start = datetime(2026, 1, 1, 8, 19)
    result = convert(make_system([
        (start, start + timedelta(minutes=20)),
        (start + timedelta(minutes=20), start + timedelta(minutes=40)),
    ]))
    assert result is not None
    assert result.hour == 8


def test_missing_dfo_windows_do_not_drop_device():
    assert convert(make_system([(None, None), (None, None)])) is not None


def test_legacy_synthetic_grid_is_contiguous_without_energy_loss():
    result = convert(make_system([(None, None), (None, None)]))
    assert result is not None
    assert result.slices[0].end_time == result.slices[1].start_time
    assert [slice.duration_minutes for slice in result.slices] == [2.0, 2.0]
    assert result.total_energy_max == pytest.approx(4)


def test_dfo_source_windows_are_not_mutated_by_conversion():
    start = datetime(2026, 1, 1, 8, 7)
    src = make_system([
        (start, start + timedelta(hours=1)),
        (start + timedelta(hours=1), start + timedelta(hours=2)),
    ])
    windows = [(x.start_time, x.end_time) for x in src.slices]
    assert convert(src) is not None
    assert [(x.start_time, x.end_time) for x in src.slices] == windows

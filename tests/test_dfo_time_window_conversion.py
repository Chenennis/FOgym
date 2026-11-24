"""DFO-to-FO conversion preserves time windows and rejects ambiguous grids."""
from datetime import datetime, timedelta

import pytest

from fo_generate.dfo import DFOSlice, DFOSystem
from run_fo_pipeline import FOPipeline


def make_system(windows):
    system = DFOSystem(time_horizon=24, device_id="d1", device_type="battery")
    for index, (start, end) in enumerate(windows):
        system.add_slice(DFOSlice(
            time_step=index,
            energy_min=1.0, energy_max=2.0, constraints=[],
            start_time=start, end_time=end,
        ))
    return system


def convert(system):
    return FOPipeline.__new__(FOPipeline)._convert_dfo_to_flexoffer(
        system, "d1", 8
    )


def test_preserves_original_nonuniform_slice_windows():
    start = datetime(2026, 1, 1, 8, 0)
    middle = start + timedelta(minutes=20)
    end = start + timedelta(hours=1)
    result = convert(make_system([(start, middle), (middle, end)]))
    assert result is not None
    assert result.start_time == start
    assert result.end_time == end
    assert [s.start_time for s in result.slices] == [start, middle]
    assert [s.end_time for s in result.slices] == [middle, end]
    assert [s.duration_minutes for s in result.slices] == pytest.approx([20, 40])
    assert result.total_energy_max == pytest.approx(4.0)


def test_without_explicit_windows_uses_complete_hour_not_two_minutes_each():
    result = convert(make_system([(None, None), (None, None)]))
    assert result is not None
    assert [s.duration_minutes for s in result.slices] == pytest.approx([30, 30])
    assert result.slices[0].end_time == result.slices[1].start_time
    assert result.slices[0].start_time == result.start_time
    assert result.slices[-1].end_time == result.end_time


def test_partial_slice_window_is_rejected():
    start = datetime(2026, 1, 1, 8, 0)
    result = convert(make_system([(start, None), (None, None)]))
    assert result is None


def test_overlapping_slice_windows_are_rejected():
    start = datetime(2026, 1, 1, 8, 0)
    result = convert(make_system([
        (start, start + timedelta(minutes=40)),
        (start + timedelta(minutes=20), start + timedelta(hours=1)),
    ]))
    assert result is None


def test_multi_hour_dfo_cannot_be_silently_compressed_to_one_hour():
    start = datetime(2026, 1, 1, 8)
    result = convert(make_system([
        (start, start + timedelta(hours=1)),
        (start + timedelta(hours=1), start + timedelta(hours=2)),
    ]))
    assert result is None

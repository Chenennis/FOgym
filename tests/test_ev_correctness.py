"""Regression tests for EV offer generation and charging limits (A3)."""
from datetime import datetime, timedelta

import pytest

from fo_generate.ev_model import EVModel, EVParameters, EVUserBehavior


START = datetime(2025, 1, 1, 18)


def make_ev(soc=0.4, fast=False, flexibility=0.8, with_behavior=True):
    params = EVParameters(
        ev_id="EV-TEST", battery_capacity=50.0,
        soc_min=0.1, soc_max=0.95, max_charging_power=10.0,
        efficiency=0.9, initial_soc=soc, fast_charge_capable=True,
    )
    behavior = EVUserBehavior(
        ev_id="EV-TEST", connection_time=START,
        disconnection_time=START + timedelta(hours=12),
        next_departure_time=START + timedelta(hours=12),
        target_soc=0.95, min_required_soc=0.6,
        fast_charge_preferred=fast, location="home", priority=3,
        charge_flexibility=flexibility,
    ) if with_behavior else None
    return EVModel(params, behavior)


def test_a3_dfo_generation_preserves_real_ev_state():
    ev = make_ev()
    ev.connect(START)
    ev.update_soc(1.0, current_time=START)
    before = vars(ev).copy()

    first = ev.generate_dfo(start_time=START, time_horizon=6)
    second = ev.generate_dfo(start_time=START, time_horizon=6)

    assert vars(ev) == before
    assert first.to_dict() == second.to_dict()
    assert len(first.slices) == 6
    assert first.slices[0].energy_max > 0
    # The midpoint forecast still advances internal SoC between slices.
    assert first.slices[1].constraints[0][1] < first.slices[0].constraints[0][1]


def test_a3_actual_charging_still_updates_real_state():
    ev = make_ev()
    ev.generate_dfo(START, 3)
    assert ev.current_soc == pytest.approx(0.4)
    ev.update_soc(2.0, current_time=START)
    assert ev.current_soc == pytest.approx(0.4 + 2.0 * 0.9 / 50.0)
    assert ev.total_charged_energy == pytest.approx(1.8)
    assert ev.last_charge_time == START


def test_a3_fast_charging_respects_remaining_capacity():
    ev = make_ev(soc=0.93, fast=True)
    lower, upper = ev.get_available_power(START)
    expected = (0.95 - 0.93) * 50.0 / 0.9
    assert lower == 0
    assert upper == pytest.approx(expected)
    dfo = ev.generate_dfo(START, 1)
    assert dfo.slices[0].energy_max == pytest.approx(expected)


def test_a3_fast_charging_preserves_full_power_when_feasible():
    ev = make_ev(fast=True)
    assert ev.get_available_power(START) == pytest.approx((0, 10.0))


def test_a3_existing_flexibility_and_urgency_rules_are_preserved():
    ev = make_ev()
    assert ev.get_available_power(START) == pytest.approx((0, 7.0))
    lower, upper = ev.get_available_power(START + timedelta(hours=6))
    assert lower == pytest.approx((0.6 - 0.4) * 50 / (6 * 0.9))
    assert upper == pytest.approx(7.0)


def test_a3_legacy_integer_horizon_call_remains_compatible():
    ev = make_ev(with_behavior=False)
    dfo = ev.generate_dfo(3)
    assert len(dfo.slices) == 3
    assert ev.current_soc == pytest.approx(0.4)
    assert ev.total_charged_energy == 0

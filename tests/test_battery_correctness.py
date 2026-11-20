"""Regression tests for battery model correctness (issue A1).

These tests intentionally describe expected physical behavior. They are expected
to fail against the uncorrected battery model on main. Do not weaken assertions
to make the current implementation pass.
"""

import pytest

from fo_generate.battery_model import BatteryModel, BatteryParameters


@pytest.fixture
def battery():
    """200 kWh battery, 100 kW power limits, fractional SoC."""
    return BatteryModel(
        BatteryParameters(
            battery_id="BAT-TEST",
            soc_min=0.20,
            soc_max=0.90,
            p_min=-100.0,
            p_max=100.0,
            efficiency=0.95,
            initial_soc=0.50,
            battery_type="test",
            capacity_kwh=200.0,
        )
    )


def test_charging_soc_is_normalized_by_capacity(battery):
    """Charging 1 kW for 1 h adds 0.95/200, not 0.95, to SoC."""
    result = battery.update_soc(power=1.0, time_step=1.0)
    assert result == pytest.approx(0.50 + 0.95 / 200.0)


def test_discharging_soc_is_normalized_by_capacity(battery):
    """Delivering 1 kWh reduces fractional SoC by 1/(efficiency*capacity)."""
    result = battery.update_soc(power=-1.0, time_step=1.0)
    assert result == pytest.approx(0.50 - 1.0 / (0.95 * 200.0))


def test_available_power_uses_energy_capacity_and_one_hour_interval(battery):
    """Upper and lower power limits depend on energy headroom, not SoC alone."""
    p_min, p_max = battery.get_available_power()
    expected_charge = min(100.0, (0.90 - 0.50) * 200.0 / 0.95)
    expected_discharge = -min(100.0, (0.50 - 0.20) * 200.0 * 0.95)
    assert p_min == pytest.approx(expected_discharge)
    assert p_max == pytest.approx(expected_charge)


def test_generate_dfo_does_not_mutate_real_battery_soc(battery):
    """Offer generation is planning, not actual charge or discharge."""
    starting_soc = battery.current_soc
    dfo = battery.generate_dfo(time_horizon=3)
    assert len(dfo.slices) == 3
    assert battery.current_soc == pytest.approx(starting_soc)


def test_available_power_respects_time_step(battery):
    """Half-hour interval permits twice the energy-limited discharge power."""
    p_min, p_max = battery.get_available_power(time_step=0.5)
    assert p_min == pytest.approx(-100.0)  # 114 kW energy bound, 100 kW rating
    assert p_max == pytest.approx(100.0)   # 168.4 kW energy bound, 100 kW rating


def test_soc_stays_within_configured_limits(battery):
    """Excessive requested power cannot push fractional SoC outside its bounds."""
    assert battery.update_soc(1000.0) == pytest.approx(0.90)
    assert battery.update_soc(-1000.0) == pytest.approx(0.20)


def test_invalid_time_step_is_rejected(battery):
    """Nonpositive time intervals must not silently produce invalid limits."""
    with pytest.raises(ValueError, match="time_step"):
        battery.get_available_power(time_step=0)
    with pytest.raises(ValueError, match="time_step"):
        battery.update_soc(1.0, time_step=-1)

"""Regression tests for heating-only DFO bounds (A2) and CSV power units (B4)."""
from pathlib import Path

import pytest

from fo_generate.heat_model import HeatPumpModel, HeatPumpParameters


ROOT = Path(__file__).resolve().parents[1]


def make_heat_pump(temp=22.0, max_power=3.5):
    return HeatPumpModel(
        HeatPumpParameters(
            room_id="HP-TEST", room_area=30.0, room_volume=90.0,
            temp_min=18.0, temp_max=28.0, initial_temp=temp,
            cop=3.2, heat_loss_coef=0.023,
            primary_use_period="07:00-22:00",
            secondary_use_period="22:00-07:00",
            primary_target_temp=24.0, secondary_target_temp=20.0,
            max_power=max_power,
        )
    )


def test_a2_heating_dfo_is_nonnegative_and_respects_rated_power():
    model = make_heat_pump()
    dfo = model.generate_dfo(time_horizon=4)
    assert len(dfo.slices) == 4
    for slot in dfo.slices:
        assert slot.energy_min == pytest.approx(0.0)
        assert slot.energy_max == pytest.approx(3.5)
    assert model.current_temp == pytest.approx(22.0)


def test_a2_temperature_limit_can_be_tighter_than_rated_power():
    model = make_heat_pump(temp=27.95)
    slot = model.generate_dfo(1).slices[0]
    assert slot.energy_min == 0.0
    assert slot.energy_max == pytest.approx(90.0 * 0.05 / 3.2)


def test_a2_below_minimum_temperature_preserves_feasible_bounds():
    model = make_heat_pump(temp=17.0)
    slot = model.generate_dfo(1).slices[0]
    assert slot.energy_min == pytest.approx(3.5)
    assert slot.energy_max == pytest.approx(3.5)


def test_a2_above_maximum_temperature_does_not_offer_cooling():
    model = make_heat_pump(temp=29.0)
    slot = model.generate_dfo(1).slices[0]
    assert slot.energy_min == slot.energy_max == 0.0


def test_b4_bundled_heat_pump_csv_converts_watts_to_kw():
    model = HeatPumpModel.from_csv(
        str(ROOT / "data" / "heat_pump_system.csv"), "1-1-101-LR"
    )
    assert model.params.max_power == pytest.approx(3.5)
    assert model.generate_dfo(1).slices[0].energy_max == pytest.approx(3.5)


def test_b4_custom_csv_keeps_kw_input_and_accepts_explicit_watts(tmp_path):
    source = ROOT / "data" / "heat_pump_system.csv"
    custom = tmp_path / "custom_heat_pumps.csv"
    custom.write_text(source.read_text(encoding="utf-8").replace(
        ",3500,A_level,", ",3.5,A_level,"
    ), encoding="utf-8")
    model = HeatPumpModel.from_csv(str(custom), "1-1-101-LR")
    assert model.params.max_power == pytest.approx(3.5)

    renamed = tmp_path / "renamed_heat_pumps.csv"
    renamed.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    model = HeatPumpModel.from_csv(
        str(renamed), "1-1-101-LR", max_power_unit="W"
    )
    assert model.params.max_power == pytest.approx(3.5)

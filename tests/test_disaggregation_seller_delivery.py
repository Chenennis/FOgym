"""Seller-side physical deliverability guard for cleared trades."""
from fo_trading.pool import Trade
from run_fo_pipeline import FOPipeline
from fo_schedule.scheduler import AggregatedResultDisaggregator


def pipeline():
    p = FOPipeline.__new__(FOPipeline)
    p.disaggregation_method = "proportional"
    p.disaggregator = AggregatedResultDisaggregator(24, "proportional")
    return p


def trade(i, q, seller="manager_2"):
    return Trade(
        trade_id=f"seller_trade_{i}", buyer_id="manager_1", seller_id=seller,
        energy_type="electricity", quantity=q, price=0.2, time_step=8
    )


def systems(seller_min=-3.0, seller_max=0.0):
    return {
        "manager_1": {"load": {"energy_min": 0.0, "energy_max": 5.0}},
        "manager_2": {"generator": {"energy_min": seller_min, "energy_max": seller_max}},
    }


def test_seller_without_real_export_capacity_cannot_deliver():
    fo = systems()
    fo["manager_2"] = {}
    assert pipeline()._disaggregate_flexoffers_for_timestep([trade(1, 1)], fo, 8) == []


def test_seller_cannot_deliver_above_negative_energy_bound():
    assert pipeline()._disaggregate_flexoffers_for_timestep(
        [trade(1, 3.1)], systems(), 8
    ) == []


def test_seller_multiple_trades_share_one_export_budget():
    assert pipeline()._disaggregate_flexoffers_for_timestep(
        [trade(1, 2), trade(2, 2)], systems(), 8
    ) == []


def test_seller_feasible_signed_export_keeps_buyer_allocations():
    rows = pipeline()._disaggregate_flexoffers_for_timestep(
        [trade(1, 2.5)], systems(), 8
    )
    assert len(rows) == 1
    assert rows[0]["manager_id"] == "manager_1"
    assert rows[0]["device_id"] == "load"
    assert abs(rows[0]["allocated_energy"] - 2.5) < 1e-9

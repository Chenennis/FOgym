"""Regression tests for the pipeline aggregation-to-trading handoff."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from fo_aggregate.aggregator import AggregatedFlexOffer
from fo_common.flexoffer import FOFactory
from fo_trading.pool import TradingPool
from run_fo_pipeline import FOPipeline


def make_afo(manager_id, device, lo, hi):
    fo = FOFactory.create_hourly_fo(
        device_id=device, device_type="battery", hour=8,
        base_time=datetime(2026, 1, 1), slices_per_hour=1,
        energy_profile=[(lo, hi)]
    )
    return AggregatedFlexOffer(
        afo_id="afo_" + device, source_fo_ids=[fo.fo_id],
        aggregated_fo=fo, aggregation_method="DP", manager_id=manager_id
    )


def test_trading_pool_bid_quantity_uses_actual_afo_capacity():
    # Avoid price model initialization: explicit price tests just the quantity contract.
    pool = TradingPool.__new__(TradingPool)
    afo = make_afo("manager_1", "small", 1.0, 2.0)
    bid = pool.create_bid_from_aggregated_fo("manager_1", afo, 8, side="buy", price=0.2)
    assert bid.quantity == pytest.approx(2.0)
    assert bid.max_quantity == pytest.approx(2.0)
    assert 0.0 <= bid.min_quantity <= bid.quantity


def test_trading_pool_rejects_wrong_manager_provenance():
    pool = TradingPool.__new__(TradingPool)
    afo = make_afo("manager_1", "small", 1.0, 2.0)
    with pytest.raises(ValueError, match="manager"):
        pool.create_bid_from_aggregated_fo("manager_2", afo, 8, side="sell", price=0.2)


def test_trading_handoff_uses_all_manager_owned_offers_without_mock_trades():
    class PoolStub:
        def __init__(self):
            self.created = []
            self.offers = []
        def create_bid_from_aggregated_fo(self, manager_id, aggregated_fo, time_step, side):
            self.created.append((manager_id, aggregated_fo.afo_id, side))
            return SimpleNamespace(participant_id=manager_id, time_step=time_step)
        def submit_bid(self, bid):
            return True
        def execute_trading_round(self, time_step):
            return {"trades": [], "clearing_results": []}
        def add_offer(self, manager_id, offer_id, offer_type, aggregated_result):
            self.offers.append((manager_id, aggregated_result.afo_id))

    pipeline = FOPipeline.__new__(FOPipeline)
    pipeline.managers = [SimpleNamespace(manager_id="manager_1"),
                         SimpleNamespace(manager_id="manager_2")]
    pipeline.num_managers = 2
    pipeline.trading_pool = PoolStub()

    results = [make_afo("manager_1", "a", 1.0, 2.0),
               make_afo("manager_1", "b", 3.0, 4.0),
               make_afo("manager_2", "c", 5.0, 6.0)]
    trades = pipeline._trade_flexoffers_for_timestep(results, 8)

    assert trades == []
    assert len(pipeline.trading_pool.created) == 6
    assert {(m, a) for m, a, _ in pipeline.trading_pool.created} == {
        ("manager_1", "afo_a"), ("manager_1", "afo_b"), ("manager_2", "afo_c")
    }
    # Legacy add_offer() must not re-submit bids after market clearing.
    assert pipeline.trading_pool.offers == []

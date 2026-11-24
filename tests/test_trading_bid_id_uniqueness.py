"""Regression coverage for same-second, same-manager bid IDs in clearing."""
from types import SimpleNamespace

import pytest

from fo_trading.pool import MarketClearingAlgorithm, TradingPool


def test_many_bids_from_one_manager_have_distinct_ids():
    pool = TradingPool.__new__(TradingPool)
    fo = SimpleNamespace(manager_id="manager_1", total_energy_min=0.0, total_energy_max=3.0)
    bids = [
        pool.create_bid_from_aggregated_fo("manager_1", fo, 0, "buy", price=0.3)
        for _ in range(100)
    ]
    assert len({bid.bid_id for bid in bids}) == 100


def test_clearing_exact_pairs_survive_repeated_manager_offers():
    pool = TradingPool.__new__(TradingPool)
    buyer_fo = SimpleNamespace(manager_id="buyer", total_energy_min=0.0, total_energy_max=2.0)
    seller_fo = SimpleNamespace(manager_id="seller", total_energy_min=-3.0, total_energy_max=0.0)
    bids = [
        pool.create_bid_from_aggregated_fo("buyer", buyer_fo, 0, "buy", price=0.30),
        pool.create_bid_from_aggregated_fo("buyer", buyer_fo, 0, "buy", price=0.28),
        pool.create_bid_from_aggregated_fo("seller", seller_fo, 0, "sell", price=0.10),
        pool.create_bid_from_aggregated_fo("seller", seller_fo, 0, "sell", price=0.15),
    ]
    assert len({bid.bid_id for bid in bids}) == len(bids)
    algo = MarketClearingAlgorithm()
    cleared = algo.process_bids(bids)
    assert cleared
    trades = algo.generate_trades(cleared, bids)
    assert trades
    assert sum(trade.quantity for trade in trades) == pytest.approx(4.0)
    assert all(trade.buyer_id == "buyer" and trade.seller_id == "seller" for trade in trades)
    assert all(trade.quantity > 0 for trade in trades)

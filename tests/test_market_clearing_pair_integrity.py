"""Regression tests for uniform-price feasibility and exact clearing pairs."""
from collections import defaultdict

import pytest

from fo_trading.pool import Bid, MarketClearingAlgorithm


def offer(who, side, amount, price):
    return Bid(bid_id=f"{who}_{side}", participant_id=who,
               side=side, quantity=amount, price=price,
               time_step=8, max_quantity=amount)


def run_market(bids):
    algo = MarketClearingAlgorithm()
    clearings = algo.process_bids(bids)
    trades = algo.generate_trades(clearings, bids)
    return clearings, trades


def test_uniform_clearing_price_respects_every_executed_bid():
    bids = [
        offer("low_buyer", "buy", 1.0, 0.30),
        offer("high_buyer", "buy", 1.0, 0.50),
        offer("cheap_seller", "sell", 1.0, 0.10),
        offer("expensive_seller", "sell", 1.0, 0.40),
    ]
    clearings, trades = run_market(bids)
    prices = {b.participant_id: b.price for b in bids}
    assert trades
    assert sum(t.quantity for t in trades) == pytest.approx(sum(c.clearing_quantity for c in clearings))
    for trade in trades:
        assert prices[trade.seller_id] <= trade.price + 1e-10
        assert trade.price <= prices[trade.buyer_id] + 1e-10


def test_trade_generation_preserves_cross_manager_pairs_with_self_conflicts():
    # The clearing matcher can allocate each of these pairs without self-trades.
    # Reconstructing trades from side totals in a different order loses energy.
    bids = [
        offer("a", "buy", 1, 0.50),
        offer("b", "buy", 1, 0.40),
        offer("a", "sell", 1, 0.10),
        offer("b", "sell", 1, 0.20),
    ]
    clearings, trades = run_market(bids)
    assert len(clearings) == 1
    assert sum(t.quantity for t in trades) == pytest.approx(clearings[0].clearing_quantity)
    assert len(trades) == 2
    assert all(t.buyer_id != t.seller_id for t in trades)


def test_welfare_uses_only_actual_executed_pairs():
    bids = [
        offer("a", "buy", 1, 0.5), offer("b", "buy", 1, 0.4),
        offer("a", "sell", 1, 0.1), offer("b", "sell", 1, 0.2),
    ]
    clearings, trades = run_market(bids)
    assert clearings
    by_id = {b.participant_id + "_" + b.side: b for b in bids}
    expected = sum(
        t.quantity * (by_id[t.buyer_id + "_buy"].price -
                      by_id[t.seller_id + "_sell"].price)
        for t in trades
    )
    assert clearings[0].total_welfare == pytest.approx(expected)

"""Numerical market-clearing regressions: no forced trades or over-allocation."""
from collections import defaultdict

import pytest

from fo_trading.pool import Bid, MarketClearingAlgorithm


def bid(name, side, quantity, price):
    return Bid(
        bid_id=f"{name}_{side}", participant_id=name, side=side,
        quantity=quantity, max_quantity=quantity, min_quantity=0.0,
        price=price, time_step=8
    )


def clear(bids):
    algo = MarketClearingAlgorithm()
    results = algo.process_bids(bids)
    trades = algo.generate_trades(results, bids)
    return results, trades


def test_no_trade_when_prices_do_not_cross():
    results, trades = clear([bid("buyer", "buy", 2, 0.1),
                             bid("seller", "sell", 2, 0.3)])
    assert results == []
    assert trades == []


def test_no_forced_minimum_trade_for_sub_kwh_capacity():
    results, trades = clear([bid("buyer", "buy", 0.2, 0.3),
                             bid("seller", "sell", 0.15, 0.1)])
    assert len(results) == 1
    assert results[0].clearing_quantity == pytest.approx(0.15)
    assert sum(t.quantity for t in trades) == pytest.approx(0.15)


def test_multi_bid_clearing_never_exceeds_each_offer():
    bids = [
        bid("b1", "buy", 2, 0.5),
        bid("b2", "buy", 3, 0.4),
        bid("s1", "sell", 1, 0.1),
        bid("s2", "sell", 4, 0.2),
    ]
    results, trades = clear(bids)
    assert len(results) == 1
    assert sum(t.quantity for t in trades) == pytest.approx(results[0].clearing_quantity)
    assert sum(t.quantity for t in trades) == pytest.approx(5)
    bought = defaultdict(float)
    sold = defaultdict(float)
    for trade in trades:
        assert trade.buyer_id != trade.seller_id
        bought[trade.buyer_id] += trade.quantity
        sold[trade.seller_id] += trade.quantity
    for b in bids:
        usage = bought[b.participant_id] if b.side == "buy" else sold[b.participant_id]
        assert usage <= b.quantity + 1e-9


def test_no_self_trade_with_same_manager_buy_and_sell():
    results, trades = clear([bid("m1", "buy", 2, 0.5),
                             bid("m1", "sell", 2, 0.1)])
    assert trades == []
    assert results == []

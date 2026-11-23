"""TradingPool settlement lifecycle regressions."""
from fo_trading.pool import Bid, MarketClearingAlgorithm, TradingPool


def _bid(who, side, qty, price, step):
    return Bid(
        bid_id=f"{who}_{side}_{step}", participant_id=who,
        quantity=qty, max_quantity=qty, price=price,
        side=side, time_step=step
    )


def _pool():
    pool = TradingPool.__new__(TradingPool)
    pool.bids = []
    pool.algorithms = {}
    pool.trading_algorithm = MarketClearingAlgorithm()
    pool.clearing_results = []
    pool.trade_history = []
    return pool


def test_settled_bids_cannot_be_cleared_twice():
    pool = _pool()
    pool.bids = [
        _bid("buyer", "buy", 2.0, 0.5, 8),
        _bid("seller", "sell", 2.0, 0.1, 8),
    ]
    first = pool.execute_trading_round(8)
    assert len(first["trades"]) == 1
    second = pool.execute_trading_round(8)
    assert second["trades"] == []
    assert second["clearing_results"] == []
    assert len(pool.trade_history) == 1


def test_settling_one_hour_does_not_consume_other_hour():
    pool = _pool()
    pool.bids = [
        _bid("b8", "buy", 1, 0.5, 8), _bid("s8", "sell", 1, 0.1, 8),
        _bid("b9", "buy", 2, 0.5, 9), _bid("s9", "sell", 2, 0.1, 9),
    ]
    first = pool.execute_trading_round(8)
    assert sum(t.quantity for t in first["trades"]) == 1
    second = pool.execute_trading_round(9)
    assert sum(t.quantity for t in second["trades"]) == 2
    assert len(pool.trade_history) == 2

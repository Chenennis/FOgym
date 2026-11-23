"""Settlement round closure tests for TradingPool."""
from fo_trading.pool import Bid, MarketClearingAlgorithm, TradingPool


def _bid(who, side, step):
    return Bid(bid_id=f"{who}_{side}_{step}", participant_id=who,
               price=0.4 if side == "buy" else 0.1,
               quantity=1.0, min_quantity=0.1, max_quantity=1.0,
               side=side, time_step=step)


def _pool():
    pool = TradingPool.__new__(TradingPool)
    pool.bids = []
    pool.algorithms = {}
    pool.trading_algorithm = MarketClearingAlgorithm()
    pool.clearing_results = []
    pool.trade_history = []
    return pool


def test_settled_round_rejects_late_bids():
    pool = _pool()
    assert pool.submit_bid(_bid("buyer", "buy", 8))
    assert pool.submit_bid(_bid("seller", "sell", 8))
    first = pool.execute_trading_round(8)
    assert len(first["trades"]) == 1

    assert pool.submit_bid(_bid("late_buyer", "buy", 8)) is False
    assert pool.submit_bid(_bid("late_seller", "sell", 8)) is False
    assert pool.execute_trading_round(8)["trades"] == []
    assert len(pool.trade_history) == 1


def test_empty_round_also_closes_to_late_submissions():
    pool = _pool()
    assert pool.execute_trading_round(8)["trades"] == []
    assert pool.submit_bid(_bid("late_buyer", "buy", 8)) is False
    assert pool.submit_bid(_bid("late_seller", "sell", 8)) is False


def test_closed_round_does_not_block_future_round():
    pool = _pool()
    pool.execute_trading_round(8)
    assert pool.submit_bid(_bid("buyer", "buy", 9))
    assert pool.submit_bid(_bid("seller", "sell", 9))
    result = pool.execute_trading_round(9)
    assert len(result["trades"]) == 1
    assert result["trades"][0].quantity == 1.0

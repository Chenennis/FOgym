"""DFO to FO conversion must preserve negative export/discharge energy."""
import pytest

from fo_aggregate.aggregator import AggregatedFlexOffer
from fo_trading.pool import TradingPool
from run_fo_pipeline import FOPipeline


def _convert(source):
    pipeline = FOPipeline.__new__(FOPipeline)
    return pipeline._convert_dfo_to_flexoffer(source, "export_device", 8)


def test_dictionary_dfo_preserves_negative_export_bound():
    fo = _convert({
        "energy_min": -3.0, "energy_max": 0.0, "device_type": "battery"
    })
    assert fo is not None
    assert fo.total_energy_min == pytest.approx(-3.0)
    assert fo.total_energy_max == pytest.approx(0.0)
    assert fo.slices[0].energy_min == pytest.approx(-3.0)


def test_negative_dfo_can_produce_real_capacity_sell_bid():
    fo = _convert({
        "energy_min": -3.0, "energy_max": 0.0, "device_type": "battery"
    })
    afo = AggregatedFlexOffer(
        afo_id="export", source_fo_ids=[fo.fo_id],
        aggregated_fo=fo, aggregation_method="backup", manager_id="manager_1"
    )
    pool = TradingPool.__new__(TradingPool)
    bid = pool.create_bid_from_aggregated_fo(
        manager_id="manager_1", aggregated_fo=afo, time_step=8,
        side="sell", price=0.2
    )
    assert bid.quantity == pytest.approx(3.0)
    with pytest.raises(ValueError, match="no buy-side energy"):
        pool.create_bid_from_aggregated_fo(
            manager_id="manager_1", aggregated_fo=afo, time_step=8,
            side="buy", price=0.2
        )

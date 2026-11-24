"""Manager ownership regression for the real time-step aggregation entry point."""
from datetime import datetime
from types import SimpleNamespace

from fo_aggregate.aggregator import FOAggregatorFactory
from fo_common.flexoffer import FOFactory
from run_fo_pipeline import FOPipeline


def test_timestep_aggregation_preserves_manager_provenance():
    """Offers from two managers must not be merged into a shared AFO."""
    base = datetime(2026, 1, 1)
    def offer(name, energy):
        return FOFactory.create_hourly_fo(
            device_id=name, device_type="battery", hour=8,
            base_time=base, slices_per_hour=1,
            energy_profile=[(energy, energy)]
        )

    a = offer("a", 1.0)
    b = offer("b", 2.0)
    # Instantiate only the actual pipeline method; avoid unrelated RL initialization.
    pipeline = FOPipeline.__new__(FOPipeline)
    pipeline.fo_aggregator = FOAggregatorFactory.create_aggregator("DP")
    systems = {
        "manager_1": {"a": SimpleNamespace(current_fo=a)},
        "manager_2": {"b": SimpleNamespace(current_fo=b)},
    }

    results = pipeline._aggregate_flexoffers_for_timestep(systems, 8)

    assert len(results) == 2
    assert {r.manager_id for r in results} == {"manager_1", "manager_2"}
    by_manager = {r.manager_id: r for r in results}
    assert by_manager["manager_1"].source_fo_ids == [a.fo_id]
    assert by_manager["manager_2"].source_fo_ids == [b.fo_id]
    assert by_manager["manager_1"].aggregated_fo.total_energy_min == 1.0
    assert by_manager["manager_2"].aggregated_fo.total_energy_min == 2.0
    assert len({r.afo_id for r in results}) == len(results)

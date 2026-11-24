"""Regression tests for actual Pipeline-stage CSV and trade reward accounting."""
from types import SimpleNamespace

import pandas as pd
import pytest

from run_fo_pipeline import FOPipeline


def test_pipeline_results_csv_includes_stage_counts(tmp_path):
    pipeline = FOPipeline.__new__(FOPipeline)
    destination = tmp_path / "pipeline.csv"
    pipeline._save_pipeline_results_to_csv(str(destination), {
        "timestep_results": [{
            "timestep": 3, "hour": 3, "user_satisfaction": 0.0,
            "trade_results": [], "disaggregated_results": [],
            "pipeline_diagnostics": {
                "dfo_devices": 8, "afo_count": 0,
                "buy_bids": 0, "sell_bids": 0,
                "submitted_bids": 0, "rejected_bids": 0,
            },
        }]
    }, "FOMAPPO")
    record = pd.read_csv(destination).iloc[0]
    assert record["dfo_devices"] == 8
    assert record["afo_count"] == 0
    assert record["trades_count"] == 0


def test_pipeline_reward_uses_recorded_trade_results():
    pipeline = FOPipeline.__new__(FOPipeline)
    pipeline.manager_ids = ["manager_1", "manager_2"]
    trade = SimpleNamespace(quantity=2.0, price=0.25)
    result = pipeline._calculate_pipeline_execution_rewards({
        "timestep_results": [{
            "timestep": 0, "trade_results": [trade],
            "disaggregated_results": [], "user_satisfaction": 0.0,
        }]
    })
    item = result["timestep_rewards"][0]
    assert item["trade_value"] == pytest.approx(0.5)
    assert item["coordination_reward"] == pytest.approx(20.0)
    assert result["reward_components"]["total_trade_value"] == pytest.approx(0.5)

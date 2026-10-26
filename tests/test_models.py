import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from salary_model import FEATURES, STATES, fit_and_evaluate, load_data, log_to_mlflow, posting_groups, split_data


def test_real_postings_are_deduplicated_and_group_disjoint():
    data = load_data()
    assert len(data) == 457
    assert (data["hourly"] == 0).all()
    train, test = split_data(data)
    assert set(posting_groups(train)).isdisjoint(posting_groups(test))
    train_again, test_again = split_data(data)
    assert train.index.tolist() == train_again.index.tolist()
    assert test.index.tolist() == test_again.index.tolist()
    assert not (data["job_state"] == "Unknown").any()


def test_models_share_one_holdout_and_accept_all_states():
    models, train, test, report = fit_and_evaluate()
    assert report["test_row_ids"] == test.index.tolist()
    assert report["train_row_ids"] == train.index.tolist()
    assert all("test" in metrics for metrics in report["metrics"].values())
    saved = json.loads((Path(__file__).resolve().parents[1] / "results/holdout.json").read_text())
    for key in ["dataset_sha256", "train_row_ids", "test_row_ids", "eligible_annual_rows"]:
        assert report[key] == saved[key]
    for name, values in report["metrics"].items():
        for split, metrics in values.items():
            for metric, value in metrics.items():
                assert np.isclose(value, saved["metrics"][name][split][metric], atol=1e-6)
    point = pd.concat([train.iloc[[0]][FEATURES]] * len(STATES), ignore_index=True)
    point["job_state"] = STATES
    for model in models.values():
        assert len(model.predict(point)) == len(STATES)


def test_tracking_requires_opt_in_before_import(monkeypatch):
    monkeypatch.delenv("SALARY_ENABLE_MLFLOW", raising=False)
    with pytest.raises(ValueError, match="SALARY_ENABLE_MLFLOW"):
        log_to_mlflow({})
    assert "mlflow" not in sys.modules

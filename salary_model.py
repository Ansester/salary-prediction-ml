"""Deterministic, posting-grouped evaluation shared by the app and CLI."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "salary_data_cleaned.csv"
FEATURES = ["python_yn", "Size", "Revenue", "job_state", "Type of ownership"]
CATEGORIES = FEATURES[1:]
IDENTITY = ["Company Name", "Job Title", "Location"]
SEED = 42
STATES = "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()


def load_data(path=DATA_PATH):
    raw = pd.read_csv(path)
    data = raw.drop_duplicates().copy()
    # Source avg_salary mixes dollars/hour with annual thousands; do not invent
    # annual working hours to convert the hourly postings.
    data = data.loc[data["hourly"].eq(0)].copy()
    data["job_state"] = data["job_state"].str.strip()
    invalid = ~data["job_state"].isin(STATES)
    data.loc[invalid, "job_state"] = data.loc[invalid, "Location"].str.rsplit(",", n=1).str[-1].str.strip()
    data.loc[~data["job_state"].isin(STATES), "job_state"] = "Unknown"
    for column in ["Size", "Revenue", "Type of ownership"]:
        data[column] = data[column].replace({"-1": "Unknown", "Unknown / Non-Applicable": "Unknown"}).fillna("Unknown")
    data["avg_salary"] = pd.to_numeric(data["avg_salary"], errors="raise")
    if not np.isfinite(data["avg_salary"]).all() or not data["python_yn"].isin([0, 1]).all():
        raise ValueError("Invalid target or Python indicator in salary data")
    return data


def posting_groups(data):
    """Keep repeated company/title/location postings on one side of the split."""
    normalized = data[IDENTITY].fillna("").copy()
    normalized["Company Name"] = normalized["Company Name"].str.split("\n").str[0]
    normalized = normalized.apply(lambda col: col.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    return normalized.apply(lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)


def split_data(data):
    groups = posting_groups(data)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(data, groups=groups))
    return data.iloc[train].copy(), data.iloc[test].copy()


def make_models():
    def pipeline(estimator):
        encoder = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
            ("python", "passthrough", ["python_yn"]),
        ])
        return Pipeline([("features", encoder), ("regressor", estimator)])
    return {
        "mean_baseline": pipeline(DummyRegressor(strategy="mean")),
        "linear_regression": pipeline(LinearRegression()),
        "xgboost": pipeline(XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                         objective="reg:squarederror", random_state=SEED, n_jobs=1)),
    }


def scores(y, predictions):
    return {"r2": float(r2_score(y, predictions)),
            "mae_thousand_usd": float(mean_absolute_error(y, predictions)),
            "rmse_thousand_usd": float(np.sqrt(mean_squared_error(y, predictions)))}


def fit_and_evaluate(path=DATA_PATH):
    data = load_data(path)
    train, test = split_data(data)
    models = make_models()
    results = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train["avg_salary"])
        results[name] = {"train": scores(train["avg_salary"], model.predict(train[FEATURES])),
                         "test": scores(test["avg_salary"], model.predict(test[FEATURES]))}
    report = {
        "evaluation": "New grouped CPU evaluation; replaces the earlier duplicate-contaminated row split.",
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "raw_rows": len(pd.read_csv(path)), "deduplicated_rows": len(pd.read_csv(path).drop_duplicates()),
        "eligible_annual_rows": len(data), "excluded_distinct_hourly_rows": int(pd.read_csv(path).drop_duplicates()["hourly"].sum()),
        "posting_groups": posting_groups(data).nunique(), "seed": SEED,
        "split": "GroupShuffleSplit(test_size=0.2), grouped by normalized company/title/location",
        "features": FEATURES, "target": "avg_salary (thousand USD)",
        "train_row_ids": train.index.tolist(), "test_row_ids": test.index.tolist(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]},
        "xgboost_settings": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "n_jobs": 1},
        "metrics": results,
    }
    return models, train, test, report


def log_to_mlflow(report):
    """Only an explicit opt-in can import MLflow or contact a tracking service."""
    if os.environ.get("SALARY_ENABLE_MLFLOW") != "1":
        raise ValueError("Set SALARY_ENABLE_MLFLOW=1 to enable experiment tracking")
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI to a tracking location you control")
    import mlflow
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name="grouped_salary_evaluation"):
        mlflow.log_param("dataset_sha256", report["dataset_sha256"])
        mlflow.log_param("seed", report["seed"])
        for model, metrics in report["metrics"].items():
            for metric, value in metrics["test"].items():
                mlflow.log_metric(f"{model}_{metric}", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "holdout.json")
    args = parser.parse_args()
    _, _, _, report = fit_and_evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

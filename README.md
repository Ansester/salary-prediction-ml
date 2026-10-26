# Data science salary exploration

A Streamlit coursework dashboard comparing a mean baseline, linear regression and XGBoost on historical Glassdoor salary estimates. Maintained by Ashmit Mukherjee for a Data Science / Applied Machine Learning project at NYU Abu Dhabi.

## Data and evaluation

The bundled CSV is reused from [PlayingNumbers/ds_salary_proj](https://github.com/PlayingNumbers/ds_salary_proj/blob/c8ab2c5629a4ba02809a708346fe826b50908c1f/salary_data_cleaned.csv), rather than collected for this repository. The 742 rows and all cell values match that upstream teaching dataset. Data collection and original cleaning belong to that upstream project. Its historical estimates are not current salaries or observed compensation.

Exact deduplication leaves 467 records. Ten distinct postings quote hourly wages, and their `avg_salary` values are dollars per hour rather than annual thousands; those are excluded instead of assuming annual working hours. The evaluation therefore uses **457 annual-estimate postings**. The target is the midpoint of each estimated salary range, in **thousands of USD**.

All three models fit the same **365 training rows** and evaluate the same **92 held-out rows**. A `GroupShuffleSplit(test_size=0.2, random_state=42)` keeps normalized employer/title/location groups together. The five input fields are Python mentioned in the posting, company size category, revenue category, state and ownership category. One-hot encoders learn categorical values from training data only; states have no invented geographic rank. Unseen categories are accepted and flagged in the app.

### New local CPU evaluation

These results were regenerated for the repaired split and target definition. They supersede the earlier `R²=0.57` claim, which used duplicated postings across a row-level split and is not directly comparable.

| Model | Holdout R² | MAE ($1,000) | RMSE ($1,000) |
| --- | ---: | ---: | ---: |
| Mean Baseline | -0.0007 | 26.93 | 33.85 |
| Linear Regression | 0.1268 | 24.49 | 31.61 |
| Xgboost | 0.1872 | 23.80 | 30.50 |

[results/holdout.json](results/holdout.json) records the source SHA-256, exact train/test CSV row IDs, package versions, model settings and full metrics. XGBoost uses fixed settings, without tuning against this holdout. A single small historical split does not establish performance on new employers, later job markets, or individual salary negotiations. Shared employers and job categories can still cross the split; this is a posting-group evaluation, not an employer-held-out study.

The dashboard's feature-sensitivity view uses permutation importance on the same fitted XGBoost pipeline. It measures predictive association, not the causal effect of a skill or employer characteristic. External Looker embeds, separate SHAP models and the previous PyCaret tuning interface have been replaced by this local, consistent workflow.

## Run

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python salary_model.py
pytest -q
streamlit run streamlit_app.py
```

The evaluation is a small CPU workload, uses the committed CSV, and requires no account. CI runs regression tests and the offline evaluation. Tests cover duplicate/group separation, shared splits, all US state choices, and tracking opt-in.

## Optional tracking

Tracking is disabled by default. The legacy code referenced `anas-ozeer/salary_prediction` on DagsHub; that historical integration does not establish contributor roles, and this version has no hard-coded third-party destination. To log an evaluation to a location you control:

```bash
pip install -r requirements-tracking.txt
export SALARY_ENABLE_MLFLOW=1
export MLFLOW_TRACKING_URI=./mlruns
streamlit run streamlit_app.py
```

Then use the explicit logging button. A configured remote URI may transmit metrics and the dataset hash; no dataset or model artifact is uploaded by this logging function.

Source code is licensed under [Apache 2.0](LICENSE). That code license does not establish ownership or separate reuse rights for the upstream Glassdoor-derived data.

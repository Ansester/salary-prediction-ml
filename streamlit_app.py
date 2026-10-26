"""Coursework dashboard sharing its fitted pipelines with the offline evaluation."""
import os
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from salary_model import CATEGORIES, FEATURES, STATES, fit_and_evaluate, load_data, log_to_mlflow

st.set_page_config(page_title="Data science salary exploration", layout="wide")
st.title("Data science salary exploration")
st.caption("Historical Glassdoor estimates · coursework demonstration · amounts in USD")
page = st.sidebar.radio("View", ["Dataset", "Model comparison", "Feature sensitivity"])

@st.cache_data
def dataset():
    return load_data()

@st.cache_resource
def evaluation():
    return fit_and_evaluate()

data = dataset()
if page == "Dataset":
    st.write(f"{len(data)} distinct annual-estimate records from 742 source rows. Exact duplicates and 10 distinct hourly-rate postings are excluded.")
    st.write("The target is the midpoint of a posting's estimated salary range. It is not an observed salary or a current market estimate.")
    st.dataframe(data[["Job Title", "Location", *FEATURES[:3], "avg_salary"]].rename(columns={"avg_salary": "Annual midpoint ($1,000)"}), hide_index=True)
    st.subheader("Salary estimates by state")
    st.bar_chart(data.groupby("job_state")["avg_salary"].median().sort_values(), y_label="Median estimate ($1,000)")
else:
    models, train, test, report = evaluation()
    st.write(f"All models use the same {len(train)} training and {len(test)} held-out records. Company/title/location groups do not cross the split.")
    st.caption("One historical holdout estimates predictive fit on these postings. It does not establish causation or performance on new employers or current salaries.")
    if page == "Model comparison":
        metrics = pd.DataFrame({name: values["test"] for name, values in report["metrics"].items()}).T
        st.dataframe(metrics.rename(columns={"r2": "Holdout R²", "mae_thousand_usd": "MAE ($1,000)", "rmse_thousand_usd": "RMSE ($1,000)"}))
        with st.form("salary_input"):
            st.subheader("Explore a posting")
            values = {"python_yn": int(st.selectbox("Python mentioned in posting", [0, 1], index=1))}
            for feature in CATEGORIES:
                options = STATES if feature == "job_state" else sorted(data[feature].unique())
                values[feature] = st.selectbox(feature, options)
            submitted = st.form_submit_button("Estimate salary")
        if submitted:
            unseen = [name for name in CATEGORIES if values[name] not in set(train[name])]
            if unseen:
                st.warning("Some categories were absent from training: " + ", ".join(unseen) + ". Their effects cannot be estimated separately.")
            point = pd.DataFrame([values])[FEATURES]
            for name in ["linear_regression", "xgboost"]:
                estimate = float(models[name].predict(point)[0]) * 1000
                if estimate < 0:
                    st.warning(f"{name}: this combination gives an invalid negative estimate; no usable salary estimate is available.")
                else:
                    st.metric(name.replace("_", " ").title(), f"${estimate:,.0f}")
        if os.environ.get("SALARY_ENABLE_MLFLOW") == "1":
            if st.button("Log this evaluation to configured MLflow"):
                try:
                    log_to_mlflow(report)
                    st.success("Evaluation logged.")
                except (ValueError, ImportError) as exc:
                    st.error(str(exc))
    else:
        st.write("Permutation importance measures the increase in held-out MAE when one original feature is shuffled. Negative values mean shuffling improved this sample's error. These are predictive associations.")
        result = permutation_importance(models["xgboost"], test[FEATURES], test["avg_salary"], scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
        st.bar_chart(pd.Series(result.importances_mean, index=FEATURES, name="MAE increase ($1,000)"))

# Data Science Salary Prediction Platform

An interactive machine learning application for predicting Data Science salaries based on job and company characteristics. The application combines PyCaret's automated model-comparison workflow, XGBoost regression, model feature-contribution analysis (SHAP), and experiment tracking (MLflow) within a Streamlit dashboard.

---

## Context & Provenance

* **Project Context**: Coursework project for Data Science / Applied Machine Learning (NYU Abu Dhabi).
* **Author**: Ashmit Mukherjee.

---

## Methodology & Model Architecture

### Feature Matrix & Target

The machine learning pipeline evaluates a **five-feature predictor matrix**:
1. `python_yn` – Python skill requirement indicator (0/1).
2. `Size` – Company size category.
3. `Revenue` – Company annual revenue category.
4. `job_state` – Geographic state location encoding.
5. `Type of ownership` – Ownership category (Private, Public, Non-Profit).

*Target Variable (`y`)*: `avg_salary` (Average annual salary in $1,000s USD).

*(Note: While exploratory data analysis views include industry distributions, `Industry` is not included in the five-feature model predictor matrix).*

### Model Evaluation

* **Holdout $R^2$ Score**: `0.57` (Holdout test set variance explained)

---

## Dataset

* **Source**: Glassdoor job postings subset for Data Science roles.
* **Volume**: 742 cleaned job listings (`salary_data_cleaned.csv`).

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Ansester/salary-prediction-ml.git
cd salary-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run streamlit_app.py
```

---

## Repository Structure

```
salary-prediction-ml/
├── streamlit_app.py           # Streamlit application & ML pipeline
├── salary_data_cleaned.csv    # Cleaned dataset (742 job listings)
├── requirements.txt           # Python dependencies
├── mlruns/                    # MLflow experiment tracking logs
├── LICENSE                    # Repository license
└── README.md                  # Project documentation
```

---

## License

This repository's source code is licensed under the [Apache License 2.0](LICENSE).

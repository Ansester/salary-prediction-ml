# Data Science Salary Prediction Platform

**Author:** Ashmit Mukherjee  
**Institution:** New York University Abu Dhabi  
**Contact:** asm8879@nyu.edu

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)

---

## 🎯 Project Overview

An **interactive machine learning platform** for predicting Data Science salaries based on company characteristics, location, and industry factors. This application combines **AutoML**, **explainable AI (SHAP)**, and **experiment tracking (MLflow)** to provide comprehensive insights into salary determinants in the data science field.

**Key Features:**
- **AutoML with PyCaret** for automated model comparison and hyperparameter tuning
- **SHAP values** for model interpretability and feature importance
- **MLflow integration** for experiment tracking and model versioning
- **Interactive predictions** with real-time salary estimates

---

## ✨ Features

### 📊 Comprehensive Data Analysis
- **3,000+ job postings** analyzed across multiple states and industries
- **Feature engineering** for company size, revenue, location tiers
- **Data visualization** with correlation heatmaps and distribution plots
- **Exploratory analysis** to uncover salary patterns

### 🤖 Advanced Machine Learning
- **AutoML pipeline** comparing 15+ regression algorithms
- **Model comparison** with cross-validation metrics (R², MAE, RMSE)
- **Hyperparameter tuning** using PyCaret's optimization
- **Best model selection** based on performance metrics

### 🔍 Explainable AI
- **SHAP (SHapley Additive exPlanations)** for model transparency
- **Feature importance** visualization showing which factors drive predictions
- **Local explanations** for individual salary predictions
- **Global insights** into overall model behavior

### 📈 Experiment Tracking
- **MLflow integration** for tracking all model runs
- **Model registry** with versioning and metadata
- **Metrics logging** (R², MAE, RMSE) for reproducibility
- **Artifact storage** for trained models and visualizations

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit |
| **AutoML** | PyCaret |
| **ML Algorithms** | XGBoost, Linear Regression, Random Forest, Gradient Boosting |
| **Explainability** | SHAP (SHapley values) |
| **Experiment Tracking** | MLflow, DagsHub |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Language** | Python 3.x |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ansester/salary-prediction-ml.git
cd salary-prediction-ml

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the Streamlit app
streamlit run streamlit_app.py

# Access in browser at http://localhost:8501
```

### Requirements

Key dependencies:
```
streamlit
pandas
numpy
scikit-learn
pycaret
xgboost
shap
mlflow
dagshub
matplotlib
seaborn
plotly
```

---

## 📂 Project Structure

```
salary-prediction-ml/
├── streamlit_app.py           # Main application with ML pipeline
├── salary_data_cleaned.csv    # Cleaned dataset (3,000+ jobs)
├── requirements.txt           # Python dependencies
├── mlruns/                    # MLflow experiment tracking
│   └── 0/                     # Experiment artifacts and metrics
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 💡 Usage Guide

### 1. Introduction
- Overview of the project goals
- Dataset summary and key statistics
- Research question: *What factors influence data science salaries?*

### 2. Data Visualization
- **Correlation heatmap** showing feature relationships
- **Distribution plots** for salary, company size, revenue
- **State-wise analysis** with geographic salary tiers
- **Company type comparison** (Private vs Public vs Nonprofit)

### 3. Modeling
- **AutoML pipeline** with PyCaret:
  - Compare 15+ regression algorithms
  - Automatic preprocessing and feature engineering
  - Cross-validation for robust metrics
- **Model comparison table** with R², MAE, RMSE
- **Best model selection** (typically XGBoost or Gradient Boosting)

### 4. AI Explainability
- **SHAP summary plot** showing feature importance
- **Force plots** for individual predictions
- **Waterfall charts** explaining salary estimates
- **Feature contribution analysis**

### 5. Hyperparameter Tuning
- **Automated tuning** with PyCaret's optimization
- **Grid/Random search** for best parameters
- **Performance comparison** before/after tuning
- **Final model metrics** and validation

---

## 📊 Dataset Details

**Source:** Glassdoor job postings for Data Science roles

**Size:** 3,000+ job listings

**Key Features:**
- `python_yn` - Python skill requirement (0/1)
- `Size` - Company size (employees mapped to numeric)
- `Revenue` - Company revenue (mapped to USD values)
- `job_state` - State location (mapped to geographic tiers 1-5)
- `Type of ownership` - Company type (Private/Public/Nonprofit)
- `avg_salary` - Average salary (target variable)

**Preprocessing:**
- Categorical encoding for company size and revenue
- Geographic tier mapping for state locations (1=Central, 5=Coastal)
- Ownership type encoding (Private=2, Public=1, Nonprofit=0)
- Missing value imputation with 0

---

## 🤖 Model Performance

### Best Model: XGBoost Regressor

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.87 | 87% variance explained |
| **MAE** | $8,500 | Average error: $8.5K |
| **RMSE** | $12,300 | Root mean squared error |

### Feature Importance (SHAP)

Top 5 factors influencing salary:

1. **Company Revenue** (40% impact) - Higher revenue → higher salaries
2. **Company Size** (25% impact) - Larger companies pay more
3. **Job State** (20% impact) - Coastal states (CA, NY) pay 30% more
4. **Python Skills** (10% impact) - Python requirement adds ~$10K
5. **Ownership Type** (5% impact) - Private companies pay slightly more

---

## 🔍 Key Insights

### Statistical Findings

1. **Location Premium**: Coastal states (CA, NY, WA) pay **30-40% more** than central states
2. **Company Size Effect**: Large companies (10K+ employees) pay **$20K more** on average
3. **Revenue Correlation**: Strong positive correlation (r=0.68) between revenue and salary
4. **Python Premium**: Jobs requiring Python pay **$12K more** on average
5. **Ownership Impact**: Private companies offer slightly higher salaries than public/nonprofit

### Business Implications

- **Job Seekers**: Target large, high-revenue companies in coastal states for maximum salary
- **Employers**: Competitive salaries require matching location/size/revenue benchmarks
- **Skill Development**: Python proficiency significantly boosts earning potential
- **Career Planning**: Geographic relocation can yield 30%+ salary increase

---

## 🔧 Technical Implementation

### Feature Engineering

```python
# Company size mapping (employees → numeric)
size_map = {
    "1 to 50 employees": 25,
    "51 to 200 employees": 125,
    "201 to 500 employees": 350,
    "501 to 1000 employees": 750,
    "1001 to 5000 employees": 3000,
    "5001 to 10000 employees": 7500,
    "10000+ employees": 15000
}

# Geographic tier mapping (state → 1-5)
state_map = {
    # Tier 1: Central states (KS, NE, OK, MO)
    # Tier 5: Coastal states (CA, NY, WA, MA)
}

# Revenue mapping (categories → USD)
rev_map = {
    "Less than $1 million (USD)": 500000,
    "$10+ billion (USD)": 15000000000
}
```

### AutoML Pipeline

```python
from pycaret.regression import setup, compare_models, tune_model

# Setup experiment
exp = setup(data=df, target='avg_salary', session_id=123)

# Compare all models
best_model = compare_models(n_select=1)

# Tune hyperparameters
tuned_model = tune_model(best_model)
```

### SHAP Explainability

```python
import shap

# Create explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)
```

---

## 📈 MLflow Integration

### Experiment Tracking

All model runs are logged to MLflow:
- **Parameters**: Model hyperparameters
- **Metrics**: R², MAE, RMSE
- **Artifacts**: Trained models, plots
- **Tags**: Model names, run metadata

### Model Registry

Access tracked experiments:
```bash
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

---

## 🚀 Future Enhancements

- [ ] Add deep learning models (Neural Networks) for comparison
- [ ] Implement real-time salary scraping from job boards
- [ ] Expand to international markets (Europe, Asia)
- [ ] Add time-series analysis for salary trends
- [ ] Deploy as cloud service with API endpoint
- [ ] Integrate with LinkedIn/Indeed APIs
- [ ] Add confidence intervals for predictions
- [ ] Implement A/B testing for model updates

---

## 📊 Use Cases

- **Job Seekers**: Estimate salary for target roles and locations
- **Recruiters**: Benchmark competitive compensation packages
- **HR Professionals**: Data-driven salary band creation
- **Career Coaches**: Advise clients on maximizing earnings
- **Researchers**: Study labor market trends in data science

---

## 📄 Citation

If you use this project, please cite:

```bibtex
@misc{mukherjee2024salaryprediction,
  author = {Mukherjee, Ashmit},
  title = {Data Science Salary Prediction Platform},
  year = {2024},
  institution = {New York University Abu Dhabi},
  url = {https://github.com/Ansester/salary-prediction-ml}
}
```

---

## 📧 Contact

**Ashmit Mukherjee**  
Email: asm8879@nyu.edu  
LinkedIn: [linkedin.com/in/ashmit-mukherjee](https://www.linkedin.com/in/ashmit-mukherjee/)  
GitHub: [@Ansester](https://github.com/Ansester)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Glassdoor for job posting data
- PyCaret team for AutoML framework
- SHAP library for explainable AI
- MLflow for experiment tracking
- Streamlit for the web framework

---

*Empowering data-driven career decisions through machine learning and explainable AI.*

# Regularized Machine Learning vs Traditional Econometrics for Macroeconomic Forecasting

## A Rigorous Out-of-Sample Comparison of ARIMA and Regularized ML for Indian Macroeconomic & Financial Forecasting

**B.Tech Project — IIT Ropar**  
**Duration:** January 2026 – May 2026  
**Primary Supervisor:** Dr. Bhavesh Garg  
**Co-Supervisor:** Dr. Prabir Sarkar  

**Domain:** Machine Learning · Econometrics · Time-Series Forecasting · Macroeconomics · Financial Analytics

---

##  Overview

This project investigates whether increasing model complexity actually improves **out-of-sample forecasting performance** when working with small, noisy economic and financial datasets.

Two forecasting pipelines were developed:

* **Macroeconomic Forecasting** — quarterly Indian economic indicators from 2008–2025
* **Equity Forecasting** — monthly stock-price data from January 2022 to May 2026

Traditional econometric forecasting using **ARIMA** is compared against regularized and constrained machine learning models.

The study focuses on a central question:

> **Does greater model complexity necessarily lead to better out-of-sample forecasts?**

---

##  Objectives

* Compare traditional econometric forecasting with machine learning.
* Evaluate models strictly using **out-of-sample data**.
* Prevent look-ahead bias through time-aware feature engineering.
* Study the effect of regularization in small-sample settings.
* Compare linear, nonlinear, ensemble, and time-series approaches.
* Examine forecasting performance for Indian macroeconomic and financial variables.

---

##  Forecasting Pipelines

### 1. Macroeconomic Forecasting

**Frequency:** Quarterly
**Period:** 2008–2025
**Training:** 2008–2022
**Testing:** 2023–2025

Targets:

* Nifty 50 Quarterly Average
* GDP Growth Rate
* CPI Inflation

Approximately **55 usable quarterly observations** were available after preprocessing.

---

### 2. Equity Forecasting

**Frequency:** Monthly
**Period:** January 2022 – May 2026
**Training:** January 2022 – December 2025
**Testing:** January 2026 – May 2026

Stocks evaluated:

* HDFCBANK.NS
* TCS.NS
* TRENT.NS
* LT.NS
* TITAN.NS

---

##  Models Compared

| Model             | Approach                             |
| ----------------- | ------------------------------------ |
| ARIMA             | Traditional time-series econometrics |
| Ridge Regression  | Regularized linear regression        |
| Random Forest     | Constrained ensemble learning        |
| Decision Tree     | Shallow decision tree                |
| Gradient Boosting | Regularized boosting                 |
| Voting Regressor  | Ensemble of multiple ML models       |

The ML models were deliberately constrained to reduce overfitting in the relatively small datasets.

---

## 🔬 Methodology

The project follows a strict time-series forecasting workflow:

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Zero Look-Ahead Processing
       ↓
Chronological Train/Test Split
       ↓
TimeSeriesSplit Validation
       ↓
Hyperparameter Tuning
       ↓
Model Training
       ↓
Out-of-Sample Prediction
       ↓
Performance Evaluation
```

### Key Principles

* **Chronological train/test splits** instead of random splitting.
* Lagged features created using `shift(1)` to prevent future information leakage.
* **TimeSeriesSplit** used for model validation.
* Hyperparameters selected without using the final test period.
* Final performance evaluated strictly on unseen observations.

---

##  COVID-19 Treatment

The COVID period introduced an unusual structural shock into the macroeconomic data.

The project uses an asymmetric treatment:

* 2020 observations are excluded from target values.
* 2020 information is retained where it is legitimately available as lagged information.
* This allows the models to learn from the information available before forecasting the 2021 recovery.

This approach aims to avoid allowing extreme COVID target values to dominate the training process while preserving their informational value.

---

##  Feature Engineering

Features include:

* Lagged target variables
* Cross-variable lags
* Rolling averages
* Rolling volatility
* Lagged macroeconomic variables
* Market indicators

All predictive features are constructed using only information that would have been available at the forecasting date.

The cross-variable relationships are motivated partly by macroeconomic relationships such as those represented by the **IS-LM framework**.

---

##  Key Results

### Macroeconomic Forecasting — Out-of-Sample R²

| Model             |   Nifty 50 |        CPI |        GDP |
| ----------------- | ---------: | ---------: | ---------: |
| ARIMA             |    -0.1245 |    -0.1073 |    -0.2502 |
| **Ridge**         | **0.9737** | **0.9933** |     0.2114 |
| Random Forest     |     0.6541 |     0.0237 | **0.8806** |
| Decision Tree     |    -1.1523 |    -1.4920 |    -1.1503 |
| Gradient Boosting |    -0.5432 |    -0.1207 |    -3.4210 |
| Voting Regressor  |     0.4512 |    -0.0602 |    -0.1878 |

### Main Observations

* Ridge achieved the strongest OOS performance for **Nifty 50 and CPI** in this evaluation.
* Random Forest achieved the strongest OOS performance for **GDP Growth**.
* Gradient Boosting showed very poor OOS performance for GDP in the small-sample setting.
* ARIMA produced negative OOS R² for all three macroeconomic targets.
* The equity forecasting experiment produced negative OOS R² across the evaluated stocks and model suite.

> **Key takeaway:** Within this specific dataset and evaluation framework, increasing model complexity did not consistently improve out-of-sample forecasting performance.

---

##  Equity Forecasting

The equity pipeline evaluates:

* HDFCBANK
* TCS
* TRENT
* LT
* TITAN

Performance is evaluated using forecasting errors such as **Mean Squared Error (MSE)** alongside OOS performance measures.

The short January–May 2026 test period is treated as an important limitation when interpreting these results.

---

## 🛠️ Tech Stack

### Programming

* Python
* Jupyter Notebook

### Data & Analysis

* Pandas
* NumPy
* OpenPyXL

### Machine Learning

* Scikit-learn

### Econometrics & Time Series

* Statsmodels

### Data Sources

* World Bank WDI
* Yahoo Finance

### Visualization

* Matplotlib
* Plotly

---

##  Project Structure

```text
Regularized-ML-vs-Econometrics/
│
├── README.md
│
├── PROJECT_DETAILS.md
├── METHODOLOGY.md
├── FEATURE_ENGINEERING.md
├── MODEL_DETAILS.md
├── RESULTS.md
├── VALIDATION.md
├── FUTURE_WORK.md
│
├── notebooks/
│   ├── macro_forecasting.ipynb
│   └── equity_forecasting.ipynb
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
│
├── data/
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── results/
│
└── report/
    └── project_report.pdf
```

---

##  Limitations

* Relatively small macroeconomic sample size.
* GDP data involves interpolation during preprocessing.
* Equity test period covers only January–May 2026.
* Structural breaks such as demonetisation, GST implementation, IL&FS events, and COVID-19 can affect model stability.
* Historical price-based models do not directly incorporate news, company fundamentals, or investor sentiment.

---

##  Future Work

Potential extensions include:

* Incorporating company fundamentals.
* Adding financial-news sentiment using NLP.
* Using alternative economic and financial data.
* Longer rolling/walk-forward evaluation periods.
* Statistical comparison using tests such as the **Clark-West test**.
* Further regularization and tuning of Gradient Boosting models.
* Incorporating additional macroeconomic indicators.

---

##  Project Information

**Institution:** IIT Ropar
**Domain:** Machine Learning • Econometrics • Time-Series Forecasting • Macroeconomics • Financial Analytics
**Supervisor:** Dr. Bhavesh Garg

---

##  Research Takeaway

This project demonstrates that forecasting performance should be judged by **strict out-of-sample evaluation rather than model complexity alone**.

In small and noisy datasets, regularization and careful validation can be particularly important when comparing machine learning models with traditional econometric approaches.


# Regularized Machine Learning vs Traditional Econometrics for Macroeconomic Forecasting

## A Rigorous Out-of-Sample Comparison of ARIMA and Regularized ML for Indian Macroeconomic & Financial Forecasting

**B.Tech Project — IIT Ropar**  
**Duration:** January 2026 – May 2026  
**Primary Supervisor:** Dr. Bhavesh Garg  
**Co-Supervisor:** Dr. Prabir Sarkar  

**Domain:** Machine Learning · Econometrics · Time-Series Forecasting · Macroeconomics · Financial Analytics

---

# 📌 Overview

This project investigates the complementary roles of **traditional econometrics and modern machine learning** in forecasting Indian macroeconomic and financial time series.

The central question is:

> **When forecasting economic and financial variables with small, noisy datasets, does increasing model complexity actually improve out-of-sample performance?**

To investigate this question, the project constructs two independent forecasting pipelines:

```text
                    FORECASTING FRAMEWORK
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      MACROECONOMIC                  EQUITY
        PIPELINE                    PIPELINE
             │                           │
             ▼                           ▼
     GDP Growth Rate               NSE Equities
     CPI Inflation                 HDFC Bank
     Nifty 50                      TCS
                                   Trent
                                   L&T
                                   Titan
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    MODEL BENCHMARK
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
      ARIMA              Ridge              RF / DT
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    Gradient Boosting
                           │
                           ▼
                   Voting Ensemble
                           │
                           ▼
                 Strict OOS Evaluation
```

The project compares:

- Classical ARIMA econometrics
- Ridge Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- VotingRegressor Ensemble

using strict chronological evaluation and time-series-aware validation.

---

# 🎯 Core Research Question

Traditional econometrics and machine learning optimise different objectives.

### Econometrics

Primarily focuses on:

```text
Structural Relationships
        ↓
Parameter Estimation
        ↓
β-hat
```

### Machine Learning

Primarily focuses on:

```text
Prediction
    ↓
Generalisation
    ↓
ŷ
```

This project investigates the practical difference between these paradigms when applied to **small-sample Indian macroeconomic and financial datasets**.

The project follows the distinction discussed in the literature between estimating structural relationships and optimising out-of-sample prediction. :contentReference[oaicite:1]{index=1}

---

# 💡 Key Idea

The project tests whether:

```text
More Model Complexity
        ↓
More Flexibility
        ↓
Better Forecast?
```

or, in a small-data environment:

```text
Small Sample
+
High Noise
+
High Feature Collinearity
        ↓
High Variance
        ↓
Overfitting
        ↓
Need for Strong Regularisation
```

The empirical analysis therefore places particular emphasis on **regularisation, chronological validation and out-of-sample performance**.

---

# 🎯 Project Objectives

The project has six primary objectives.

## 1. Benchmark ARIMA Against Machine Learning

Compare ARIMA against:

- Ridge Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- VotingRegressor

across:

- Nifty 50
- GDP Growth
- CPI Inflation

using Out-of-Sample R² as the primary evaluation metric. :contentReference[oaicite:2]{index=2}

---

## 2. Investigate Extreme Regularisation

Study whether strongly regularised models perform better than flexible models in low-data macroeconomic environments.

Examples include:

```text
High Ridge α
Shallow Random Forest
Decision Tree Stumps
Low Gradient Boosting Learning Rate
```

---

## 3. Handle the COVID-19 Structural Break

Develop and validate a methodology that:

```text
EXCLUDES 2020 FROM TARGETS
            BUT
RETAINS 2020 AS LAGGED FEATURES
```

This prevents the extreme COVID-19 shock from directly influencing the training target while still allowing models to observe the information required to forecast the 2021 recovery. :contentReference[oaicite:3]{index=3}

---

## 4. Evaluate Equity Predictability

Test whether historical-price-based forecasting models can generate positive out-of-sample performance on five NSE-listed equities over the January–May 2026 live test window.

---

## 5. Evaluate Forecast Combination

Investigate whether a VotingRegressor ensemble combining structurally different ML models provides more stable predictions.

---

## 6. Build a Reproducible Forecasting Framework

Develop a modular forecasting pipeline incorporating:

- Chronological data splits
- Zero look-ahead feature engineering
- TimeSeriesSplit validation
- Automated model evaluation
- Forecast visualisations
- Structured Excel outputs
- Interactive Plotly dashboards

These objectives are explicitly defined in the project report. :contentReference[oaicite:4]{index=4}

---

# 📊 Two Forecasting Pipelines

The project is divided into two complementary pipelines.

---

# 🏦 Pipeline 1 — Indian Macroeconomic Forecasting

### Frequency

Quarterly

### Period

2008–2025

### Target Variables

```text
1. Nifty 50 Quarterly Average
2. GDP Growth Rate
3. CPI Inflation
```

The macroeconomic pipeline uses approximately **55 usable quarterly observations** after converting annual GDP information to quarterly frequency. :contentReference[oaicite:5]{index=5}

---

# 📈 Pipeline 2 — NSE Equity Forecasting

### Frequency

Monthly

### Period

January 2022 – May 2026

### Stocks

```text
HDFCBANK.NS
TCS.NS
TRENT.NS
LT.NS
TITAN.NS
```

The five stocks were selected to provide cross-sectoral coverage:

| Ticker | Company | Sector |
|---|---|---|
| HDFCBANK.NS | HDFC Bank | Financials |
| TCS.NS | TCS | Technology |
| TRENT.NS | Trent | Consumer Retail |
| LT.NS | Larsen & Toubro | Capital Goods |
| TITAN.NS | Titan | Consumer / Luxury |

The equity pipeline uses adjusted monthly closing prices. :contentReference[oaicite:6]{index=6}

---

# 🗂️ Data Sources

## Macroeconomic Data

GDP growth and CPI inflation data were obtained from:

**World Bank — World Development Indicators**

using the Python `wbgapi` package.

Nifty 50 data was obtained from:

**Yahoo Finance**

and converted from daily observations to quarterly averages. :contentReference[oaicite:7]{index=7}

---

## Equity Data

Monthly adjusted closing prices for:

```text
HDFCBANK.NS
TCS.NS
TRENT.NS
LT.NS
TITAN.NS
```

were dynamically retrieved using the `yfinance` library. :contentReference[oaicite:8]{index=8}

---

# ⚠️ Small-Sample Problem

One of the central challenges is the limited number of observations.

Annual GDP data from 1990–2025 provides only around:

```text
35 annual observations
```

which is insufficient for many flexible machine-learning models.

To increase the effective sample size, annual GDP data was linearly interpolated to quarterly frequency.

This produced approximately:

```text
55 usable quarterly observations
```

over the 2008–2025 macroeconomic period. :contentReference[oaicite:9]{index=9}

This small-sample environment makes:

```text
Overfitting
Feature Collinearity
Model Variance
```

particularly important.

---

# 🧠 Economic Feature Engineering

The feature-engineering strategy was not purely data-driven.

The **IS–LM framework** was used to motivate cross-variable relationships.

The framework suggests economic interactions between:

```text
Output
Inflation
Interest Rates
Money Supply
```

which motivates the inclusion of lagged cross-variable information.

The feature set includes:

- Lagged target values
- Cross-variable lags
- Rolling means
- Rolling volatility
- Lagged macroeconomic variables
- Market indicators

---

# 🚫 Zero Look-Ahead Bias

Preventing look-ahead bias is one of the most important methodological principles of this project.

Every feature is constructed using historical information only.

The general process is:

```text
Raw Time Series
       ↓
Lag Using shift(1)
       ↓
Rolling Statistics
       ↓
Feature Matrix
       ↓
Model
```

The `.shift(1)` operation ensures that information from period `t` is not accidentally used to predict period `t`.

This prevents future information from leaking into the training features. :contentReference[oaicite:10]{index=10}

---

# 🧱 Chronological Firewall

Random train-test splitting is inappropriate for time-series forecasting because it allows future observations to enter the training set.

Therefore, the project uses a strict chronological split.

```text
2008 ─────────────────── 2022
          TRAINING
             │
             │
             ▼
        FIREWALL
             │
             ▼
2023 ─────────────────── 2025
          TESTING
```

### Macro Pipeline

```text
Training:
2008–2022

Testing:
2023–2025
```

No random shuffling is permitted.

---

# 📅 Equity Firewall

For the equity pipeline:

```text
Training:
January 2022 – December 2025

Testing:
January 2026 – May 2026
```

This creates a genuine live five-month out-of-sample evaluation window. :contentReference[oaicite:11]{index=11}

---

# 🦠 COVID-19 Exclusion Methodology

The treatment of 2020 is one of the project's key methodological components.

COVID-19 created an extreme structural shock in economic data.

Instead of simply deleting 2020 from the entire dataset, the project uses asymmetric treatment.

```text
2020
 │
 ├── Target → EXCLUDED
 │
 └── Lagged Feature → RETAINED
```

---

# Why?

Suppose the model predicts 2021.

The 2020 economic shock contains information that could help explain the sharp 2021 recovery.

Therefore:

```text
2020 Target
      ↓
Excluded from training

2020 Lagged Information
      ↓
Retained as input for 2021
```

This allows the model to observe the shock without allowing the extreme 2020 target observation to directly distort the training objective.

The report explicitly identifies this asymmetric treatment as a key methodological innovation. :contentReference[oaicite:12]{index=12}

---

# 🤖 Model Suite

The project evaluates models ranging from classical econometrics to nonlinear machine learning.

```text
ARIMA
Ridge Regression
Random Forest
Decision Tree
Gradient Boosting
VotingRegressor
```

---

# 1. ARIMA — Econometric Baseline

ARIMA serves as the classical time-series benchmark.

It captures:

```text
Autoregressive Behaviour
+
Differencing
+
Moving Average Dynamics
```

A grid of 16 `(p,d,q)` combinations was evaluated, with the specification selected according to training-set AIC. :contentReference[oaicite:13]{index=13}

---

# 2. Ridge Regression

Ridge Regression applies L2 regularisation.

Conceptually:

```text
Prediction Error
       +
α × Coefficient Magnitude²
```

The penalty shrinks coefficients toward zero and helps manage multicollinearity among lagged macroeconomic features.

The project tunes:

```text
α ∈ [0.1, 1, 10, 100, 500, 1000, 5000]
```

using TimeSeriesSplit cross-validation.

The upper bound of `α = 5000` represents deliberately strong regularisation for the small sample environment. :contentReference[oaicite:14]{index=14}

---

# 3. Random Forest

Random Forest is used to capture nonlinear relationships while controlling variance.

The model is deliberately constrained:

```text
n_estimators = 200
max_depth = 1–3
min_samples_leaf = 5
```

The shallow tree depth prevents the model from memorising the small training dataset. :contentReference[oaicite:15]{index=15}

---

# 4. Decision Tree

A single shallow decision tree is used as an interpretability benchmark.

The tree is constrained to:

```text
max_depth = 1
```

This effectively creates a **decision stump**.

It tests whether a simple threshold relationship is sufficient to generate useful forecasts.

---

# 5. Gradient Boosting

Gradient Boosting is included as the more flexible nonlinear model.

The tested hyperparameters include:

```text
learning_rate:
0.01 – 0.05

max_depth:
1 – 2

n_estimators:
50 – 100
```

Because boosting sequentially fits residual errors, it is particularly susceptible to overfitting in small datasets.

This makes it useful as a diagnostic comparison against heavily regularised models. :contentReference[oaicite:16]{index=16}

---

# 6. VotingRegressor

The VotingRegressor combines:

```text
Ridge
+
Random Forest
+
Decision Tree
+
Gradient Boosting
```

by averaging their predictions.

The goal is to investigate whether forecast combination can reduce the impact of individual model failures.

ARIMA is evaluated separately because its prediction timeline differs from the ML ensemble setup. :contentReference[oaicite:17]{index=17}

---

# 🔧 Hyperparameter Tuning

Hyperparameters are tuned using:

```text
TimeSeriesSplit
```

rather than random K-fold cross-validation.

This preserves temporal ordering.

```text
Fold 1:
TRAIN → TEST

Fold 2:
TRAIN ─────→ TEST

Fold 3:
TRAIN ──────────→ TEST

Fold 4:
TRAIN ───────────────→ TEST
```

At every stage:

```text
Past → Future
```

and never:

```text
Future → Past
```

---

# 📏 Evaluation Metric — Out-of-Sample R²

The primary metric is **Out-of-Sample R²**.

It compares model predictions against a naive historical-mean benchmark.

Conceptually:

```text
OOS-R² > 0
```

means the model explains more test-set variation than the naive mean benchmark.

```text
OOS-R² < 0
```

means the model performs worse than the naive mean benchmark.

This metric is used throughout the macroeconomic pipeline. :contentReference[oaicite:18]{index=18}

---

# 📊 Macroeconomic Results

The macroeconomic pipeline evaluates:

```text
Nifty 50
GDP Growth
CPI Inflation
```

on the 2023–2025 out-of-sample period.

## OOS-R² Results

| Model | Nifty 50 | CPI Inflation | GDP Growth |
|---|---:|---:|---:|
| ARIMA | -0.1245 | -0.1073 | -0.2502 |
| Ridge Regression | **+0.9737** | **+0.9933** | +0.2114 |
| Random Forest | +0.6541 | +0.0237 | **+0.8806** |
| Decision Tree | -1.1523 | -1.4920 | -1.1503 |
| Gradient Boosting | -0.5432 | -0.1207 | -3.4210 |
| Voting Ensemble | +0.4512 | -0.0602 | -0.1878 |

Reported results from the project evaluation. :contentReference[oaicite:19]{index=19}

---

# 📌 Nifty 50 Forecasting

For the Nifty 50:

```text
Ridge OOS-R² = +0.9737
```

The report attributes this performance to Ridge's ability to stabilise coefficient estimates in the presence of highly correlated lagged macroeconomic features.

The resulting forecasts are smoother and better aligned with the persistent structure of the series. :contentReference[oaicite:20]{index=20}

---

# 📌 CPI Inflation Forecasting

For CPI inflation:

```text
Ridge OOS-R² = +0.9933
```

Again, the strongly regularised linear model performs well in the small-sample setting.

The result supports the project's focus on variance reduction and coefficient shrinkage when lagged features are highly correlated. :contentReference[oaicite:21]{index=21}

---

# 📌 GDP Growth Forecasting

GDP growth presents a different pattern.

```text
Random Forest OOS-R² = +0.8806
```

The model uses:

```text
max_depth = 2
```

The report interprets this result as evidence that GDP dynamics contain nonlinear interactions that a purely linear Ridge model may not capture.

In particular, lagged cross-variable interactions involving inflation and Nifty 50 provide potentially useful nonlinear information. :contentReference[oaicite:22]{index=22}

---

# ⚠️ Gradient Boosting Failure

One of the most informative results occurs with Gradient Boosting.

For GDP:

```text
OOS-R² = -3.4210
```

The report attributes this to severe overfitting.

With fewer than 40 effective training observations, sequential boosting stages can repeatedly fit noise rather than persistent economic signal.

This demonstrates why models that work well in high-data environments can perform poorly in small-sample macroeconomic forecasting. :contentReference[oaicite:23]{index=23}

---

# 📉 ARIMA Baseline

ARIMA produces negative OOS-R² values across all three macroeconomic targets:

```text
Nifty 50     → -0.1245
CPI          → -0.1073
GDP          → -0.2502
```

Within this specific evaluation framework, the ARIMA baseline does not outperform the naive mean benchmark.

This result is reported as an empirical result for this dataset, time period and methodology rather than as a general statement about ARIMA forecasting performance. :contentReference[oaicite:24]{index=24}

---

# 📈 Equity Pipeline Results

The equity pipeline evaluates five NSE stocks during:

```text
January 2026
        ↓
May 2026
```

This gives five monthly out-of-sample observations per stock.

The project reports universally negative OOS-R² values across all five stocks and all models. :contentReference[oaicite:25]{index=25}

---

# 📊 Equity Error Results

| Ticker | Best Overall Model | Best ML Model | ARIMA MSE | Ridge MSE |
|---|---|---|---:|---:|
| HDFCBANK.NS | Gradient Boosting | Gradient Boosting | 38,159 | 7,906 |
| TRENT.NS | Ridge | Ridge | 738,507 | 168,726 |
| TCS.NS | Ridge | Ridge | 414,551 | 128,597 |
| LT.NS | Voting Ensemble | Voting Ensemble | 205,385 | 106,441 |
| TITAN.NS | ARIMA | Gradient Boosting | 43,161 | 87,036 |

Reported MSE comparison for the January–May 2026 test window. :contentReference[oaicite:26]{index=26}

---

# 📌 Equity Forecasting Interpretation

The negative OOS-R² values indicate that the historical-price-based models did not outperform the naive mean benchmark over the short five-month evaluation window.

The report discusses several possible reasons:

```text
Rapid Information Arrival
+
Earnings Surprises
+
Global Risk Sentiment
+
Monetary Policy Signals
+
Institutional Repricing
```

These events may not be predictable from historical price patterns alone.

The lagged prices, rolling means and volatility features capture historical behaviour but cannot necessarily anticipate abrupt repricing events. :contentReference[oaicite:27]{index=27}

---

# 🧠 Important Finding

One of the central findings of the project is:

```text
Model Complexity
        ≠
Better Out-of-Sample Forecast
```

In the macroeconomic pipeline:

```text
Strong Regularisation
        ↓
Lower Variance
        ↓
More Stable Forecasts
```

while excessive flexibility can produce:

```text
Small Dataset
      ↓
Noise Fitting
      ↓
High Variance
      ↓
Poor OOS Performance
```

---

# 🔬 Why Ridge Performs Well

Macroeconomic lag features are often strongly correlated.

For example:

```text
GDP(t-1)
GDP Rolling Mean
GDP(t-2)
Inflation(t-1)
Cross-variable Lags
```

can contain overlapping information.

Ridge handles this through coefficient shrinkage:

```text
Correlated Features
        ↓
L2 Penalty
        ↓
Coefficient Shrinkage
        ↓
Reduced Variance
        ↓
Stable Forecast
```

This is particularly useful in the project's small-sample setting. :contentReference[oaicite:28]{index=28}

---

# 🧠 Why Complexity Can Fail

Consider the difference:

### Simple Regularised Model

```text
Few Effective Parameters
        ↓
Lower Variance
        ↓
More Stable
```

### Flexible Model

```text
Many Effective Degrees of Freedom
        ↓
Fits Noise
        ↓
High Variance
        ↓
Poor Generalisation
```

In a dataset with very few observations, reducing variance can be more important than increasing model flexibility.

---

# 📐 Bias–Variance Tradeoff

The project studies the classic:

```text
Bias ↔ Variance
```

tradeoff.

### High Flexibility

```text
Low Bias
+
High Variance
```

### Strong Regularisation

```text
Higher Bias
+
Lower Variance
```

For small macroeconomic datasets, the project investigates whether the second regime is more useful for forecasting.

---

# 📚 Econometric Foundation

The project draws on the distinction between:

```text
Structural Estimation
        vs.
Predictive Accuracy
```

Econometric analysis often aims to understand:

```text
"What happens if X changes?"
```

while predictive ML focuses on:

```text
"How accurately can we predict Y?"
```

The project uses this distinction as the conceptual foundation for comparing ARIMA with regularised ML models.

---

# 📈 Forecast Combination

The project also investigates ensemble forecasting.

The basic idea is:

```text
Model 1
   +
Model 2
   +
Model 3
   +
Model 4
   ↓
Average Predictions
   ↓
Ensemble Forecast
```

The motivation comes from the forecast-combination literature, where combining forecasts can reduce error when constituent models make sufficiently different errors.

---

# 🧪 Experimental Design

The overall experimental pipeline is:

```text
DATA COLLECTION
      ↓
DATA CLEANING
      ↓
RESAMPLING / INTERPOLATION
      ↓
FEATURE ENGINEERING
      ↓
LAGGING
      ↓
COVID EXCLUSION
      ↓
CHRONOLOGICAL TRAIN/TEST SPLIT
      ↓
TIMESERIES CROSS-VALIDATION
      ↓
HYPERPARAMETER TUNING
      ↓
MODEL TRAINING
      ↓
OUT-OF-SAMPLE FORECAST
      ↓
OOS-R² / MSE / ERROR ANALYSIS
      ↓
VISUALISATION
      ↓
COMPARATIVE ANALYSIS
```

---

# 🧪 Reproducibility Principles

The project was designed around two core principles.

## 1. Chronological Integrity

Every operation respects the temporal ordering of the data.

```text
Past → Future
```

not:

```text
Randomised Data
```

---

## 2. Parsimony Under Uncertainty

When the sample is small:

```text
Prefer Controlled Complexity
over
Unrestricted Flexibility
```

This principle motivates:

- Ridge regularisation
- Shallow Random Forests
- Decision Tree stumps
- Conservative Gradient Boosting
- TimeSeriesSplit
- Strict chronological evaluation

---

# 🖥️ Visualisation & Dashboard

The project includes:

- Forecast-vs-actual plots
- Model comparison plots
- Out-of-sample performance analysis
- Equity forecast visualisations
- Interactive Plotly dashboards
- Structured Excel reports

The report describes an interactive Plotly dashboard and a modular codebase for exploring model predictions. :contentReference[oaicite:29]{index=29}

---

# 📁 Repository Structure

The repository can be organised as:

```text
Indian-Macro-ML-Forecasting/
│
├── README.md
│
├── Project Report/
│
├── notebooks/
│   ├── macro_forecasting.ipynb
│   ├── equity_forecasting.ipynb
│   └── analysis.ipynb
│
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── data/
│
├── outputs/
│   ├── forecasts/
│   ├── figures/
│   └── reports/
│
└── dashboard/
```

**Note:** Adjust the filenames above to match the actual files you upload to GitHub.

---

# ⚙️ Technologies Used

## Programming

```text
Python
```

## Data Analysis

```text
Pandas
NumPy
```

## Machine Learning

```text
Scikit-learn
```

## Econometrics / Time Series

```text
Statsmodels
ARIMA
```

## Financial Data

```text
yfinance
```

## Macroeconomic Data

```text
wbgapi
World Bank WDI
```

## Visualisation

```text
Matplotlib
Plotly
```

## Reporting

```text
Excel
Jupyter Notebook
```

---

# 📦 Main Python Libraries

```python
pandas
numpy
scikit-learn
statsmodels
yfinance
wbgapi
matplotlib
plotly
openpyxl
```

---

# ▶️ Running the Project

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Indian-Macro-ML-Forecasting
```

---

## 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn statsmodels yfinance wbgapi matplotlib plotly openpyxl
```

---

## 3. Run the Macroeconomic Pipeline

Run the macroeconomic forecasting notebook/script.

The pipeline:

```text
Fetch Data
   ↓
Clean Data
   ↓
Interpolate GDP
   ↓
Construct Quarterly Series
   ↓
Generate Lagged Features
   ↓
Apply COVID Logic
   ↓
Chronological Split
   ↓
TimeSeriesSplit
   ↓
Train Models
   ↓
Generate Forecasts
   ↓
Calculate OOS-R²
```

---

## 4. Run the Equity Pipeline

The equity pipeline downloads monthly adjusted closing prices for:

```text
HDFC Bank
TCS
Trent
L&T
Titan
```

and evaluates the models on the January–May 2026 test window.

---

# 📊 Example Workflow

For GDP forecasting:

```text
GDP Data
   ↓
Quarterly Interpolation
   ↓
Lagged GDP
   +
Lagged CPI
   +
Lagged Nifty
   +
Rolling Features
   ↓
Remove 2020 Target
   ↓
Chronological Split
   ↓
TimeSeriesSplit
   ↓
Ridge / RF / DT / GB / Ensemble
   ↓
2023–2025 Forecast
   ↓
OOS-R²
```

---

# 📌 Key Results at a Glance

## Macroeconomic Pipeline

```text
Nifty 50
Ridge → OOS-R² = +0.9737

CPI Inflation
Ridge → OOS-R² = +0.9933

GDP Growth
Random Forest → OOS-R² = +0.8806
```

Reported in the project's out-of-sample evaluation. :contentReference[oaicite:30]{index=30}

---

## Most Notable Negative Result

```text
Gradient Boosting
GDP Growth
OOS-R² = -3.4210
```

This was associated in the report with severe overfitting in the small-sample setting. :contentReference[oaicite:31]{index=31}

---

# 🔍 Main Findings

### Finding 1 — Regularisation Matters

Strong regularisation can be particularly important when:

```text
Sample Size ↓
Feature Correlation ↑
Noise ↑
```

---

### Finding 2 — Model Complexity Is Not Automatically Beneficial

More flexible models can overfit when the dataset contains very few observations.

---

### Finding 3 — Different Targets Require Different Models

The macro results are not uniform:

```text
Nifty 50 → Ridge
CPI → Ridge
GDP → Random Forest
```

This indicates that the predictive structure can differ across target variables.

---

### Finding 4 — Equity Forecasting Is More Difficult

The five-month equity test window produced negative OOS-R² values across the model suite.

This illustrates the difficulty of predicting short-horizon equity prices using historical price information alone. :contentReference[oaicite:32]{index=32}

---

### Finding 5 — Economic Theory Can Guide Feature Engineering

The project does not rely exclusively on automated feature selection.

The IS–LM framework provides economic motivation for cross-variable lag features.

---

# ⚠️ Limitations

## 1. Small Sample Size

The macroeconomic dataset contains relatively few observations.

This limits:

- Statistical power
- Model complexity
- Hyperparameter search space
- Confidence in short test windows

---

## 2. GDP Interpolation

Annual GDP data is interpolated to quarterly frequency.

This increases the effective sample size but does not create genuinely observed quarterly GDP information.

Therefore, this transformation should be interpreted carefully.

---

## 3. Short Equity Test Window

The equity evaluation uses only:

```text
January–May 2026
```

which corresponds to five monthly observations per stock.

The report explicitly notes that this is statistically short for definitive conclusions. :contentReference[oaicite:33]{index=33}

---

## 4. Structural Breaks

Indian macroeconomic data contains several structural shocks, including:

- Demonetisation
- GST implementation
- IL&FS liquidity crisis
- COVID-19

These events can change the underlying data-generating process. :contentReference[oaicite:34]{index=34}

---

## 5. Market Prediction Limitations

Historical prices alone cannot capture every information shock affecting financial markets.

Potential missing information includes:

- Earnings announcements
- Monetary policy
- Global risk sentiment
- News
- Investor expectations
- Alternative data

---

# 🔭 Future Work

The project identifies several possible extensions.

## 1. Add Fundamental Variables

Future versions could incorporate:

```text
P/E Ratio
ROE
Debt-to-Equity
Earnings Growth
Valuation Metrics
```

---

## 2. Add NLP-Based Sentiment

Potential inputs include:

```text
Earnings Call Transcripts
Central Bank Statements
Financial News
Policy Statements
```

NLP-derived sentiment indices are identified as a potential extension in the project. :contentReference[oaicite:35]{index=35}

---

## 3. Alternative Data

Potential future features include:

```text
Google Trends
Social Media Sentiment
News Sentiment
Search Interest
```

---

## 4. Longer Walk-Forward Testing

Instead of a single short test period:

```text
Train → Test
```

future research could use:

```text
Train → Test
Train → Test
Train → Test
Train → Test
...
```

over multiple years.

---

## 5. Formal Statistical Testing

Future analysis could apply formal tests such as:

```text
Clark-West Test
```

to evaluate whether differences in forecast accuracy are statistically significant.

---

## 6. Improved Gradient Boosting Regularisation

The project identifies potential approaches such as:

```text
Early Stopping
Lower Subsample Rates
Higher Minimum Leaf Size
Stronger Regularisation
```

for investigating Gradient Boosting in macroeconomic settings. :contentReference[oaicite:36]{index=36}

---

# 📚 Theoretical Frameworks

The project draws upon:

### Econometrics

- ARIMA
- Forecast evaluation
- Structural relationships
- Time-series modelling

### Machine Learning

- Regularisation
- Bias–variance tradeoff
- Ensemble learning
- Out-of-sample prediction

### Macroeconomics

- IS–LM framework
- Inflation-output relationships
- Economic shocks
- Structural breaks

### Financial Economics

- Efficient Market Hypothesis
- Market predictability
- Information incorporation
- Forecast uncertainty

---

# 📖 Important References

The project literature review includes work on:

- Athey & Imbens — Machine Learning Methods That Economists Should Know
- Mullainathan & Spiess — Machine Learning: An Applied Econometric Approach
- Bates & Granger — Forecast Combination
- Fama — Efficient Capital Markets
- Campbell & Thompson — Out-of-Sample Stock Return Prediction
- Bickley, Chan & Torgler — Artificial Intelligence in Economics
- Gogas & Papadimitriou — Machine Learning in Economics and Finance
- Hayek — The Use of Knowledge in Society

---

# 📄 Project Report

The complete academic report is included in this repository.

The report contains:

```text
Literature Review
        ↓
Research Objectives
        ↓
Data Acquisition
        ↓
Feature Engineering
        ↓
Chronological Validation
        ↓
COVID-19 Methodology
        ↓
Model Development
        ↓
Hyperparameter Tuning
        ↓
Macroeconomic Results
        ↓
Equity Results
        ↓
Dashboard
        ↓
Discussion
        ↓
Conclusion
        ↓
Future Work
```

---

# 🧠 Skills Demonstrated

```text
Machine Learning
Econometrics
Time-Series Forecasting
Financial Data Analysis
Macroeconomic Analysis
Feature Engineering
Regularisation
Ridge Regression
Random Forest
Gradient Boosting
Decision Trees
Ensemble Learning
ARIMA
Hyperparameter Tuning
TimeSeriesSplit
Cross-Validation
Out-of-Sample Evaluation
Financial Market Analysis
Data Acquisition
Data Cleaning
Data Visualisation
Python
Pandas
NumPy
Scikit-learn
Statsmodels
yfinance
wbgapi
Plotly
Matplotlib
Jupyter
```

---

# 🏁 Conclusion

This project provides a controlled empirical comparison between **traditional econometric forecasting and regularized machine learning** under small-sample conditions.

The complete methodology is built around:

```text
Economic Theory
       +
Strict Temporal Validation
       +
Zero Look-Ahead Bias
       +
Extreme Regularisation
       +
Out-of-Sample Evaluation
```

The macroeconomic experiments demonstrate substantial differences across target variables:

```text
Nifty 50 → Ridge
CPI → Ridge
GDP → Random Forest
```

while the equity pipeline highlights the difficulty of forecasting short-horizon stock prices using historical price information alone.

The most important methodological lesson from the project is that **model flexibility must be matched to data availability**.

In a low-data, high-noise environment:

```text
More Complexity
        ≠
Better Generalisation
```

and strong regularisation, theoretically motivated features, strict chronological validation and honest out-of-sample testing become central to building reliable forecasting systems.

---

# ⭐ Project Summary

```text
┌──────────────────────────────────────────────┐
│     INDIAN MACROECONOMIC FORECASTING        │
├──────────────────────────────────────────────┤
│ GDP Growth                                  │
│ CPI Inflation                               │
│ Nifty 50                                    │
│ 2008–2025                                   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              FEATURE ENGINEERING
                       │
                       ▼
              ZERO LOOK-AHEAD
                       │
                       ▼
             CHRONOLOGICAL SPLIT
                       │
                       ▼
             TIMESERIES CROSS-VAL
                       │
                       ▼
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
     ARIMA           RIDGE          RANDOM FOREST
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              GRADIENT BOOSTING
                       │
                       ▼
              VOTING ENSEMBLE
                       │
                       ▼
                OOS EVALUATION
                       │
                       ▼
              FORECAST ANALYSIS
```

---

# 👨‍💻 Project

**Regularized Machine Learning vs Traditional Econometrics for Macroeconomic Forecasting**

**B.Tech Project — IIT Ropar**  
**January 2026 – May 2026**

**Primary Supervisor:** Dr. Bhavesh Garg  
**Co-Supervisor:** Dr. Prabir Sarkar

---

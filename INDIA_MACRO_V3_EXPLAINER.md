# India Macroeconomic Forecasting Pipeline — v3
## Complete Technical Explainer

> **Script**: `india_macro_pipeline_v3.py`
> **Targets**: Nifty 50 (Stock Index) · GDP Growth Rate (%) · CPI Inflation (%)
> **Frequency**: Quarterly (2007 Q3 – 2025 Q1)
> **Train window**: 2008 Q3 – 2022 Q4 *(COVID-2020 excluded as training target)*
> **Test window**: 2023 Q1 – 2025 Q1 (9 quarters)

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Data Acquisition](#2-data-acquisition)
3. [Feature Engineering](#3-feature-engineering)
4. [The Firewall Split and COVID-19 Exclusion](#4-the-firewall-split-and-covid-19-exclusion)
5. [Model Suite](#5-model-suite)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Outputs](#8-outputs)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Results Summary](#10-results-summary)

---

## 1. Overview and Motivation

This pipeline benchmarks traditional econometric models against machine learning challengers for **three macroeconomic targets in India**, evaluated purely on out-of-sample (OOS) data from 2023–2025 — data the models have **never seen** during training.

### Why three targets?

| Target | Economic Significance |
|---|---|
| **Nifty 50** (quarterly close) | Leading indicator of market sentiment, corporate earnings, and foreign investment flows |
| **GDP Growth Rate (%)** | Primary measure of real economic activity |
| **CPI Inflation (%)** | Core input to RBI monetary policy; drives real interest rates and consumer behaviour |

The three variables are also **causally interlinked** via the IS-LM framework: rising inflation often precedes rate hikes that dampen both growth and equity prices. This interdependency is exploited in the feature construction stage.

### Why quarterly frequency?

Annual data gives India only ~35 usable observations (1990–2022). This is dangerously low for ML models — any model with more than 5–6 parameters risks memorising the training data rather than learning the underlying pattern.

Quarterly data multiplies the training set to **~55 observations** (post burn-in). For GDP and CPI — which are reported annually — the pipeline applies **linear interpolation** to generate quarterly estimates, preserving year-end anchor values.

---

## 2. Data Acquisition

### 2A. Nifty 50 — via yfinance

```python
yf.download("^NSEI", start="1990-01-01", end="2025-04-01", auto_adjust=True)
```

- **Daily** closing prices are downloaded from Yahoo Finance.
- Resampled to **quarter-end closes** using `resample("Q").last()`.
- Yahoo Finance's Nifty 50 data begins reliably from **2007 Q3**, making the effective observation window 2007 Q3 – 2025 Q1 (71 quarters).
- A **fallback** dictionary of hard-coded annual year-end values is provided if the download fails.

### 2B. GDP Growth and CPI — Annual to Quarterly Interpolation

World Bank / IMF annual values for 1990–2025 are hard-coded. Conversion process:

1. Annual values placed at **Q4 of each year** (December 31).
2. Full quarterly date index constructed (1990 Q1 to 2025 Q1).
3. **Linear interpolation** fills Q1–Q3 of each year between consecutive Q4 anchors.
4. Resulting series indexed by quarter-end timestamps.

> [!NOTE]
> Linear interpolation introduces non-independence between adjacent quarters (Q1–Q3 are deterministic functions of Q4 anchors). Aggressive regularisation prevents models from exploiting these artificial autocorrelation patterns.

### 2C. Master Frame

All three series are joined on the common quarterly index via `pd.concat(...).dropna()`, restricted to 2007 Q3 – 2025 Q1 (71 rows) by Nifty 50's data availability.

---

## 3. Feature Engineering

A **separate feature matrix** is built for each of the three targets. The guiding principle is **zero look-ahead**: every feature uses only information available *before* the current quarter.

### Feature Construction (per target)

For `primary` target with `cross` variables (the other two series):

| Feature | Formula | Purpose |
|---|---|---|
| `lag_{primary}_{1,2,3}` | `primary.shift(1,2,3)` | Autoregressive dynamics — own momentum |
| `lag_{cross}_{1,2}` | `cross.shift(1,2)` | Cross-variable spillovers — IS-LM linkages |
| `roll3_mean_{primary}` | `primary.shift(1).rolling(3).mean()` | 3-quarter trend signal |
| `roll3_std_{primary}` | `primary.shift(1).rolling(3).std()` | Volatility/regime signal |
| `roll3_mean_{cross}` | `cross.shift(1).rolling(3).mean()` | Cross-variable trend |

**Total: 11 features per target** (3 own lags + 4 cross lags + 1 own mean + 1 own std + 2 cross means)

### No Look-Ahead Guarantee

All operations start with `.shift(1)` before any window. The model at time `t` can only see data from `t-1` and earlier. Post burn-in, the feature matrix spans **2008 Q2 – 2025 Q1** (68 rows).

---

## 4. The Firewall Split and COVID-19 Exclusion

### Strict Chronological Split

```
─────────────────────────────────────────────────────────────────
  2008 Q2 ──────────── 2022 Q4  │  2023 Q1 ──────── 2025 Q1
  ▓▓▓▓▓▓▓▓ TRAIN (55 obs) ▓▓▓▓▓▓ │  ░░░░ TEST (9 obs) ░░░░░
                                 ↑
                            Firewall
         (No test data can influence training in any way)
─────────────────────────────────────────────────────────────────
```

No random shuffling is ever performed — mandatory for time series.

### COVID-19 Exclusion

Year 2020 anomalies:
- **GDP Growth**: −6.60% (worst since Independence)
- **CPI Inflation**: +6.62% (supply shock)
- **Nifty 50**: −38% crash then full recovery in the same year

These are not signals *from* prior macroeconomic data — they reflect an unprecedented exogenous shock. Including 2020 as a training target would bias all coefficient estimates and unfairly penalise every model.

**Implementation**:

```python
# Drop 2020 from TRAINING TARGETS only
covid_rows = y_train_full.index.year.isin([2020])
X_train    = X_train_full[~covid_rows]   # 2020 feature rows KEPT as lag inputs
y_train    = y_train_full[~covid_rows]   # 2020 target rows DROPPED
```

> [!IMPORTANT]
> 2020 feature values are **kept in X_train** because the 2021 row needs `lag_gdp_1 = −6.60` (the 2020 crash value) to correctly model the 2021 rebound. Removing 2020 entirely from X would destroy this crucial context. The model knows a shock happened — it is simply not penalised for being unable to predict it.

**Final split sizes**:

| Split | Rows | Period |
|---|---|---|
| X_train / y_train | 55 obs | 2008 Q2 – 2022 Q4, excluding 4 quarters of 2020 |
| X_test / y_test | 9 obs | 2023 Q1 – 2025 Q1 |

---

## 5. Model Suite

Six models are trained for each of the three targets.

### A. ARIMA Baseline

The gold standard econometric time-series model (univariate — no feature matrix used).

| Target | Order | Rationale |
|---|---|---|
| Nifty 50 | (2, 1, 0) | I(1) differencing for non-stationary price series; AR(2) captures momentum |
| GDP Growth | (2, 0, 0) | Stationary series; AR(2) for short-term autocorrelation |
| CPI Inflation | (2, 0, 0) | Same as GDP |

Generates 9 unconditional step-ahead forecasts. Cannot use cross-variable information — serves as the conservative econometric benchmark.

### B. Ridge Regression

Regularised linear model with L2 penalty, chosen over LASSO because macro features are highly collinear.

```
Loss = Σ(yᵢ − ŷᵢ)² + α · Σβⱼ²
```

L2 penalty shrinks all coefficients proportionally without eliminating any — stable under collinearity.

**Pipeline**: `StandardScaler → Ridge` (StandardScaler mandatory since features span vastly different scales)

**Alpha grid (extreme shrinkage)**:
```
α ∈ {10, 50, 100, 500, 1000, 5000, 10000}
```

Starting at α = 10 (10× the typical default) reflects the low-data regime. Values up to 10,000 force near-zero coefficients — the primary defence against overfitting.

### C. Random Forest

Ensemble of 300 decision trees, each trained on a bootstrap sample and random feature subset.

```
ŷ_RF = (1/300) · Σ T_b(x)
```

**Anti-overfitting constraints**:

```python
"max_depth"        : [2, 3],       # at most 8 leaf nodes per tree
"min_samples_leaf" : [5, 10, 15],  # each leaf needs ≥5 samples
```

With 55 training examples and a max of 8 leaves, each leaf averages ~7 samples — enough for a reliable mean but not enough to memorise individual outliers. Random feature subsampling (`sqrt(11) ≈ 3` features per split) provides additional decorrelation.

### D. Decision Tree

Single recursive binary partitioning tree — transparent and interpretable.

**Anti-overfitting constraints**:

```python
"max_depth"        : [2, 3],
"min_samples_leaf" : [5, 10, 15],
"min_samples_split": [5, 10],    # node must have ≥5 samples before it can split
```

Most interpretable model in the suite — splits can be visualised as if-then flowcharts showing which features the model considers most discriminative.

### E. Gradient Boosting

Sequential residual fitting — each new tree corrects the prior ensemble's errors:

```
F_M(x) = F_0 + η·h_1(x) + η·h_2(x) + … + η·h_M(x)
```

Each `h_m` is a shallow tree fit to the residuals of the prior ensemble, scaled by learning rate η.

**Anti-overfitting constraints**:

```python
"learning_rate": [0.01, 0.05],   # very slow — 1–5% contribution per tree
"max_depth"    : [1, 2],          # depth-1 = single-split stumps
# Fixed: n_estimators=200, subsample=0.8
```

`subsample=0.8` — each tree trains on 80% of training rows randomly sampled (stochastic gradient boosting), decorrelating trees and further reducing variance.

### F. Ensemble (VotingRegressor)

```
ŷ_Ensemble = (ŷ_Ridge + ŷ_RF + ŷ_DT + ŷ_GB) / 4
```

Uses best-tuned instances from each GridSearchCV, re-fitted on the full training set. Exploits **model diversity**: linear errors from Ridge are orthogonal to step-function errors from trees and direction errors from boosting — averaging them produces more robust predictions.

ARIMA is excluded because it operates in a different input space (no X matrix).

---

## 6. Hyperparameter Tuning

### TimeSeriesSplit — 5 Folds

```
Fold 1:  Train ─── 2010 Q3  │  Val  2010 Q4 – 2012 Q4
Fold 2:  Train ─── 2012 Q4  │  Val  2013 Q1 – 2015 Q1
Fold 3:  Train ─── 2015 Q1  │  Val  2015 Q2 – 2017 Q2
Fold 4:  Train ─── 2017 Q2  │  Val  2017 Q3 – 2019 Q3
Fold 5:  Train ─── 2019 Q3  │  Val  2019 Q4 – 2022 Q4
                              ↑  Always forward-only (no data leakage)
```

Applied **only within training data** (2008–2022). The test set (2023–2025) is never observed during CV.

Best hyperparameters are selected by average CV MSE across all 5 folds. After CV, each model is **re-fitted on all 55 training observations** before predicting on the test set.

Standard k-fold CV is prohibited for time series because random assignment of rows to folds allows a model to train on 2020 data while being evaluated on 2015 data — producing unrealistically optimistic scores that do not reflect deployment performance.

---

## 7. Evaluation Metrics

### Mean Squared Error (MSE)

```
MSE = (1/n) · Σᵢ (yᵢ − ŷᵢ)²
```

Lower is better. Large errors penalised quadratically — one very wrong prediction matters more than several moderately wrong ones.

### Out-of-Sample R² (Campbell and Thompson, 2008)

```
OOS-R² = 1 − [ Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ_train)² ]
```

The denominator benchmark is **ȳ_train** — the historical mean of the training series applied to every future period as a naive forecast.

| Value | Interpretation |
|---|---|
| > 0 | Model beats the naive historical mean — genuinely predictive |
| = 0 | Equal to naive mean |
| < 0 | **Worse than simply predicting the training average — generalisation failure** |

> [!WARNING]
> Negative OOS-R² means the model's predictions are further from truth than a simple constant forecast would be. This is a clear signal of overfitting.

The benchmark ȳ_train is computed from training data only — it never sees the test set, preventing circularity.

---

## 8. Outputs

### Charts (PNG)

| File | Contents |
|---|---|
| `india_macro_pipeline_v3.png` | All 3 targets: overview panel + 6 model mini-panels each (combined figure) |
| `india_nifty50_all_models.png` | Nifty 50: full-history overview + all 6 model close-ups (per-target figure) |
| `india_gdp_growth_all_models.png` | GDP Growth: same layout |
| `india_inflation_all_models.png` | CPI Inflation: same layout |

**Overview panel**: actual series (blue line), COVID grey band, training region shade, firewall dashed line, all 6 model OOS predictions with R² values in legend.

**Model mini-panel**: actual (circles) vs predicted (squares), shaded error band between actual and predicted, MSE and OOS-R² in panel title.

### Excel Workbook (`india_macro_forecasting_results.xlsx`)

| Sheet | Tab colour | Contents |
|---|---|---|
| **Summary** | Gold | 6 models × 3 targets grid. Best MSE highlighted green; R² coloured green (positive) or red (negative) |
| **Nifty 50** | Orange | Quarter-by-quarter Actual + all 6 model predictions. OOS MSE and R² pinned as sub-header rows |
| **GDP Growth** | Green | Same structure |
| **CPI Inflation** | Blue | Same structure |
| **All Predictions** | Purple | Consolidated view: all 3 targets side-by-side, 22 columns total |

---

## 9. Key Design Decisions

### Ridge over LASSO

LASSO (L1 penalty) zeros out some coefficients arbitrarily under collinearity — creating unstable variable selection in small samples. Ridge (L2) shrinks all coefficients proportionally without eliminating any, producing more stable coefficient vectors for macro data where all lagged features likely carry partial signal.

### Extreme Alpha Values (10 – 10,000)

Standard tutorials suggest α ∈ {0.001 – 1.0}. This pipeline uses α ∈ {10 – 10,000} because with 55 observations and 11 features, the OLS estimator without regularisation has a condition number in the thousands due to feature collinearity. Extreme α is required to stabilise the system.

### Shallow Trees (max_depth ≤ 3)

A depth-3 tree has at most 8 leaf nodes. With 55 training examples each leaf averages ~7 samples — adequate for a reliable mean but insufficient to memorise individual recession quarters as decision rules. A depth-5 tree (32 leaves) would average fewer than 2 examples per leaf — guaranteed overfitting.

### Very Low Gradient Boosting Learning Rate

At η = 0.01 each of 200 trees contributes only 1% of its prediction to the ensemble. The cumulative update is a very conservative correction to the initial mean. This implicit regularisation is appropriate when training data is scarce and the signal-to-noise ratio is low.

### COVID-2020 Treatment

Three possible treatments and their trade-offs:

| Option | Problem |
|---|---|
| Include 2020 in y_train | Biases all coefficient estimates; forces model to explain an unexplainable shock |
| Remove 2020 from both X and y | Destroys 2021 feature values; model loses the key information that a crash occurred |
| **Remove 2020 from y only (chosen)** | Model knows about the shock via lag features but is not penalised for not predicting it |

---

## 10. Results Summary

*(Test window: 2023 Q1 – 2025 Q1)*

### Nifty 50

| Model | OOS MSE | OOS-R² |
|---|---|---|
| ARIMA (2,1,0) | 17,545,783 | +0.9036 |
| **Ridge** ⭐ | **4,785,760** | **+0.9737** |
| Random Forest | 42,352,219 | +0.7673 |
| Decision Tree | 33,416,384 | +0.8164 |
| Gradient Boosting | 30,742,242 | +0.8311 |
| Ensemble | 23,898,540 | +0.8687 |

### GDP Growth Rate

| Model | OOS MSE | OOS-R² |
|---|---|---|
| ARIMA (2,0,0) | 0.7637 | +0.0575 |
| Ridge | 0.0998 | +0.8769 |
| **Random Forest** ⭐ | **0.0968** | **+0.8806** |
| Decision Tree | 0.2064 | +0.7453 |
| Gradient Boosting | 3.6291 | −3.4791 |
| Ensemble | 0.3406 | +0.5796 |

### CPI Inflation

| Model | OOS MSE | OOS-R² |
|---|---|---|
| ARIMA (2,0,0) | 4.0339 | −0.0647 |
| **Ridge** ⭐ | **0.0255** | **+0.9933** |
| Random Forest | 0.0516 | +0.9864 |
| Decision Tree | 0.3067 | +0.9191 |
| Gradient Boosting | 0.0961 | +0.9746 |
| Ensemble | 0.0401 | +0.9894 |

### Key Takeaways

1. **Ridge dominates Nifty 50 and CPI** — extreme regularisation on a linear model outperforms all tree-based challengers. Test-period dynamics are well-captured by stable linear combinations of lagged features.

2. **Random Forest wins GDP Growth** — slight edge suggesting mild non-linear interactions in GDP dynamics that a linear model cannot fully capture.

3. **Gradient Boosting fails on GDP** (OOS-R² = −3.48) — catastrophic generalisation failure despite aggressive constraints. Even extreme regularisation cannot save a boosted model with insufficient training data.

4. **ARIMA is strong for Nifty 50** (OOS-R² = +0.90) due to the random-walk-with-drift structure captured by I(1) differencing. Ridge still wins decisively (+0.974) by exploiting cross-variable information unavailable to ARIMA.

5. **ARIMA fails for CPI** (−0.065) — CPI dynamics in 2023–2025 are better explained by cross-variable signals (Nifty 50 and GDP lags) than by CPI's own history alone. This validates the cross-variable feature engineering approach.

6. **The Ensemble is defensive, not offensive** — never wins outright but avoids catastrophic failure. Its GDP OOS-R² of +0.58 (vs Gradient Boosting's −3.48) illustrates the core value of ensemble averaging in low-data, high-uncertainty regimes.

---

*India Macroeconomic BTP Research Project — Pipeline v3 — April 2026.*

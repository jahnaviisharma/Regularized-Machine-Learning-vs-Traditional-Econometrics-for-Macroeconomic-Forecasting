# Regularized Machine Learning vs Traditional Econometrics for Macroeconomic Forecasting

### Bridging Econometrics and Machine Learning for Indian Macroeconomic and Financial Forecasting

**B.Tech Course Project | CP302 | Indian Institute of Technology Ropar**  
**Duration:** January 2026 – May 2026

---

## 📌 Project Overview

This project investigates the complementary roles of **classical econometrics and modern machine learning** for forecasting Indian macroeconomic and financial time-series data.

The central objective is to compare the forecasting performance of the classical **ARIMA** model with several regularized machine learning models under a challenging **small-sample, high-noise, time-dependent data environment**.

The project develops two independent forecasting pipelines:

1. **Macroeconomic Forecasting Pipeline**
   - Nifty 50
   - GDP Growth
   - CPI Inflation

2. **Equity Forecasting Pipeline**
   - HDFC Bank
   - TCS
   - Trent
   - Larsen & Toubro
   - Titan

A major emphasis of the project is on preventing **look-ahead bias and temporal data leakage**. All feature engineering, model selection, validation, and testing follow chronological ordering.

---

# 🎯 Problem Statement

Machine learning models can achieve strong predictive performance in large datasets, but macroeconomic forecasting presents a very different environment.

Economic datasets often have:

- Small numbers of observations
- Strong multicollinearity
- High noise
- Non-stationarity
- Structural breaks
- Persistent temporal dependencies

These characteristics create a high risk of **overfitting**, especially for flexible machine learning models.

The project therefore investigates:

> **How do regularized machine learning models compare with traditional econometric forecasting methods when applied to small-sample Indian macroeconomic and financial time-series data?**

The study focuses particularly on whether strong regularization can improve out-of-sample generalization.

---

# 🎯 Objectives

The project was designed around the following objectives:

### 1. Benchmark ARIMA against Machine Learning

Compare the classical ARIMA econometric baseline against:

- Ridge Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- VotingRegressor

across multiple macroeconomic and financial targets.

### 2. Investigate Regularization

Study whether strong regularization improves forecasting performance in low-data environments.

### 3. Handle COVID-19 Structural Break

Develop a methodology that prevents the extreme COVID-19 observation from dominating model training while still allowing post-COVID predictions to use historical information from 2020.

### 4. Evaluate NSE Equity Forecasting

Test whether historical price-based features can generate useful out-of-sample forecasts for five major NSE-listed equities.

### 5. Evaluate Ensemble Forecasting

Investigate whether combining structurally different ML models improves forecast stability.

### 6. Build a Reproducible Forecasting Framework

Develop a modular Python-based framework with:

- Chronological data splits
- Zero look-ahead feature engineering
- TimeSeriesSplit cross-validation
- Automated model evaluation
- Forecast visualizations
- Structured results

---

# 🏗️ Project Architecture

The project consists of two independent pipelines:

```text
                         DATA SOURCES
                              │
              ┌───────────────┴───────────────┐
              │                               │
       MACROECONOMIC DATA              EQUITY DATA
              │                               │
       GDP / CPI / Nifty             NSE Stock Prices
              │                               │
              └───────────────┬───────────────┘
                              │
                       DATA PROCESSING
                              │
                     FEATURE ENGINEERING
                              │
                   ZERO LOOK-AHEAD DESIGN
                              │
                  CHRONOLOGICAL FIREWALL
                              │
                ┌─────────────┴─────────────┐
                │                           │
          ARIMA BASELINE              ML MODELS
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                       Ridge       Random Forest     Tree
                         │              │              │
                    Gradient Boosting + Voting Ensemble
                         │
                         └──────────────┬──────────────┘
                                        │
                               TIME-SERIES CV
                                        │
                                  FORECASTING
                                        │
                               OUT-OF-SAMPLE
                                  EVALUATION

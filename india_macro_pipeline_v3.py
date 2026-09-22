"""
===============================================================================
  India Macroeconomic Forecasting Pipeline 
  ─────────────────────────────────────────────
  Target Variables:
    1. Stock Price Index  (Nifty 50, sourced via yfinance)
    2. GDP Growth Rate    (%)   — World Bank / IMF India annual estimates
    3. Inflation Rate     (CPI, %) — World Bank / IMF India annual estimates

  Coverage   : 1990 Q1 – 2025 Q1  (Quarterly observations)
  Data source: yfinance (Nifty 50) + hard-coded annual macro data resampled
               to quarterly via linear interpolation

  ─────────────────────────────────────────────────────────────────────────────
  KEY METHODOLOGICAL DESIGN (v3)
  ─────────────────────────────────────────────────────────────────────────────
  1. QUARTERLY FREQUENCY — gives ML models ~130 obs (vs ~35 annual) to learn
     from, dramatically reducing overfitting risk.

  2. THREE-TARGET LOOP — separate model suite per target:
       • Nifty 50 quarterly close price (log-level used as feature, raw for y)
       • GDP growth rate (annualised, quarterly interpolated)
       • CPI inflation rate (annualised, quarterly interpolated)

  3. FIREWALL SPLIT (strict chronological, NO shuffle):
       Training  : 1990 Q1 – 2022 Q4  (post-feature-burn-in)
       EXCLUDED  : 2020 rows as TRAINING TARGETS only (COVID anomaly).
                   2020 lag-values still appear as look-back features for
                   Q1-2021 rows — giving the model "memory" of the shock
                   without fitting the anomaly as a training target.
       Test      : 2023 Q1 – 2025 Q1

  4. FEATURE ENGINEERING (no look-ahead):
       • Own lags 1, 2, 3 (autoregressive)
       • Cross-variable lags 1, 2 (IS-LM interdependency)
       • 3-quarter rolling mean (own + cross)
       • 3-quarter rolling std  (own)
       All rolling/lag operations use .shift(1) before the window.

  5. EXTREME REGULARIZATION (anti-overfitting):
       • Ridge (replaces LASSO) — handles collinear macro features;
         alpha grid spans 10 → 10 000 (extreme shrinkage).
       • Random Forest  — max_depth ≤ 3, min_samples_leaf ≥ 5 (shallow stumps)
       • Decision Tree  — max_depth ≤ 3
       • Gradient Boost — learning_rate ∈ {0.01, 0.05}, max_depth ∈ {1, 2}

  6. ENSEMBLE — VotingRegressor averaging Ridge + RF + DT + GB predictions.

  7. METRICS — OOS MSE + OOS-R² (Campbell & Thompson 2008 benchmark = train mean)
===============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all envs
import matplotlib.pyplot       as plt
import matplotlib.gridspec     as gridspec
import matplotlib.patches      as mpatches

from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import Ridge
from sklearn.tree             import DecisionTreeRegressor
from sklearn.ensemble         import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import GridSearchCV, TimeSeriesSplit
from sklearn.metrics          import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

try:
    import openpyxl
    from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                                  GradientFill)
    from openpyxl.utils  import get_column_letter
    from openpyxl.chart  import BarChart, Reference
    _OPENPYXL = True
except ImportError:
    _OPENPYXL = False
    print("[WARNING] openpyxl not installed — Excel export skipped.")
    print("          Run:  pip install openpyxl")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[WARNING] yfinance not installed.  Run:  pip install yfinance")
    print("          Nifty 50 data will be synthesised from known annual values.")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS & COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-03-31"
COVID_YEARS = [2020]             # rows whose YEAR falls here are excluded as targets

PALETTE = {
    "Actual"             : "#58a6ff",
    "ARIMA (Baseline)"   : "#f85149",
    "Ridge"              : "#ffa657",
    "Random Forest"      : "#3fb950",
    "Decision Tree"      : "#d2a8ff",
    "Gradient Boosting"  : "#ff7b72",
    "Ensemble"           : "#e3b341",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA ACQUISITION
#     2A.  Nifty 50 — via yfinance (quarterly close prices)
#     2B.  GDP Growth & CPI — hard-coded World Bank / IMF annual series
#          resampled to quarterly via linear interpolation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 76)
print("STEP 1 │ Data Acquisition")
print("=" * 76)

# ── 2A : Nifty 50 ─────────────────────────────────────────────────────────────
def fetch_nifty50_quarterly() -> pd.Series:
    """
    Download Nifty 50 (^NSEI) daily prices via yfinance and resample to
    quarterly end-of-quarter close. Falls back to synthetic data if yfinance
    is unavailable or the download fails.
    """
    if _YF_AVAILABLE:
        try:
            print("  Downloading Nifty 50 (^NSEI) from Yahoo Finance …", end="", flush=True)
            raw_daily_raw = yf.download(
                "^NSEI",
                start="1990-01-01",
                end="2025-04-01",
                progress=False,
                auto_adjust=True,
            )
            # Handle both old (single-level) and new (multi-level) column structures
            if isinstance(raw_daily_raw.columns, pd.MultiIndex):
                raw_daily = raw_daily_raw["Close"].squeeze().dropna()
            else:
                raw_daily = raw_daily_raw["Close"].dropna()
            # resample to quarterly (last trading day of each quarter)
            nifty_q = raw_daily.resample("Q").last().dropna()
            nifty_q.index = nifty_q.index.to_period("Q").to_timestamp("Q")
            print(f" done  ({len(nifty_q)} quarters, "
                  f"{nifty_q.index[0].date()} – {nifty_q.index[-1].date()})")
            return nifty_q.rename("nifty50")
        except Exception as exc:
            print(f"\n  [WARNING] yfinance download failed: {exc}")
            print("            Falling back to hard-coded Nifty 50 annual data.")

    # ── FALLBACK: hard-coded approximate Nifty 50 year-end closing values ──
    print("  Using hard-coded Nifty 50 annual data (interpolated to quarterly).")
    nifty_annual = {
        1990: 480,   1991: 1000,  1992: 2615,  1993: 1678,  1994: 3926,
        1995: 3110,  1996: 3085,  1997: 3658,  1998: 3055,  1999: 5006,
        2000: 3972,  2001: 3262,  2002: 3377,  2003: 5590,  2004: 6073,
        2005: 8000,  2006: 13786, 2007: 20286, 2008: 9647,  2009: 17464,
        2010: 20509, 2011: 15454, 2012: 19427, 2013: 21170, 2014: 27499,
        2015: 26117, 2016: 26626, 2017: 33940, 2018: 36068, 2019: 41253,
        2020: 47751, 2021: 58253, 2022: 60840, 2023: 72240, 2024: 78000,
        2025: 74000,
    }
    dates  = pd.date_range("1990-12-31", periods=len(nifty_annual), freq="A")
    annual = pd.Series(list(nifty_annual.values()), index=dates, name="nifty50")
    # resample to quarterly by linear interpolation
    q_idx  = pd.date_range("1990-03-31", "2025-03-31", freq="Q")
    nifty_q = (
        annual.reindex(annual.index.union(q_idx))
              .interpolate("linear")
              .reindex(q_idx)
    )
    nifty_q.index = nifty_q.index.to_period("Q").to_timestamp("Q")
    return nifty_q.rename("nifty50")

nifty_q = fetch_nifty50_quarterly()

# ── 2B : Annual macroeconomic data → quarterly interpolation ──────────────────
YEARS = list(range(1990, 2026))

GDP_GROWTH_ANNUAL = [
    5.53,   # 1990
    1.06,   # 1991  BOP crisis
    5.44,   # 1992
    4.75,   # 1993
    6.38,   # 1994
    7.57,   # 1995
    7.55,   # 1996
    4.05,   # 1997
    6.19,   # 1998
    8.85,   # 1999
    3.84,   # 2000
    5.00,   # 2001
    3.87,   # 2002
    7.86,   # 2003
    7.92,   # 2004
    9.28,   # 2005
    9.26,   # 2006
    9.80,   # 2007  Pre-GFC peak
    3.89,   # 2008  GFC slowdown
    8.48,   # 2009
   10.26,   # 2010
    6.64,   # 2011
    5.46,   # 2012
    6.39,   # 2013
    7.41,   # 2014
    8.00,   # 2015
    8.26,   # 2016
    6.80,   # 2017  Demonetisation drag
    6.45,   # 2018
    3.73,   # 2019  Pre-COVID slowdown
   -6.60,   # 2020  ◄─ COVID-19 ANOMALY
    8.95,   # 2021
    7.24,   # 2022
    8.20,   # 2023
    7.00,   # 2024  advance estimate
    6.50,   # 2025  IMF projection
]

CPI_ANNUAL = [
     9.00,  # 1990
    13.87,  # 1991  Reform-era supply shock
    11.79,  # 1992
     6.36,  # 1993
    10.22,  # 1994
    10.22,  # 1995
     9.00,  # 1996
     7.16,  # 1997
    13.23,  # 1998  Drought + currency depreciation
     4.67,  # 1999
     4.01,  # 2000
     3.68,  # 2001
     4.37,  # 2002
     3.81,  # 2003
     3.77,  # 2004
     4.25,  # 2005
     6.14,  # 2006
     6.37,  # 2007
     8.35,  # 2008  Commodity price surge
    10.88,  # 2009
    12.00,  # 2010  Food inflation
     8.86,  # 2011
     9.30,  # 2012
    10.92,  # 2013  Twin-deficit pressure
     6.37,  # 2014
     4.91,  # 2015
     4.95,  # 2016
     2.49,  # 2017  Demonetisation demand collapse
     4.86,  # 2018
     7.66,  # 2019
     6.62,  # 2020  ◄─ COVID-19 ANOMALY
     5.13,  # 2021
     6.70,  # 2022
     5.65,  # 2023
     4.85,  # 2024  advance estimate
     4.50,  # 2025  projection
]

def annual_to_quarterly(values: list, years: list) -> pd.Series:
    """
    Convert a list of annual values to quarterly frequency via linear
    interpolation.  Annual value is placed at Q4 of each year; Q1-Q3
    are interpolated between consecutive Q4 points.
    Returns a Series indexed by quarter-end timestamps.
    """
    q4_dates = pd.to_datetime([f"{y}-12-31" for y in years])
    annual_s  = pd.Series(values, index=q4_dates)

    # Build full quarterly index from 1990-Q1 to 2025-Q1
    q_idx = pd.date_range("1990-03-31", "2025-03-31", freq="Q")

    combined = (
        annual_s
        .reindex(annual_s.index.union(q_idx))
        .sort_index()
        .interpolate("linear")
        .reindex(q_idx)
    )
    combined.index = combined.index.to_period("Q").to_timestamp("Q")
    return combined

gdp_q = annual_to_quarterly(GDP_GROWTH_ANNUAL, YEARS).rename("gdp_growth")
cpi_q = annual_to_quarterly(CPI_ANNUAL,        YEARS).rename("inflation")

print(f"\n  Nifty 50   : {len(nifty_q)} quarterly observations")
print(f"  GDP Growth : {len(gdp_q)} quarterly observations (interpolated from annual)")
print(f"  CPI        : {len(cpi_q)} quarterly observations (interpolated from annual)")

# ── 2C : Align all series to common quarterly index ───────────────────────────
master = pd.concat([nifty_q, gdp_q, cpi_q], axis=1).dropna()
master.index.name = "quarter"

print(f"\n  Combined master frame : {master.shape}"
      f"  [{master.index[0]} – {master.index[-1]}]")
print(master.head(8).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
#     For each primary target build a feature matrix using:
#       • Own lags 1, 2, 3             (autoregressive dynamics)
#       • Cross-variable lags 1, 2     (2 other variables each lag 1-2)
#       • 3-quarter rolling mean (own + 1 cross)
#       • 3-quarter rolling std  (own)
#     All rolling/lag operations use .shift(1) before the window → zero leakage.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("STEP 2 │ Feature Engineering  (lag + rolling, no look-ahead)")
print("=" * 76)

TARGETS = {
    "nifty50"   : {"label": "Nifty 50 (Quarterly Close)",   "arima": (2, 1, 0)},
    "gdp_growth": {"label": "GDP Growth Rate (%)",           "arima": (2, 0, 0)},
    "inflation" : {"label": "Inflation Rate — CPI (%)",      "arima": (2, 0, 0)},
}

def build_feature_matrix(primary: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a backward-looking feature matrix for `primary` target.
    Cross variables are all columns in df EXCEPT primary.
    All features derived from .shift(1) to prevent look-ahead leakage.
    """
    crosses = [c for c in df.columns if c != primary]
    feat = pd.DataFrame(index=df.index)

    # Own lags 1-3
    for lag in [1, 2, 3]:
        feat[f"lag_{primary}_{lag}"] = df[primary].shift(lag)

    # Cross-variable lags 1-2
    for cross in crosses:
        for lag in [1, 2]:
            feat[f"lag_{cross}_{lag}"] = df[cross].shift(lag)

    # Rolling stats (from .shift(1) → no look-ahead)
    shifted_own = df[primary].shift(1)
    feat[f"roll3_mean_{primary}"] = shifted_own.rolling(3).mean()
    feat[f"roll3_std_{primary}"]  = shifted_own.rolling(3).std()

    for cross in crosses:
        feat[f"roll3_mean_{cross}"] = df[cross].shift(1).rolling(3).mean()

    feat.dropna(inplace=True)
    return feat

features = {}
for tgt in TARGETS:
    features[tgt] = build_feature_matrix(tgt, master)
    print(f"\n  {tgt:12s} feature matrix: {features[tgt].shape}"
          f"  [{features[tgt].index[0]} – {features[tgt].index[-1]}]")
    print(f"              Columns: {list(features[tgt].columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  "FIREWALL" SPLIT WITH COVID-19 EXCLUSION
#     ─────────────────────────────────────────
#     Training  : 1990 – 2022 Q4 (inclusive of all quarters through end of 2022)
#     EXCLUDED  : 2020 rows are DROPPED from the training target vector only.
#                 Their lag values still appear in 2021 feature rows — see above.
#     Testing   : 2023 Q1 – 2025 Q1
#
#     NOTE: The split is performed AFTER feature burn-in (dropping NaN from lags).
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("STEP 3 │ COVID-19 Exclusion + Chronological Firewall Split")
print("=" * 76)
print(f"\n  Training window   : start – {TRAIN_END}  (excl. 2020 as target)")
print(f"  *** EXCLUDED      : year 2020  (COVID-19 anomaly — dropped from y_train)")
print(f"  Test window       : {TEST_START} – {TEST_END}")

def make_splits(feat_df: pd.DataFrame, y_series: pd.Series):
    """
    Chronological split with explicit COVID-year exclusion from TRAINING targets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Align y to feature index (after burn-in NaN drop)
    y = y_series.reindex(feat_df.index)

    # Full chronological masks
    mask_train_full = feat_df.index <= pd.Timestamp(TRAIN_END)
    mask_test       = (feat_df.index >= pd.Timestamp(TEST_START)) & \
                      (feat_df.index <= pd.Timestamp(TEST_END))

    X_train_full = feat_df[mask_train_full]
    y_train_full = y[mask_train_full]
    X_test       = feat_df[mask_test]
    y_test       = y[mask_test]

    # Drop COVID year 2020 from the TRAINING TARGET (not from X — lags keep it)
    covid_rows = y_train_full.index.year.isin(COVID_YEARS)
    X_train    = X_train_full[~covid_rows]
    y_train    = y_train_full[~covid_rows]

    return X_train, X_test, y_train, y_test

splits = {}
for tgt in TARGETS:
    feat_df  = features[tgt]
    y_series = master[tgt].reindex(feat_df.index)
    X_tr, X_te, y_tr, y_te = make_splits(feat_df, y_series)
    splits[tgt] = {"X_train": X_tr, "X_test": X_te,
                   "y_train": y_tr, "y_test":  y_te}
    print(f"\n  {tgt:12s} → "
          f"Train {X_tr.index[0].date()} – {X_tr.index[-1].date()} "
          f"({len(X_tr)} obs, 2020 excluded from y)  |  "
          f"Test {X_te.index[0].date()} – {X_te.index[-1].date()} "
          f"({len(X_te)} obs)")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """
    Out-of-sample R² (Campbell & Thompson 2008).
    Benchmark: historical mean of the training set.
    Positive → model beats naïve mean; Negative → model is worse.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_bm  = np.sum((y_true - train_mean) ** 2)
    return float(1.0 - ss_res / ss_bm)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MASTER MODEL PIPELINE  (runs for each target)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    target_name : str,
    target_col  : str,
    y_train     : pd.Series,
    y_test      : pd.Series,
    X_train     : pd.DataFrame,
    X_test      : pd.DataFrame,
    arima_order : tuple = (2, 0, 0),
    n_cv_splits : int   = 5,
) -> dict:
    """
    Execute the full modelling suite for a single target.
    Returns a dict with model results (predictions, MSE, OOS-R²).
    """
    SEP = "─" * 76
    print(f"\n{'═'*76}")
    print(f"  ▶▶  TARGET : {target_name.upper()}")
    print(f"      Train  : {y_train.index[0].date()} – {y_train.index[-1].date()}"
          f"  ({len(y_train)} obs, COVID-2020 dropped as target)")
    print(f"      Test   : {y_test.index[0].date()} – {y_test.index[-1].date()}"
          f"  ({len(y_test)} obs)")
    print(f"{'═'*76}")

    results    = {}
    train_mean = float(y_train.mean())

    # ──────────────────────────────────────────────────────────────────────
    # A  ARIMA BASELINE  (econometric benchmark)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  [A] ARIMA{arima_order} — Econometric Baseline")
    print(SEP)

    arima_fit = ARIMA(y_train.values, order=arima_order).fit()
    arima_fc  = arima_fit.forecast(steps=len(y_test))

    mse_a = mean_squared_error(y_test, arima_fc)
    r2_a  = oos_r2(y_test.values, arima_fc, train_mean)
    print(f"  OOS MSE = {mse_a:.4f}   OOS-R² = {r2_a:+.4f}")
    results["ARIMA (Baseline)"] = {"preds": arima_fc, "MSE": mse_a, "OOS_R2": r2_a}

    # ──────────────────────────────────────────────────────────────────────
    # TimeSeriesSplit — shared across all GridSearchCV calls
    # ──────────────────────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    print(f"\n{SEP}")
    print(f"  TimeSeriesSplit: {n_cv_splits} folds (training data only)")
    for fold, (tr, va) in enumerate(tscv.split(X_train), 1):
        print(f"    Fold {fold}: train …{X_train.index[tr[-1]].date()}"
              f"  ({len(tr)} obs)  │  val {X_train.index[va[0]].date()}"
              f"–{X_train.index[va[-1]].date()}  ({len(va)} obs)")
    print(SEP)

    # ──────────────────────────────────────────────────────────────────────
    # B  MODEL A — RIDGE  (StandardScaler + Ridge, L2 penalty)
    #    Extreme alpha values → forces maximum shrinkage to fight collinearity
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [B] Model A — Ridge  (StandardScaler + Ridge, L2 extreme shrinkage)")
    print("      α ∈ {10, 50, 100, 500, 1000, 5000, 10000}")
    print(SEP)

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge()),
    ])
    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid={"ridge__alpha": [10, 50, 100, 500, 1000, 5000, 10_000]},
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)

    best_alpha = ridge_grid.best_params_["ridge__alpha"]
    print(f"  Best α = {best_alpha}   CV MSE = {-ridge_grid.best_score_:.4f}")

    preds_r = ridge_grid.best_estimator_.predict(X_test)
    mse_r   = mean_squared_error(y_test, preds_r)
    r2_r    = oos_r2(y_test.values, preds_r, train_mean)
    print(f"  OOS MSE = {mse_r:.4f}   OOS-R² = {r2_r:+.4f}")
    results["Ridge"] = {"preds": preds_r, "MSE": mse_r, "OOS_R2": r2_r}

    # ──────────────────────────────────────────────────────────────────────
    # C  MODEL B — RANDOM FOREST  (shallow stumps, high min_samples_leaf)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [C] Model B — Random Forest  (shallow stumps, high min_samples_leaf)")
    print("      max_depth ∈ {2, 3}   min_samples_leaf ∈ {5, 10, 15}")
    print(SEP)

    rf_grid = GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        param_grid={
            "max_depth"        : [2, 3],
            "min_samples_leaf" : [5, 10, 15],
        },
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    ).fit(X_train, y_train)

    print(f"  Best params = {rf_grid.best_params_}   CV MSE = {-rf_grid.best_score_:.4f}")
    preds_rf = rf_grid.best_estimator_.predict(X_test)
    mse_rf   = mean_squared_error(y_test, preds_rf)
    r2_rf    = oos_r2(y_test.values, preds_rf, train_mean)
    print(f"  OOS MSE = {mse_rf:.4f}   OOS-R² = {r2_rf:+.4f}")
    results["Random Forest"] = {"preds": preds_rf, "MSE": mse_rf, "OOS_R2": r2_rf}

    # ──────────────────────────────────────────────────────────────────────
    # D  MODEL C — DECISION TREE  (max_depth ≤ 3)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [D] Model C — Decision Tree  (max_depth ≤ 3)")
    print(SEP)

    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid={
            "max_depth"        : [2, 3],
            "min_samples_leaf" : [5, 10, 15],
            "min_samples_split": [5, 10],
        },
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)

    best_dt = dt_grid.best_estimator_
    print(f"  Best params = {dt_grid.best_params_}   CV MSE = {-dt_grid.best_score_:.4f}")
    print(f"  Actual depth: {best_dt.get_depth()}   Leaves: {best_dt.get_n_leaves()}")
    preds_dt = best_dt.predict(X_test)
    mse_dt   = mean_squared_error(y_test, preds_dt)
    r2_dt    = oos_r2(y_test.values, preds_dt, train_mean)
    print(f"  OOS MSE = {mse_dt:.4f}   OOS-R² = {r2_dt:+.4f}")
    results["Decision Tree"] = {"preds": preds_dt, "MSE": mse_dt, "OOS_R2": r2_dt}

    # ──────────────────────────────────────────────────────────────────────
    # E  MODEL D — GRADIENT BOOSTING  (low lr, shallow trees)
    #    F_m(x) = F_{m-1}(x) + η · h_m(x)   (h_m fits residuals of F_{m-1})
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [E] Model D — Gradient Boosting  (low lr + shallow depth)")
    print("      learning_rate ∈ {0.01, 0.05}   max_depth ∈ {1, 2}")
    print(SEP)

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(
            random_state=42, subsample=0.8, n_estimators=200
        ),
        param_grid={
            "learning_rate": [0.01, 0.05],
            "max_depth"    : [1, 2],
        },
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)

    print(f"  Best params = {gb_grid.best_params_}   CV MSE = {-gb_grid.best_score_:.4f}")
    preds_gb = gb_grid.best_estimator_.predict(X_test)
    mse_gb   = mean_squared_error(y_test, preds_gb)
    r2_gb    = oos_r2(y_test.values, preds_gb, train_mean)
    print(f"  OOS MSE = {mse_gb:.4f}   OOS-R² = {r2_gb:+.4f}")
    results["Gradient Boosting"] = {"preds": preds_gb, "MSE": mse_gb, "OOS_R2": r2_gb}

    # ──────────────────────────────────────────────────────────────────────
    # F  ENSEMBLE — VotingRegressor (Ridge + RF + DT + GB)
    #    ŷ = 0.25 · ŷ_Ridge + 0.25 · ŷ_RF + 0.25 · ŷ_DT + 0.25 · ŷ_GB
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [F] Ensemble — VotingRegressor (Ridge + RF + DT + GB)")
    print("      ŷ = (ŷ_Ridge + ŷ_RF + ŷ_DT + ŷ_GB) / 4")
    print(SEP)

    ensemble = VotingRegressor(estimators=[
        ("ridge", ridge_grid.best_estimator_),
        ("rf",    rf_grid.best_estimator_),
        ("dt",    dt_grid.best_estimator_),
        ("gb",    gb_grid.best_estimator_),
    ]).fit(X_train, y_train)

    preds_ens = ensemble.predict(X_test)
    mse_ens   = mean_squared_error(y_test, preds_ens)
    r2_ens    = oos_r2(y_test.values, preds_ens, train_mean)
    print(f"  OOS MSE = {mse_ens:.4f}   OOS-R² = {r2_ens:+.4f}")
    results["Ensemble"] = {"preds": preds_ens, "MSE": mse_ens, "OOS_R2": r2_ens}

    # ──────────────────────────────────────────────────────────────────────
    # G  RESULTS TABLE
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  OUT-OF-SAMPLE BENCHMARKING — {target_name}  (2023 Q1 – 2025 Q1)")
    print(SEP)
    print(f"  {'Model':<24}  {'OOS MSE':>12}  {'OOS-R²':>10}")
    print(f"  {'─'*50}")
    best_mse = min(v["MSE"] for v in results.values())
    for name, v in results.items():
        tag = "  ◀ BEST" if abs(v["MSE"] - best_mse) < 1e-9 else ""
        print(f"  {name:<24}  {v['MSE']:>12.4f}  {v['OOS_R2']:>+10.4f}{tag}")

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN LOOP — run pipeline for all three targets
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("STEP 4 │ Running Full Pipeline for All Three Targets")
print("=" * 76)

all_results = {}

for tgt_col, tgt_meta in TARGETS.items():
    sp = splits[tgt_col]
    all_results[tgt_col] = run_pipeline(
        target_name = tgt_meta["label"],
        target_col  = tgt_col,
        y_train     = sp["y_train"],
        y_test      = sp["y_test"],
        X_train     = sp["X_train"],
        X_test      = sp["X_test"],
        arima_order = tgt_meta["arima"],
        n_cv_splits = 5,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
#     Figure A (combined overview)  : 3-target overview rows + 6 mini-panels each
#     Figure B (per-target deep)    : One PNG per target — overview + all 6 models
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("STEP 5 │ Generating Visualisations")
print("=" * 76)


# ─── shared plot helpers ──────────────────────────────────────────────────────
def style_ax(ax, bg="white"):
    ax.set_facecolor(bg)
    ax.tick_params(colors="#333333", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5, linestyle="--")


def plot_overview(ax, y_full, results, tgt_label, X_test):
    """Wide panel: full history + all OOS predictions + COVID band."""
    style_ax(ax)
    all_idx = y_full.index

    ax.axvspan(pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31"),
               color="#f87171", alpha=0.12, label="2020 excluded (COVID)")
    ax.axvspan(all_idx[0], pd.Timestamp(TRAIN_END),
               color="#e0e7ff", alpha=0.45, label="Training window (–2022)")
    ax.plot(all_idx, y_full.values, color=PALETTE["Actual"],
            linewidth=1.9, alpha=0.92, label="Actual", zorder=5)
    ax.axvline(pd.Timestamp(TEST_START), color="#6b7280",
               linewidth=1.2, linestyle="--",
               label="Firewall │ Test ≥ 2023-Q1", zorder=6)

    test_idx = X_test.index
    for name, v in results.items():
        ax.plot(test_idx, v["preds"],
                color=PALETTE[name], linestyle="--", linewidth=1.15,
                alpha=0.90, label=f"{name} (R²={v['OOS_R2']:+.3f})", zorder=7)

    ax.set_title(f"{tgt_label}  —  Full History & OOS Predictions",
                 color="#111111", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Value", color="#444444", fontsize=8)
    ax.legend(framealpha=0.85, labelcolor="#111111", edgecolor="#cccccc",
              fontsize=5.8, ncol=4, loc="upper left")
    ax.set_xlim(all_idx[0], pd.Timestamp(TEST_END) + pd.DateOffset(months=3))


def plot_mini(ax, y_test, preds, name, mse, r2, test_idx, bg="white"):
    """Small panel: one model vs actual for the test window."""
    style_ax(ax, bg=bg)
    ax.fill_between(test_idx, y_test.values, preds,
                    alpha=0.12, color=PALETTE[name])
    ax.plot(test_idx, y_test.values,
            color=PALETTE["Actual"], linewidth=1.6,
            marker="o", markersize=4, label="Actual", zorder=5)
    ax.plot(test_idx, preds,
            color=PALETTE[name], linewidth=1.2, linestyle="--",
            marker="s", markersize=3.5, alpha=0.95,
            label=name, zorder=6)
    ax.set_title(f"{name}\nMSE={mse:.4f}   OOS-R²={r2:+.4f}",
                 color="#111111", fontsize=7, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=5.5, rotation=40)
    ax.legend(framealpha=0.85, labelcolor="#111111", edgecolor="#cccccc",
              fontsize=5.5, loc="best")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A  —  Combined Overview  (3 targets × [overview + 6 mini-panels])
# Layout per target: 1 overview row (3-wide) + 2 mini rows (3-each) = 3 rows
# Total grid height: 3 targets × 3 rows = 9 rows
# ─────────────────────────────────────────────────────────────────────────────
ROWS_PER_TGT = 3   # 1 overview row + 2 mini rows (3 panels each = 6 models)
N_COLS       = 3
ALL_MODELS   = ["ARIMA (Baseline)", "Ridge", "Random Forest",
                "Decision Tree", "Gradient Boosting", "Ensemble"]

fig_a = plt.figure(figsize=(24, 9 * ROWS_PER_TGT))
fig_a.patch.set_facecolor("white")
gs_a = gridspec.GridSpec(
    len(TARGETS) * ROWS_PER_TGT, N_COLS,
    figure=fig_a, hspace=0.72, wspace=0.30,
    height_ratios=([1.6, 1.0, 1.0] * len(TARGETS)),
)

for t_idx, (tgt_col, tgt_meta) in enumerate(TARGETS.items()):
    sp       = splits[tgt_col]
    results  = all_results[tgt_col]
    y_full   = master[tgt_col].reindex(features[tgt_col].index)
    y_test   = sp["y_test"]
    X_test   = sp["X_test"]
    test_idx = X_test.index
    base_r   = t_idx * ROWS_PER_TGT

    # Overview row
    ax_ov = fig_a.add_subplot(gs_a[base_r, :])
    plot_overview(ax_ov, y_full, results, tgt_meta["label"], X_test)

    # 6 mini-panels across 2 rows
    for m_idx, mname in enumerate(ALL_MODELS):
        mini_row = base_r + 1 + (m_idx // N_COLS)   # row 1 or row 2
        mini_col = m_idx % N_COLS
        ax_m = fig_a.add_subplot(gs_a[mini_row, mini_col])
        v    = results[mname]
        plot_mini(ax_m, y_test, v["preds"], mname,
                  v["MSE"], v["OOS_R2"], test_idx)

fig_a.suptitle(
    "India Macroeconomic Forecasting — v3  (Quarterly, 3 Targets)\n"
    "Train: 2008–2022 Q4  │  COVID-2020 excluded from y_train  │  Test: 2023 Q1 – 2025 Q1\n"
    "Models: ARIMA · Ridge · Random Forest · Decision Tree · Gradient Boosting · Ensemble",
    color="#111111", fontsize=11, fontweight="bold", y=1.001,
)

out_combined = "india_macro_pipeline_v3.png"
fig_a.savefig(out_combined, dpi=150, bbox_inches="tight",
              facecolor=fig_a.get_facecolor())
print(f"\n  [Fig A] Combined overview + all model panels  → {out_combined}")
plt.close(fig_a)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B  —  Per-target detailed PNGs (1 per target)
# Each figure: top = overview, bottom 2 rows = all 6 model mini-panels
# ─────────────────────────────────────────────────────────────────────────────
per_target_pngs = {}
for tgt_col, tgt_meta in TARGETS.items():
    sp       = splits[tgt_col]
    results  = all_results[tgt_col]
    y_full   = master[tgt_col].reindex(features[tgt_col].index)
    y_test   = sp["y_test"]
    X_test   = sp["X_test"]
    test_idx = X_test.index
    label    = tgt_meta["label"]

    fig_b = plt.figure(figsize=(22, 16))
    fig_b.patch.set_facecolor("white")
    gs_b = gridspec.GridSpec(
        3, N_COLS, figure=fig_b,
        hspace=0.68, wspace=0.30,
        height_ratios=[1.8, 1.0, 1.0],
    )

    # Overview
    ax_ov = fig_b.add_subplot(gs_b[0, :])
    plot_overview(ax_ov, y_full, results, label, X_test)

    # All 6 model panels
    for m_idx, mname in enumerate(ALL_MODELS):
        row = 1 + (m_idx // N_COLS)
        col = m_idx % N_COLS
        ax_m = fig_b.add_subplot(gs_b[row, col])
        v    = results[mname]
        plot_mini(ax_m, y_test, v["preds"], mname,
                  v["MSE"], v["OOS_R2"], test_idx)

    fig_b.suptitle(
        f"India — {label}  (Quarterly, 2008–2025)\n"
        "Train: 2008 Q3 – 2022 Q4  │  COVID-2020 excluded  │  "
        "Test: 2023 Q1 – 2025 Q1",
        color="#111111", fontsize=10.5, fontweight="bold", y=1.002,
    )
    safe_name = tgt_col.replace(" ", "_")
    out_b = f"india_{safe_name}_all_models.png"
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight",
                  facecolor=fig_b.get_facecolor())
    per_target_pngs[tgt_col] = out_b
    print(f"  [Fig B] {label:<35} → {out_b}")
    plt.close(fig_b)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  EXCEL EXPORT
#     Sheet 1 : Summary — OOS MSE & OOS-R² for every model × target
#     Sheet 2 : Nifty 50 predictions (quarter-by-quarter)
#     Sheet 3 : GDP Growth predictions
#     Sheet 4 : CPI Inflation predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("STEP 6 │ Excel Export")
print("=" * 76)

XL_OUT = "india_macro_forecasting_results.xlsx"

if not _OPENPYXL:
    print("  [SKIP] openpyxl not available. Install via: pip install openpyxl")
else:
    wb = openpyxl.Workbook()

    # ── Colour palette for Excel ──────────────────────────────────────────────
    XL_BG_HEADER  = "1F2937"   # dark header
    XL_BG_BEST    = "14532D"   # dark green  — best model row
    XL_BG_OV_ROW  = "111827"   # alternating row dark
    XL_BG_ALT_ROW = "1C2432"   # alternating row slightly lighter
    XL_FG_WHITE   = "FFFFFF"
    XL_FG_GOLD    = "F59E0B"
    XL_FG_GREEN   = "34D399"
    XL_FG_RED     = "F87171"
    XL_FG_BLUE    = "60A5FA"
    XL_FG_GREY    = "9CA3AF"

    def hdr_font(bold=True, color=XL_FG_WHITE, size=10):
        return Font(bold=bold, color=color, size=size, name="Calibri")

    def cell_font(bold=False, color=XL_FG_WHITE, size=9):
        return Font(bold=bold, color=color, size=size, name="Calibri")

    def fill(hex_color):
        return PatternFill("solid", fgColor=hex_color)

    def center():
        return Alignment(horizontal="center", vertical="center", wrap_text=False)

    def left():
        return Alignment(horizontal="left",   vertical="center")

    def thin_border():
        s = Side(style="thin", color="374151")
        return Border(left=s, right=s, top=s, bottom=s)

    def set_col_width(ws, col_letter, width):
        ws.column_dimensions[col_letter].width = width

    # ═════════════════════════════════════════════════════════════════════════
    # SHEET 1 — SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    ws_sum = wb.active
    ws_sum.title = "Summary"
    ws_sum.sheet_view.showGridLines = False
    ws_sum.sheet_properties.tabColor = "F59E0B"

    # Title block
    ws_sum.merge_cells("A1:H1")
    title_cell = ws_sum["A1"]
    title_cell.value   = "India Macroeconomic Forecasting — v3  |  Out-of-Sample Results"
    title_cell.font    = Font(bold=True, color=XL_FG_GOLD, size=13, name="Calibri")
    title_cell.fill    = fill("0F172A")
    title_cell.alignment = center()

    ws_sum.merge_cells("A2:H2")
    sub_cell = ws_sum["A2"]
    sub_cell.value = (
        "Train: 2008 Q3 – 2022 Q4  (COVID-2020 excluded from y_train)  "
        "│  Test: 2023 Q1 – 2025 Q1  │  OOS-R²: Campbell & Thompson (2008)"
    )
    sub_cell.font      = cell_font(color=XL_FG_GREY, size=8)
    sub_cell.fill      = fill("0F172A")
    sub_cell.alignment = center()

    ws_sum.row_dimensions[1].height = 22
    ws_sum.row_dimensions[2].height = 16

    # Header row
    hdrs = ["", "Model",
            "Nifty 50\nOOS MSE",  "Nifty 50\nOOS-R²",
            "GDP Growth\nOOS MSE", "GDP Growth\nOOS-R²",
            "CPI Inflation\nOOS MSE", "CPI Inflation\nOOS-R²"]
    for c_idx, h in enumerate(hdrs, 1):
        cell = ws_sum.cell(row=4, column=c_idx, value=h)
        cell.font      = hdr_font(size=9)
        cell.fill      = fill(XL_BG_HEADER)
        cell.alignment = center()
        cell.border    = thin_border()
    ws_sum.row_dimensions[4].height = 30

    # Model rows
    all_msm = {}
    for tgt_col in TARGETS:
        all_msm[tgt_col] = min(v["MSE"] for v in all_results[tgt_col].values())

    for r_idx, mname in enumerate(ALL_MODELS, 5):
        bg = XL_BG_OV_ROW if r_idx % 2 == 1 else XL_BG_ALT_ROW
        ws_sum.cell(row=r_idx, column=1, value=r_idx - 4).font = cell_font(color=XL_FG_GREY)
        ws_sum.cell(row=r_idx, column=1).fill = fill(bg)
        ws_sum.cell(row=r_idx, column=1).alignment = center()
        ws_sum.cell(row=r_idx, column=1).border = thin_border()

        ws_sum.cell(row=r_idx, column=2, value=mname)
        ws_sum.cell(row=r_idx, column=2).font = cell_font(bold=(mname == "Ensemble"))
        ws_sum.cell(row=r_idx, column=2).fill = fill(bg)
        ws_sum.cell(row=r_idx, column=2).alignment = left()
        ws_sum.cell(row=r_idx, column=2).border = thin_border()

        col = 3
        for tgt_col in TARGETS:
            v   = all_results[tgt_col][mname]
            mse = round(v["MSE"],    4)
            r2  = round(v["OOS_R2"], 4)
            is_best_mse = abs(v["MSE"] - all_msm[tgt_col]) < 1e-9

            c_mse = ws_sum.cell(row=r_idx, column=col,   value=mse)
            c_r2  = ws_sum.cell(row=r_idx, column=col+1, value=r2)

            for c in [c_mse, c_r2]:
                this_bg = XL_BG_BEST if is_best_mse else bg
                c.fill      = fill(this_bg)
                c.alignment = center()
                c.border    = thin_border()
                c.number_format = "0.0000"

            # Colour OOS-R² by sign
            r2_color = XL_FG_GREEN if r2 >= 0 else XL_FG_RED
            c_mse.font = cell_font(bold=is_best_mse,
                                    color=(XL_FG_GOLD if is_best_mse else XL_FG_WHITE))
            c_r2.font  = cell_font(bold=is_best_mse, color=r2_color)
            col += 2

    # Add ◀ BEST legend
    ws_sum.merge_cells("A12:H12")
    leg = ws_sum["A12"]
    leg.value     = "★ Green highlight = best MSE for that target   |   OOS-R² green = positive (beats naïve mean)  /  red = negative"
    leg.font      = cell_font(color=XL_FG_GREY, size=8)
    leg.fill      = fill("0F172A")
    leg.alignment = center()

    # Column widths
    widths = [4, 22, 14, 12, 14, 12, 16, 12]
    for i, w in enumerate(widths, 1):
        ws_sum.column_dimensions[get_column_letter(i)].width = w

    # ═════════════════════════════════════════════════════════════════════════
    # SHEETS 2-4 — Per-target quarter-by-quarter predictions
    # ═════════════════════════════════════════════════════════════════════════
    TGT_SHEET_NAMES = {
        "nifty50"   : "Nifty 50",
        "gdp_growth": "GDP Growth",
        "inflation" : "CPI Inflation",
    }
    TGT_TAB_COLORS = {
        "nifty50"   : "ffa657",
        "gdp_growth": "3fb950",
        "inflation" : "58a6ff",
    }

    for tgt_col, sheet_name in TGT_SHEET_NAMES.items():
        ws = wb.create_sheet(title=sheet_name)
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.tabColor = TGT_TAB_COLORS[tgt_col]

        label = TARGETS[tgt_col]["label"]
        sp    = splits[tgt_col]
        y_te  = sp["y_test"]
        results = all_results[tgt_col]
        best_mse_val = min(v["MSE"] for v in results.values())
        test_qtrs = [str(q.date()) for q in sp["X_test"].index]

        # Title
        n_data_cols = 2 + len(ALL_MODELS)   # Quarter + Actual + 6 models
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1,   end_column=n_data_cols)
        t = ws.cell(row=1, column=1,
                    value=f"India — {label}  |  Quarterly OOS Predictions (2023 Q1 – 2025 Q1)")
        t.font = Font(bold=True, color=XL_FG_GOLD, size=12, name="Calibri")
        t.fill = fill("0F172A")
        t.alignment = center()
        ws.row_dimensions[1].height = 22

        # Sub-header
        ws.merge_cells(start_row=2, start_column=1,
                       end_row=2,   end_column=n_data_cols)
        s = ws.cell(row=2, column=1,
                    value=(
                        "Train (excluded COVID-2020 from y_train): 2008 Q3 – 2022 Q4  "
                        "│  Test: 2023 Q1 – 2025 Q1  "
                        "│  ★ = best MSE model"
                    ))
        s.font = cell_font(color=XL_FG_GREY, size=8)
        s.fill = fill("0F172A")
        s.alignment = center()

        # Header row (row 4)
        hdrs2 = ["Quarter", "Actual"] + ALL_MODELS
        for c_idx, h in enumerate(hdrs2, 1):
            cell = ws.cell(row=4, column=c_idx, value=h)
            cell.font = hdr_font(size=9)
            cell.fill = fill(XL_BG_HEADER)
            cell.alignment = center()
            cell.border = thin_border()
        ws.row_dimensions[4].height = 22

        # MSE / R² sub-header rows (rows 5 & 6)
        ws.cell(row=5, column=1, value="OOS MSE").font   = cell_font(color=XL_FG_GREY, size=8)
        ws.cell(row=5, column=1).fill = fill("111827")
        ws.cell(row=5, column=1).alignment = center()
        ws.cell(row=6, column=1, value="OOS-R²").font    = cell_font(color=XL_FG_GREY, size=8)
        ws.cell(row=6, column=1).fill = fill("111827")
        ws.cell(row=6, column=1).alignment = center()
        ws.cell(row=5, column=2, value="—").font = cell_font(color=XL_FG_GREY, size=8)
        ws.cell(row=5, column=2).fill = fill("111827")
        ws.cell(row=5, column=2).alignment = center()
        ws.cell(row=6, column=2, value="—").font = cell_font(color=XL_FG_GREY, size=8)
        ws.cell(row=6, column=2).fill = fill("111827")
        ws.cell(row=6, column=2).alignment = center()

        for m_idx, mname in enumerate(ALL_MODELS):
            col       = 3 + m_idx
            v         = results[mname]
            is_best   = abs(v["MSE"] - best_mse_val) < 1e-9
            mse_color = XL_FG_GOLD if is_best else XL_FG_WHITE
            r2_color  = XL_FG_GREEN if v["OOS_R2"] >= 0 else XL_FG_RED
            bg2       = XL_BG_BEST if is_best else "111827"

            c_mse = ws.cell(row=5, column=col, value=round(v["MSE"],    4))
            c_r2  = ws.cell(row=6, column=col, value=round(v["OOS_R2"], 4))
            for c, clr in [(c_mse, mse_color), (c_r2, r2_color)]:
                c.font = cell_font(bold=is_best, color=clr, size=8)
                c.fill = fill(bg2)
                c.alignment = center()
                c.number_format = "0.0000"

        # Data rows (from row 7)
        for q_idx, (qdate, actual) in enumerate(zip(test_qtrs, y_te.values)):
            row  = 7 + q_idx
            bg_r = XL_BG_OV_ROW if q_idx % 2 == 0 else XL_BG_ALT_ROW

            c_q = ws.cell(row=row, column=1, value=qdate)
            c_q.font = cell_font(color=XL_FG_BLUE)
            c_q.fill = fill(bg_r)
            c_q.alignment = center()
            c_q.border = thin_border()

            c_a = ws.cell(row=row, column=2, value=round(float(actual), 4))
            c_a.font = cell_font(bold=True, color=XL_FG_WHITE)
            c_a.fill = fill(bg_r)
            c_a.alignment = center()
            c_a.border = thin_border()
            c_a.number_format = "0.0000"

            for m_idx, mname in enumerate(ALL_MODELS):
                col   = 3 + m_idx
                pred  = results[mname]["preds"][q_idx]
                err   = float(pred) - float(actual)
                is_best = abs(results[mname]["MSE"] - best_mse_val) < 1e-9
                this_bg = XL_BG_BEST if is_best else bg_r
                c_p = ws.cell(row=row, column=col, value=round(float(pred), 4))
                c_p.font = cell_font(
                    bold=is_best,
                    color=(XL_FG_GOLD if is_best
                           else XL_FG_GREEN if abs(err) < abs(actual) * 0.05
                           else XL_FG_WHITE)
                )
                c_p.fill = fill(this_bg)
                c_p.alignment = center()
                c_p.border = thin_border()
                c_p.number_format = "0.0000"

        # Auto-width columns
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 12
        for m_idx in range(len(ALL_MODELS)):
            ws.column_dimensions[get_column_letter(3 + m_idx)].width = 15

    # ═════════════════════════════════════════════════════════════════════════
    # SHEET 5 — ALL PREDICTIONS  (consolidated: all targets × all models)
    # Rows   : one per test quarter (2023 Q1 – 2025 Q1)
    # Columns: Quarter | [per target: Actual · ARIMA · Ridge · RF · DT · GB · Ensemble]
    # Sub-header rows (rows 5-6): OOS MSE and OOS-R² for every (target, model) pair
    # ═════════════════════════════════════════════════════════════════════════
    ws_all = wb.create_sheet(title="All Predictions")
    ws_all.sheet_view.showGridLines = False
    ws_all.sheet_properties.tabColor = "A855F7"     # purple tab

    # ── fixed reference data ──────────────────────────────────────────────────
    # use nifty50 test index as the canonical quarter list (all targets share it)
    ref_tgt      = "nifty50"
    test_qtrs    = [str(q.date()) for q in splits[ref_tgt]["X_test"].index]
    n_test       = len(test_qtrs)

    TGT_LABELS_SHORT = {
        "nifty50"   : "Nifty 50",
        "gdp_growth": "GDP Growth (%)",
        "inflation" : "CPI Inflation (%)",
    }
    PER_TGT_COLS = 1 + len(ALL_MODELS)   # Actual + 6 models = 7 cols per target
    TOTAL_DATA_COLS = 1 + len(TARGETS) * PER_TGT_COLS   # Quarter + 3×7 = 22

    # ── Title (row 1) ─────────────────────────────────────────────────────────
    ws_all.merge_cells(start_row=1, start_column=1,
                       end_row=1,   end_column=TOTAL_DATA_COLS)
    tc = ws_all.cell(row=1, column=1,
                     value="India Macroeconomic Forecasting — v3  |  "
                           "Consolidated OOS Predictions  (All Targets × All Models)")
    tc.font      = Font(bold=True, color=XL_FG_GOLD, size=13, name="Calibri")
    tc.fill      = fill("0F172A")
    tc.alignment = center()
    ws_all.row_dimensions[1].height = 22

    # ── Sub-title (row 2) ─────────────────────────────────────────────────────
    ws_all.merge_cells(start_row=2, start_column=1,
                       end_row=2,   end_column=TOTAL_DATA_COLS)
    sc = ws_all.cell(row=2, column=1,
                     value="Test window: 2023 Q1 – 2025 Q1  │  "
                           "Train: 2008 Q3 – 2022 Q4  (COVID-2020 excluded from y_train)  │  "
                           "★ = best MSE model per target")
    sc.font      = cell_font(color=XL_FG_GREY, size=8)
    sc.fill      = fill("0F172A")
    sc.alignment = center()
    ws_all.row_dimensions[2].height = 14

    # ── Target group headers (row 3) ──────────────────────────────────────────
    ws_all.cell(row=3, column=1, value="").fill = fill("0F172A")
    for t_idx, (tgt_col, tgt_short) in enumerate(TGT_LABELS_SHORT.items()):
        col_start = 2 + t_idx * PER_TGT_COLS
        col_end   = col_start + PER_TGT_COLS - 1
        ws_all.merge_cells(start_row=3, start_column=col_start,
                           end_row=3,   end_column=col_end)
        tg = ws_all.cell(row=3, column=col_start, value=tgt_short)
        tg.font      = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
        tg.fill      = fill({"nifty50": "92400E",
                             "gdp_growth": "14532D",
                             "inflation": "1E3A5F"}[tgt_col])
        tg.alignment = center()
        tg.border    = thin_border()
    ws_all.row_dimensions[3].height = 18

    # ── Column headers (row 4): Quarter | [Actual, ARIMA, Ridge, RF, DT, GB, Ens] × 3 ──
    ws_all.cell(row=4, column=1, value="Quarter")
    ws_all.cell(row=4, column=1).font      = hdr_font(size=9)
    ws_all.cell(row=4, column=1).fill      = fill(XL_BG_HEADER)
    ws_all.cell(row=4, column=1).alignment = center()
    ws_all.cell(row=4, column=1).border    = thin_border()
    ws_all.column_dimensions["A"].width    = 14

    col_map = {}   # (tgt_col, col_name) → excel column index
    for t_idx, tgt_col in enumerate(TARGETS):
        col_base = 2 + t_idx * PER_TGT_COLS
        sub_hdrs = ["Actual"] + ALL_MODELS
        for s_idx, sh in enumerate(sub_hdrs):
            c_idx = col_base + s_idx
            c = ws_all.cell(row=4, column=c_idx, value=sh)
            c.font      = hdr_font(size=8)
            c.fill      = fill(XL_BG_HEADER)
            c.alignment = center()
            c.border    = thin_border()
            ws_all.column_dimensions[get_column_letter(c_idx)].width = 14
            col_map[(tgt_col, sh)] = c_idx
    ws_all.row_dimensions[4].height = 22

    # ── OOS MSE sub-header (row 5) ────────────────────────────────────────────
    ws_all.cell(row=5, column=1, value="OOS MSE")
    ws_all.cell(row=5, column=1).font      = cell_font(color=XL_FG_GREY, size=8)
    ws_all.cell(row=5, column=1).fill      = fill("111827")
    ws_all.cell(row=5, column=1).alignment = center()

    # ── OOS-R² sub-header (row 6) ─────────────────────────────────────────────
    ws_all.cell(row=6, column=1, value="OOS-R²")
    ws_all.cell(row=6, column=1).font      = cell_font(color=XL_FG_GREY, size=8)
    ws_all.cell(row=6, column=1).fill      = fill("111827")
    ws_all.cell(row=6, column=1).alignment = center()

    for tgt_col in TARGETS:
        res_t        = all_results[tgt_col]
        best_mse_t   = min(v["MSE"] for v in res_t.values())

        # Actual column in rows 5-6: just dashes
        c_act_mse = ws_all.cell(row=5, column=col_map[(tgt_col, "Actual")], value="—")
        c_act_r2  = ws_all.cell(row=6, column=col_map[(tgt_col, "Actual")], value="—")
        for c in [c_act_mse, c_act_r2]:
            c.font = cell_font(color=XL_FG_GREY, size=8)
            c.fill = fill("111827")
            c.alignment = center()

        for mname in ALL_MODELS:
            v        = res_t[mname]
            is_best  = abs(v["MSE"] - best_mse_t) < 1e-9
            bg2      = XL_BG_BEST if is_best else "111827"
            r2_color = XL_FG_GREEN if v["OOS_R2"] >= 0 else XL_FG_RED

            c_mse = ws_all.cell(row=5, column=col_map[(tgt_col, mname)],
                                 value=round(v["MSE"], 4))
            c_r2  = ws_all.cell(row=6, column=col_map[(tgt_col, mname)],
                                 value=round(v["OOS_R2"], 4))
            c_mse.font = cell_font(bold=is_best,
                                   color=XL_FG_GOLD if is_best else XL_FG_WHITE,
                                   size=8)
            c_r2.font  = cell_font(bold=is_best, color=r2_color, size=8)
            for c in [c_mse, c_r2]:
                c.fill          = fill(bg2)
                c.alignment     = center()
                c.number_format = "0.0000"

    ws_all.row_dimensions[5].height = 14
    ws_all.row_dimensions[6].height = 14

    # ── Data rows (rows 7 → 7+n_test-1) ──────────────────────────────────────
    for q_idx, qdate in enumerate(test_qtrs):
        row  = 7 + q_idx
        bg_r = XL_BG_OV_ROW if q_idx % 2 == 0 else XL_BG_ALT_ROW

        # Quarter label
        cq = ws_all.cell(row=row, column=1, value=qdate)
        cq.font      = cell_font(bold=True, color=XL_FG_BLUE)
        cq.fill      = fill(bg_r)
        cq.alignment = center()
        cq.border    = thin_border()

        for tgt_col in TARGETS:
            res_t      = all_results[tgt_col]
            best_mse_t = min(v["MSE"] for v in res_t.values())
            actual     = float(splits[tgt_col]["y_test"].values[q_idx])

            # Actual value
            ca = ws_all.cell(row=row,
                             column=col_map[(tgt_col, "Actual")],
                             value=round(actual, 4))
            ca.font          = cell_font(bold=True, color=XL_FG_WHITE)
            ca.fill          = fill(bg_r)
            ca.alignment     = center()
            ca.border        = thin_border()
            ca.number_format = "0.0000"

            # Each model's prediction
            for mname in ALL_MODELS:
                v       = res_t[mname]
                pred    = float(v["preds"][q_idx])
                err     = pred - actual
                is_best = abs(v["MSE"] - best_mse_t) < 1e-9
                this_bg = XL_BG_BEST if is_best else bg_r
                close   = abs(actual) > 1e-9 and abs(err) / abs(actual) < 0.05

                cp = ws_all.cell(row=row,
                                 column=col_map[(tgt_col, mname)],
                                 value=round(pred, 4))
                cp.font = cell_font(
                    bold=is_best,
                    color=(XL_FG_GOLD  if is_best
                           else XL_FG_GREEN if close
                           else XL_FG_WHITE)
                )
                cp.fill          = fill(this_bg)
                cp.alignment     = center()
                cp.border        = thin_border()
                cp.number_format = "0.0000"

        ws_all.row_dimensions[row].height = 16

    # ── Legend row ────────────────────────────────────────────────────────────
    legend_row = 7 + n_test + 1
    ws_all.merge_cells(start_row=legend_row, start_column=1,
                       end_row=legend_row,   end_column=TOTAL_DATA_COLS)
    lg = ws_all.cell(row=legend_row, column=1,
                     value="★ Gold / dark-green = best-MSE model per target  │  "
                           "Green text = prediction within 5 % of actual  │  "
                           "OOS-R² red = negative (worse than naïve mean)")
    lg.font      = cell_font(color=XL_FG_GREY, size=8)
    lg.fill      = fill("0F172A")
    lg.alignment = center()

    # ─── save ─────────────────────────────────────────────────────────────────
    wb.save(XL_OUT)
    print(f"\n  Excel workbook saved → {XL_OUT}")
    print("  Sheets: Summary | Nifty 50 | GDP Growth | CPI Inflation | All Predictions")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CONSOLIDATED FINAL SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("FINAL SUMMARY — India Macro Forecasting v3  (Quarterly, 3 Targets)")
print("=" * 76)
print(f"\n  Frequency  : Quarterly  (Nifty 50 from yfinance; GDP/CPI interpolated)")
print(f"  Train      : 2008 Q3 – 2022 Q4  (2020 obs dropped as training targets)")
print(f"  Test       : 2023 Q1 – 2025 Q1")
print(f"  Features   : Own lags 1-3 + cross lags 1-2 + rolling mean/std (3-period)")
print(f"  CV         : TimeSeriesSplit (5 folds, training data only)")
print(f"  OOS-R²     : Campbell & Thompson (2008) — benchmark = train set mean")

for tgt_col, tgt_meta in TARGETS.items():
    results = all_results[tgt_col]
    label   = tgt_meta["label"]
    print(f"\n  ┌─ {label} {'─' * max(1, 55 - len(label))}")
    print(f"  │  {'Model':<24}  {'OOS MSE':>12}  {'OOS-R²':>10}")
    print(f"  │  {'─'*50}")
    best_mse = min(v["MSE"] for v in results.values())
    for name, v in results.items():
        tag = "  ◀ BEST" if abs(v["MSE"] - best_mse) < 1e-9 else ""
        print(f"  │  {name:<24}  {v['MSE']:>12.4f}  {v['OOS_R2']:>+10.4f}{tag}")
    print(f"  └{'─' * 55}")

print("\n  Key design choices:")
print("  • Ridge (L2) chosen over LASSO — handles highly collinear macro features.")
print("  • Extreme α (10–10 000) forces maximum coefficient shrinkage.")
print("  • max_depth ≤ 3 for RF/DT; lr ≤ 0.05 for GB → prevents memorising noise.")
print("  • COVID-2020 kept as LAG FEATURE for 2021 rows but not as training target.")
print(f"\n  Output files:")
print(f"    Combined PNG  → {out_combined}")
for tgt_col, png in per_target_pngs.items():
    print(f"    Per-target    → {png}")
if _OPENPYXL:
    print(f"    Excel sheet   → {XL_OUT}")
print("\n" + "=" * 76)
print("Pipeline complete.")
print("=" * 76)

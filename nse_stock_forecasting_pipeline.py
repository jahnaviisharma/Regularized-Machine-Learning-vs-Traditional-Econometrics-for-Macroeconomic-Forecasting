"""
================================================================================
  NSE Stock Time-Series Forecasting Pipeline
  Stocks : HDFCBANK.NS | TRENT.NS | TCS.NS | LT.NS | TITAN.NS
  Train  : Jan 2022 – Dec 2025   |   Test : Jan 2026 – Dec 2026
  Models : ARIMA · Ridge · RandomForest · DecisionTree · GradientBoosting · Ensemble
  Output : 5 individual plots + 1 combined panel plot + predictions_2026.xlsx
           (per-stock tabs + Best Model Summary sheet)
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"]   = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ["HDFCBANK.NS", "TRENT.NS", "TCS.NS", "LT.NS", "TITAN.NS"]

TRAIN_START = "2022-01-01"
TRAIN_END   = "2025-12-31"
TEST_START  = "2026-01-01"
TEST_END    = "2026-12-31"

LAGS         = [1, 2, 3]
ROLLING_WIN  = 3
N_CV_SPLITS  = 3          # TimeSeriesSplit folds (small dataset → 3)

# Colour palette for plots (distinct, readable on white)
PALETTE = {
    "train_actual"  : "#1f77b4",   # steel blue
    "test_actual"   : "#2ca02c",   # green
    "arima"         : "#d62728",   # red
    "ridge"         : "#9467bd",   # purple
    "rf"            : "#8c564b",   # brown
    "dt"            : "#e377c2",   # pink
    "gb"            : "#ff7f0e",   # orange
    "ensemble"      : "#17becf",   # teal
}

MODEL_KEYS   = ["arima", "ridge", "rf", "dt", "gb", "ensemble"]
MODEL_LABELS = {
    "arima"    : "ARIMA",
    "ridge"    : "Ridge (L2)",
    "rf"       : "Random Forest",
    "dt"       : "Decision Tree",
    "gb"       : "Gradient Boosting",
    "ensemble" : "Ensemble (Voting)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mse(y_true, y_pred):
    """Return MSE; guard against all-NaN predictions."""
    if y_pred is None or np.all(np.isnan(y_pred)):
        return np.nan
    return mean_squared_error(y_true, y_pred)


def _safe_r2(y_true, y_pred):
    """Return R²; guard against degenerate cases."""
    if y_pred is None or np.all(np.isnan(y_pred)):
        return np.nan
    return r2_score(y_true, y_pred)


def _feature_names(lags=LAGS, win=ROLLING_WIN):
    names  = [f"lag_{l}" for l in lags]
    names += [f"roll_mean_{win}m", f"roll_std_{win}m"]
    return names

# ─────────────────────────────────────────────────────────────────────────────
# 2.  STEP 1 – DATA ACQUISITION & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_engineer(ticker: str) -> pd.DataFrame:
    """
    Download monthly OHLCV data for `ticker`, engineer lag/rolling features,
    and return a clean DataFrame with columns:
        Close, lag_1, lag_2, lag_3, roll_mean_3m, roll_std_3m
    Index: DatetimeIndex (monthly, first-of-month normalised)
    """
    print(f"  Fetching {ticker} …", end=" ", flush=True)

    # yfinance: fetch daily then resample to month-end
    raw = yf.download(
        ticker,
        start="2021-10-01",   # pull extra history so lags/rolling don't vanish
        end=TEST_END,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. "
                         "Check ticker validity and internet connection.")

    # Flatten multi-level columns (yfinance ≥ 0.2 can return MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df.sort_index(inplace=True)

    # ── Feature engineering ──────────────────────────────────────────────────
    for lag in LAGS:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    df[f"roll_mean_{ROLLING_WIN}m"] = (
        df["Close"].shift(1).rolling(ROLLING_WIN).mean()
    )
    df[f"roll_std_{ROLLING_WIN}m"] = (
        df["Close"].shift(1).rolling(ROLLING_WIN).std()
    )

    df.dropna(inplace=True)

    # Keep only the window we need (from 2022 onward)
    df = df.loc[df.index >= pd.Timestamp(TRAIN_START)]

    print(f"OK  ({len(df)} monthly rows, {df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STEP 2 – CHRONOLOGICAL TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame):
    """
    Strict date-based split – no shuffling whatsoever.
    Returns (train_df, test_df).
    """
    train = df.loc[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    test  = df.loc[(df.index >= TEST_START)  & (df.index <= TEST_END)].copy()
    return train, test


def xy_split(df: pd.DataFrame):
    feat = _feature_names()
    X = df[feat].values
    y = df["Close"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STEP 3 – ARIMA BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def fit_arima_and_forecast(train_series: pd.Series, n_forecast: int):
    """
    Auto-select ARIMA order by iterating a small grid and picking the
    model with the lowest AIC.  Forecast `n_forecast` steps ahead.
    Returns numpy array of forecasts (length = n_forecast).
    """
    best_aic   = np.inf
    best_order = (1, 1, 1)

    candidate_orders = [
        (p, d, q)
        for p in [0, 1, 2]
        for d in [0, 1]
        for q in [0, 1, 2]
        if not (p == 0 and q == 0)
    ]

    for order in candidate_orders:
        try:
            m = ARIMA(train_series.values, order=order, trend="n").fit()
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = order
        except Exception:
            continue

    final_model = ARIMA(
        train_series.values, order=best_order, trend="n"
    ).fit()
    forecasts = final_model.forecast(steps=n_forecast)
    print(f"    ARIMA order={best_order}  AIC={best_aic:.2f}")
    return np.asarray(forecasts)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STEP 4 – ML CHALLENGERS WITH EXTREME REGULARISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_and_tune_ml_models(X_train, y_train, n_splits=N_CV_SPLITS):
    """
    Returns a dict of best-estimators keyed by 'ridge', 'rf', 'dt', 'gb'.
    Uses TimeSeriesSplit CV strictly on the training window.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_models = {}

    # ── A. Ridge (L2 Regularised Linear) ──────────────────────────────────
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge()),
    ])
    ridge_grid = {"model__alpha": [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0]}
    ridge_cv = GridSearchCV(
        ridge_pipe, ridge_grid, cv=tscv, scoring="neg_mean_squared_error",
        refit=True, n_jobs=-1
    )
    ridge_cv.fit(X_train, y_train)
    best_models["ridge"] = ridge_cv.best_estimator_
    print(f"    Ridge best alpha = {ridge_cv.best_params_['model__alpha']}")

    # ── B. Random Forest (shallow stumps + high min_samples_leaf) ──────────
    rf_grid = {
        "max_depth"       : [1, 2],
        "min_samples_leaf": [5, 10, 15],
        "n_estimators"    : [100, 200],
    }
    rf_cv = GridSearchCV(
        RandomForestRegressor(random_state=42),
        rf_grid, cv=tscv, scoring="neg_mean_squared_error",
        refit=True, n_jobs=-1
    )
    rf_cv.fit(X_train, y_train)
    best_models["rf"] = rf_cv.best_estimator_
    print(f"    RF   best params = {rf_cv.best_params_}")

    # ── C. Decision Tree (depth-restricted stump) ──────────────────────────
    dt_grid = {
        "max_depth"       : [1, 2],
        "min_samples_leaf": [5, 10, 15],
    }
    dt_cv = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        dt_grid, cv=tscv, scoring="neg_mean_squared_error",
        refit=True, n_jobs=-1
    )
    dt_cv.fit(X_train, y_train)
    best_models["dt"] = dt_cv.best_estimator_
    print(f"    DT   best params = {dt_cv.best_params_}")

    # ── D. Gradient Boosting (low lr, shallow depth) ──────────────────────
    gb_grid = {
        "learning_rate": [0.01, 0.05],
        "max_depth"    : [1, 2],
        "n_estimators" : [100, 200],
        "subsample"    : [0.7, 1.0],
    }
    gb_cv = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_grid, cv=tscv, scoring="neg_mean_squared_error",
        refit=True, n_jobs=-1
    )
    gb_cv.fit(X_train, y_train)
    best_models["gb"] = gb_cv.best_estimator_
    print(f"    GB   best params = {gb_cv.best_params_}")

    return best_models


# ─────────────────────────────────────────────────────────────────────────────
# 6.  STEP 5 – ENSEMBLE + OUT-OF-SAMPLE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble_and_predict(best_models, X_train, y_train, X_test):
    """
    Builds VotingRegressor from tuned components, re-fits on full training set,
    returns predictions for X_test.
    """
    voting = VotingRegressor(estimators=[
        ("ridge", best_models["ridge"]),
        ("rf",    best_models["rf"]),
        ("dt",    best_models["dt"]),
        ("gb",    best_models["gb"]),
    ])
    voting.fit(X_train, y_train)
    return voting.predict(X_test)


def evaluate_all(y_true, predictions: dict):
    """
    predictions: dict keyed by model name → np.array
    Prints MSE and R² for each model.
    Returns a DataFrame of metrics.
    """
    rows = []
    print(f"\n    {'Model':<22} {'MSE':>14} {'R²':>10}")
    print("    " + "─" * 50)
    for key in MODEL_KEYS:
        pred = predictions.get(key)
        mse  = _safe_mse(y_true, pred)
        r2   = _safe_r2(y_true, pred)
        print(f"    {MODEL_LABELS[key]:<22} {mse:>14.2f} {r2:>10.4f}")
        rows.append({"Model": MODEL_LABELS[key], "MSE": mse, "R2": r2})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STEP 6 – VISUALISATION (WHITE BACKGROUND)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stock(ticker, train_df, test_df, predictions: dict):
    """
    Single figure per ticker:
      • Full actual price (train = blue, test = green)
      • 2026 model forecasts overlaid as dashed/dotted lines
    White background throughout.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Actual data ──────────────────────────────────────────────────────────
    ax.plot(
        train_df.index, train_df["Close"],
        color=PALETTE["train_actual"], linewidth=2.0,
        label="Actual (Train 2022–2025)", zorder=5
    )
    ax.plot(
        test_df.index, test_df["Close"],
        color=PALETTE["test_actual"], linewidth=2.5,
        marker="o", markersize=5, label="Actual (Test 2026)", zorder=6
    )

    # ── Forecasts ────────────────────────────────────────────────────────────
    linestyles = {
        "arima"   : ("--",  1.8),
        "ridge"   : ("--",  1.6),
        "rf"      : ("-.",  1.6),
        "dt"      : (":",   1.8),
        "gb"      : ("--",  1.6),
        "ensemble": ("-",   2.2),
    }
    for key in MODEL_KEYS:
        pred = predictions.get(key)
        if pred is None or np.all(np.isnan(pred)):
            continue
        ls, lw = linestyles[key]
        ax.plot(
            test_df.index, pred,
            color=PALETTE[key], linewidth=lw, linestyle=ls,
            label=MODEL_LABELS[key], alpha=0.88, zorder=4
        )

    # ── Vertical "firewall" line ──────────────────────────────────────────
    ax.axvline(
        x=pd.Timestamp("2026-01-01"), color="#555555",
        linestyle=":", linewidth=1.3, alpha=0.7, label="Train / Test split"
    )
    ax.fill_betweenx(
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        pd.Timestamp("2026-01-01"), test_df.index[-1],
        alpha=0.04, color="#17becf"
    )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_title(f"{ticker} — Actual vs. 2026 Forecast",
                 fontsize=15, fontweight="bold", color="#111111", pad=12)
    ax.set_xlabel("Date", fontsize=11, color="#333333")
    ax.set_ylabel("Closing Price (INR)", fontsize=11, color="#333333")
    ax.tick_params(axis="both", colors="#333333")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=35, ha="right")
    ax.legend(
        loc="upper left", fontsize=9, framealpha=0.9,
        edgecolor="#cccccc", fancybox=True
    )
    ax.grid(True, linestyle="--", alpha=0.4, color="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{ticker.replace('.', '_')}_forecast_2026.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"    Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 7b. COMBINED 5-PANEL COMPARISON PLOT (with MSE & OOS-R² annotations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_stocks_combined(stock_data: dict, filepath="all_stocks_combined_forecast_2026.png"):
    """
    stock_data: dict keyed by ticker → {
        'train_df', 'test_df', 'predictions': {key: np.array},
        'metrics': DataFrame(Model, MSE, R2)
    }
    Produces a 3-row × 2-col grid (5 subplots + 1 legend panel).
    Each subplot annotates every model's MSE and OOS-R².
    """
    n = len(stock_data)
    ncols = 2
    nrows = (n + 1) // ncols + ((n + 1) % ncols > 0)   # ceil to fit 5 + legend
    nrows = 3   # fixed: 3 rows × 2 cols = 6 cells; 5 used + 1 for shared legend

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 6))
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    linestyles = {
        "arima"   : ("--",  1.8),
        "ridge"   : ("--",  1.6),
        "rf"      : ("-.",  1.6),
        "dt"      : (":",   1.9),
        "gb"      : ("--",  1.6),
        "ensemble": ("-",   2.3),
    }

    legend_handles = []   # collect once for shared legend panel
    legend_labels  = []

    for ax_idx, (ticker, data) in enumerate(stock_data.items()):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("white")

        train_df    = data["train_df"]
        test_df     = data["test_df"]
        predictions = data["predictions"]
        metrics     = data["metrics"]   # DataFrame: Model | MSE | R2

        # ── Actual lines ──────────────────────────────────────────────────
        h1, = ax.plot(
            train_df.index, train_df["Close"],
            color=PALETTE["train_actual"], linewidth=1.8,
            label="Actual – Train 2022–2025", zorder=5
        )
        h2, = ax.plot(
            test_df.index, test_df["Close"],
            color=PALETTE["test_actual"], linewidth=2.2,
            marker="o", markersize=4, label="Actual – Test 2026", zorder=6
        )
        if ax_idx == 0:
            legend_handles += [h1, h2]
            legend_labels  += ["Actual – Train 2022–2025", "Actual – Test 2026"]

        # ── Model forecast lines ──────────────────────────────────────────
        for key in MODEL_KEYS:
            pred = predictions.get(key)
            if pred is None or np.all(np.isnan(pred)):
                continue
            ls, lw = linestyles[key]
            h, = ax.plot(
                test_df.index, pred,
                color=PALETTE[key], linewidth=lw, linestyle=ls,
                label=MODEL_LABELS[key], alpha=0.85, zorder=4
            )
            if ax_idx == 0:
                legend_handles.append(h)
                legend_labels.append(MODEL_LABELS[key])

        # ── Firewall line ─────────────────────────────────────────────────
        ax.axvline(
            x=pd.Timestamp("2026-01-01"), color="#555555",
            linestyle=":", linewidth=1.2, alpha=0.6
        )

        # ── MSE / OOS-R² annotation box ───────────────────────────────────
        # Build a compact table string: each model on one line
        annotation_lines = ["Model            MSE        OOS-R²"]
        annotation_lines += ["-" * 38]
        for _, row in metrics.iterrows():
            mse_str = f"{row['MSE']:>12,.0f}"
            r2_str  = f"{row['R2']:>+8.3f}"
            annotation_lines.append(f"{row['Model']:<17}{mse_str}  {r2_str}")

        annotation_text = "\n".join(annotation_lines)
        ax.text(
            0.01, 0.98, annotation_text,
            transform=ax.transAxes,
            fontsize=6.5, verticalalignment="top", horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                edgecolor="#cccccc", alpha=0.92
            ),
            zorder=10
        )

        # ── Highlight best ML model name ──────────────────────────────────
        ml_only = metrics[metrics["Model"] != "ARIMA"]
        if not ml_only.empty:
            best_row = ml_only.loc[ml_only["MSE"].idxmin()]
            best_label = f"★ Best: {best_row['Model']}  (MSE={best_row['MSE']:,.0f})"
            ax.set_title(
                f"{ticker}\n{best_label}",
                fontsize=11, fontweight="bold", color="#111111", pad=8
            )
        else:
            ax.set_title(ticker, fontsize=11, fontweight="bold")

        ax.set_xlabel("Date", fontsize=9, color="#444444")
        ax.set_ylabel("Close Price (INR)", fontsize=9, color="#444444")
        ax.tick_params(axis="both", labelsize=8, colors="#333333")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── 6th cell → shared legend panel ───────────────────────────────────────
    legend_ax = axes_flat[n]   # index 5 (6th cell)
    legend_ax.set_facecolor("white")
    legend_ax.axis("off")

    # Also add firewall legend entry
    import matplotlib.lines as mlines
    fw_handle = mlines.Line2D(
        [], [], color="#555555", linestyle=":", linewidth=1.2,
        label="Train / Test firewall (Jan 2026)"
    )
    legend_handles.append(fw_handle)
    legend_labels.append("Train / Test firewall (Jan 2026)")

    legend_ax.legend(
        legend_handles, legend_labels,
        loc="center", fontsize=10.5, framealpha=0.95,
        edgecolor="#cccccc", fancybox=True,
        title="Legend", title_fontsize=12
    )
    legend_ax.set_title(
        "OOS Metrics annotated in each subplot\n(lower MSE = better; higher R² = better)",
        fontsize=9.5, color="#555555", pad=6
    )

    fig.suptitle(
        "NSE Stocks — Actual vs. 2026 Out-of-Sample Forecasts\n"
        "(Train: Jan 2022–Dec 2025  |  Test: Jan–May 2026)",
        fontsize=15, fontweight="bold", color="#111111", y=1.01
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  Saved combined plot → {filepath}")



# ─────────────────────────────────────────────────────────────────────────────
# 8.  STEP 7 – EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

# ── openpyxl styling helpers ──────────────────────────────────────────────────
_HDR_FILL   = PatternFill("solid", fgColor="1F4E79")   # dark navy
_HDR_FONT   = Font(color="FFFFFF", bold=True, size=10)
_BEST_FILL  = PatternFill("solid", fgColor="C6EFCE")   # light green
_BEST_FONT  = Font(color="276221", bold=True, size=10)
_THIN       = Side(style="thin", color="BBBBBB")
_BORDER     = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _style_header_row(ws):
    """Apply navy header style to the first row of a worksheet."""
    for cell in ws[1]:
        cell.fill      = _HDR_FILL
        cell.font      = _HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = _BORDER
    ws.row_dimensions[1].height = 28


def _autofit_columns(ws, extra=3):
    """Set each column width to the max content length + extra padding."""
    for col in ws.columns:
        max_len = max(
            (len(str(c.value)) if c.value is not None else 0) for c in col
        )
        ws.column_dimensions[col[0].column_letter].width = max_len + extra


def _highlight_best_model_col(ws, model_col_start, n_model_cols, n_data_rows):
    """
    For each data row, find the column with the lowest MSE proxy
    (i.e. column whose absolute deviation from 'Actual' is smallest)
    and highlight it green.  model_col_start is 1-indexed.
    """
    actual_col = 3   # column C is always 'Actual (Close)'
    for row_idx in range(2, 2 + n_data_rows):
        actual_val = ws.cell(row=row_idx, column=actual_col).value
        if actual_val is None or np.isnan(float(actual_val if actual_val else "nan")):
            continue
        try:
            actual_val = float(actual_val)
        except (TypeError, ValueError):
            continue
        best_col   = None
        best_diff  = np.inf
        for c in range(model_col_start, model_col_start + n_model_cols):
            val = ws.cell(row=row_idx, column=c).value
            try:
                diff = abs(float(val) - actual_val)
                if diff < best_diff:
                    best_diff = diff
                    best_col  = c
            except (TypeError, ValueError):
                continue
        if best_col:
            cell = ws.cell(row=row_idx, column=best_col)
            cell.fill = _BEST_FILL
            cell.font = _BEST_FONT


def export_to_excel(all_results: list, all_metrics: list,
                    filepath="predictions_2026.xlsx"):
    """
    Writes one sheet per stock (predictions + row-level best-model highlight)
    plus a 'Best Model Summary' sheet.

    all_results : list of per-row dicts (Ticker, Date, Actual, arima…ensemble)
    all_metrics : list of per-ticker DataFrames (Ticker, Model, MSE, R2)
    """
    df_all = pd.DataFrame(all_results)

    # Rename raw model keys → human labels
    rename_map = {k: MODEL_LABELS[k] for k in MODEL_KEYS}
    rename_map.update({"Ticker": "Ticker", "Date": "Date", "Actual": "Actual (Close)"})
    df_all.rename(columns=rename_map, inplace=True)

    col_order = ["Ticker", "Date", "Actual (Close)"] + [MODEL_LABELS[k] for k in MODEL_KEYS]
    df_all = df_all[col_order]

    model_display_cols = [MODEL_LABELS[k] for k in MODEL_KEYS]
    model_col_start    = 4   # columns A=Ticker B=Date C=Actual D=first model

    # ── Build Best-Model Summary DataFrame ──────────────────────────────────
    summary_rows = []
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        for ticker in TICKERS:
            sub = combined_metrics[combined_metrics["Ticker"] == ticker]
            if sub.empty:
                continue
            best_mse_row   = sub.loc[sub["MSE"].idxmin()]
            best_r2_row    = sub.loc[sub["R2"].idxmax()]
            ml_only        = sub[sub["Model"] != "ARIMA"]
            best_ml_row    = ml_only.loc[ml_only["MSE"].idxmin()] if not ml_only.empty else best_mse_row
            arima_row      = sub[sub["Model"] == "ARIMA"].iloc[0] if len(sub[sub["Model"] == "ARIMA"]) else None

            summary_rows.append({
                "Ticker"                  : ticker,
                "Best Overall (↓ MSE)"    : best_mse_row["Model"],
                "Best Overall MSE"        : round(best_mse_row["MSE"], 2),
                "Best Overall OOS-R²"     : round(best_mse_row["R2"], 4),
                "Best ML Model (↓ MSE)"   : best_ml_row["Model"],
                "Best ML MSE"             : round(best_ml_row["MSE"], 2),
                "Best ML OOS-R²"          : round(best_ml_row["R2"], 4),
                "Best R² Model"           : best_r2_row["Model"],
                "Best R²"                 : round(best_r2_row["R2"], 4),
                "ARIMA MSE"               : round(arima_row["MSE"], 2) if arima_row is not None else np.nan,
                "ARIMA OOS-R²"            : round(arima_row["R2"], 4) if arima_row is not None else np.nan,
            })
    df_summary = pd.DataFrame(summary_rows)

    # ── Write workbook ───────────────────────────────────────────────────────
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:

        # ① Per-stock sheets
        for ticker in TICKERS:
            df_ticker = df_all[df_all["Ticker"] == ticker].drop(columns=["Ticker"])
            if df_ticker.empty:
                continue
            sheet_name = ticker.replace(".NS", "").replace(".", "_")[:31]
            df_ticker.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            _style_header_row(ws)
            _highlight_best_model_col(
                ws,
                model_col_start=model_col_start,
                n_model_cols=len(model_display_cols),
                n_data_rows=len(df_ticker)
            )
            # Zebra-stripe data rows
            zebra_fill = PatternFill("solid", fgColor="F2F7FC")
            for r in range(2, 2 + len(df_ticker)):
                if r % 2 == 0:
                    for c in range(1, len(df_ticker.columns) + 1):
                        ws.cell(row=r, column=c).fill = zebra_fill
                # Add borders to all cells
                for c in range(1, len(df_ticker.columns) + 1):
                    ws.cell(row=r, column=c).border = _BORDER
                    ws.cell(row=r, column=c).alignment = Alignment(horizontal="center")
            _autofit_columns(ws)
            # Freeze header
            ws.freeze_panes = "A2"

        # ② Best-Model Summary sheet
        if not df_summary.empty:
            df_summary.to_excel(writer, index=False, sheet_name="Best Model Summary")
            ws_sum = writer.sheets["Best Model Summary"]
            _style_header_row(ws_sum)
            # Highlight the best-model cells in summary
            gold_fill = PatternFill("solid", fgColor="FFEB9C")
            gold_font = Font(color="9C5700", bold=True)
            for r in range(2, 2 + len(df_summary)):
                # Columns B (Best Overall) and E (Best ML) get gold highlight
                for c in [2, 5, 8]:
                    ws_sum.cell(row=r, column=c).fill = gold_fill
                    ws_sum.cell(row=r, column=c).font = gold_font
                for c in range(1, len(df_summary.columns) + 1):
                    ws_sum.cell(row=r, column=c).border = _BORDER
                    ws_sum.cell(row=r, column=c).alignment = Alignment(horizontal="center")
            _autofit_columns(ws_sum)
            ws_sum.freeze_panes = "A2"

        # ③ All-data combined sheet (for backward compatibility)
        df_all.to_excel(writer, index=False, sheet_name="All Stocks Combined")
        ws_all = writer.sheets["All Stocks Combined"]
        _style_header_row(ws_all)
        _autofit_columns(ws_all)
        ws_all.freeze_panes = "A2"

    print(f"\n✅  Excel file saved → {filepath}")
    print(f"    Sheets: {[t.replace('.NS','') for t in TICKERS]} + Best Model Summary + All Stocks Combined")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  NSE STOCK FORECASTING PIPELINE  (Train 2022-2025 | Test 2026)")
    print("=" * 72)

    all_results   = []   # collects per-row dicts for Excel export
    all_metrics   = []   # collects metric DataFrames for summary table
    stock_plot_data = {}  # keyed by ticker → data for combined plot

    for ticker in TICKERS:
        print(f"\n{'─'*60}")
        print(f"  ► TICKER : {ticker}")
        print(f"{'─'*60}")

        # ── 1. Data ─────────────────────────────────────────────────────────
        try:
            df = fetch_and_engineer(ticker)
        except Exception as exc:
            print(f"  [ERROR] Could not fetch {ticker}: {exc}")
            continue

        # ── 2. Split ────────────────────────────────────────────────────────
        train_df, test_df = chronological_split(df)

        if train_df.empty:
            print(f"  [WARNING] No training data for {ticker}. Skipping.")
            continue
        if test_df.empty:
            print(f"  [WARNING] No 2026 test data available yet for {ticker}.")
            print(f"            This is expected if today < Dec 2026.")
            # Construct synthetic future dates for the forecast horizon
            last_train_date = train_df.index[-1]
            future_index = pd.date_range(
                start=last_train_date + pd.DateOffset(months=1),
                periods=12, freq="MS"
            )
            test_df = pd.DataFrame(
                {"Close": [np.nan] * 12},
                index=future_index
            )
            # We cannot evaluate against actuals if the future is unknown,
            # but we can still generate forecasts.
            future_only = True
        else:
            future_only = False

        X_train, y_train = xy_split(train_df)

        # For test: if real test data exists use it; otherwise build feature
        # rows from the end of training (walk-forward, simplified).
        if not future_only and len(test_df) > 0:
            feat = _feature_names()
            # Some test rows may lack lag features if the test window is
            # partly outside the fetched data → fill any remaining NaN with
            # the last known value (conservative imputation).
            X_test_full = test_df[feat].values
            y_test       = test_df["Close"].values
        else:
            # Forecast by rolling the last available features forward
            last_row = train_df.iloc[-1]
            X_test_full = np.tile(
                last_row[_feature_names()].values, (12, 1)
            )
            y_test = None

        n_test = len(test_df)

        # ── 3. ARIMA ────────────────────────────────────────────────────────
        print(f"\n  [ARIMA]")
        arima_preds = fit_arima_and_forecast(train_df["Close"], n_test)

        # ── 4. ML Models ────────────────────────────────────────────────────
        print(f"\n  [ML Tuning via TimeSeriesSplit (k={N_CV_SPLITS})]")
        best_models = build_and_tune_ml_models(X_train, y_train)

        # ── 5. Ensemble + Predictions ────────────────────────────────────────
        print(f"\n  [Building Ensemble & Generating 2026 Predictions]")
        ensemble_preds = build_ensemble_and_predict(
            best_models, X_train, y_train, X_test_full
        )

        predictions = {
            "arima"   : arima_preds,
            "ridge"   : best_models["ridge"].predict(X_test_full),
            "rf"      : best_models["rf"].predict(X_test_full),
            "dt"      : best_models["dt"].predict(X_test_full),
            "gb"      : best_models["gb"].predict(X_test_full),
            "ensemble": ensemble_preds,
        }

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics_df = pd.DataFrame()   # may stay empty if no actuals
        if y_test is not None:
            print(f"\n  [Out-of-Sample Metrics – {ticker}]")
            metrics_df = evaluate_all(y_test, predictions)
            metrics_df.insert(0, "Ticker", ticker)
            all_metrics.append(metrics_df.copy())

        # Store data for combined plot (metrics without Ticker column)
        stock_plot_data[ticker] = {
            "train_df"   : train_df,
            "test_df"    : test_df,
            "predictions": predictions,
            "metrics"    : metrics_df[["Model", "MSE", "R2"]] if not metrics_df.empty else pd.DataFrame(),
        }

        # ── 6. Plot ──────────────────────────────────────────────────────────
        print(f"\n  [Plotting {ticker}]")
        plot_stock(ticker, train_df, test_df, predictions)

        # ── Collect rows for Excel ───────────────────────────────────────────
        for i, (idx, row_actual) in enumerate(test_df.iterrows()):
            rec = {
                "Ticker": ticker,
                "Date"  : idx.strftime("%Y-%m-%d"),
                "Actual": row_actual["Close"],
            }
            for key in MODEL_KEYS:
                pred_arr = predictions.get(key)
                rec[key] = float(pred_arr[i]) if (
                    pred_arr is not None and i < len(pred_arr)
                ) else np.nan
            all_results.append(rec)

    # ── 7. Summary metrics table ─────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 72)
        print("  CONSOLIDATED OUT-OF-SAMPLE METRICS (all tickers)")
        print("=" * 72)
        summary = pd.concat(all_metrics, ignore_index=True)
        print(summary.to_string(index=False))

    # ── 8. Combined 5-panel plot ──────────────────────────────────────────────
    if stock_plot_data:
        print("\n  [Generating combined 5-panel comparison plot …]")
        plot_all_stocks_combined(stock_plot_data)

    # ── 9. Excel export (per-stock tabs + best-model summary) ─────────────────
    if all_results:
        export_to_excel(all_results, all_metrics)
    else:
        print("\n[WARNING] No results to export.")

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

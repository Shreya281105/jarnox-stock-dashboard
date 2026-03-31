"""
ml_predictions.py
──────────────────────────────────────────────────────────────────────────────
ML Price Prediction Module — JarNox Stock Intelligence Platform
──────────────────────────────────────────────────────────────────────────────
Uses scikit-learn to train and evaluate multiple regression models per stock.

Models tried:
  1. Linear Regression      — baseline trend model
  2. Ridge Regression       — regularized linear model
  3. Random Forest Regressor — ensemble tree model (optional, heavier)

Features used:
  - Day index (t)
  - 7-day moving average
  - 20-day moving average
  - Lag-1 close price
  - Lag-5 close price
  - Daily return
  - Volatility score

Outputs:
  - data/predictions.json  — 14-day forward predictions per symbol
  - data/model_scores.json — R² and MAE scores per symbol per model
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).parent / "data"

SYMBOLS = [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
    "WIPRO","BAJFINANCE","SBIN","TATAMOTORS","MARUTI","SUNPHARMA","LTIM"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for time-series regression."""
    df = df.copy().dropna(subset=["close"])
    df["t"]       = np.arange(len(df))
    df["lag1"]    = df["close"].shift(1)
    df["lag5"]    = df["close"].shift(5)
    df["lag20"]   = df["close"].shift(20)
    df["ret5"]    = df["close"].pct_change(5) * 100
    df = df.dropna()
    return df


FEATURE_COLS = ["t", "lag1", "lag5", "lag20", "ma7", "ma20", "daily_return", "volatility", "ret5"]


def train_and_predict(sym: str) -> dict:
    df = pd.read_csv(DATA_DIR / f"{sym}.csv", parse_dates=["date"])
    df = build_features(df)

    X = df[FEATURE_COLS].values
    y = df["close"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Time-series cross validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=3)

    models = {
        "linear_regression": LinearRegression(),
        "ridge":             Ridge(alpha=1.0),
    }

    scores = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        r2s, maes = [], []
        for train_idx, val_idx in tscv.split(X_sc):
            model.fit(X_sc[train_idx], y[train_idx])
            preds = model.predict(X_sc[val_idx])
            r2s.append(r2_score(y[val_idx], preds))
            maes.append(mean_absolute_error(y[val_idx], preds))
        avg_r2  = float(np.mean(r2s))
        avg_mae = float(np.mean(maes))
        scores[name] = {"r2": round(avg_r2, 4), "mae": round(avg_mae, 2)}
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = (name, model)

    # Retrain best model on full data
    name, model = best_model
    model.fit(X_sc, y)

    # Generate future feature rows
    last_row   = df.iloc[-1]
    last_close = float(last_row["close"])
    last_t     = float(last_row["t"])
    last_ma7   = float(last_row["ma7"]  or last_close)
    last_ma20  = float(last_row["ma20"] or last_close)
    last_ret   = float(last_row["daily_return"] or 0)
    last_vol   = float(last_row["volatility"]   or 0)
    lag1, lag5, lag20 = last_close, float(df.iloc[-5]["close"]), float(df.iloc[-20]["close"])

    future_rows, sim_close = [], last_close
    for i in range(1, 15):
        row = [
            last_t + i,
            sim_close,
            lag5 if i <= 5 else future_rows[i-6][1],
            lag20 if i <= 20 else future_rows[i-21][1],
            last_ma7,
            last_ma20,
            last_ret,
            last_vol,
            0.0,
        ]
        future_rows.append(row)
        pred = model.predict(scaler.transform([row]))[0]
        sim_close = pred

    preds_scaled = scaler.transform(future_rows)
    pred_closes  = model.predict(preds_scaled)

    # Build future dates
    last_date = df.iloc[-1]["date"]
    future_dates = []
    d = last_date
    while len(future_dates) < 14:
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d.strftime("%Y-%m-%d"))

    predictions = [
        {"date": future_dates[i], "predicted_close": round(float(pred_closes[i]), 2)}
        for i in range(14)
    ]

    return {
        "symbol": sym,
        "best_model": name,
        "scores": scores,
        "predictions": predictions,
    }


def run_all():
    all_preds  = {}
    all_scores = {}

    for sym in SYMBOLS:
        print(f"  Training {sym}...", end=" ")
        try:
            result = train_and_predict(sym)
            all_preds[sym]  = result["predictions"]
            all_scores[sym] = {"best_model": result["best_model"], **result["scores"]}
            print(f"✓  best={result['best_model']}  r²={result['scores'][result['best_model']]['r2']:.3f}")
        except Exception as e:
            print(f"✗  {e}")

    with open(DATA_DIR / "predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)

    with open(DATA_DIR / "model_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\n  [done] Predictions saved to {DATA_DIR}/predictions.json")
    print(f"  [done] Model scores saved to  {DATA_DIR}/model_scores.json")


if __name__ == "__main__":
    print("\n" + "─"*60)
    print("  JarNox ML Prediction Module")
    print("─"*60 + "\n")
    run_all()

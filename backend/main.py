"""
JarNox Stock Intelligence Platform — Backend API
Author: Intern Candidate
Stack: FastAPI + Pandas + NumPy + Scikit-learn
Data Source: NSE-style simulated OHLCV data (bhavcopy format)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="JarNox Stock Intelligence API",
    description="""
## 📈 Stock Data Intelligence Platform

A mini financial data platform built for the JarNox internship assignment.

### Features
- Real NSE-style stock data (bhavcopy format)
- Moving averages, volatility scores, daily returns
- Stock comparison & correlation analysis
- ML-based price predictions (Linear Regression)
- Top gainers / losers / sector analysis

### Data Source
NSE Bhavcopy CSVs format — OHLCV data for 12 major NSE stocks.
""",
    version="1.0.0",
    contact={"name": "Internship Candidate", "email": "candidate@example.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─────────────────────────────────────────────────────────────
# Data Loaders (cached for performance)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def load_stock(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found.")
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

@lru_cache(maxsize=1)
def load_meta() -> dict:
    with open(os.path.join(DATA_DIR, "stocks_meta.json")) as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_predictions() -> dict:
    with open(os.path.join(DATA_DIR, "predictions.json")) as f:
        return json.load(f)

def get_all_symbols():
    return list(load_meta().keys())


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def df_to_records(df: pd.DataFrame) -> list:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df.replace({np.nan: None}).to_dict(orient="records")

def compute_correlation(sym1: str, sym2: str) -> float:
    df1 = load_stock(sym1)[["date", "close"]].rename(columns={"close": sym1})
    df2 = load_stock(sym2)[["date", "close"]].rename(columns={"close": sym2})
    merged = df1.merge(df2, on="date")
    return round(merged[sym1].corr(merged[sym2]), 4)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dashboard.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/companies", tags=["Companies"])
def get_companies():
    """Returns all available companies with metadata."""
    meta = load_meta()
    result = []
    for sym, info in meta.items():
        df = load_stock(sym)
        latest = df.iloc[-1]
        prev    = df.iloc[-2]
        change  = round(latest["close"] - prev["close"], 2)
        change_pct = round((change / prev["close"]) * 100, 3)
        result.append({
            "symbol": sym,
            "name": info["name"],
            "sector": info["sector"],
            "close": round(latest["close"], 2),
            "change": change,
            "change_pct": change_pct,
            "volume": int(latest["volume"]),
        })
    return {"count": len(result), "companies": result}


@app.get("/data/{symbol}", tags=["Stock Data"])
def get_stock_data(
    symbol: str,
    days: int = Query(30, ge=7, le=365, description="Number of trading days"),
    include_ma: bool = Query(True, description="Include moving averages"),
    include_volatility: bool = Query(True, description="Include volatility score"),
):
    """Returns OHLCV + calculated metrics for a given symbol."""
    symbol = symbol.upper()
    df = load_stock(symbol)
    df = df.tail(days).copy()
    cols = ["date", "open", "high", "low", "close", "volume", "daily_return"]
    if include_ma:
        cols += ["ma7", "ma20", "ma50"]
    if include_volatility:
        cols += ["volatility"]
    available = [c for c in cols if c in df.columns]
    return {
        "symbol": symbol,
        "name": load_meta()[symbol]["name"],
        "sector": load_meta()[symbol]["sector"],
        "days": days,
        "data": df_to_records(df[available])
    }


@app.get("/summary/{symbol}", tags=["Stock Data"])
def get_summary(symbol: str):
    """Returns 52-week high, low, average close, and other key stats."""
    symbol = symbol.upper()
    df = load_stock(symbol)
    one_year_ago = df["date"].max() - pd.DateOffset(weeks=52)
    df_52 = df[df["date"] >= one_year_ago]
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    return {
        "symbol": symbol,
        "name": load_meta()[symbol]["name"],
        "sector": load_meta()[symbol]["sector"],
        "latest_close": round(latest["close"], 2),
        "change": round(latest["close"] - prev["close"], 2),
        "change_pct": round((latest["close"] - prev["close"]) / prev["close"] * 100, 3),
        "52w_high": round(df_52["high"].max(), 2),
        "52w_low":  round(df_52["low"].min(), 2),
        "52w_avg_close": round(df_52["close"].mean(), 2),
        "avg_volume_30d": int(df.tail(30)["volume"].mean()),
        "avg_daily_return_30d": round(df.tail(30)["daily_return"].mean(), 4),
        "current_volatility": round(latest["volatility"], 4) if not pd.isna(latest["volatility"]) else None,
        "total_trading_days": len(df),
    }


@app.get("/compare", tags=["Analysis"])
def compare_stocks(
    symbol1: str = Query(..., example="INFY"),
    symbol2: str = Query(..., example="TCS"),
    days: int = Query(90, ge=7, le=365),
):
    """Compare two stocks — returns returns, correlation, volatility side-by-side."""
    s1, s2 = symbol1.upper(), symbol2.upper()
    df1 = load_stock(s1).tail(days)[["date", "close", "daily_return", "volume"]].rename(
        columns={"close": "close_1", "daily_return": "return_1", "volume": "volume_1"}
    )
    df2 = load_stock(s2).tail(days)[["date", "close", "daily_return", "volume"]].rename(
        columns={"close": "close_2", "daily_return": "return_2", "volume": "volume_2"}
    )
    merged = df1.merge(df2, on="date")
    correlation = round(merged["close_1"].corr(merged["close_2"]), 4)

    meta = load_meta()
    def stock_stats(sym, col_close, col_ret):
        return {
            "symbol": sym,
            "name": meta[sym]["name"],
            "sector": meta[sym]["sector"],
            "latest_close": round(merged[col_close].iloc[-1], 2),
            "avg_daily_return": round(merged[col_ret].mean(), 4),
            "total_return_pct": round(
                (merged[col_close].iloc[-1] / merged[col_close].iloc[0] - 1) * 100, 3
            ),
            "volatility": round(merged[col_close].pct_change().std() * 100, 4),
        }

    return {
        "period_days": days,
        "correlation": correlation,
        "interpretation": (
            "Strongly correlated" if abs(correlation) > 0.85
            else "Moderately correlated" if abs(correlation) > 0.5
            else "Weakly correlated"
        ),
        "stock1": stock_stats(s1, "close_1", "return_1"),
        "stock2": stock_stats(s2, "close_2", "return_2"),
        "timeline": df_to_records(merged),
    }


@app.get("/gainers", tags=["Market"])
def top_gainers(limit: int = Query(5, ge=1, le=12)):
    """Returns top gaining stocks today."""
    meta = load_meta()
    result = []
    for sym in meta:
        df = load_stock(sym)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        chg = round((latest["close"] - prev["close"]) / prev["close"] * 100, 3)
        result.append({"symbol": sym, "name": meta[sym]["name"], "sector": meta[sym]["sector"],
                        "close": round(latest["close"], 2), "change_pct": chg})
    result.sort(key=lambda x: x["change_pct"], reverse=True)
    return {"gainers": result[:limit]}


@app.get("/losers", tags=["Market"])
def top_losers(limit: int = Query(5, ge=1, le=12)):
    """Returns top losing stocks today."""
    meta = load_meta()
    result = []
    for sym in meta:
        df = load_stock(sym)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        chg = round((latest["close"] - prev["close"]) / prev["close"] * 100, 3)
        result.append({"symbol": sym, "name": meta[sym]["name"], "sector": meta[sym]["sector"],
                        "close": round(latest["close"], 2), "change_pct": chg})
    result.sort(key=lambda x: x["change_pct"])
    return {"losers": result[:limit]}


@app.get("/predict/{symbol}", tags=["ML Predictions"])
def predict_price(symbol: str, days: int = Query(14, ge=3, le=14)):
    """Returns ML-based (Linear Regression) price predictions for next N days."""
    symbol = symbol.upper()
    preds = load_predictions()
    if symbol not in preds:
        raise HTTPException(status_code=404, detail=f"No predictions for '{symbol}'.")
    return {
        "symbol": symbol,
        "name": load_meta()[symbol]["name"],
        "model": "Linear Regression (sklearn)",
        "note": "Indicative only — based on historical price trend",
        "predictions": preds[symbol][:days],
    }


@app.get("/correlation", tags=["Analysis"])
def full_correlation_matrix():
    """Returns a full correlation matrix across all stocks (last 90 days)."""
    symbols = get_all_symbols()
    closes = {}
    for sym in symbols:
        df = load_stock(sym).tail(90)
        closes[sym] = df.set_index("date")["close"]
    combined = pd.DataFrame(closes)
    corr = combined.corr().round(4)
    return {"matrix": corr.to_dict(), "symbols": symbols}


@app.get("/sector-performance", tags=["Analysis"])
def sector_performance():
    """Aggregates 30-day performance grouped by sector."""
    meta = load_meta()
    sector_data = {}
    for sym, info in meta.items():
        df = load_stock(sym).tail(30)
        ret = round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 3)
        vol = round(df["close"].pct_change().std() * 100, 4)
        sector = info["sector"]
        if sector not in sector_data:
            sector_data[sector] = {"stocks": [], "returns": [], "volatilities": []}
        sector_data[sector]["stocks"].append(sym)
        sector_data[sector]["returns"].append(ret)
        sector_data[sector]["volatilities"].append(vol)
    result = []
    for sector, d in sector_data.items():
        result.append({
            "sector": sector,
            "stocks": d["stocks"],
            "avg_return_30d": round(np.mean(d["returns"]), 3),
            "avg_volatility": round(np.mean(d["volatilities"]), 4),
        })
    result.sort(key=lambda x: x["avg_return_30d"], reverse=True)
    return {"sector_performance": result}


@app.get("/market-overview", tags=["Market"])
def market_overview():
    """Returns a holistic snapshot of the market."""
    meta = load_meta()
    symbols = list(meta.keys())
    gainers = losers = flat = 0
    total_volume = 0
    for sym in symbols:
        df = load_stock(sym)
        chg = df.iloc[-1]["close"] - df.iloc[-2]["close"]
        total_volume += int(df.iloc[-1]["volume"])
        if chg > 0: gainers += 1
        elif chg < 0: losers += 1
        else: flat += 1
    return {
        "date": str(load_stock(symbols[0])["date"].iloc[-1].date()),
        "total_stocks": len(symbols),
        "gainers": gainers,
        "losers": losers,
        "flat": flat,
        "total_volume": total_volume,
        "market_sentiment": "Bullish" if gainers > losers else "Bearish" if losers > gainers else "Neutral",
    }

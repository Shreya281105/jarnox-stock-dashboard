"""
data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
NSE Bhavcopy Data Pipeline
─────────────────────────────────────────────────────────────────────────────
This script demonstrates how to download real NSE Bhavcopy (daily equity
CSV files) from NSE India's public archive, clean, and store them.

Since the sandbox blocks external network access, this script also
includes a fallback: generate_mock_data() which produces statistically
realistic OHLCV data using Geometric Brownian Motion (GBM) — the same
model used by the Black-Scholes options pricing formula.

To run with real data:
    python data_ingestion.py --real

To run with mock data (default):
    python data_ingestion.py
"""

import os
import io
import zipfile
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SYMBOLS_OF_INTEREST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "WIPRO", "BAJFINANCE", "SBIN", "TATAMOTORS", "MARUTI",
    "SUNPHARMA", "LTIM",
]

STOCK_META = {
    "RELIANCE":   {"name": "Reliance Industries",          "sector": "Energy",   "base": 2850, "vol": 0.015},
    "TCS":        {"name": "Tata Consultancy Services",    "sector": "IT",       "base": 3900, "vol": 0.012},
    "INFY":       {"name": "Infosys",                      "sector": "IT",       "base": 1750, "vol": 0.013},
    "HDFCBANK":   {"name": "HDFC Bank",                    "sector": "Banking",  "base": 1620, "vol": 0.014},
    "ICICIBANK":  {"name": "ICICI Bank",                   "sector": "Banking",  "base": 1180, "vol": 0.016},
    "WIPRO":      {"name": "Wipro",                        "sector": "IT",       "base": 480,  "vol": 0.017},
    "BAJFINANCE": {"name": "Bajaj Finance",                "sector": "Finance",  "base": 7100, "vol": 0.020},
    "SBIN":       {"name": "State Bank of India",          "sector": "Banking",  "base": 790,  "vol": 0.018},
    "TATAMOTORS": {"name": "Tata Motors",                  "sector": "Auto",     "base": 980,  "vol": 0.022},
    "MARUTI":     {"name": "Maruti Suzuki",                "sector": "Auto",     "base": 12400,"vol": 0.011},
    "SUNPHARMA":  {"name": "Sun Pharmaceutical",           "sector": "Pharma",   "base": 1580, "vol": 0.014},
    "LTIM":       {"name": "LTIMindtree",                  "sector": "IT",       "base": 5200, "vol": 0.016},
}


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Real NSE Bhavcopy Download
# ─────────────────────────────────────────────────────────────────────────────

def get_bhavcopy_url(date: datetime) -> str:
    """Construct the NSE Bhavcopy archive URL for a given date."""
    mon = date.strftime("%b").upper()
    yr  = date.strftime("%Y")
    day = date.strftime("%d")
    return f"https://archives.nseindia.com/content/historical/EQUITIES/{yr}/{mon}/cm{day}{mon}{yr}bhav.csv.zip"


def fetch_bhavcopy(date: datetime) -> pd.DataFrame | None:
    """
    Fetch NSE Bhavcopy CSV for a given date.
    NSE Bhavcopy columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, ...
    """
    url = get_bhavcopy_url(date)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    df.columns = df.columns.str.strip()
                    # Filter for EQ series only
                    df = df[df["SERIES"].str.strip() == "EQ"]
                    df = df[df["SYMBOL"].isin(SYMBOLS_OF_INTEREST)].copy()
                    df["date"]   = date.strftime("%Y-%m-%d")
                    df["symbol"] = df["SYMBOL"].str.strip()
                    df["open"]   = pd.to_numeric(df["OPEN"], errors="coerce")
                    df["high"]   = pd.to_numeric(df["HIGH"], errors="coerce")
                    df["low"]    = pd.to_numeric(df["LOW"], errors="coerce")
                    df["close"]  = pd.to_numeric(df["CLOSE"], errors="coerce")
                    df["volume"] = pd.to_numeric(df["TOTTRDQTY"], errors="coerce")
                    return df[["date","symbol","open","high","low","close","volume"]]
        print(f"  [skip] {date.date()} — HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [error] {date.date()} — {e}")
    return None


def ingest_real_data(days_back: int = 400):
    """Download and store last N trading days of NSE Bhavcopy data."""
    all_frames = []
    end = datetime.today()
    checked = 0
    cur = end

    while len(all_frames) < days_back and checked < days_back * 2:
        if cur.weekday() < 5:  # weekdays only
            df = fetch_bhavcopy(cur)
            if df is not None and len(df) > 0:
                all_frames.append(df)
                print(f"  [ok] {cur.date()} — {len(df)} records")
        cur -= timedelta(days=1)
        checked += 1

    if not all_frames:
        print("No real data fetched — falling back to mock data.")
        return generate_mock_data()

    combined = pd.concat(all_frames).sort_values(["symbol", "date"])
    _compute_and_save(combined)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Geometric Brownian Motion Mock Data
# ─────────────────────────────────────────────────────────────────────────────

def gbm_prices(s0: float, mu: float, sigma: float, n: int) -> np.ndarray:
    """
    Geometric Brownian Motion — standard financial price simulation.
    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1), dt = 1/252 (daily)
    """
    dt = 1 / 252
    Z  = np.random.standard_normal(n)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return s0 * np.exp(np.cumsum(log_returns))


def generate_mock_data(days: int = 400) -> pd.DataFrame:
    """
    Generate realistic NSE-style OHLCV data using GBM.
    Annual drift (mu) = 12% for Indian large-caps.
    """
    np.random.seed(42)
    end = datetime.today()
    trading_days = [end - timedelta(days=i) for i in range(days*2) if (end - timedelta(days=i)).weekday() < 5]
    trading_days = sorted(trading_days[:days])

    all_rows = []
    for sym, info in STOCK_META.items():
        prices = gbm_prices(info["base"], mu=0.12, sigma=info["vol"]*16, n=days)
        for i, (d, price) in enumerate(zip(trading_days, prices)):
            spread = price * 0.005
            open_p  = price + np.random.uniform(-spread, spread)
            close_p = price
            high_p  = max(open_p, close_p) + abs(np.random.normal(0, price*0.004))
            low_p   = min(open_p, close_p) - abs(np.random.normal(0, price*0.004))
            volume  = max(int(np.random.normal(3e6, 8e5) * info["base"]/1000), 50000)
            all_rows.append({
                "date": d.strftime("%Y-%m-%d"), "symbol": sym,
                "open": round(open_p, 2), "high": round(high_p, 2),
                "low": round(low_p, 2),   "close": round(close_p, 2),
                "volume": volume
            })

    combined = pd.DataFrame(all_rows).sort_values(["symbol","date"])
    _compute_and_save(combined)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _compute_and_save(combined: pd.DataFrame):
    """Compute derived metrics and save per-symbol CSVs."""
    import json

    for sym in SYMBOLS_OF_INTEREST:
        df = combined[combined["symbol"] == sym].copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Handle missing values
        df["open"]  = df["open"].fillna(method="ffill")
        df["close"] = df["close"].fillna(method="ffill")
        df["high"]  = df["high"].fillna(df["close"])
        df["low"]   = df["low"].fillna(df["close"])
        df["volume"]= df["volume"].fillna(df["volume"].median())

        # Daily return = (close - open) / open
        df["daily_return"] = ((df["close"] - df["open"]) / df["open"] * 100).round(4)

        # Moving averages
        df["ma7"]  = df["close"].rolling(7).mean().round(2)
        df["ma20"] = df["close"].rolling(20).mean().round(2)
        df["ma50"] = df["close"].rolling(50).mean().round(2)

        # Volatility score (20-day coefficient of variation × 100)
        df["volatility"] = (
            df["close"].rolling(20).std() / df["close"].rolling(20).mean() * 100
        ).round(4)

        # Custom metric: Momentum Score (close vs 20-day MA, normalized)
        df["momentum_score"] = ((df["close"] - df["ma20"]) / df["ma20"] * 100).round(4)

        df.to_csv(DATA_DIR / f"{sym}.csv", index=False)
        print(f"  [saved] {sym}.csv — {len(df)} rows")

    # Save metadata
    import json
    with open(DATA_DIR / "stocks_meta.json", "w") as f:
        json.dump({k: {"name": v["name"], "sector": v["sector"]}
                   for k, v in STOCK_META.items()}, f, indent=2)
    print(f"\n  [done] Data saved to {DATA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JarNox Data Ingestion Pipeline")
    parser.add_argument("--real", action="store_true", help="Fetch real NSE Bhavcopy data")
    parser.add_argument("--days", type=int, default=400, help="Number of trading days")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print("  JarNox Stock Data Pipeline")
    print(f"  Mode: {'NSE Bhavcopy (real)' if args.real else 'GBM Simulation (mock)'}")
    print(f"  Days: {args.days}")
    print(f"{'─'*60}\n")

    if args.real:
        ingest_real_data(args.days)
    else:
        generate_mock_data(args.days)

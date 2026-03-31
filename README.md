# 📈 JarNox Stock Intelligence Platform

> A mini financial data platform built as part of the JarNox Software Internship Assignment.

---

## 🗂️ Project Structure

```
jarnox-dashboard/
│
├── backend/
│   └── main.py              # FastAPI application — all REST endpoints
│
├── frontend/
│   ├── dashboard.html       # Interactive web dashboard (standalone)
│   └── data.js              # Pre-embedded stock data (for offline use)
│
├── data/
│   ├── *.csv                # Per-symbol OHLCV + metrics (NSE Bhavcopy format)
│   ├── stocks_meta.json     # Company metadata (name, sector)
│   ├── predictions.json     # ML price forecasts
│   └── model_scores.json    # Model evaluation scores
│
├── data_ingestion.py        # Data pipeline (real NSE + GBM simulation)
├── ml_predictions.py        # ML training + forecasting module
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Layer       | Technology                              |
|-------------|-----------------------------------------|
| Language    | Python 3.12                             |
| Backend     | FastAPI + Uvicorn                       |
| Data        | Pandas, NumPy                           |
| ML          | scikit-learn (LinearRegression, Ridge)  |
| Frontend    | Vanilla HTML + JS + Chart.js            |
| Container   | Docker + Docker Compose                 |
| Data Source | NSE Bhavcopy CSVs / GBM Simulation      |

---

## 🚀 Quick Start

### Option A — Run Locally (Python)

```bash
# 1. Clone / unzip the project
cd jarnox-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data (mock GBM simulation — no external API needed)
python data_ingestion.py

# 4. Train ML models + generate predictions
python ml_predictions.py

# 5. Start the API
uvicorn backend.main:app --reload --port 8000

# 6. Open the dashboard
# Simply open frontend/dashboard.html in your browser
# (data.js is pre-embedded — works fully offline)
```

### Option B — Docker

```bash
# Build and run
docker-compose up --build

# API available at: http://localhost:8000
# Docs at:          http://localhost:8000/docs
```

### Option C — Real NSE Data

```bash
# Fetch last 400 trading days from NSE Bhavcopy archives
python data_ingestion.py --real --days 400
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + endpoint map |
| `/companies` | GET | All companies with latest prices + change |
| `/data/{symbol}` | GET | OHLCV + MA7/MA20/MA50 + daily return + volatility |
| `/summary/{symbol}` | GET | 52-week high/low/avg, momentum, volatility |
| `/compare` | GET | Side-by-side comparison + Pearson correlation |
| `/gainers` | GET | Top gaining stocks (today) |
| `/losers` | GET | Top losing stocks (today) |
| `/predict/{symbol}` | GET | ML price forecast for next 14 days |
| `/correlation` | GET | Full correlation matrix (all stocks) |
| `/sector-performance` | GET | 30-day performance grouped by sector |
| `/market-overview` | GET | Bulls vs Bears, volume, sentiment |

**API Docs (Swagger UI):** `http://localhost:8000/docs`
**ReDoc:** `http://localhost:8000/redoc`

---

## 📊 Data Pipeline

### Real Data (NSE Bhavcopy)
NSE publishes daily OHLCV data for all equities as compressed CSV files — called **Bhavcopy** — at:
```
https://archives.nseindia.com/content/historical/EQUITIES/{YEAR}/{MON}/cm{DD}{MON}{YEAR}bhav.csv.zip
```
The ingestion script downloads, unzips, filters for EQ series, and extracts OHLCV.

### Mock Data (GBM Simulation)
When real data isn't accessible, the pipeline uses **Geometric Brownian Motion** — the same mathematical model used in Black-Scholes options pricing:

```
S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × Z)
```
Where:
- `μ = 0.12` (12% annual drift, typical for Indian large-caps)
- `σ` = per-stock annual volatility (calibrated from historical ranges)
- `Z ~ N(0,1)` — standard normal shock

This produces statistically realistic price paths with proper fat-tail behavior.

---

## 🧮 Calculated Metrics

| Metric | Formula |
|---|---|
| Daily Return | `(close - open) / open × 100` |
| MA7 / MA20 / MA50 | Rolling mean of close prices |
| Volatility Score | `rolling_std(20) / rolling_mean(20) × 100` |
| Momentum Score | `(close - MA20) / MA20 × 100` |
| Pearson Correlation | Standard Pearson r between close price series |

---

## 🤖 ML Prediction Model

**Models trained:** Linear Regression, Ridge Regression  
**Features used:**
- Day index (t) — captures trend
- Lag-1, Lag-5, Lag-20 close prices — autocorrelation
- MA7, MA20 — trend context
- Daily return + Volatility — momentum signals

**Evaluation:** Time-series cross-validation (`TimeSeriesSplit`, 3 folds) to prevent data leakage. Best model selected by R² score.

**Scores are saved in** `data/model_scores.json`.

---

## 💡 Custom / Creative Features

Beyond the required tasks, this project adds:

1. **Volatility Score** — a rolling coefficient of variation that flags when a stock is unusually volatile
2. **Momentum Score** — measures how far the current price is from its 20-day average (positive = bullish momentum)
3. **Pearson Correlation Matrix** — `/correlation` endpoint returns full pairwise correlations; useful for portfolio diversification analysis
4. **Sector Performance Aggregation** — `/sector-performance` groups stocks by GICS-style sectors and computes average 30-day returns
5. **Market Sentiment** — `/market-overview` counts gainers vs losers and labels the session Bullish / Bearish / Neutral
6. **Normalized Compare Chart** — dashboard comparison panel normalizes both stocks to base=100 for fair visual comparison
7. **GBM Simulation fallback** — when external APIs are blocked, the pipeline generates statistically sound data using financial math

---

## 🎨 Dashboard Features

- **Live ticker strip** — animated scrolling top bar with all 12 stocks
- **Searchable sidebar** — filter by symbol or company name
- **OHLCV stat cards** — open, high, low, volume, 52W range
- **3 chart modes** — Price (with MA overlays), Daily Returns (bar), Volume (bar)
- **5 time ranges** — 2W / 1M / 3M / 6M / 1Y
- **Stock comparison panel** — normalized return chart + Pearson correlation
- **ML forecast panel** — 10-day ahead predictions with directional arrows
- **Top Gainers / Losers** — clickable, switches active stock
- **Sector performance bars** — color-coded by sector

---

## 🐳 Docker Notes

```bash
# Build image
docker build -t jarnox-api .

# Run container
docker run -p 8000:8000 jarnox-api

# With docker-compose (recommended)
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📬 Submission

- **GitHub:** *(add your repo link here)*
- **Live Demo:** *(add Render / Railway deployment link here)*
- **Contact:** *(your email)*

---

*Built with focus on clean code, financial correctness, and a polished user experience.*

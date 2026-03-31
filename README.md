📈 JarNox Stock Intelligence Platform

A mini financial data intelligence platform built as part of the JarNox Software Internship Assignment.
It provides stock analytics, market insights, and machine learning–based price forecasts through a REST API and an interactive dashboard.

🌐 Live Demo

Dashboard:

https://jarnox-stock-dashboard.onrender.com

API Documentation:

https://jarnox-stock-dashboard.onrender.com/docs

🗂️ Project Structure
jarnox-stock-dashboard/
│
├── backend/
│   └── main.py              # FastAPI application — all REST endpoints
│
├── frontend/
│   ├── dashboard.html       # Interactive web dashboard
│   └── data.js              # Embedded dataset for dashboard rendering
│
├── data/
│   ├── *.csv                # Per-stock OHLCV + metrics
│   ├── stocks_meta.json     # Company metadata (name, sector)
│   ├── predictions.json     # ML model predictions
│   └── model_scores.json    # Model evaluation results
│
├── data_ingestion.py        # Data pipeline
├── ml_predictions.py        # ML training + forecasting
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Container orchestration
└── requirements.txt

⚙️ Tech Stack
Layer	               Technology
Language-----	------Python 3.12
Backend	------- ----FastAPI
Data Processing-----	Pandas, NumPy
Machine Learning	------scikit-learn
Visualization-----------	Chart.js
Frontend	---------------HTML + JavaScript
Containerization---------	Docker
Deployment-----------------	Render


🚀 Quick Start
Run Locally
# Clone repo
git clone <repo-link>

cd jarnox-stock-dashboard

# Install dependencies
pip install -r requirements.txt

# Generate stock dataset
python data_ingestion.py

# Train ML models
python ml_predictions.py

# Run API
uvicorn backend.main:app --reload

Open:

http://localhost:8000

API docs:

http://localhost:8000/docs
🐳 Run Using Docker
docker-compose up --build

Access API:

http://localhost:8000
📡 API Endpoints
Endpoint	Description
/companies---	List of all companies with latest price
/data/{symbol}	--OHLCV data and technical indicators
/summary/{symbol}---	Key statistics for a stock
/compare	---Compare two stocks
/gainers----	Top gaining stocks
/losers---	Top losing stocks
/predict/{symbol}---	ML price forecast
/correlation---Correlation matrix
/sector-performance	Sector-wise performance
/market-overview	Overall market sentiment

Swagger Docs:

/docs
📊 Data Pipeline

The platform supports two data generation modes:

Real NSE Data

Historical stock data can be downloaded from NSE Bhavcopy archives.

GBM Simulation

If external data access is unavailable, stock prices are simulated using Geometric Brownian Motion (GBM):

S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × Z)

This produces realistic financial time-series behavior.

🧮 Calculated Metrics
Metric	Description
Moving Averages	MA7, MA20, MA50
Daily Returns	Percentage change
Volatility Score	Rolling volatility
Momentum Score	Deviation from MA20
Correlation	Pearson correlation
🤖 Machine Learning Model

Models used:

Linear Regression
Ridge Regression

Features:

Lagged closing prices
Moving averages
Daily returns
Volatility
Time index

Evaluation:

TimeSeriesSplit cross-validation

Best model selected using R² score.

🎨 Dashboard Features

Interactive financial dashboard including:

• Live stock ticker
• Interactive price charts
• Moving average overlays
• Time range filters
• Stock comparison panel
• ML price predictions
• Top gainers & losers
• Sector performance visualization

Charts are powered by Chart.js.

💡 Unique Features

This project goes beyond the basic assignment requirements:

📊 Market Sentiment Engine

Classifies market state as Bullish / Bearish / Neutral based on stock movements.

📉 Correlation Analysis

Generates a full correlation matrix to identify diversification opportunities.

📈 Momentum Indicator

Tracks trend strength relative to moving averages.

🔁 Simulation Fallback

Automatically generates realistic stock data if real APIs fail.

📊 Sector Aggregation

Groups stocks by sector to analyze sector performance.

🌍 Deployment

The project is deployed using Docker and hosted on Render.

Live demo:

https://jarnox-stock-dashboard.onrender.com
📬 Author

Shreya Agrawal

GitHub:

https://github.com/Shreya281105
⭐ Notes

This project was built with emphasis on:

• Clean architecture
• Realistic financial modeling
• Interactive data visualization
• Scalable API design

# Crypto-Asset Sentiment & Volatility Regime Analysis

### 📊 Project Overview
This project investigates the causal relationship between **Social Media Sentiment (Twitter)** and **Bitcoin Market Volatility** during the 2021–2023 market cycle.

Using **Natural Language Processing (VADER)** on a dataset of 200,000+ tweets, it tests the hypothesis that "Crowd Fear" acts as a leading indicator for market crashes. The analysis reveals a significant **regime disconnect** in the 2022 Bear Market, proving that institutional macro-economic factors (Fed Rates) overpowered retail sentiment, rendering social signals statistically insignificant for predictive trading during this period.

### 🚀 Key Features
* **NLP Sentiment Engine:** Processes unstructured text data using **VADER** to generate a daily "Fear & Greed" index, handling emojis and crypto-slang.
* **Hypothesis Testing:** Utilizes **Granger Causality** tests to statistically validate (or disprove) lead-lag relationships between sentiment and price action.
* **Regime Analysis:** Visualizes **Rolling Correlations** to identify specific market environments where social signals decouple from price performance.
* **Reusable Library:** Features a modular `src` package with automated data caching and robust error handling for multi-year time series.

### 🛠️ Tech Stack
* **Python 3.10+**
* **Data Engineering:** `pandas`, `yfinance`
* **NLP & Stats:** `nltk` (VADER), `statsmodels` (Granger Causality), `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

### 📂 Repository Structure
```text
├── data/                   # Raw CSVs (Excluded from Git due to size)
├── notebooks/              # Jupyter Notebooks for narrative analysis
│   └── crypto_analysis.ipynb
├── output/                 # Generated visualizations (Heatmaps, Lead-Lag charts)
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── data_loader.py      # Robust Kaggle & Yahoo Finance ingestion
│   ├── sentiment_engine.py # VADER scoring & aggregation logic
│   └── plotting.py         # Publication-ready charting modules
├── main.py                 # Orchestrator script for the full pipeline
├── requirements.txt        # Project dependencies
└── README.md
```

### 📈 Key Insights

1.  **The "Toxic Positivity" Divergence:** Contrary to standard behavioral finance theory, sentiment did not crash alongside prices in 2022. The data reveals a "HODL Culture" bias where retail sentiment remained neutral/positive (0.0–0.2) despite a >50% drawdown in Bitcoin price, making raw sentiment a dangerous contra-indicator during bear markets.
    
2.  **Market Efficiency and Alpha Decay:** Granger Causality tests across multiple lags (1–5 days) yielded p-values > 0.05, failing to reject the null hypothesis. This suggests the market is highly efficient; any predictive signal in public social data is likely arbitraged away by High-Frequency Trading (HFT) algorithms before it appears in daily aggregated data.
    
3.  **Structural Decoupling:** The Correlation Matrix shows a near-zero relationship ($r \approx 0$) between Sentiment and Volatility. This confirms that volatility is bimodal—driven equally by "Panic Selling" (Fear) and "FOMO Rallies" (Greed)—canceling out linear predictive signals.

### 💻 How to Run

1.  **Clone the repository**
```bash
git clone [https://github.com/Amsyar0689/crypto-sentiment-analysis.git](https://github.com/Amsyar0689/crypto-sentiment-analysis.git)
cd crypto-sentiment-analysis
```

2.  **Install Dependencies**
```bash
pip install -r requirements.txt
```

3.  **Download the Data (Important!)**  Due to GitHub file size limits, the raw Tweet dataset is not included.
*   Download the **Bitcoin Tweets** dataset from [Kaggle] (https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets).
*   Rename the file to tweets.csv.
*   Place it inside the data/ folder.

4.  **Run the Pipeline**  To execute the ingestion, scoring, and statistical tests:
```bash
python main.py
```

5.  **Explore the Analysis**  Open the Jupyter Notebook for the deep-dive visual storytelling:
```bash
jupyter notebook notebooks/crypto_analysis.ipynb
```
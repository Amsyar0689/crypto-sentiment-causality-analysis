import os
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from src.data_loader import DataLoader
from src.sentiment_engine import SentimentEngine
from src.plotting import Visualizer

if not os.path.exists('output'):
    os.makedirs('output')

def run_pipeline():
    # --- CONFIGURATION ---
    TWEETS_PATH = 'data/tweets.csv'
    # Use a small sample (0.5%) for development speed. Increase to 5-10% for final run.
    SAMPLE_FRAC = 0.05 
    
    # 1. LOAD DATA
    loader = DataLoader()
    
    # Load Tweets
    tweets_df = loader.load_tweets_csv(TWEETS_PATH, sample_fraction=SAMPLE_FRAC)
    if tweets_df.empty:
        print("Pipeline Stopped: No tweets loaded.")
        return

    # Fetch Crypto (Get range based on tweet dates)
    min_date = tweets_df['date'].min().strftime('%Y-%m-%d')
    max_date = tweets_df['date'].max().strftime('%Y-%m-%d')
    crypto_df = loader.fetch_crypto_data(start_date=min_date, end_date=max_date)

    if crypto_df.empty:
        print("Pipeline Stopped: No crypto data fetched.")
        return

    # 2. SENTIMENT ANALYSIS
    engine = SentimentEngine()
    daily_sentiment = engine.process_tweets(tweets_df)
    
    # 3. MERGE DATA
    # Combines Price, Volatility, and Sentiment into one timeline
    final_df = engine.merge_data(daily_sentiment, crypto_df)
    
    print(f"\nData Merged! Total Days: {len(final_df)}")
    print(final_df.head())
    
    # Save for potential Notebook usage
    final_df.to_csv('output/final_dataset.csv')

    # 4. VISUALIZATION
    viz = Visualizer()
    viz.plot_sentiment_vs_price(final_df)
    viz.plot_correlation_heatmap(final_df)

    # Granger Causality: Does Sentiment (X) cause Volatility (Y)?
    print("\n--- Running Granger Causality Test ---")
    print("Hypothesis: Does today's Sentiment predict tomorrow's Volatility?")
    
    # We test lags of 1 to 5 days
    max_lag = 5
    test_data = final_df[['Volatility', 'sentiment_score']]
    
    try:
        grangercausalitytests(test_data, maxlag=max_lag, verbose=True)
    except Exception as e:
        print(f"Stats Error: {e}")
        print("Not enough data points for Granger test. Try increasing SAMPLE_FRAC.")

if __name__ == "__main__":
    run_pipeline()
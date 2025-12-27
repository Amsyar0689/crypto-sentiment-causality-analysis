# src/data_loader.py

import pandas as pd
import yfinance as yf
import re

class DataLoader:
    """
    Fetches Crypto prices and ingests Social Media (Tweet) dumps.
    """

    def fetch_crypto_data(self, ticker='BTC-USD', start_date='2021-01-01', end_date='2024-01-01'):
        """
        Fetches historical crypto OHLC data from Yahoo Finance.
        Returns: Daily dataframe with 'Close' and 'Volume'.
        """
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
            
            if df.empty:
                print("Warning: No data fetched from Yahoo Finance.")
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                else:
                    print(f"Error: 'Close' column missing. Columns found: {df.columns.tolist()}")
                    return pd.DataFrame()
            
            # Calculate 7-day rolling window for volatility (Standard Deviation of Returns)
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=7).std()
            
            return df[['Close', 'Volume', 'Volatility']].dropna()
            
        except Exception as e:
            print(f"Error fetching crypto data: {e}")
            return pd.DataFrame()

    def load_tweets_csv(self, filepath, sample_fraction=0.1):
        """
        Loads the large Kaggle CSV.
        Args:
            sample_fraction (float): Loads only X% of data to save RAM (default 10%).
        """
        print(f"Loading tweets from {filepath} (Sampling {sample_fraction*100}%)...")
        
        try:
            df = pd.read_csv(filepath, usecols=['date', 'text'])
            
            # Randomly sample the data to prevent crashing RAM
            df = df.sample(frac=sample_fraction, random_state=42)
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')            
            df = df.dropna(subset=['date'])            
            df = df.sort_values(by='date')
            print(f"Loaded {len(df)} tweets.")

            return df
            
        except ValueError as v:
            print(f"Column Error: {v}")
            print("Check if your CSV headers match 'date' and 'text'.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def clean_text(self, text):
        """
        Basic NLP cleaning.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    loader = DataLoader()
    
    btc_df = loader.fetch_crypto_data()
    print("\n--- BTC Data Head ---")
    print(btc_df.head())
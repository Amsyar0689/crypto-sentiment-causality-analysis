import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    """
    Visualizes the relationship between Social Sentiment and Market Data.
    """

    def __init__(self):
        sns.set_theme(style="whitegrid")

    def plot_sentiment_vs_price(self, df):
        """
        Dual-axis chart: Bitcoin Price (Line) vs. Sentiment (Bar/Area).
        """
        print("Generating Sentiment vs. Price chart...")
        
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Axis 1: Bitcoin Price
        color = 'tab:gray'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bitcoin Price (USD)', color=color)
        ax1.plot(df.index, df['Close'], color=color, alpha=0.6, label='BTC Price')
        ax1.tick_params(axis='y', labelcolor=color)

        # Axis 2: Sentiment Score
        # We use a rolling mean to smooth out the noise for a cleaner chart
        ax2 = ax1.twinx()  
        color = 'tab:purple'
        ax2.set_ylabel('Daily Sentiment (VADER)', color=color)
        
        # Plot raw sentiment as faint bars
        ax2.bar(df.index, df['sentiment_score'], color=color, alpha=0.3, label='Daily Sentiment', width=1.0)
        
        # Plot 7-day moving average of sentiment as a solid line
        df['Sentiment_MA'] = df['sentiment_score'].rolling(window=7).mean()
        ax2.plot(df.index, df['Sentiment_MA'], color='indigo', linewidth=2, label='7-Day Sentiment Trend')
        
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, color='black', linewidth=1, linestyle='--') # Zero line

        plt.title('Does Social Mood Follow Price? (Bitcoin vs. Twitter Sentiment)', fontsize=16)
        fig.tight_layout()
        
        plt.savefig('../output/price_vs_sentiment.png')
        print("Saved to ../output/price_vs_sentiment.png")
        plt.close()

    def plot_correlation_heatmap(self, df):
        """
        Heatmap showing correlation between Sentiment, Volatility, and Returns.
        """
        print("Generating Correlation Heatmap...")
        
        # Select only numeric columns of interest
        cols = ['Close', 'Volume', 'Volatility', 'sentiment_score', 'tweet_volume']
        corr_matrix = df[cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title('Correlation Matrix: Crypto vs. Social', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('../output/correlation_matrix.png')
        print("Saved to ../output/correlation_matrix.png")
        plt.close()
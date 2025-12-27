import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentEngine:
    """
    Analyzes text sentiment using VADER and aggregates it by time.
    """

    def __init__(self):
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
        
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, text):
        """
        Returns a compound score between -1 (Negative) and +1 (Positive).
        """
        if not isinstance(text, str):
            return 0.0
        return self.analyzer.polarity_scores(text)['compound']

    def process_tweets(self, tweets_df):
        """
        Scoring and Aggregation Pipeline.
        1. Scores every tweet.
        2. Groups by Date to find Daily Average Sentiment.
        """
        print("Calculating sentiment scores (this may take a moment)...")
        
        # 1. Apply VADER to the text column
        tweets_df['sentiment_score'] = tweets_df['text'].apply(self.get_sentiment_score)
        
        # 2. Resample/Group by Day
        tweets_df.set_index('date', inplace=True)
        
        # Aggregation:
        # - mean: The average sentiment of the crowd that day
        # - count: The "Volume" of tweets (how loud is the crowd?)
        daily_sentiment = tweets_df.resample('D').agg({
            'sentiment_score': 'mean',
            'text': 'count'
        })
        
        # Rename columns for clarity
        daily_sentiment.rename(columns={'text': 'tweet_volume'}, inplace=True)
        
        return daily_sentiment.dropna()

    def merge_data(self, sentiment_df, crypto_df):
        """
        Inner joins the Daily Sentiment with Daily Crypto Price/Volatility.
        """
        print("Merging Sentiment and Crypto data...")
        
        # Ensure indices match (remove timezone info if present)
        if sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_localize(None)
        
        if crypto_df.index.tz is not None:
            crypto_df.index = crypto_df.index.tz_localize(None)
            
        # Merge on Date index
        merged_df = pd.merge(sentiment_df, crypto_df, left_index=True, right_index=True, how='inner')
        
        return merged_df

if __name__ == "__main__":
    engine = SentimentEngine()
    print("VADER initialized successfully.")
    print(f"Test Score for 'Bitcoin is crashing! 😭': {engine.get_sentiment_score('Bitcoin is crashing! 😭')}")
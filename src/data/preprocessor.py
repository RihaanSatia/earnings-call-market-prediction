import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


class EarningsDataPreprocessor:
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the three datasets."""
        transcripts = pd.read_csv(self.raw_dir / "earnings_calls_transcripts.csv")
        sentiment = pd.read_csv(self.raw_dir / "earnings_calls_transcript_sentiment.csv")
        prices = pd.read_csv(self.raw_dir / "earnings_calls_stock_prices.csv")
        
        return transcripts, sentiment, prices
    
    def preprocess_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def aggregate_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate paragraph-level sentiment to call-level."""
        sentiment_df = self.preprocess_dates(sentiment_df)
        
        # Convert labels to numeric
        label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        sentiment_df['sentiment_numeric'] = sentiment_df['label'].map(label_map)
        
        # Aggregate by company and date
        agg_sentiment = sentiment_df.groupby(['company', 'date']).agg({
            'sentiment_numeric': ['mean', 'std', 'count'],
            'label': lambda x: x.value_counts().index[0]  # most frequent
        }).reset_index()
        
        # Flatten column names
        agg_sentiment.columns = ['company', 'date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 'dominant_sentiment']
        agg_sentiment['sentiment_std'] = agg_sentiment['sentiment_std'].fillna(0)
        
        return agg_sentiment
    
    def calculate_market_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and returns from stock prices."""
        prices_df = self.preprocess_dates(prices_df)
        prices_df = prices_df.sort_values(['company', 'date'])
        
        # Calculate daily returns
        prices_df['daily_return'] = prices_df.groupby('company')['adj_close'].pct_change()
        
        # Calculate rolling volatility (10-day window)
        prices_df['volatility'] = prices_df.groupby('company')['daily_return'].rolling(
            window=10, min_periods=5
        ).std().reset_index(0, drop=True)
        
        return prices_df
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        df = df.copy()
        df = df.sort_values(['company', 'date'])
        
        # Calculate forward-looking metrics (3-day window)
        df['future_volatility'] = df.groupby('company')['volatility'].shift(-3)
        df['future_return'] = df.groupby('company')['daily_return'].shift(-3)
        
        # Binary targets
        vol_threshold = df['future_volatility'].quantile(0.75)
        return_threshold = 0.02
        
        df['volatility_spike'] = (df['future_volatility'] > vol_threshold).astype(int)
        df['significant_return'] = (df['future_return'].abs() > return_threshold).astype(int)
        df['target'] = ((df['volatility_spike'] == 1) | (df['significant_return'] == 1)).astype(int)
        
        return df
    
    def merge_datasets(self, transcripts: pd.DataFrame, sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on company and date."""
        transcripts = self.preprocess_dates(transcripts)
        agg_sentiment = self.aggregate_sentiment(sentiment)
        prices_features = self.calculate_market_features(prices)
        
        # Merge transcripts with sentiment
        merged = transcripts.merge(agg_sentiment, on=['company', 'date'], how='inner')
        
        # Merge with market data (find closest market date)
        merged_final = []
        for _, row in merged.iterrows():
            company_prices = prices_features[prices_features['company'] == row['company']]
            closest_date = company_prices[company_prices['date'] <= row['date']]['date'].max()
            
            if pd.notna(closest_date):
                market_data = company_prices[company_prices['date'] == closest_date].iloc[0]
                combined_row = {**row.to_dict(), **market_data.to_dict()}
                combined_row['date'] = row['date']  # Keep earnings call date
                merged_final.append(combined_row)
        
        result = pd.DataFrame(merged_final)
        return self.create_targets(result)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleaning."""
        df = df.copy()
        
        # Remove rows with missing targets
        df = df.dropna(subset=['target'])
        
        # Remove very short transcripts
        df['transcript_length'] = df['transcript'].str.len()
        df = df[df['transcript_length'] >= 1000]
        
        # Basic feature engineering
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        
        return df
    
    def process_pipeline(self) -> pd.DataFrame:
        """Run complete preprocessing pipeline."""
        print("Loading raw data...")
        transcripts, sentiment, prices = self.load_raw_data()
        
        print("Merging datasets...")
        merged_df = self.merge_datasets(transcripts, sentiment, prices)
        
        print("Final cleaning...")
        final_df = self.clean_data(merged_df)
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Target distribution: {final_df['target'].value_counts().to_dict()}")
        
        # Save processed data
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "processed_earnings_data.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        return final_df
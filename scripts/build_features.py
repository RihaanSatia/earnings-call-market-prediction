#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.feature_engineering import EarningsFeatureEngineer
from features.market_features import MarketFeatureEngineer


def main():
    print("Building features for earnings prediction model...")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load raw data
    transcripts_df = pd.read_csv(data_dir / "raw" / "earnings_calls_transcripts.csv")
    
    # Initialize feature engineer
    feature_engineer = EarningsFeatureEngineer()
    market_engineer = MarketFeatureEngineer()
    
    # Load and prepare market data
    market_df = market_engineer.load_and_prepare_market_data(data_dir)
    
    print(f"Loaded {len(transcripts_df)} transcripts")
    print(f"Market data: {len(market_df)} price records")
    
    # Process full dataset
    features_df = feature_engineer.process_full_dataset(transcripts_df, market_df)
    
    # Save processed features
    output_path = data_dir / "processed" / "earnings_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"\nFeatures saved to: {output_path}")
    print(f"Feature columns: {len(features_df.columns)}")
    
    # Display feature summary
    feature_cols = [col for col in features_df.columns if col not in ['company', 'date', 'transcript']]
    print(f"Generated {len(feature_cols)} features:")
    
    # Group features by type
    sentiment_features = [col for col in feature_cols if col.startswith('lm_')]
    readability_features = [col for col in feature_cols if any(x in col for x in ['flesch', 'grade', 'fog', 'readability'])]
    topic_features = [col for col in feature_cols if col.startswith('topic_')]
    market_features = [col for col in feature_cols if any(x in col for x in ['volatility', 'return', 'volume'])]
    
    print(f"  Sentiment features: {len(sentiment_features)}")
    print(f"  Readability features: {len(readability_features)}")
    print(f"  Topic features: {len(topic_features)}")
    print(f"  Market features: {len(market_features)}")
    print(f"  Other features: {len(feature_cols) - len(sentiment_features) - len(readability_features) - len(topic_features) - len(market_features)}")


if __name__ == "__main__":
    main()
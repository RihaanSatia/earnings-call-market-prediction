#!/usr/bin/env python3

from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessor import EarningsDataPreprocessor

def main():
    processor = EarningsDataPreprocessor()
    processed_df = processor.process_pipeline()
    
    print("\nDataset info:")
    print(f"Shape: {processed_df.shape}")
    print(f"Companies: {processed_df['company'].nunique()}")
    print(f"Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    print(f"Target balance: {processed_df['target'].mean():.3f}")

if __name__ == "__main__":
    main()
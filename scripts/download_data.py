#!/usr/bin/env python3

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def download_earnings_call_data():
    """Download and save the Hugging Face earnings call dataset."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading jlh-ibm/earnings_call dataset from Hugging Face...")
    
    try:
        # Load all three configurations of the dataset
        configurations = ["stock_prices", "transcript-sentiment", "transcripts"]
        datasets = {}
        dataframes = {}
        
        for config in configurations:
            print(f"Loading {config} configuration...")
            datasets[config] = load_dataset("jlh-ibm/earnings_call", config, trust_remote_code=True)
            
            # Convert to pandas DataFrame
            if 'train' in datasets[config]:
                df = datasets[config]['train'].to_pandas()
            else:
                # If no split, get the first available split
                split_name = list(datasets[config].keys())[0]
                df = datasets[config][split_name].to_pandas()
            
            dataframes[config] = df
            print(f"{config} - Shape: {df.shape}, Columns: {list(df.columns)}")
            
            # Save each configuration to separate CSV
            output_path = data_dir / f"earnings_calls_{config.replace('-', '_')}.csv"
            df.to_csv(output_path, index=False)
            print(f"{config} data saved to: {output_path}")
        
        # Display overview of each dataset
        for config, df in dataframes.items():
            print(f"\n=== {config.upper()} Dataset Overview ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df.head(2))
            print("-" * 50)
        
        return dataframes
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    datasets = download_earnings_call_data()
    if datasets is not None:
        print("Dataset download completed successfully!")
        print(f"Downloaded {len(datasets)} configurations: {list(datasets.keys())}")
    else:
        print("Failed to download dataset.")
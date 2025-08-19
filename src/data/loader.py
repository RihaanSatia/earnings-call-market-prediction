from pathlib import Path
import pandas as pd
from datasets import load_dataset
from typing import Dict, Optional


class EarningsDataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_huggingface(self) -> Dict[str, pd.DataFrame]:
        """Download earnings call data from Hugging Face."""
        configurations = ["stock_prices", "transcript-sentiment", "transcripts"]
        dataframes = {}
        
        for config in configurations:
            print(f"Downloading {config}...")
            dataset = load_dataset("jlh-ibm/earnings_call", config, trust_remote_code=True)
            
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
            
            dataframes[config] = df
            
            # Save to raw data directory
            filename = f"earnings_calls_{config.replace('-', '_')}.csv"
            df.to_csv(self.raw_dir / filename, index=False)
            print(f"Saved {config} ({df.shape[0]} rows) to {filename}")
        
        return dataframes
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from CSV files."""
        dataframes = {}
        
        for file_path in self.raw_dir.glob("earnings_calls_*.csv"):
            config_name = file_path.stem.replace("earnings_calls_", "").replace("_", "-")
            df = pd.read_csv(file_path)
            dataframes[config_name] = df
            print(f"Loaded {config_name}: {df.shape}")
        
        return dataframes
    
    def get_data(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """Get earnings call data, downloading if necessary."""
        if force_download or not list(self.raw_dir.glob("earnings_calls_*.csv")):
            return self.download_from_huggingface()
        else:
            return self.load_raw_data()
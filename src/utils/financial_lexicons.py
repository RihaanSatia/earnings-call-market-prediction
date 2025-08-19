import pandas as pd
from pathlib import Path
from typing import Dict, Set


class FinancialLexicons:
    def __init__(self, cache_dir: str = "data/lexicons"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize lexicon sets
        self.lm_positive = set()
        self.lm_negative = set()
        self.lm_uncertainty = set()
        self.lm_litigious = set()
        self.lm_constraining = set()
        
        self._load_loughran_mcdonald()
    
    def _load_loughran_mcdonald(self):
        """Load Loughran-McDonald financial dictionary from local file."""
        file_path = self.cache_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Loughran-McDonald dictionary not found at {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Extract words by category (non-zero values indicate category membership)
            self.lm_positive = set(df[df['Positive'] != 0]['Word'].str.lower())
            self.lm_negative = set(df[df['Negative'] != 0]['Word'].str.lower())
            self.lm_uncertainty = set(df[df['Uncertainty'] != 0]['Word'].str.lower())
            self.lm_litigious = set(df[df['Litigious'] != 0]['Word'].str.lower())
            self.lm_constraining = set(df[df['Constraining'] != 0]['Word'].str.lower())
            
            print(f"Loaded Loughran-McDonald lexicons:")
            print(f"  Positive: {len(self.lm_positive)} words")
            print(f"  Negative: {len(self.lm_negative)} words")
            print(f"  Uncertainty: {len(self.lm_uncertainty)} words")
            print(f"  Litigious: {len(self.lm_litigious)} words")
            print(f"  Constraining: {len(self.lm_constraining)} words")
            
        except Exception as e:
            print(f"Error loading Loughran-McDonald lexicon: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using Loughran-McDonald lexicon."""
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return {
                'lm_positive_score': 0.0,
                'lm_negative_score': 0.0,
                'lm_uncertainty_score': 0.0,
                'lm_litigious_score': 0.0,
                'lm_constraining_score': 0.0,
                'lm_net_sentiment': 0.0
            }
        
        positive_count = sum(1 for word in words if word in self.lm_positive)
        negative_count = sum(1 for word in words if word in self.lm_negative)
        uncertainty_count = sum(1 for word in words if word in self.lm_uncertainty)
        litigious_count = sum(1 for word in words if word in self.lm_litigious)
        constraining_count = sum(1 for word in words if word in self.lm_constraining)
        
        return {
            'lm_positive_score': positive_count / word_count,
            'lm_negative_score': negative_count / word_count,
            'lm_uncertainty_score': uncertainty_count / word_count,
            'lm_litigious_score': litigious_count / word_count,
            'lm_constraining_score': constraining_count / word_count,
            'lm_net_sentiment': (positive_count - negative_count) / word_count
        }
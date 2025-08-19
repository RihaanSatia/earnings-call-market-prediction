import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from utils.financial_lexicons import FinancialLexicons
from features.market_features import MarketFeatureEngineer


class EarningsFeatureEngineer:
    def __init__(self):
        self.lexicons = FinancialLexicons()
        self.market_engineer = MarketFeatureEngineer()
        
        # Initialize TF-IDF for topic modeling
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Initialize LDA for topic modeling
        self.lda = LatentDirichletAllocation(
            n_components=5,
            random_state=42,
            max_iter=10
        )
        
        self.is_fitted = False
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive text features from earnings transcript."""
        features = {}
        
        # Sentiment features using Loughran-McDonald
        sentiment_features = self.lexicons.analyze_sentiment(text)
        features.update(sentiment_features)
        
        # Readability features
        features.update(self._extract_readability_features(text))
        
        # Financial linguistic features
        features.update(self._extract_linguistic_features(text))
        
        return features
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features."""
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            fk_grade = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            automated_readability = textstat.automated_readability_index(text)
        except:
            flesch_score = fk_grade = gunning_fog = automated_readability = 0
        
        sentences = text.split('.')
        words = text.split()
        
        return {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': fk_grade,
            'gunning_fog_index': gunning_fog,
            'automated_readability_index': automated_readability,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract financial-specific linguistic features."""
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)
        
        # Financial performance indicators
        revenue_mentions = text_lower.count('revenue') + text_lower.count('sales')
        profit_mentions = text_lower.count('profit') + text_lower.count('earnings') + text_lower.count('income')
        guidance_mentions = text_lower.count('guidance') + text_lower.count('outlook') + text_lower.count('forecast')
        
        # Future-oriented language
        future_words = ['will', 'expect', 'anticipate', 'plan', 'intend', 'goal']
        future_mentions = sum(text_lower.count(word) for word in future_words)
        
        # Numbers and financial metrics
        number_mentions = len([word for word in words if any(char.isdigit() for char in word)])
        percent_mentions = text_lower.count('%') + text_lower.count('percent')
        
        return {
            'revenue_mentions_per_1k': (revenue_mentions / max(word_count, 1)) * 1000,
            'profit_mentions_per_1k': (profit_mentions / max(word_count, 1)) * 1000,
            'guidance_mentions_per_1k': (guidance_mentions / max(word_count, 1)) * 1000,
            'future_mentions_per_1k': (future_mentions / max(word_count, 1)) * 1000,
            'number_mentions_per_1k': (number_mentions / max(word_count, 1)) * 1000,
            'percent_mentions_per_1k': (percent_mentions / max(word_count, 1)) * 1000
        }
    
    def fit_topic_model(self, transcripts: List[str]) -> None:
        """Fit topic model on the corpus of transcripts."""
        print("Fitting topic model...")
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(transcripts)
        
        # Fit LDA
        self.lda.fit(tfidf_matrix)
        
        self.is_fitted = True
        print("Topic model fitted successfully")
    
    def extract_topic_features(self, text: str) -> Dict[str, float]:
        """Extract topic distribution features."""
        if not self.is_fitted:
            return {f'topic_{i}': 0.0 for i in range(5)}
        
        try:
            tfidf_vector = self.tfidf.transform([text])
            topic_dist = self.lda.transform(tfidf_vector)[0]
            
            return {f'topic_{i}': float(topic_dist[i]) for i in range(len(topic_dist))}
        except:
            return {f'topic_{i}': 0.0 for i in range(5)}
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on earnings date."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        df['is_january'] = (df['month'] == 1).astype(int)
        
        # Market timing features
        df['days_since_2016'] = (df['date'] - pd.Timestamp('2016-01-01')).dt.days
        
        return df
    
    def process_full_dataset(self, transcripts_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """Process complete dataset with all features."""
        print("Processing earnings dataset...")
        
        # Fit topic model first
        self.fit_topic_model(transcripts_df['transcript'].tolist())
        
        results = []
        
        for idx, row in transcripts_df.iterrows():
            print(f"Processing {idx+1}/{len(transcripts_df)}: {row['company']} {row['date']}")
            
            # Extract text features
            text_features = self.extract_text_features(row['transcript'])
            
            # Extract topic features
            topic_features = self.extract_topic_features(row['transcript'])
            
            # Extract market context features
            market_features = self.market_engineer.calculate_pre_earnings_features(
                market_df, row['date'], row['company']
            )
            
            # Calculate target variables
            target_features = self.market_engineer.calculate_target_variables(
                market_df, row['date'], row['company']
            )
            
            # Combine all features
            combined_features = {
                'company': row['company'],
                'date': row['date'],
                'transcript': row['transcript'],
                **text_features,
                **topic_features,
                **market_features,
                **target_features
            }
            
            results.append(combined_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Create binary targets
        df = self.market_engineer.create_binary_targets(df)
        
        print(f"\nFeature engineering completed!")
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:")
        print(f"  Volatility spikes: {df['volatility_spike'].sum()} ({df['volatility_spike'].mean():.1%})")
        print(f"  Significant returns: {df['significant_return'].sum()} ({df['significant_return'].mean():.1%})")
        print(f"  Combined target: {df['target'].sum()} ({df['target'].mean():.1%})")
        
        return df
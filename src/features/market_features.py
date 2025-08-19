import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path


class MarketFeatureEngineer:
    def __init__(self):
        self.volatility_window = 10
        self.post_earnings_window = 3
        
    def load_and_prepare_market_data(self, data_dir: Path) -> pd.DataFrame:
        """Load and prepare stock price data."""
        prices_df = pd.read_csv(data_dir / "raw" / "earnings_calls_stock_prices.csv")
        
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        prices_df = prices_df.sort_values(['company', 'date']).reset_index(drop=True)
        
        return self._calculate_returns_and_volatility(prices_df)
    
    def _calculate_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and volatility metrics."""
        df = df.copy()
        
        df['daily_return'] = df.groupby('company')['adj_close'].pct_change()
        
        df['rolling_volatility'] = df.groupby('company')['daily_return'].rolling(
            window=self.volatility_window, min_periods=5
        ).std().reset_index(0, drop=True)
        
        df['avg_volume'] = df.groupby('company')['volume'].rolling(
            window=self.volatility_window, min_periods=5
        ).mean().reset_index(0, drop=True)
        
        df['volume_ratio'] = df['volume'] / df['avg_volume']
        
        return df
    
    def calculate_target_variables(self, market_df: pd.DataFrame, earnings_date: str, company: str) -> Dict[str, float]:
        """Calculate target variables for a specific earnings call."""
        earnings_date = pd.to_datetime(earnings_date)
        company_data = market_df[market_df['company'] == company].copy()
        
        # Get post-earnings data
        post_earnings_start = earnings_date + pd.Timedelta(days=1)
        post_earnings_end = earnings_date + pd.Timedelta(days=7)
        
        post_data = company_data[
            (company_data['date'] >= post_earnings_start) & 
            (company_data['date'] <= post_earnings_end)
        ].head(self.post_earnings_window)
        
        if len(post_data) < 2:
            return {
                'post_earnings_return': 0.0,
                'post_earnings_max_volatility': 0.0,
                'post_earnings_avg_volume_ratio': 1.0,
                'abs_return': 0.0
            }
        
        post_returns = post_data['daily_return'].dropna()
        post_volatility = post_data['rolling_volatility'].dropna()
        
        if post_returns.empty or post_volatility.empty:
            return {
                'post_earnings_return': 0.0,
                'post_earnings_max_volatility': 0.0,
                'post_earnings_avg_volume_ratio': 1.0,
                'abs_return': 0.0
            }
        
        cumulative_return = (1 + post_returns).prod() - 1
        max_volatility = post_volatility.max()
        avg_volume_ratio = post_data['volume_ratio'].mean()
        
        return {
            'post_earnings_return': float(cumulative_return),
            'post_earnings_max_volatility': float(max_volatility),
            'post_earnings_avg_volume_ratio': float(avg_volume_ratio),
            'abs_return': float(abs(cumulative_return))
        }
    
    def calculate_pre_earnings_features(self, market_df: pd.DataFrame, earnings_date: str, company: str) -> Dict[str, float]:
        """Calculate pre-earnings market context features."""
        earnings_date = pd.to_datetime(earnings_date)
        company_data = market_df[market_df['company'] == company].copy()
        
        pre_earnings_data = company_data[company_data['date'] <= earnings_date]
        if pre_earnings_data.empty:
            return {
                'pre_earnings_volatility': 0.0,
                'pre_earnings_volume_ratio': 1.0,
                'return_30d': 0.0,
                'volatility_30d_avg': 0.0,
            }
        
        latest_data = pre_earnings_data.iloc[-1]
        
        date_30_days = earnings_date - pd.Timedelta(days=30)
        data_30d = company_data[
            (company_data['date'] >= date_30_days) & 
            (company_data['date'] <= earnings_date)
        ]
        
        return {
            'pre_earnings_volatility': float(latest_data.get('rolling_volatility', 0)),
            'pre_earnings_volume_ratio': float(latest_data.get('volume_ratio', 1)),
            'return_30d': float(data_30d['daily_return'].sum()) if not data_30d.empty else 0.0,
            'volatility_30d_avg': float(data_30d['rolling_volatility'].mean()) if not data_30d.empty else 0.0,
        }
    
    def create_binary_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary classification targets."""
        df = df.copy()
        
        volatility_threshold = df['post_earnings_max_volatility'].quantile(0.75)
        return_threshold = 0.02
        
        df['volatility_spike'] = (df['post_earnings_max_volatility'] > volatility_threshold).astype(int)
        df['significant_return'] = (df['abs_return'] > return_threshold).astype(int)
        df['target'] = ((df['volatility_spike'] == 1) | (df['significant_return'] == 1)).astype(int)
        
        return df
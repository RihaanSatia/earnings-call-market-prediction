import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class EarningsPredictor:
    def __init__(self, model_dir: str = "models/trained"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = None
        self.target_columns = ['volatility_spike', 'significant_return', 'target']
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets for modeling."""
        # Define feature columns (exclude metadata and targets)
        exclude_cols = [
            'company', 'date', 'transcript', 
            'post_earnings_return', 'post_earnings_max_volatility',
            'post_earnings_avg_volume_ratio', 'abs_return',
            'volatility_spike', 'significant_return', 'target'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df[self.target_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        print(f"Features prepared: {X.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        
        return X, y
    
    def time_series_split(self, df: pd.DataFrame, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-based train/test splits."""
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(df_sorted):
            splits.append((train_idx, test_idx))
        
        return splits
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train multiple models with time-series cross-validation."""
        print("Training predictive models...")
        
        X, y = self.prepare_features(df)
        
        # Get time-series splits
        splits = self.time_series_split(df)
        
        # Define models
        model_configs = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                verbosity=-1
            )
        }
        
        results = {}
        
        # Train models for each target
        for target in self.target_columns:
            print(f"\nTraining models for target: {target}")
            target_results = {}
            
            for model_name, model in model_configs.items():
                print(f"  Training {model_name}...")
                
                # Cross-validation scores
                cv_scores = []
                
                for fold, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[target].iloc[train_idx], y[target].iloc[test_idx]
                    
                    # Scale features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    # Train model
                    if model_name == 'lightgbm':
                        model.fit(X_train_scaled, y_train)
                    else:
                        model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate AUC
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                        cv_scores.append(auc_score)
                    except:
                        cv_scores.append(0.5)
                
                # Final model training on full dataset
                X_scaled = self.scaler.fit_transform(X)
                model.fit(X_scaled, y[target])
                
                # Store model and results
                model_key = f"{model_name}_{target}"
                self.models[model_key] = {
                    'model': model,
                    'scaler': self.scaler,
                    'target': target
                }
                
                target_results[model_name] = {
                    'cv_auc_mean': np.mean(cv_scores),
                    'cv_auc_std': np.std(cv_scores),
                    'cv_scores': cv_scores
                }
                
                print(f"    CV AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            results[target] = target_results
        
        return results
    
    def predict(self, df: pd.DataFrame, model_name: str = 'lightgbm', target: str = 'target') -> Dict[str, Any]:
        """Make predictions on new data."""
        model_key = f"{model_name}_{target}"
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found. Available models: {list(self.models.keys())}")
        
        model_info = self.models[model_key]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare features
        X, _ = self.prepare_features(df)
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_name,
            'target': target
        }
    
    def get_feature_importance(self, model_name: str = 'lightgbm', target: str = 'target') -> pd.DataFrame:
        """Get feature importance from trained model."""
        model_key = f"{model_name}_{target}"
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        model = self.models[model_key]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For logistic regression, use coefficient magnitude
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        for model_key, model_info in self.models.items():
            model_path = self.model_dir / f"{model_key}.joblib"
            joblib.dump(model_info, model_path)
        
        # Save feature columns
        feature_path = self.model_dir / "feature_columns.joblib"
        joblib.dump(self.feature_columns, feature_path)
        
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        feature_path = self.model_dir / "feature_columns.joblib"
        if feature_path.exists():
            self.feature_columns = joblib.load(feature_path)
        
        for model_file in self.model_dir.glob("*.joblib"):
            if model_file.stem != "feature_columns":
                model_key = model_file.stem
                self.models[model_key] = joblib.load(model_file)
        
        print(f"Loaded {len(self.models)} models from {self.model_dir}")
    
    def evaluate_model(self, df: pd.DataFrame, model_name: str = 'lightgbm', target: str = 'target') -> Dict[str, float]:
        """Evaluate model performance."""
        X, y = self.prepare_features(df)
        
        # Get time-series splits for evaluation
        splits = self.time_series_split(df)
        
        # Use the last split for final evaluation
        train_idx, test_idx = splits[-1]
        X_test, y_test = X.iloc[test_idx], y[target].iloc[test_idx]
        
        # Predict
        pred_results = self.predict(df.iloc[test_idx], model_name, target)
        
        # Calculate metrics
        y_pred = pred_results['predictions']
        y_prob = pred_results['probabilities']
        
        auc_score = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        return {
            'roc_auc': auc_score,
            'pr_auc': pr_auc,
            'accuracy': (y_pred == y_test).mean(),
            'positive_rate': y_test.mean(),
            'n_samples': len(y_test)
        }
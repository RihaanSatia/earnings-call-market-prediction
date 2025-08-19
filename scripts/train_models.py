#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.predictive_models import EarningsPredictor


def main():
    print("Training earnings prediction models...")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load processed features
    features_path = data_dir / "processed" / "earnings_features.csv"
    if not features_path.exists():
        print(f"Features file not found at {features_path}")
        print("Please run 'python scripts/build_features.py' first")
        return
    
    df = pd.read_csv(features_path)
    print(f"Loaded feature dataset: {df.shape}")
    
    # Initialize predictor
    predictor = EarningsPredictor()
    
    # Train models
    results = predictor.train_models(df)
    
    # Save models
    predictor.save_models()
    
    # Print results summary
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    for target, target_results in results.items():
        print(f"\nTarget: {target}")
        print("-" * 40)
        for model_name, scores in target_results.items():
            print(f"{model_name:20}: {scores['cv_auc_mean']:.3f} ± {scores['cv_auc_std']:.3f}")
    
    # Evaluate best models
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    for target in ['volatility_spike', 'significant_return', 'target']:
        print(f"\nTarget: {target}")
        print("-" * 40)
        
        for model_name in ['logistic_regression', 'random_forest', 'lightgbm']:
            try:
                eval_results = predictor.evaluate_model(df, model_name, target)
                print(f"{model_name:20}: ROC-AUC={eval_results['roc_auc']:.3f}, PR-AUC={eval_results['pr_auc']:.3f}")
            except Exception as e:
                print(f"{model_name:20}: Error - {e}")
    
    # Show feature importance for best model
    print("\n" + "="*60)
    print("TOP 10 FEATURES (LightGBM, Combined Target)")
    print("="*60)
    
    try:
        importance_df = predictor.get_feature_importance('lightgbm', 'target')
        print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    print(f"\nModels saved to: {predictor.model_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.financial_lexicons import FinancialLexicons


def main():
    print("Testing Financial Lexicons...")
    
    # Initialize lexicons
    lexicons = FinancialLexicons()
    
    # Test sentences
    test_texts = [
        "We are confident about strong revenue growth and excellent profitability this quarter.",
        "The company faces significant challenges and uncertainty in the current market environment.",
        "Our results were disappointing with declining margins and weak performance.",
        "We expect continued expansion and positive momentum in our key business segments."
    ]
    
    print("\nTesting sentiment analysis:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        print("-" * 60)
        
        # Get sentiment features
        features = lexicons.analyze_sentiment(text)
        
        print("Loughran-McDonald scores:")
        print(f"  Positive: {features['lm_positive_score']:.4f}")
        print(f"  Negative: {features['lm_negative_score']:.4f}")
        print(f"  Uncertainty: {features['lm_uncertainty_score']:.4f}")
        print(f"  Litigious: {features['lm_litigious_score']:.4f}")
        print(f"  Constraining: {features['lm_constraining_score']:.4f}")
        print(f"  Net sentiment: {features['lm_net_sentiment']:.4f}")


if __name__ == "__main__":
    main()
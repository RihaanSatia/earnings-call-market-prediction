A machine learning project to predict significant market reactions (volatility spikes or abnormal returns) following earnings calls, using natural language processing of call transcripts and market context features.

## 📋 Project Overview

This project aims to predict the probability of significant market reactions in the 3 days following an earnings call by analyzing:
- **Text features**: Sentiment analysis, topic modeling, and structural patterns from earnings call transcripts
- **Market context**: Pre-call volatility, abnormal returns, and sector information
- **Binary classification**: Volatility spike (>75th percentile) or abnormal return (>±2%)

## 🎯 Key Features

- **Advanced NLP Pipeline**: FinBERT sentiment analysis, Loughran-McDonald financial dictionaries, LLM topic tagging
- **Feature Engineering**: Text structural analysis, sentiment gap detection, market context integration
- **Multiple Models**: Logistic regression with L2 regularization and gradient boosted trees (XGBoost/LightGBM)
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Brier score, calibration plots, SHAP explainability

## 📊 Dataset

- **Primary**: Hugging Face `jlh-ibm/earnings_call` dataset (188 calls, 2016-2020)
- **Optional Extensions**: Motley Fool, Finnhub API, Yahoo Finance (2021-2024)

## 🛠️ Technology Stack

- **Data Processing**: Pandas, NumPy
- **NLP**: Hugging Face Transformers (FinBERT), Sentence-Transformers
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Visualization**: Matplotlib, Seaborn, SHAP
- **API**: FastAPI, Uvicorn

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone github-personal:RihaanSatia/earnings-call-market-prediction.git
cd earnings-call-market-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

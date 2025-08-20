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

- **Primary**: Hugging Face `jlh-ibm/earnings_call` dataset (188 calls, 2016-2020), extended further.

## 🛠️ Technology Stack

- **Data Processing**: Pandas, NumPy
- **NLP**: Hugging Face Transformers (FinBERT), Sentence-Transformers
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **API**: FastAPI, Uvicorn

## 🚀 Quick Start

### Docker Setup (Recommended)

```bash
git clone github-personal:RihaanSatia/earnings-call-market-prediction.git
cd earnings-call-market-prediction
docker-compose up --build
```

Services:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **Jupyter Lab**: http://localhost:8888

Individual services:
```bash
docker-compose up earnings-ml    
docker-compose up jupyter       
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

import os

def create_project_structure():
    """
    Creates Portfolio Risk Analyzer structure in CURRENT directory
    (No subdirectory created - current folder becomes the repo)
    """
    
    # Use current directory as base
    base_dir = "."
    
    # Define folder structure
    folders = [
        # Data folders
        "data/raw",
        "data/processed",
        "data/external",
        
        # Notebooks
        "notebooks",
        
        # Source code
        "src/data",
        "src/analysis",
        "src/models",
        "src/visualization",
        "src/utils",
        
        # Models
        "models",
        
        # Reports
        "reports/figures",
        "reports/results",
        
        # Tests
        "tests",
        
        # Documentation
        "docs",
    ]
    
    # Define files to create
    files = {
        # Root level files
        "README.md": """# Portfolio Risk Analyzer

## Overview
A comprehensive ML-powered portfolio risk analysis tool.

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Reusable source code modules
- `models/`: Saved trained models
- `reports/`: Generated figures and results
- `docs/`: Project documentation

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run notebooks in order (01 to 05)
2. Or use modules from src/

## Features
- Multi-stock data collection
- Portfolio risk metrics (VaR, CVaR, Sharpe)
- ML forecasting (Random Forest, LSTM)
- Interactive visualizations

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, Plotly
""",
        
        "requirements.txt": """# Core Data Science
pandas==2.0.0
numpy==1.24.0

# Data Fetching
yfinance==0.2.28

# Visualization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.15.0

# Machine Learning
scikit-learn==1.3.0
xgboost==2.0.0

# Deep Learning
tensorflow==2.13.0

# Bayesian (Optional)
pymc3==5.0.0
arviz==0.15.0

# Utilities
pyyaml==6.0
jupyter==1.0.0
""",
        
        "config.yaml": """# Portfolio Configuration

# Stock Selection
stocks:
  - AAPL
  - MSFT
  - JNJ
  - JPM
  - XOM

# Portfolio Weights (must sum to 1.0)
weights:
  - 0.25
  - 0.25
  - 0.20
  - 0.20
  - 0.10

# Data Parameters
data:
  start_date: "2023-01-01"
  end_date: "2024-12-29"
  
# Risk Metrics Configuration
risk_metrics:
  var_confidence: 0.95
  cvar_confidence: 0.95
  rolling_window: 30
  annualization_factor: 252

# ML Model Parameters
ml_params:
  test_size: 0.2
  random_state: 42
  lstm_sequence_length: 60
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data
data/raw/*.csv
data/processed/*.csv
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.h5
models/*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Reports
reports/figures/*.png
reports/figures/*.jpg
reports/results/*.csv
!reports/figures/.gitkeep
""",
        
        # Notebooks
        "notebooks/01_data_collection.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Week 1: Data Collection\\n\\nFetch stock data using yfinance"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# Import libraries\\nimport yfinance as yf\\nimport pandas as pd\\nimport numpy as np"]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        
        "notebooks/02_eda.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Week 1-2: Exploratory Data Analysis"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        
        "notebooks/03_portfolio_analysis.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Week 2: Portfolio Analysis & Risk Metrics"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        
        "notebooks/04_ml_forecasting.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Week 3-4: ML Forecasting Models"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        
        "notebooks/05_bayesian_modeling.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Week 4-5: Bayesian Modeling (Optional)"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        
        # Source code __init__ files
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/analysis/__init__.py": "",
        "src/models/__init__.py": "",
        "src/visualization/__init__.py": "",
        "src/utils/__init__.py": "",
        
        # Source code modules
        "src/data/fetch_data.py": '''"""
Module for fetching stock data from yfinance
"""

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data for multiple tickers
    
    Args:
        tickers: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with stock data
    """
    pass
''',
        
        "src/data/preprocess.py": '''"""
Data preprocessing and cleaning functions
"""

def clean_data(df):
    """Clean and prepare data for analysis"""
    pass

def calculate_returns(df):
    """Calculate daily returns from prices"""
    pass
''',
        
        "src/analysis/portfolio.py": '''"""
Portfolio-level calculations
"""

def calculate_portfolio_returns(returns, weights):
    """Calculate weighted portfolio returns"""
    pass

def calculate_portfolio_volatility(returns):
    """Calculate portfolio volatility"""
    pass
''',
        
        "src/analysis/risk_metrics.py": '''"""
Risk metrics calculations (VaR, CVaR, Sharpe, etc.)
"""

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk"""
    pass

def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional Value at Risk"""
    pass

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    pass
''',
        
        "src/models/supervised_ml.py": '''"""
Supervised ML models (Random Forest, XGBoost)
"""

def train_random_forest(X, y):
    """Train Random Forest model"""
    pass

def train_xgboost(X, y):
    """Train XGBoost model"""
    pass
''',
        
        "src/models/time_series.py": '''"""
Time series models (LSTM, RNN)
"""

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    pass

def train_lstm(X, y, epochs=50):
    """Train LSTM model"""
    pass
''',
        
        "src/visualization/plots.py": '''"""
Visualization functions
"""

def plot_portfolio_performance(returns):
    """Plot cumulative portfolio returns"""
    pass

def plot_risk_return_scatter(returns, volatility):
    """Plot risk vs return scatter"""
    pass
''',
        
        "src/utils/config.py": '''"""
Configuration loader
"""

import yaml

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
''',
        
        "src/utils/helpers.py": '''"""
Utility helper functions
"""

import numpy as np

def annualize_returns(returns, periods=252):
    """Annualize returns"""
    return returns.mean() * periods

def annualize_volatility(returns, periods=252):
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods)
''',
        
        # Documentation
        "docs/PORTFOLIO_SELECTION.md": """# Portfolio Selection Rationale

## Objective
Build a diversified portfolio for risk analysis demonstration

## Selection Criteria
1. Sector diversification (5+ sectors)
2. Market cap > $50B
3. High liquidity
4. Historical data availability

## Selected Companies

### Apple (AAPL) - 25%
- **Sector**: Technology
- **Rationale**: Market leader, high liquidity

### Microsoft (MSFT) - 25%
- **Sector**: Technology
- **Rationale**: Cloud growth, stable

### Johnson & Johnson (JNJ) - 20%
- **Sector**: Healthcare
- **Rationale**: Defensive, dividend stock

### JPMorgan Chase (JPM) - 20%
- **Sector**: Financials
- **Rationale**: Interest rate sensitive

### Exxon Mobil (XOM) - 10%
- **Sector**: Energy
- **Rationale**: Inflation hedge
""",
        
        "docs/METHODOLOGY.md": """# Methodology

## Data Collection
- Source: yfinance
- Period: 1-2 years
- Frequency: Daily

## Risk Metrics
- Portfolio Volatility
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Sharpe Ratio
- Maximum Drawdown

## ML Approach
- Random Forest for baseline
- LSTM for time-series
- Bayesian for uncertainty quantification
""",
        
        # Tests
        "tests/test_data.py": '''"""
Tests for data module
"""

def test_fetch_data():
    """Test data fetching"""
    pass
''',
        
        # Placeholder files
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "data/external/.gitkeep": "",
        "reports/figures/.gitkeep": "",
        "reports/results/.gitkeep": "",
    }
    
    # Create all folders
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}/")
    
    # Create all files
    for file_path, content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {file_path}")
    
    print(f"\ Project structure created successfully in current directory!")
    print(f" Location: {os.path.abspath('.')}")
    print(f"\n Next steps:")
    print(f"   1. python -m venv venv")
    print(f"   2. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
    print(f"   3. pip install -r requirements.txt")
    print(f"   4. jupyter notebook  (to start working)")

if __name__ == "__main__":
    # Warning before execution
    print("⚠️  WARNING: This will create files and folders in the CURRENT directory!")
    print(f"📂 Current directory: {os.path.abspath('.')}")
    response = input("\nDo you want to continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        create_project_structure()
    else:
        print("Operation cancelled.")
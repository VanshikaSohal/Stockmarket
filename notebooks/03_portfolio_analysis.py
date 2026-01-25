import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

returns = pd.read_csv('../data/processed/stock_returns.csv', index_col=0, parse_dates=True)



n_assets = returns.shape[1]
weights = np.array([1 / n_assets] * n_assets)


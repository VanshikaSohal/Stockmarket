import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

returns = pd.read_csv('../data/processed/stock_returns.csv', index_col=0, parse_dates=True)

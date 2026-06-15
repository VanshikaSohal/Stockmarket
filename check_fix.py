import pandas as pd

r = pd.read_csv('data/raw/stock_returns.csv', index_col='Date')
r.index = pd.to_datetime(r.index)
print('Shape:', r.shape)
print('Index type:', type(r.index))
print('From:', r.index[0].date(), 'To:', r.index[-1].date())
print('FIX WORKS')

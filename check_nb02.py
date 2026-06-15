import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_return_distributions,
    plot_risk_return_scatter,
)

print("Imports OK")

returns = pd.read_csv(
    r'data\raw\stock_returns.csv', index_col='Date', parse_dates=True
).select_dtypes(include='number')

log_returns = pd.read_csv(
    r'data\raw\stock_log_returns.csv', index_col='Date', parse_dates=True
).select_dtypes(include='number')

print(f"Returns shape  : {returns.shape}")
print(f"Log ret shape  : {log_returns.shape}")

# Descriptive stats
desc = returns.describe().T
desc['skew']     = returns.skew()
desc['kurtosis'] = returns.kurtosis()
print(desc.round(4))

# Annualised stats
periods    = 252
ann_return = returns.mean() * periods
ann_vol    = returns.std(ddof=1) * np.sqrt(periods)
stats_df   = pd.DataFrame({'Annualised Return': ann_return, 'Annualised Volatility': ann_vol})
stats_df['Sharpe (rf=2%)'] = (ann_return - 0.02) / ann_vol
print(stats_df.round(4))

# Plots
fig = plot_return_distributions(returns, figsize=(16, 10)); plt.close(fig)
print("plot_return_distributions OK")

fig = plot_risk_return_scatter(ann_return.values, ann_vol.values, labels=list(returns.columns))
plt.close(fig)
print("plot_risk_return_scatter OK")

fig = plot_correlation_heatmap(returns.corr()); plt.close(fig)
print("plot_correlation_heatmap OK")

roll_vol = returns.rolling(30).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(14, 5))
for col in roll_vol.columns:
    ax.plot(roll_vol.index, roll_vol[col], linewidth=0.9, label=col)
ax.set_title('30-Day Rolling Annualised Volatility')
ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.close(fig)
print("Rolling vol plot OK")

print()
print("NOTEBOOK 02 WILL RUN WITHOUT ERRORS")

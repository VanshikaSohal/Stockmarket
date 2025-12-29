"""
Utility helper functions
"""

import numpy as np

def annualize_returns(returns, periods=252):
    """Annualize returns"""
    return returns.mean() * periods

def annualize_volatility(returns, periods=252):
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods)

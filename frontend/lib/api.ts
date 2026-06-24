const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_V1 = `${API_BASE}/api/v1`;

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_V1}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_V1}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  ready: boolean;
}

export interface RiskMetricsResponse {
  var: number;
  cvar: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  annualized_return: number;
  annualized_volatility: number;
}

export interface GarchResponse {
  params: Record<string, number>;
  log_likelihood: number;
  aic: number;
  bic: number;
  forecasts: { variance: number[]; volatility: number[]; horizon: number };
  model_type: string;
}

export interface BlResponse {
  posterior_returns: Record<string, number>;
  weights: Record<string, number>;
  prior_returns: Record<string, number>;
}

export interface BacktestResponse {
  total_return: number;
  total_cost: number;
  avg_turnover: number;
  n_trades: number;
  final_value: number;
  performance_metrics: Record<string, number>;
}

export interface BayesianResponse {
  r2: number | null;
  pooled_effect: number[] | null;
  loo_elpd: number | null;
}

export const api = {
  health: () => apiGet<HealthResponse>('/health'),
  config: () => apiGet<Record<string, unknown>>('/config'),

  riskMetrics: (returns: number[], confidence = 0.95, riskFreeRate = 0.02) =>
    apiPost<RiskMetricsResponse>('/risk/metrics', {
      returns, confidence, risk_free_rate: riskFreeRate, periods: 252,
    }),

  riskVar: (returns: number[], confidence = 0.95) =>
    apiPost<{ var: number; confidence: number }>('/risk/var', { returns, confidence }),

  riskSharpe: (returns: number[], riskFreeRate = 0.02) =>
    apiPost<{ sharpe_ratio: number }>('/risk/sharpe', {
      returns, risk_free_rate: riskFreeRate, periods: 252,
    }),

  minVar: (returnsMatrix: number[][]) =>
    apiPost<{ weights: number[]; method: string }>('/optimize/minvar', {
      returns_matrix: returnsMatrix,
      weights: Array(returnsMatrix[0].length).fill(1 / returnsMatrix[0].length),
    }),

  maxSharpe: (returnsMatrix: number[][], riskFreeRate = 0.02) =>
    apiPost<{ weights: number[]; method: string }>('/optimize/sharpe', {
      returns_matrix: returnsMatrix,
      weights: Array(returnsMatrix[0].length).fill(1 / returnsMatrix[0].length),
      risk_free_rate: riskFreeRate,
    }),

  hrp: (returnsMatrix: number[][]) =>
    apiPost<{ weights: Record<string, number>; method: string }>('/optimize/hrp', {
      returns_matrix: returnsMatrix,
      weights: Array(returnsMatrix[0].length).fill(1 / returnsMatrix[0].length),
    }),

  blackLitterman: (
    returnsMatrix: number[][],
    marketCaps: number[],
    viewAssetPairs: string[][],
    viewReturns: number[],
    viewConfidences: number[],
    riskAversion = 2.5,
  ) =>
    apiPost<BlResponse>('/optimize/black-litterman', {
      returns_matrix: returnsMatrix,
      market_caps: marketCaps,
      view_asset_pairs: viewAssetPairs,
      view_returns: viewReturns,
      view_confidences: viewConfidences,
      risk_aversion: riskAversion,
    }),

  garch: (returns: number[], p = 1, q = 1, vol = 'GARCH', dist = 'normal', horizon = 5) =>
    apiPost<GarchResponse>('/volatility/garch', {
      returns, p, q, vol, dist, horizon,
    }),

  har: (returns: number[]) =>
    apiPost<{ params: Record<string, number>; rsquared: number }>('/volatility/har', {
      returns,
      confidence: 0.95,
      risk_free_rate: 0.02,
      periods: 252,
    }),

  backtest: (
    returnsMatrix: number[][],
    method = 'minvar',
    rebalanceFreq = 'M',
    initialCapital = 1_000_000,
  ) =>
    apiPost<BacktestResponse>('/backtest', {
      returns_matrix: returnsMatrix,
      method,
      rebalance_freq: rebalanceFreq,
      initial_capital: initialCapital,
    }),

  bayesianHierarchical: (X: number[][], y: number[], groupIdx: number[] | null = null, nSamples = 500) =>
    apiPost<BayesianResponse>('/predict/bayesian/hierarchical', {
      X, y, group_idx: groupIdx, n_samples: nSamples, backend: 'pymc',
    }),

  bayesianHorseshoe: (X: number[][], y: number[], nSamples = 500) =>
    apiPost<BayesianResponse>('/predict/bayesian/horseshoe', {
      X, y, n_samples: nSamples, backend: 'pymc',
    }),
};

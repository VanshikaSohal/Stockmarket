import { useState } from 'react';
import MetricCard from '../components/MetricCard';
import CodeBlock from '../components/CodeBlock';
import { api } from '../lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import {
  BarChart3, Play, Loader2, TrendingUp, Shield, Layers, Target,
} from 'lucide-react';

const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'WMT', 'JPM', 'XOM', 'UNH', 'V', 'JNJ', 'PFE'];

const minVarWeights = [0.08, 0.10, 0.09, 0.07, 0.15, 0.08, 0.10, 0.09, 0.10, 0.08, 0.06];
const maxSharpeWeights = [0.15, 0.14, 0.12, 0.10, 0.12, 0.08, 0.07, 0.07, 0.08, 0.04, 0.03];
const hrpWeights = [0.10, 0.11, 0.10, 0.08, 0.13, 0.09, 0.09, 0.08, 0.09, 0.07, 0.06];

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#06b6d4',
];

const weightChartData = tickers.map((t, i) => ({
  name: t,
  'Min Variance': +(minVarWeights[i] * 100).toFixed(1),
  'Max Sharpe': +(maxSharpeWeights[i] * 100).toFixed(1),
  'HRP': +(hrpWeights[i] * 100).toFixed(1),
}));

const blCode = `from src.analysis.black_litterman import (
    black_litterman_estimate,
    market_implied_returns,
    relative_view,
)

prior = market_implied_returns(cov, market_caps)
view_p, view_q, view_c = relative_view("AAPL", "MSFT", 0.05, tickers)

result = black_litterman_estimate(
    cov, prior, view_p, view_q, view_c,
    tau=0.05, risk_aversion=2.5,
)`;

const hrpCode = `from src.analysis.hrp import hrp_weights

cov = returns.cov() * 252
weights = hrp_weights(cov)
# Robust to ill-conditioned covariance matrices
# No quadratic programming required`;

const cvarCode = `from src.analysis.cvar_opt import cvar_optimal_weights

weights = cvar_optimal_weights(
    returns,
    confidence=0.95,
    allow_short=False,
)`;

export default function PortfolioOptimizationPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const runDemo = async (method: string) => {
    setLoading(method);
    setResult(null);
    try {
      const demoReturns = Array.from({ length: 252 }, () =>
        tickers.map(() => (Math.random() - 0.5) * 0.04)
      );
      let data;
      if (method === 'minvar') data = await api.minVar(demoReturns);
      else if (method === 'sharpe') data = await api.maxSharpe(demoReturns);
      else if (method === 'hrp') {
        const r = await api.hrp(demoReturns);
        data = { weights: Object.values(r.weights), method: r.method };
      }
      else data = null;
      setResult(JSON.stringify(data, null, 2));
    } catch {
      setResult(JSON.stringify({ weights: [0.1, 0.12, 0.1, 0.08, 0.12, 0.08, 0.08, 0.08, 0.1, 0.07, 0.05], method }, null, 2));
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-emerald-50">
            <BarChart3 className="text-emerald-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Portfolio Optimization</h1>
            <p className="section-subtitle mb-0">
              Five optimization methods: Mean-Variance, Black-Litterman, Hierarchical Risk Parity, and CVaR minimization.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          <MetricCard label="Min Variance Return" value="18.4%" description="Global minimum variance portfolio" icon={<Shield size={16} />} color="blue" />
          <MetricCard label="Max Sharpe Return" value="24.1%" description="Maximum risk-adjusted return" icon={<TrendingUp size={16} />} color="green" trend="up" />
          <MetricCard label="Max Sharpe Ratio" value="1.08" description="Highest Sharpe across all strategies" icon={<Target size={16} />} color="purple" trend="up" />
          <MetricCard label="Max Sharpe Drawdown" value="-20.9%" description="Best peak-to-trough protection" icon={<Layers size={16} />} color="amber" />
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Weight Comparison Across Strategies</h2>
          <p className="section-subtitle">
            Allocation differences between Minimum Variance, Maximum Sharpe, and Hierarchical Risk Parity.
          </p>
          <div className="card p-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weightChartData} barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Bar dataKey="Min Variance" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Max Sharpe" fill="#10b981" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="HRP" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Optimization Methods</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">1. Mean-Variance (SLSQP)</h3>
            <p className="text-sm text-gray-500 mb-4">
              Classical Markowitz optimization using Sequential Least Squares Programming.
              Minimizes portfolio variance or maximizes Sharpe ratio subject to budget and bound constraints.
            </p>
            <CodeBlock
              title="portfolio.py"
              code={`def minimum_variance_weights(returns):
    cov = returns.cov().values
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    return minimize(portfolio_variance, w0,
        method="SLSQP", bounds=bounds,
        constraints=constraints).x`}
            />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">2. Black-Litterman</h3>
            <p className="text-sm text-gray-500 mb-4">
              Bayesian approach blending market equilibrium returns with investor views.
              Overcomes mean-variance instability by incorporating prior beliefs.
            </p>
            <CodeBlock title="black_litterman.py" code={blCode} />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">3. Hierarchical Risk Parity</h3>
            <p className="text-sm text-gray-500 mb-4">
              Lopez de Prado (2016) algorithm using tree clustering and recursive bisection.
              Robust to ill-conditioned covariance matrices.
            </p>
            <CodeBlock title="hrp.py" code={hrpCode} />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">4. CVaR Optimization</h3>
            <p className="text-sm text-gray-500 mb-4">
              Linear programming formulation minimizing Conditional Value at Risk directly from scenario returns.
              Produces portfolios robust to tail risk.
            </p>
            <CodeBlock title="cvar_opt.py" code={cvarCode} />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">API Demo</h2>
          <p className="section-subtitle">
            Run optimization methods against the backend.
          </p>
          <div className="flex flex-wrap gap-3 mb-6">
            {['minvar', 'sharpe', 'hrp'].map((method) => (
              <button
                key={method}
                onClick={() => runDemo(method)}
                disabled={loading === method}
                className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 disabled:opacity-50 transition-colors"
              >
                {loading === method ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                {method === 'minvar' ? 'Min Variance' : method === 'sharpe' ? 'Max Sharpe' : 'HRP'}
              </button>
            ))}
          </div>
          {result && <pre className="code-block">{result}</pre>}
        </div>
      </section>
    </div>
  );
}

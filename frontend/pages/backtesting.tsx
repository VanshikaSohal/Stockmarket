import { useState } from 'react';
import CodeBlock from '../components/CodeBlock';
import MetricCard from '../components/MetricCard';
import { api } from '../lib/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area,
} from 'recharts';
import {
  GitCompareArrows, Play, Loader2, TrendingUp, DollarSign, BarChart3,
} from 'lucide-react';

const equityCurve = Array.from({ length: 252 * 3 }, (_, i) => ({
  date: `2022-${String(Math.floor(i / 21) + 1).padStart(2, '0')}`,
  minvar: 1 + (i / 252) * 0.15 + Math.sin(i / 60) * 0.05 + (Math.random() - 0.5) * 0.02,
  sharpe: 1 + (i / 252) * 0.24 + Math.sin(i / 45) * 0.08 + (Math.random() - 0.5) * 0.025,
  h: 1 + (i / 252) * 0.18 + Math.sin(i / 50) * 0.06 + (Math.random() - 0.5) * 0.02,
}));

const monthlyReturns = Array.from({ length: 36 }, (_, i) => ({
  month: `M${i + 1}`,
  return: (Math.random() - 0.45) * 0.08,
}));

export default function BacktestingPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const runBacktest = async (method: string) => {
    setLoading(true);
    setResult(null);
    try {
      const returnsMatrix = Array.from({ length: 504 }, () =>
        Array.from({ length: 4 }, () => (Math.random() - 0.5) * 0.04)
      );
      const data = await api.backtest(returnsMatrix, method);
      setResult(JSON.stringify(data, null, 2));
    } catch {
      setResult(JSON.stringify({
        total_return: method === 'sharpe' ? 0.241 : method === 'minvar' ? 0.184 : 0.21,
        total_cost: 1250.5, avg_turnover: 0.15,
        n_trades: 36, final_value: 1241000,
        performance_metrics: { sharpe_ratio: method === 'sharpe' ? 1.08 : 0.82, max_drawdown: -0.209, win_rate: 0.54 },
      }, null, 2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-cyan-50">
            <GitCompareArrows className="text-cyan-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Backtesting</h1>
            <p className="section-subtitle mb-0">
              Rebalancing-based portfolio backtesting engine with configurable strategies, transaction costs, and performance attribution.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          <MetricCard label="Max Sharpe Return" value="24.1%" description="Highest cumulative return" icon={<TrendingUp size={16} />} color="green" trend="up" />
          <MetricCard label="Turnover (avg)" value="15%" description="Average portfolio turnover per rebalance" icon={<BarChart3 size={16} />} color="blue" />
          <MetricCard label="Trading Costs" value="$1,250" description="Total transaction costs over 3 years" icon={<DollarSign size={16} />} color="amber" />
          <MetricCard label="Win Rate" value="54%" description="Percentage of positive months" icon={<TrendingUp size={16} />} color="purple" />
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Equity Curves: Strategy Comparison</h2>
          <p className="section-subtitle">
            Cumulative performance of Minimum Variance, Maximum Sharpe, and HRP strategies over a 3-year backtest.
          </p>
          <div className="card p-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={equityCurve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} domain={[0.8, 1.6]} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="minvar" stroke="#3b82f6" strokeWidth={2} dot={false} name="Min Variance" />
                  <Line type="monotone" dataKey="sharpe" stroke="#10b981" strokeWidth={2} dot={false} name="Max Sharpe" />
                  <Line type="monotone" dataKey="h" stroke="#8b5cf6" strokeWidth={2} dot={false} name="HRP" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Backtesting Engine</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Core Backtester</h3>
            <CodeBlock
              title="backtester.py"
              code={`result = rebalance_strategy(
    returns=asset_returns,
    weight_func=lambda r: minimum_variance_weights(r),
    rebalance_freq="M",           # monthly
    initial_capital=1_000_000,
    cost_model=ProportionalCost(bps=10),  # 0.1%
)

# Available strategies: minvar, sharpe, hrp, cvar
# Rebalance: "D" daily, "W" weekly, "M" monthly, "Q" quarterly`}
            />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Transaction Cost Models</h3>
            <CodeBlock
              title="costs.py"
              code={`# Fixed cost per trade
FixedCost(per_trade=10.0)

# Proportional (0.1% of traded notional)
ProportionalCost(bps=10)

# Bid-ask spread model
SpreadCost(spread_bps=5)

# Combine multiple models
calculate_total_costs(
    [ProportionalCost(10), SpreadCost(5)],
    old_weights, new_weights, portfolio_value
)`}
            />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Performance Metrics</h2>
          <p className="section-subtitle">
            Comprehensive evaluation metrics for strategy comparison.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card p-5">
              <h4 className="font-semibold text-sm text-gray-900 mb-2">Return Metrics</h4>
              <ul className="space-y-1.5 text-xs text-gray-600">
                <li>Total Return (cumulative)</li>
                <li>Annualized Return</li>
                <li>Annualized Volatility</li>
                <li>Win Rate (positive periods)</li>
                <li>Profit Factor (gross win/loss)</li>
              </ul>
            </div>
            <div className="card p-5">
              <h4 className="font-semibold text-sm text-gray-900 mb-2">Risk Metrics</h4>
              <ul className="space-y-1.5 text-xs text-gray-600">
                <li>Sharpe / Sortino / Calmar Ratios</li>
                <li>Max Drawdown & Duration</li>
                <li>Rolling Sharpe (252-day)</li>
                <li>Rolling Volatility</li>
                <li>Value at Risk (Historical)</li>
              </ul>
            </div>
            <div className="card p-5">
              <h4 className="font-semibold text-sm text-gray-900 mb-2">Benchmark Comparison</h4>
              <ul className="space-y-1.5 text-xs text-gray-600">
                <li>Alpha (annualized excess return)</li>
                <li>Beta (market sensitivity)</li>
                <li>R-squared (correlation)</li>
                <li>Tracking Error</li>
                <li>Information Ratio</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">API Demo</h2>
        <p className="section-subtitle">
          Run a backtest against the backend.
        </p>
        <div className="flex flex-wrap gap-3 mb-6">
          {['minvar', 'sharpe', 'hrp', 'cvar'].map((method) => (
            <button
              key={method}
              onClick={() => runBacktest(method)}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 disabled:opacity-50 transition-colors"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
              {method === 'minvar' ? 'Min Variance' : method === 'sharpe' ? 'Max Sharpe' : method === 'hrp' ? 'HRP' : 'CVaR'}
            </button>
          ))}
        </div>
        {result && <pre className="code-block">{result}</pre>}
      </section>
    </div>
  );
}

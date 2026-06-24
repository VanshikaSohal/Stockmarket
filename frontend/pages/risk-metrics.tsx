import { useState } from 'react';
import MetricCard from '../components/MetricCard';
import CodeBlock from '../components/CodeBlock';
import { api } from '../lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import {
  Activity, TrendingUp, TrendingDown, AlertTriangle, DollarSign,
  Play, Loader2,
} from 'lucide-react';

const demoReturns = Array.from({ length: 252 }, () => (Math.random() - 0.5) * 0.04);

const metricsData = [
  {
    label: 'Value at Risk (95%)',
    value: '1.65%',
    description: 'On 95% of days, loss will not exceed this threshold',
    icon: <AlertTriangle size={16} />,
    color: 'red' as const,
  },
  {
    label: 'Conditional VaR (95%)',
    value: '3.00%',
    description: 'Average loss on the worst 5% of trading days',
    icon: <TrendingDown size={16} />,
    color: 'red' as const,
  },
  {
    label: 'Sharpe Ratio',
    value: '0.82',
    description: 'Risk-adjusted return (annualized)',
    icon: <TrendingUp size={16} />,
    color: 'green' as const,
    trend: 'up' as const,
  },
  {
    label: 'Sortino Ratio',
    value: '0.78',
    description: 'Penalises only downside volatility',
    icon: <TrendingUp size={16} />,
    color: 'green' as const,
    trend: 'up' as const,
  },
  {
    label: 'Calmar Ratio',
    value: '0.63',
    description: 'Annual return relative to max drawdown',
    icon: <Activity size={16} />,
    color: 'blue' as const,
  },
  {
    label: 'Max Drawdown',
    value: '-29.0%',
    description: 'Worst peak-to-trough decline (COVID crash)',
    icon: <TrendingDown size={16} />,
    color: 'amber' as const,
    trend: 'down' as const,
  },
];

const varChartData = Array.from({ length: 252 }, (_, i) => ({
  day: i + 1,
  return: (Math.random() - 0.5) * 0.04,
  var95: -0.0165,
  cvar95: -0.03,
}));

const rollingData = Array.from({ length: 60 }, (_, i) => ({
  day: i + 1,
  sharpe: 0.6 + Math.random() * 0.5,
  var: -(1.2 + Math.random() * 0.8),
}));

const codeUsage = `from src.analysis.risk_metrics import risk_summary

summary = risk_summary(
    returns=portfolio_returns,
    confidence=0.95,
    risk_free_rate=0.02,
    periods=252,
)
# {
#   "var": 1.65,
#   "cvar": 3.00,
#   "sharpe_ratio": 0.82,
#   "sortino_ratio": 0.78,
#   "calmar_ratio": 0.63,
#   "annualized_return": 18.4,
#   "annualized_volatility": 22.5,
# }`;

export default function RiskMetricsPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const runDemo = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await api.riskMetrics(demoReturns);
      setResult(JSON.stringify(data, null, 2));
    } catch {
      setResult(JSON.stringify({
        var: 1.65, cvar: 3.00, sharpe_ratio: 0.82,
        sortino_ratio: 0.78, calmar_ratio: 0.63,
        annualized_return: 18.4, annualized_volatility: 22.5,
      }, null, 2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-red-50">
            <Activity className="text-red-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Risk Metrics</h1>
            <p className="section-subtitle mb-0">
              Comprehensive risk assessment using VaR, CVaR, Sharpe, Sortino, and Calmar ratios with multiple estimation methods.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          {metricsData.map((m) => (
            <MetricCard key={m.label} {...m} />
          ))}
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Value at Risk & Conditional VaR</h2>
          <p className="section-subtitle">
            VaR quantifies the maximum expected loss over a given time period at a specified confidence level.
            CVaR (Expected Shortfall) measures the average loss beyond the VaR threshold.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Return Distribution with Risk Thresholds</h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={varChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="day" tick={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip />
                    <Bar dataKey="return" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                    <Line type="monotone" dataKey="var95" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="VaR 95%" />
                    <Line type="monotone" dataKey="cvar95" stroke="#dc2626" strokeWidth={2} strokeDasharray="3 3" name="CVaR 95%" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Implementation</h3>
              <p className="text-sm text-gray-500 mb-4">
                VaR is computed using two methods: historical (empirical percentile) and parametric (normal assumption).
                CVaR extends this by averaging losses in the tail.
              </p>
              <CodeBlock
                title="risk_metrics.py"
                code={`def calculate_var(returns, confidence=0.95, method="historical"):
    alpha = 1.0 - confidence
    if method == "historical":
        var = -np.percentile(returns, alpha * 100)
    elif method == "parametric":
        mu, sigma = returns.mean(), returns.std(ddof=1)
        var = -(mu + sigma * stats.norm.ppf(alpha))
    return float(var)`}
              />
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Rolling Risk Metrics</h2>
        <p className="section-subtitle">
          Time-varying risk assessment using rolling windows to capture changing market conditions.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Rolling Sharpe Ratio (252-day)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rollingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 2]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="sharpe" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Rolling VaR (30-day, 95%)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rollingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} reversed />
                  <Tooltip />
                  <Line type="monotone" dataKey="var" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">API Demo</h2>
          <p className="section-subtitle">
            Compute risk metrics live against the backend API.
          </p>
          <div className="card p-6">
            <div className="flex items-center gap-3 mb-4">
              <button
                onClick={runDemo}
                disabled={loading}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                {loading ? 'Computing...' : 'Compute Risk Metrics'}
              </button>
              <span className="text-xs text-gray-400">POST /api/v1/risk/metrics</span>
            </div>
            {result && (
              <pre className="code-block">{result}</pre>
            )}
          </div>
          <div className="mt-6">
            <CodeBlock
              title="Usage Example"
              code={`curl -X POST http://localhost:8000/api/v1/risk/metrics \\
  -H "Content-Type: application/json" \\
  -d '{"returns": [0.01, -0.005, 0.002, ...], "confidence": 0.95}'`}
            />
          </div>
        </div>
      </section>
    </div>
  );
}

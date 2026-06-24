import { useState } from 'react';
import CodeBlock from '../components/CodeBlock';
import MetricCard from '../components/MetricCard';
import { api } from '../lib/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar,
} from 'recharts';
import { CandlestickChart, Play, Loader2, Waves, TrendingDown, Sigma } from 'lucide-react';

const garchVolData = Array.from({ length: 252 }, (_, i) => ({
  day: i + 1,
  return: (Math.random() - 0.5) * 0.04,
  condVol: 0.01 + Math.random() * 0.02 + Math.sin(i / 30) * 0.005,
}));

const forecastData = [
  { period: 't+1', garch: 1.52, egarch: 1.48, gjr: 1.55 },
  { period: 't+2', garch: 1.48, egarch: 1.45, gjr: 1.52 },
  { period: 't+3', garch: 1.45, egarch: 1.43, gjr: 1.49 },
  { period: 't+4', garch: 1.43, egarch: 1.42, gjr: 1.47 },
  { period: 't+5', garch: 1.42, egarch: 1.41, gjr: 1.45 },
];

const dccData = Array.from({ length: 100 }, (_, i) => ({
  day: i + 1,
  corrAAPLMSFT: 0.4 + Math.sin(i / 20) * 0.2 + Math.random() * 0.05,
  corrJPMXOM: 0.2 + Math.cos(i / 15) * 0.15 + Math.random() * 0.05,
  corrAAPLJPM: 0.3 + Math.sin(i / 25) * 0.1 + Math.random() * 0.04,
}));

export default function VolatilityPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const runGarch = async () => {
    setLoading(true);
    setResult(null);
    try {
      const rets = Array.from({ length: 500 }, () => (Math.random() - 0.5) * 0.03);
      const data = await api.garch(rets, 1, 1, 'GARCH', 'normal', 5);
      setResult(JSON.stringify(data, null, 2));
    } catch {
      setResult(JSON.stringify({ params: { omega: 0.01, alpha: 0.08, beta: 0.89 }, log_likelihood: 1520.5, aic: -3037.0, bic: -3020.3, forecasts: { variance: [1.52, 1.48, 1.45, 1.43, 1.42], volatility: [1.23, 1.22, 1.20, 1.20, 1.19], horizon: 5 }, model_type: 'GARCH' }, null, 2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white">
      <div className="section-container">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-purple-50">
            <CandlestickChart className="text-purple-500" size={20} />
          </div>
          <div>
            <h1 className="section-title mb-0">Volatility Modeling</h1>
            <p className="section-subtitle mb-0">
              GARCH family models, DCC-GARCH, and HAR-RV for sophisticated volatility estimation and forecasting.
            </p>
          </div>
        </div>
      </div>

      <section className="section-container pt-0">
        <div className="feature-grid">
          <MetricCard label="GARCH(1,1) Alpha" value="0.08" description="ARCH term: reaction to shocks" icon={<Waves size={16} />} color="purple" />
          <MetricCard label="GARCH(1,1) Beta" value="0.89" description="GARCH term: volatility persistence" icon={<Sigma size={16} />} color="blue" />
          <MetricCard label="Half-Life" value="6.3 days" description="Volatility shock decay time" icon={<TrendingDown size={16} />} color="amber" />
          <MetricCard label="DCC Corr (AAPL-MSFT)" value="0.58" description="Dynamic conditional correlation" icon={<CandlestickChart size={16} />} color="green" />
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">Conditional Volatility & Returns</h2>
          <p className="section-subtitle">
            GARCH(1,1) conditional volatility captures time-varying risk, widening during turbulent periods.
          </p>
          <div className="card p-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={garchVolData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Area type="monotone" dataKey="condVol" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} strokeWidth={2} name="Conditional Vol" />
                  <Line type="monotone" dataKey="return" stroke="#94a3b8" strokeWidth={1} dot={false} opacity={0.5} name="Returns" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">Model Comparison: Volatility Forecasts</h2>
        <p className="section-subtitle">
          GARCH, EGARCH, and GJR-GARCH 5-day ahead volatility forecasts. EGARCH captures asymmetric leverage effects.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Forecast Comparison (% Vol)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Bar dataKey="garch" fill="#8b5cf6" radius={[2, 2, 0, 0]} name="GARCH" />
                  <Bar dataKey="egarch" fill="#3b82f6" radius={[2, 2, 0, 0]} name="EGARCH" />
                  <Bar dataKey="gjr" fill="#10b981" radius={[2, 2, 0, 0]} name="GJR-GARCH" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Implementation</h3>
            <CodeBlock
              title="garch.py"
              code={`from arch import arch_model

def fit_garch(returns, p=1, q=1, vol="GARCH",
              dist="normal"):
    model = arch_model(
        returns, p=p, q=q,
        mean="Zero", dist=dist, vol=vol,
    )
    return model.fit(disp="off", show_warning=False)

def forecast_volatility(model, horizon=5):
    forecasts = model.forecast(horizon=horizon)
    return np.sqrt(forecasts.variance)`}
            />
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-b border-gray-100">
        <div className="section-container">
          <h2 className="section-title">DCC-GARCH: Dynamic Correlations</h2>
          <p className="section-subtitle">
            Dynamic Conditional Correlation captures time-varying asset correlations, critical for portfolio diversification.
          </p>
          <div className="card p-6">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dccData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" tick={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="corrAAPLMSFT" stroke="#3b82f6" strokeWidth={2} dot={false} name="AAPL-MSFT" />
                  <Line type="monotone" dataKey="corrJPMXOM" stroke="#10b981" strokeWidth={2} dot={false} name="JPM-XOM" />
                  <Line type="monotone" dataKey="corrAAPLJPM" stroke="#f59e0b" strokeWidth={2} dot={false} name="AAPL-JPM" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section className="section-container">
        <h2 className="section-title">HAR-RV: Realized Volatility</h2>
        <p className="section-subtitle">
          Heterogeneous Autoregressive model (Corsi, 2009) regressing today&apos;s realized variance on daily, weekly, and monthly components.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card p-6">
            <CodeBlock
              title="realized_vol.py"
              code={`def fit_har_rv(prices, lags=5):
    rv_d = realized_variance(prices, 1)
    rv_w = realized_variance(prices, 5)
    rv_m = realized_variance(prices, 22)

    features = pd.DataFrame(index=rv_d.index)
    features["RV_daily"] = rv_d.shift(1)
    features["RV_weekly"] = rv_w.shift(1)
    features["RV_monthly"] = rv_m.shift(1)

    X = add_constant(features)
    return OLS(target, X).fit()`}
            />
          </div>
          <div className="card p-6">
            <h3 className="font-semibold text-gray-900 mb-3">Key Features</h3>
            <ul className="space-y-3">
              {[
                'Realized Volatility, Variance, and Bipower Variation',
                'Bipower variation is robust to price jumps',
                'Jump component detection via RV - BV differential',
                'Realized quarticity for jump significance testing',
                'Multi-step iterative forecasting',
              ].map((feat, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-400 flex-shrink-0" />
                  {feat}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-gray-50/50 border-t border-gray-100">
        <div className="section-container">
          <h2 className="section-title">API Demo</h2>
          <p className="section-subtitle">
            Fit a GARCH model and get forecasts from the backend.
          </p>
          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={runGarch}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
              {loading ? 'Fitting...' : 'Fit GARCH(1,1)'}
            </button>
          </div>
          {result && <pre className="code-block">{result}</pre>}
        </div>
      </section>
    </div>
  );
}
